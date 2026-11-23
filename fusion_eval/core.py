"""Exact, memory-bounded searches for three two-model fusion families."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch

GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
FAMILIES = ("global", "dead", "hierarchical")


@dataclass(frozen=True)
class Candidate:
    family: str
    a: float
    b: float

    def record(self):
        return asdict(self)


def candidates(grid=GRID):
    return ([Candidate("global", a, a) for a in grid]
            + [Candidate("dead", a, b) for a in grid for b in grid]
            + [Candidate("hierarchical", a, b) for a in grid for b in grid])


def health_lookup(metadata=None):
    if metadata is None:
        metadata = json.loads((Path(__file__).resolve().parents[1]
                               / "coralscapes_metadata/v2.json").read_text())
    legacy_health = [0] * 40
    for i in (0, 21, 27):
        legacy_health[i] = 4  # Unmapped predictions remain errors for valid targets.
    for group, ids in ((1, (6, 17, 22, 25, 28, 31, 34, 36)),
                       (2, (4, 16, 19, 33)), (3, (3, 20, 23, 32, 37))):
        for i in ids:
            legacy_health[i] = group
    result = torch.full((96,), 4, dtype=torch.long)
    for source, target in metadata["fine_to_legacy_id"].items():
        result[int(source)] = legacy_health[int(target)]
    return result


def blend(first, second, weight):
    if weight == 1:
        return first
    if weight == 0:
        return second
    return first * weight + second * (1 - weight)


def maximum(values, ids):
    scores, positions = values.max(dim=0)
    return scores, ids[positions]


def choose(first, second):
    """Match torch.argmax's smallest-native-ID tie break across disjoint groups."""
    sa, ia = first
    sb, ib = second
    use_first = (sa > sb) | ((sa == sb) & (ia < ib))
    return torch.where(use_first, sa, sb), torch.where(use_first, ia, ib)


def conditional(probabilities):
    mass = probabilities.sum(dim=0)
    safe = probabilities / mass.clamp_min(torch.finfo(probabilities.dtype).tiny)
    safe = torch.where(mass.unsqueeze(0) > 0, safe,
                       torch.full_like(safe, 1 / probabilities.shape[0]))
    return mass, safe


def predict_all(first, second, groups, grid=GRID):
    """Yield all 55 exact native argmax maps without materializing 55x96 scores.

    Inputs are class probabilities [C,H,W]. E2 normalization is omitted because
    its positive per-pixel normalizer cannot change argmax. E3 uses normalized
    conditional distributions, including a separate unmapped/void group.
    """
    if first.shape != second.shape or first.ndim != 3 or len(groups) != first.shape[0]:
        raise ValueError("Expected equal [C,H,W] maps and one group per native class")
    groups = groups.to(first.device)
    all_ids = torch.arange(len(groups), device=first.device)
    dead_ids, rest_ids = all_ids[groups == 3], all_ids[groups != 3]
    if not len(dead_ids) or not len(rest_ids):
        raise ValueError("Both Dead and non-Dead classes are required")
    dead, rest = {}, {}
    fd, sd = first[dead_ids], second[dead_ids]
    fr, sr = first[rest_ids], second[rest_ids]
    for weight in grid:
        dead[weight] = maximum(blend(fd, sd, weight), dead_ids)
        rest[weight] = maximum(blend(fr, sr, weight), rest_ids)
    for weight in grid:
        yield Candidate("global", weight, weight), choose(dead[weight], rest[weight])[1]
    for a in grid:
        for b in grid:
            yield Candidate("dead", a, b), choose(dead[a], rest[b])[1]
    del dead, rest, fd, sd, fr, sr

    marginal_first, marginal_second, conditional_max = {}, {}, {}
    for group in sorted(set(groups.tolist())):
        ids = all_ids[groups == group]
        marginal_first[group], cf = conditional(first[ids])
        marginal_second[group], cs = conditional(second[ids])
        conditional_max[group] = {b: maximum(blend(cf, cs, b), ids) for b in grid}
    for a in grid:
        masses = {g: blend(marginal_first[g], marginal_second[g], a)
                  for g in conditional_max}
        for b in grid:
            winner = None
            for group in conditional_max:
                score, label = conditional_max[group][b]
                current = (masses[group] * score, label)
                winner = current if winner is None else choose(winner, current)
            yield Candidate("hierarchical", a, b), winner[1]


def fuse(first, second, groups, candidate):
    """Full probability map for a selected inference recipe."""
    if candidate.family == "global":
        return blend(first, second, candidate.a)
    groups = groups.to(first.device)
    if candidate.family == "dead":
        weights = torch.where(groups == 3, candidate.a, candidate.b)[:, None, None]
        scores = first * weights + second * (1 - weights)
        total = scores.sum(0, keepdim=True)
        return torch.where(total > 0, scores / total.clamp_min(1e-30),
                           torch.full_like(scores, 1 / scores.shape[0]))
    if candidate.family != "hierarchical":
        raise ValueError(candidate.family)
    result = torch.zeros_like(first)
    for group in sorted(set(groups.tolist())):
        ids = torch.where(groups == group)[0]
        hf, cf = conditional(first[ids])
        hs, cs = conditional(second[ids])
        result[ids] = blend(hf, hs, candidate.a) * blend(cf, cs, candidate.b)
    return result


def predict_selected(first, second, groups, candidate):
    """Use the same max-before-normalization arithmetic as the grid search."""
    groups = groups.to(first.device)
    ids = torch.arange(len(groups), device=first.device)
    if candidate.family == "global":
        return blend(first, second, candidate.a).argmax(0)
    if candidate.family == "dead":
        dead, rest = ids[groups == 3], ids[groups != 3]
        return choose(maximum(blend(first[dead], second[dead], candidate.a), dead),
                      maximum(blend(first[rest], second[rest], candidate.b), rest))[1]
    if candidate.family != "hierarchical":
        raise ValueError(candidate.family)
    winner = None
    for group in sorted(set(groups.tolist())):
        selected = ids[groups == group]
        hf, cf = conditional(first[selected])
        hs, cs = conditional(second[selected])
        score, label = maximum(blend(cf, cs, candidate.b), selected)
        current = (blend(hf, hs, candidate.a) * score, label)
        winner = current if winner is None else choose(winner, current)
    return winner[1]


class ConfusionCounter:
    """Reuse target encoding and shard CUDA counters to avoid hot-bin contention."""
    def __init__(self, target, classes=96, blocks=None):
        self.valid = target != 0
        self.classes = classes
        labels = target[self.valid].long()
        self.blocks = blocks if blocks is not None else (32 if target.is_cuda and labels.numel() > 65536 else 1)
        if self.blocks < 1:
            raise ValueError("Counter shard count must be positive")
        offsets = torch.arange(labels.numel(), device=target.device) // max(1, (labels.numel() + self.blocks - 1) // self.blocks)
        self.codes = labels * classes + offsets * classes * classes

    def __call__(self, prediction):
        encoded = self.codes + prediction[self.valid].long()
        return torch.bincount(encoded, minlength=self.blocks * self.classes * self.classes).reshape(
            self.blocks, self.classes, self.classes).sum(0)


def confusion(target, prediction, classes=96):
    return ConfusionCounter(target, classes)(prediction)


def summarize(counts, groups):
    counts = counts.to(dtype=torch.float64, device="cpu")
    tp = counts.diag()
    union = counts.sum(0) + counts.sum(1) - tp
    present = union > 0
    present[0] = False
    native_iou = torch.where(union > 0, tp / union.clamp_min(1), torch.nan)
    groups = groups.cpu()
    positions = (groups[:, None] * 5 + groups[None, :]).flatten()
    hc = torch.zeros(25, dtype=torch.float64)
    hc.scatter_add_(0, positions, counts.flatten())
    hc = hc.reshape(5, 5)
    hc[4] = 0  # Drop health-unknown targets, retain health-unknown predictions.
    htp = hc.diag()
    hu = hc.sum(0) + hc.sum(1) - htp
    hiou = torch.where(hu > 0, htp / hu.clamp_min(1), torch.nan)

    def finite(value):
        value = float(value)
        return value if __import__("math").isfinite(value) else None

    return dict(native_miou=finite(native_iou[present].mean()),
                native_pixel_accuracy=finite(tp.sum() / counts.sum().clamp_min(1)),
                coral_miou=finite(hiou[1:4].nanmean()),
                health_miou=finite(hiou[:4].nanmean()),
                background_iou=finite(hiou[0]), healthy_iou=finite(hiou[1]),
                bleached_iou=finite(hiou[2]), dead_iou=finite(hiou[3]),
                dead_precision=finite(htp[3] / hc[:, 3].sum().clamp_min(1)),
                dead_recall=finite(htp[3] / hc[3].sum().clamp_min(1)),
                health_pixel_accuracy=finite(htp.sum() / hc.sum().clamp_min(1)),
                native_per_class_iou=[finite(v) for v in native_iou],
                health_confusion=hc.long())


def select_indices(records, counts, groups):
    """Selection uses validation native mIoU only; prefer tied/simple weights."""
    metrics = [summarize(c, groups) for c in counts]
    selected = {}
    for family in FAMILIES:
        pool = [i for i, c in enumerate(records) if c.family == family]
        selected[family] = max(pool, key=lambda i: (metrics[i]["native_miou"],
                                                   -abs(records[i].a - records[i].b),
                                                   records[i].a))
    return selected, metrics
