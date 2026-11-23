"""Joint error counts, annotated-area cover errors and fixed referral diagnostics."""

import math
import numpy as np
import torch


def health_counts(native_counts, groups):
    groups = groups.cpu()
    positions = (groups[:, None] * 5 + groups[None, :]).flatten()
    flat = native_counts.cpu().reshape(-1, 96 * 96)
    result = torch.zeros(len(flat), 25, dtype=flat.dtype)
    result.scatter_add_(1, positions.expand(len(flat), -1), flat)
    result = result.reshape(*native_counts.shape[:-2], 5, 5)
    result[..., 4, :] = 0
    return result


def joint_errors(target, first, second, groups):
    valid = target != 0
    result = {}
    for name, truth, a, b, mask in (("native", target, first, second, valid),
            ("health", groups[target], groups[first], groups[second], valid & (groups[target] < 4))):
        ca, cb = (a == truth)[mask], (b == truth)[mask]
        result[name] = torch.stack(((ca & cb).sum(), (ca & ~cb).sum(),
                                    (~ca & cb).sum(), (~ca & ~cb).sum())).cpu()
    disagreement = first != second
    same_health = groups[first] == groups[second]
    result["disagreement_strata"] = torch.stack(((valid & disagreement & same_health).sum(),
                                                 (valid & ~same_health).sum())).cpu()
    return result


def cover_metrics(health_counts):
    counts = health_counts.double()
    total = float(counts.sum())
    if total == 0:
        return dict(dead_true_cover_pp=None, dead_pred_cover_pp=None,
                    dead_cover_error_pp=None, health_pixel_error=None)
    truth = 100 * float(counts[3].sum()) / total
    predicted = 100 * float(counts[:, 3].sum()) / total
    return dict(dead_true_cover_pp=truth, dead_pred_cover_pp=predicted,
                dead_cover_error_pp=abs(predicted - truth),
                health_pixel_error=1 - float(counts.diag().sum()) / total)


def pixel_histogram(score, target, prediction, groups, bins=512):
    truth = groups[target]
    valid = (target != 0) & (truth < 4)
    error = groups[prediction] != truth
    position = (score[valid].clamp(0, 1) * bins).long().clamp_max(bins - 1)
    return torch.bincount(position * 2 + error[valid].long(), minlength=bins * 2).reshape(bins, 2).cpu()


def retained_pixel_risk(histogram, coverage=.8):
    counts = np.asarray(histogram, dtype=np.float64)
    total = counts.sum()
    if total == 0:
        return None
    remaining, errors = total * coverage, 0.
    for correct, wrong in counts:
        size = correct + wrong
        take = min(remaining, size)
        if size:
            errors += wrong * take / size
        remaining -= take
        if remaining <= 1e-9:
            break
    return errors / (total * coverage)


def ranks(values):
    values = np.asarray(values)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    average = (ends - counts + 1 + ends) / 2
    return average[inverse]


def auroc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    positive, negative = int(labels.sum()), int((~labels).sum())
    if not positive or not negative:
        return None
    return float((ranks(scores)[labels].sum() - positive * (positive + 1) / 2) / (positive * negative))


def correlation(first, second):
    a, b = ranks(first), ranks(second)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def retained_image_mean(values, scores, keep):
    """Average over random ordering within a tied uncertainty threshold."""
    values, scores = np.asarray(values), np.asarray(scores)
    threshold = np.partition(scores, keep - 1)[keep - 1]
    below, tied = scores < threshold, scores == threshold
    fraction = (keep - int(below.sum())) / int(tied.sum())
    return float((values[below].sum() + fraction * values[tied].sum()) / keep)


def reliability(rows):
    valid = [r for r in rows if r["dead_cover_error_pp"] is not None]
    if not valid:
        return dict(images=0)
    errors = np.array([r["dead_cover_error_pp"] for r in valid])
    health_errors = np.array([r["health_pixel_error"] for r in valid])
    keep = max(1, math.ceil(.8 * len(valid)))
    result = dict(images=len(valid), dead_cover_mae_pp=float(errors.mean()),
                  mean_health_pixel_error=float(health_errors.mean()),
                  large_cover_error_threshold_pp=5., predictors={})
    for name in valid[0]["uncertainty"]:
        scores = np.array([r["uncertainty"][name] for r in valid])
        result["predictors"][name] = dict(
            spearman_cover_error=correlation(scores, errors),
            auroc_cover_error_gt5pp=auroc(errors > 5., scores),
            retained80_dead_cover_mae_pp=retained_image_mean(errors, scores, keep),
            retained80_mean_health_error=retained_image_mean(health_errors, scores, keep),
            retained_images=keep)
    result["by_site"] = {site: dict(images=sum(r["site"] == site for r in valid),
        dead_cover_mae_pp=float(np.mean([r["dead_cover_error_pp"] for r in valid if r["site"] == site])))
        for site in sorted({r["site"] for r in valid})}
    return result
