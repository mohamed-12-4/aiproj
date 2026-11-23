"""Probability calibration and conservative, spatial health routing."""

from dataclasses import asdict, dataclass
import math

import torch
from torch.nn import functional as F

WEIGHTS = (0., .25, .5, .75, 1.)
TEMPERATURES = (.25, .5, .75, 1., 1.5, 2., 3., 4.)
CALIBRATION_SITE = "site6"
PATCH = 32


def normalize(scores):
    total = scores.sum(0, keepdim=True)
    return torch.where(total > 0, scores / total.clamp_min(1e-30),
                       torch.full_like(scores, 1 / len(scores)))


def temperature(probabilities, value):
    if value <= 0:
        raise ValueError("Temperature must be positive")
    if value == 1:
        return probabilities
    return (probabilities.clamp_min(1e-30).log() / value).softmax(0)


def fit_temperature(log_probabilities, targets, grid=TEMPERATURES):
    """Fit just one scalar on the designated calibration site, never selection/test."""
    if log_probabilities.ndim != 2 or len(log_probabilities) != len(targets) or not len(targets):
        raise ValueError("Expected nonempty [samples,classes] calibration log probabilities")
    rows = []
    for value in grid:
        nll = float(F.cross_entropy(log_probabilities / value, targets))
        rows.append(dict(temperature=value, nll=nll))
    winner = min(rows, key=lambda r: (r["nll"], abs(math.log(r["temperature"])) ))
    return dict(**winner, search=rows, samples=len(targets))


def mixture_prediction(first, second, weight, first_prediction=None, second_prediction=None):
    if weight == 1:
        return first.argmax(0) if first_prediction is None else first_prediction
    if weight == 0:
        return second.argmax(0) if second_prediction is None else second_prediction
    return (first * weight + second * (1 - weight)).argmax(0)


def health_probabilities(probabilities, groups):
    groups = groups.to(probabilities.device)
    return torch.stack([probabilities[groups == group].sum(0) for group in range(5)])


def entropy(probabilities):
    return -(probabilities * probabilities.clamp_min(1e-30).log()).sum(0) / math.log(len(probabilities))


def patch_mean(value, patch=PATCH):
    singleton = value.ndim == 2
    x = value[None] if singleton else value
    pooled = F.avg_pool2d(x[None], patch, stride=patch, ceil_mode=True, count_include_pad=False)
    # Repetition, rather than interpolation, preserves actual patch boundaries for odd fixtures.
    expanded = pooled[0].repeat_interleave(patch, -2).repeat_interleave(patch, -1)
    expanded = expanded[..., :x.shape[-2], :x.shape[-1]]
    return expanded[0] if singleton else expanded


@dataclass(frozen=True)
class Gate:
    min_advantage: float | None
    min_entropy: float = 0.

    def record(self):
        return asdict(self)


def gate_grid():
    return [Gate(None)] + [Gate(delta, uncertainty) for delta in (0., .05, .1, .2)
                          for uncertainty in (0., .25, .5)]


def gate_features(first_health, second_health, first_prediction, second_prediction, groups,
                  patch=PATCH):
    """Features use predictions only; ground truth is deliberately absent."""
    groups = groups.to(first_prediction.device)
    hfirst, hsecond = groups[first_prediction], groups[second_prediction]
    first_support = first_health.gather(0, hfirst[None])[0]
    second_support = second_health.gather(0, hsecond[None])[0]
    frequency = patch_mean(F.one_hot(hsecond, 5).permute(2, 0, 1).float(), patch)
    consistency = frequency.gather(0, hsecond[None])[0]
    return dict(eligible=(hfirst != hsecond) & (hsecond < 4) & (consistency >= .75),
                advantage=patch_mean(second_support - first_support, patch),
                uncertainty=patch_mean(entropy(first_health), patch))


def route(first_prediction, second_prediction, features, candidate):
    if candidate.min_advantage is None:
        return first_prediction, torch.zeros_like(first_prediction, dtype=torch.bool)
    selected = (features["eligible"] & (features["advantage"] > candidate.min_advantage)
                & (features["uncertainty"] >= candidate.min_entropy))
    return torch.where(selected, second_prediction, first_prediction), selected


def health_js(first, second):
    mean = (first + second) * .5
    log_mean = mean.clamp_min(1e-30).log()
    value = .5 * ((first * (first.clamp_min(1e-30).log() - log_mean)).sum(0)
                  + (second * (second.clamp_min(1e-30).log() - log_mean)).sum(0))
    return (value / math.log(2)).clamp(0, 1)


def uncertainty_features(first_health, second_health, first_prediction, second_prediction, groups,
                         patch=PATCH):
    groups = groups.to(first_prediction.device)
    hf, hs = groups[first_prediction], groups[second_prediction]
    roi = ((hf >= 1) & (hf <= 3)) | ((hs >= 1) & (hs <= 3))
    if not bool(roi.any()):
        roi = torch.ones_like(roi)
    js = health_js(first_health, second_health)
    patch_mass = F.avg_pool2d(roi[None, None].float(), patch, patch, ceil_mode=True,
                              count_include_pad=False)[0, 0]
    patch_js = F.avg_pool2d((js * roi)[None, None], patch, patch, ceil_mode=True,
                            count_include_pad=False)[0, 0]
    spatial = patch_js[patch_mass > 0] / patch_mass[patch_mass > 0]
    scores = dict(lora_entropy=float(entropy(first_health)[roi].mean()),
                  reinpp_entropy=float(entropy(second_health)[roi].mean()),
                  health_js=float(js[roi].mean()),
                  health_disagreement=float((hf[roi] != hs[roi]).float().mean()),
                  spatial_health_js=float(torch.quantile(spatial, .9)))
    return scores, js
