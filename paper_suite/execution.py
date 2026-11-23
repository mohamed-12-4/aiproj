"""Precision and decoder checkpointing without changing model state names."""

from contextlib import contextmanager, nullcontext
from functools import wraps

import torch
from torch.utils.checkpoint import checkpoint


def precision_dtype(cfg):
    precision = cfg["experiment"].get("precision", "fp16")
    if precision not in ("fp16", "bf16"):
        raise ValueError(f"Unsupported precision: {precision}")
    return torch.bfloat16 if precision == "bf16" else torch.float16


def checkpoint_decoder_module(module):
    """Checkpoint a decoder, keeping BatchNorm running updates to one per batch.

    Recomputing BatchNorm normally updates its running statistics twice. During
    recomputation, functional_call gives the module cloned buffers instead.
    Parameter names and gradients, original buffers, and dropout RNG are retained.
    """
    original = module.forward
    state = {"recomputing": False, "functional": False}

    @contextmanager
    def recompute_context():
        state["recomputing"] = True
        try:
            yield
        finally:
            state["recomputing"] = False

    def compute(*args, **kwargs):
        if not state["recomputing"]:
            return original(*args, **kwargs)
        buffers = {name: value.detach().clone() for name, value in module.named_buffers()}
        state["functional"] = True
        try:
            return torch.func.functional_call(module, buffers, args, kwargs, strict=False)
        finally:
            state["functional"] = False

    @wraps(original)
    def forward(*args, **kwargs):
        if state["functional"] or not module.training or not torch.is_grad_enabled():
            return original(*args, **kwargs)
        return checkpoint(compute, *args, use_reentrant=False, preserve_rng_state=True,
                          context_fn=lambda: (nullcontext(), recompute_context()), **kwargs)

    module.forward = forward


class AttributeProxy:
    def __init__(self, wrapped, **overrides):
        self._wrapped = wrapped
        self._overrides = overrides

    def __getattr__(self, name):
        return self._overrides[name] if name in self._overrides else getattr(self._wrapped, name)


def author_torch_proxy(precision):
    """Override the author's hardcoded FP16 contexts locally to its train module."""
    if precision == "fp16":
        return torch
    if precision != "bf16":
        raise ValueError(precision)

    def autocast(device_type, dtype=None, enabled=True, cache_enabled=None):
        if device_type == "cuda":
            dtype = torch.bfloat16
        return torch.autocast(device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)

    def scaler(*args, **kwargs):
        return torch.amp.GradScaler("cuda", enabled=False)

    cuda_amp = AttributeProxy(torch.cuda.amp, GradScaler=scaler)
    cuda = AttributeProxy(torch.cuda, amp=cuda_amp)
    return AttributeProxy(torch, autocast=autocast, cuda=cuda)
