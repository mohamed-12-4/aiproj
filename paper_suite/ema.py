"""EMA of effective adapted weights, with transactional evaluation swapping.

EMA(W0 + scaling * B@A) is not EMA(W0) + scaling * EMA(B)@EMA(A).
Keep the first expression exactly, independently of the adapter parameterization.
"""

from contextlib import contextmanager

import torch
from torch import nn


class EffectiveEMA:
    def __init__(self, model, decay=0.999):
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0,1)")
        self.decay, self.num_updates = float(decay), 0
        self.adapters, excluded = {}, set()
        for name, module in model.named_modules():
            if hasattr(module, "base_layer") and hasattr(module, "lora_A"):
                if list(module.lora_A.keys()) != ["default"]:
                    raise ValueError("This comparison supports exactly one unmerged LoRA adapter")
                weight_name = name + ".base_layer.weight"
                a_name, b_name = name + ".lora_A.default.weight", name + ".lora_B.default.weight"
                self.adapters[weight_name] = (module, a_name, b_name, "peft")
                excluded.update((a_name, b_name))
            elif isinstance(module, nn.Linear) and hasattr(module, "lora_A") and hasattr(module, "scaling"):
                weight_name = name + ".weight"
                a_name, b_name = name + ".lora_A.weight", name + ".lora_B.weight"
                self.adapters[weight_name] = (module, a_name, b_name, "soma")
                excluded.update((a_name, b_name))
        params = dict(model.named_parameters())
        self.names = [name for name, p in params.items() if (p.requires_grad or name in self.adapters) and name not in excluded]
        self.shadow = {name: self._value(name, params).detach().float().clone() for name in self.names}
        self.buffers = {name: value.detach().clone() for name, value in model.named_buffers()}

    def _value(self, name, params):
        if name not in self.adapters:
            return params[name].float()
        module, a, b, kind = self.adapters[name]
        if kind == "peft":
            if module.merged:
                raise ValueError("EMA must be updated before any LoRA merge")
            scale = module.scaling["default"]
        else:
            if module.merged:
                raise ValueError("EMA must be updated before any SoMA merge")
            scale = module.scaling
        return params[name].float() + (params[b].float() @ params[a].float()) * scale

    @torch.no_grad()
    def update(self, model):
        params = dict(model.named_parameters())
        for name in self.names:
            self.shadow[name].lerp_(self._value(name, params), 1 - self.decay)
        # Running statistics and integer counters are copied, not averaged.
        for name, value in model.named_buffers():
            self.buffers[name].copy_(value)
        self.num_updates += 1

    @contextmanager
    def apply(self, model):
        params, buffers = dict(model.named_parameters()), dict(model.named_buffers())
        affected = set(self.names) | {spec[2] for spec in self.adapters.values()}
        saved = {name: params[name].detach().clone() for name in affected}
        saved_buffers = {name: value.detach().clone() for name, value in buffers.items()}
        try:
            with torch.no_grad():
                for name in self.names:
                    params[name].copy_(self.shadow[name])
                for _, _, b, _ in self.adapters.values():
                    # Base weight is now the complete effective EMA weight.
                    # Zero B so the adapter cannot be added a second time.
                    params[b].zero_()
                for name, value in self.buffers.items():
                    buffers[name].copy_(value)
            yield model
        finally:
            with torch.no_grad():
                for name, value in saved.items():
                    params[name].copy_(value)
                for name, value in saved_buffers.items():
                    buffers[name].copy_(value)

    def state_dict(self):
        return {"decay": self.decay, "num_updates": self.num_updates, "names": self.names,
                "shadow": self.shadow, "buffers": self.buffers,
                "kind": "effective_weight_ema"}

    def load_state_dict(self, state):
        if state["kind"] != "effective_weight_ema" or set(state["names"]) != set(self.names):
            raise ValueError("EMA checkpoint architecture does not match")
        self.decay, self.num_updates = state["decay"], state["num_updates"]
        for name in self.names:
            self.shadow[name].copy_(state["shadow"][name])
        for name in self.buffers:
            self.buffers[name].copy_(state["buffers"][name])

    def inference_state(self, model):
        with self.apply(model):
            # Same module layout, full effective weights in base, B=0. This is
            # an inference checkpoint, never a training-resume checkpoint.
            return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
