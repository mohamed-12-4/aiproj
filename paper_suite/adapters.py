"""Compatibility ports of the authors' SoMA and Rein-G modules to HF DINOv3."""

from contextlib import contextmanager
from functools import wraps
import importlib.util
import re
import sys

import torch
from torch import nn
from torch.nn import functional as F

from .recipe import AUTHOR, PROJECT


def load_source(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_model_class():
    if str(AUTHOR) not in sys.path:
        sys.path.insert(0, str(AUTHOR))
    from dinov3_dpt_seg.model import Dinov3DPTSegmenter
    return Dinov3DPTSegmenter


def install_soma(encoder, settings):
    source = load_source("paper_soma_author", PROJECT / "third_party/soma_author/rein/peft/sora.py")

    class MatchedSomaLinear(source.Linear):
        def forward(self, x):
            # Upstream's default dropout is zero and its forward omits the
            # configured dropout. Honor the matched CoralScapes dropout here.
            result = F.linear(x, self.weight, self.bias)
            adapted = self.lora_B(self.lora_A(self.lora_dropout(x)))
            return result + adapted * self.scaling

    targets = [(n, m) for n, m in encoder.named_modules() if isinstance(m, nn.Linear)]
    if not targets:
        raise ValueError("No SoMA target linears found")
    for name, original in targets:
        parent_name, _, child = name.rpartition(".")
        parent = encoder.get_submodule(parent_name) if parent_name else encoder
        adapted = MatchedSomaLinear(original.in_features, original.out_features,
                                    bias=original.bias is not None, r=settings["rank"],
                                    lora_alpha=settings["alpha"], lora_dropout=settings["dropout"],
                                    first_eigen=False, merge_weights=False).to(original.weight.device)
        adapted.weight = original.weight
        adapted.bias = original.bias
        adapted.sora_init("sora")  # Exact SVD of the minor components, not randomized top-r SVD.
        adapted.weight.requires_grad_(False)
        if adapted.bias is not None:
            adapted.bias.requires_grad_(False)
        setattr(parent, child, adapted)
    return len(targets)


def install_rein(encoder, settings):
    source = load_source("paper_rein_author", PROJECT / "third_party/rein_author/simple_reins/reins.py")
    cfg = encoder.config
    encoder.reins = source.Reins(num_layers=cfg.num_hidden_layers, embed_dims=cfg.hidden_size,
                                patch_size=cfg.patch_size, token_length=settings["token_length"],
                                scale_init=settings["scale_init"]).to(next(encoder.parameters()).device)
    prefix = 1 + cfg.num_register_tokens

    def make_forward(original, index):
        @wraps(original)
        def forward(*args, **kwargs):
            output = original(*args, **kwargs)
            if not isinstance(output, torch.Tensor):
                raise TypeError("Expected HF DINOv3 block to return one tensor")
            patches = encoder.reins(output[:, prefix:], index, batch_first=True, has_cls_token=False)
            return torch.cat((output[:, :prefix], patches), dim=1)
        return forward

    for i, layer in enumerate(encoder.layer):
        # HF records hidden states inside forward, before Module forward hooks.
        # Refine the returned tensor there so DPT sees every tapped Rein block,
        # including the final block. Keep the original layer type and state keys.
        layer.forward = make_forward(layer.forward, i)
    return cfg.num_hidden_layers


def build_model(kwargs, experiment, device=None):
    # All variants first construct the same author LoRA model, so the random DPT
    # initialization and the post-initialization RNG states match the control.
    model = source_model_class()(**kwargs)
    if device is not None:
        model.to(device)
    method = experiment["method"]
    if method == "eva":
        from peft import EvaConfig
        settings = experiment["eva"]
        # Construct the ordinary author LoRA model first so DPT initialization
        # and the following training RNG are identical to the control.
        model.encoder.peft_config["default"].init_lora_weights = "eva"
        model.encoder.peft_config["default"].eva_config = EvaConfig(
            rho=settings["rho"], tau=settings["tau"], whiten=settings["whiten"],
            adjust_scaling_factors=settings["adjust_scaling_factors"], use_label_mask=False)
    elif method != "lora":
        with torch.random.fork_rng():
            model.encoder = model.encoder.unload()  # Initial LoRA B is zero; retain the pretrained weights.
            model.lora_enabled = False
            model.encoder.requires_grad_(False)
            if method == "decoder_ft":
                # Include every parameter in the optimizer before freezing at
                # the first epoch. Unfreezing later keeps the same LR schedule.
                model.encoder.requires_grad_(True)
            elif method == "soma":
                model.adapter_locations = install_soma(model.encoder, experiment["soma"])
            elif method == "rein_g":
                model.adapter_locations = install_rein(model.encoder, experiment["rein_g"])
            else:
                raise ValueError(method)
    if experiment.get("gradient_checkpointing"):
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if experiment.get("decoder_checkpointing"):
        from .execution import checkpoint_decoder_module
        checkpoint_decoder_module(model.neck)
        checkpoint_decoder_module(model.seg_head)
    return model


def set_epoch_phase(model, experiment, epoch):
    if experiment["method"] == "decoder_ft":
        train_encoder = epoch >= experiment["decoder_warmup_epochs"]
        model.encoder.requires_grad_(train_encoder)
        return "full_finetuning" if train_encoder else "decoder_warmup"
    return experiment["method"]
