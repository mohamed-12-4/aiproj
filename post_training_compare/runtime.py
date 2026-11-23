"""Keep published architectures; reuse cached backbone and reference PyTorch ops."""
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import types

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch

from .assets import ROOT, OUT

BASE = ROOT / ".cache/huggingface/hub/models--facebook--dinov3-vitl16-pretrain-lvd1689m/snapshots/ea8dc2863c51be0a264bab82070e3e8836b02d51"


def configure():
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    # Segmentation import registries enumerate many unused C++ ops. Every
    # unsupported op fails explicitly; only deformable attention is replaced
    # with MMCV's own differentiable PyTorch reference implementation.
    if "mmcv._ext" not in sys.modules and importlib.util.find_spec("mmcv._ext") is None:
        extension = types.ModuleType("mmcv._ext")
        extension.__file__ = __file__
        def unsupported(name):
            if name.startswith("__"): raise AttributeError(name)
            def fail(*args, **kwargs):
                raise RuntimeError(f"Compiled MMCV op {name} is unavailable in this local runtime")
            return fail
        extension.__getattr__ = unsupported
        sys.modules["mmcv._ext"] = extension
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch
    def reference(value, shapes, starts, locations, weights, im2col_step):
        return multi_scale_deformable_attn_pytorch(value, shapes, locations, weights)
    MultiScaleDeformableAttnFunction.apply = staticmethod(reference)
    author = ROOT / "third_party/rein_plus_author/projects/rein++"
    if str(author) not in sys.path: sys.path.insert(0, str(author))
    return "MMCV differentiable PyTorch reference deformable attention"


def build_dpt(gamma=1.0, device="cuda"):
    from transformers import AutoImageProcessor
    from safetensors.torch import load_file
    assets = json.loads((OUT / "assets.json").read_text())["model"]["local"]
    cfg = json.loads(Path(assets["config.json"]).read_text())
    spec=importlib.util.spec_from_file_location("coralscapes_released_inference",assets["coralscapes_hub_model.py"])
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    kwargs=module._segmenter_kwargs_from_hub(cfg)
    kwargs["encoder_id"]=str(BASE)  # Identical cached weights; no gated Hub request.
    # The published config names a slow DINOv3 processor that this Transformers
    # release does not implement. Its supported fast counterpart has the same
    # normalization; resizing is explicitly disabled for native RSG patches.
    processor=AutoImageProcessor.from_pretrained(BASE,local_files_only=True,use_fast=True,do_resize=False)
    published_processor=json.loads(Path(assets["preprocessor_config.json"]).read_text())
    if processor.image_mean!=published_processor["image_mean"] or processor.image_std!=published_processor["image_std"]:
        raise ValueError("Published image normalization differs from the cached processor")
    model=module.Dinov3DPTSegmenter(**kwargs,image_processor=processor)
    state = load_file(assets["model.safetensors"])
    # Transformers flattened the old DINOv3 encoder wrapper. This is a pure
    # bijective key rename, with every released tensor preserved unchanged.
    renamed={k.replace("encoder.model.layer.","encoder.layer.",1):v for k,v in state.items()}
    if len(renamed)!=len(state):raise RuntimeError("Released-key conversion collision")
    state=renamed
    model.load_state_dict(state, strict=True)
    del state
    if gamma != 1.0:
        from safetensors import safe_open
        with safe_open(BASE / "model.safetensors", framework="pt") as base:
            for name, parameter in model.encoder.named_parameters():
                match = re.match(r"layer\.(\d+)\.", name)
                if match:
                    factor = gamma + (1-gamma) * int(match[1]) / 23
                    original = base.get_tensor(name)
                    parameter.data.lerp_(original, 1-factor)
    return model.eval().requires_grad_(False).to(device)


def hf_to_native(backbone):
    """Invert Meta-to-HF naming and QKV splitting without another 1.2 GB download."""
    from safetensors import safe_open
    converted = {}
    simple = {"cls_token": "embeddings.cls_token", "storage_tokens": "embeddings.register_tokens",
              "mask_token": "embeddings.mask_token", "patch_embed.proj.weight": "embeddings.patch_embeddings.weight",
              "patch_embed.proj.bias": "embeddings.patch_embeddings.bias", "norm.weight": "norm.weight", "norm.bias": "norm.bias"}
    with safe_open(BASE / "model.safetensors", framework="pt") as source:
        for name, destination in backbone.state_dict().items():
            if name.startswith("reins."): continue
            if name == "rope_embed.periods":
                converted[name] = destination.clone()
                continue
            if name in simple:
                converted[name] = source.get_tensor(simple[name]).reshape(destination.shape)
                continue
            match = re.match(r"blocks\.(\d+)\.(.+)", name)
            if not match: raise ValueError(f"Unmapped native backbone parameter: {name}")
            layer, suffix = match.groups()
            prefix = f"layer.{layer}."
            if suffix == "attn.qkv.weight":
                value = torch.cat([source.get_tensor(prefix + f"attention.{v}_proj.weight") for v in "qkv"])
            elif suffix == "attn.qkv.bias":
                q = source.get_tensor(prefix + "attention.q_proj.bias")
                v = source.get_tensor(prefix + "attention.v_proj.bias")
                value = torch.cat([q, torch.zeros_like(q), v])
            elif suffix == "attn.qkv.bias_mask":
                value = torch.ones_like(destination)
                value[len(value)//3:2*len(value)//3] = 0
            else:
                suffix = suffix.replace("attn.proj.", "attention.o_proj.")
                suffix = suffix.replace("mlp.fc1.", "mlp.up_proj.").replace("mlp.fc2.", "mlp.down_proj.")
                suffix = suffix.replace("ls1.gamma", "layer_scale1.lambda1").replace("ls2.gamma", "layer_scale2.lambda1")
                value = source.get_tensor(prefix + suffix)
            if value.shape != destination.shape: raise ValueError(f"Backbone shape mismatch: {name}")
            converted[name] = value
    loaded = backbone.load_state_dict(converted, strict=False)
    if loaded.unexpected_keys or any(not n.startswith("reins.") for n in loaded.missing_keys):
        raise ValueError(f"Backbone conversion mismatch: {loaded}")


def build_rein(device="cuda", trainable=False):
    configure()
    from reinpp_training.model import register, _dense_tokens, as_float
    from reinpp_training.config import model_config
    from mmseg.registry import MODELS
    from mmengine.config import ConfigDict
    register()
    model = MODELS.build(ConfigDict(model_config()))
    hf_to_native(model.backbone)
    record = json.loads((OUT / "rein_checkpoint.json").read_text())
    checkpoint = torch.load(record["path"], map_location="cpu", mmap=True, weights_only=True)
    if checkpoint["epoch"] + 1 != 40: raise ValueError("Expected the selected 40-epoch checkpoint")
    state = dict(checkpoint["model"])
    tokens = state.pop("backbone.reins.effective_ema_tokens")
    loaded = model.load_state_dict(state, strict=False)
    allowed = {"backbone.reins.learnable_tokens_a", "backbone.reins.learnable_tokens_b"}
    invalid = [n for n in loaded.missing_keys if (not n.startswith("backbone.") or n.startswith("backbone.reins.")) and n not in allowed]
    if invalid or loaded.unexpected_keys: raise ValueError(f"Rein++ state mismatch: {invalid}, {loaded.unexpected_keys}")
    model.requires_grad_(False)
    module = model.backbone.reins
    module.register_parameter("effective_ema_tokens", torch.nn.Parameter(tokens, requires_grad=trainable))
    module.get_tokens = types.MethodType(_dense_tokens, module)
    if trainable:
        module.requires_grad_(True)
        module.learnable_tokens_a.requires_grad_(False)
        module.learnable_tokens_b.requires_grad_(False)
        model.decode_head.requires_grad_(True)
    model.to(device)
    original = model.backbone.forward
    def backbone_forward(*args, **kwargs):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
            value = original(*args, **kwargs)
        return as_float(value)
    model.backbone.forward = backbone_forward
    return model.eval()


def rein_outputs(model, image):
    features = model.backbone(image)
    classes, masks = model.decode_head(features)
    # Exactly the Mask2Former semantic inference equation, without label mapping.
    scores = torch.einsum("bqc,bqhw->bchw", classes[-1].softmax(-1)[..., :-1], masks[-1].sigmoid())
    scores = torch.nn.functional.interpolate(scores, image.shape[-2:], mode="bilinear", align_corners=False)
    return scores, classes, masks
