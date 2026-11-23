"""Build the unmodified official native model and published optimizer groups."""
from contextlib import contextmanager
from functools import wraps
import importlib.util
import sys
from pathlib import Path
from types import MethodType

import torch

from .config import PROJECT, model_config, optimizer_config

BASE_NAME = "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
BASE_SHA = "8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035"


def register():
    from mmseg.utils import register_all_modules as register_seg
    from mmdet.utils import register_all_modules as register_det
    register_det(init_default_scope=False)
    register_seg(init_default_scope=True)
    import reinpp
    reinpp.register_modules()


def as_float(tree):
    if isinstance(tree,torch.Tensor):return tree.float() if tree.is_floating_point() else tree
    if isinstance(tree,list):return [as_float(x) for x in tree]
    if isinstance(tree,tuple):return tuple(as_float(x) for x in tree)
    if isinstance(tree,dict):return {k:as_float(x) for k,x in tree.items()}
    return tree


def build_model(device="cuda"):
    register()
    from mmseg.registry import MODELS
    from mmengine.config import ConfigDict
    from reinpp.checkpoints import tensor_state
    model=MODELS.build(ConfigDict(model_config()))
    model.init_weights()
    base_path=Path("/workspace/pretrained/reinpp")/BASE_NAME
    state=tensor_state(torch.load(str(base_path),map_location="cpu",mmap=True,weights_only=True))
    loaded=model.backbone.load_state_dict(state,strict=False)
    bad=[name for name in loaded.missing_keys if not name.startswith("reins.")]
    if bad or loaded.unexpected_keys:
        raise ValueError(f"Original Meta backbone mismatch: missing={bad}, unexpected={loaded.unexpected_keys}")
    del state
    model.backbone.requires_grad_(False)
    model.backbone.reins.requires_grad_(True)
    model.to(device)
    original=model.backbone.forward
    @wraps(original)
    def forward(*args,**kwargs):
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=str(device).startswith("cuda")):
            features=original(*args,**kwargs)
        # MMCV's deformable-attention operators use FP32 in this configuration.
        return as_float(features)
    model.backbone.forward=forward
    return model


def build_optimizer(model):
    source=PROJECT/"third_party/rein_plus_author/rein/optimizers/peft_optimizer_constructor.py"
    if not source.exists():source=Path("/app/author_optimizer.py")
    module=sys.modules.get("reinpp_author_optimizer")
    if module is None:
        spec=importlib.util.spec_from_file_location("reinpp_author_optimizer",source)
        module=importlib.util.module_from_spec(spec)
        sys.modules["reinpp_author_optimizer"]=module
        spec.loader.exec_module(module)
    return module.PEFTOptimWrapperConstructor(**optimizer_config())(model)


def adapted_state(model):
    return {name:value.detach().cpu().clone() for name,value in model.state_dict().items()
            if name.startswith(("backbone.reins.","decode_head."))}


def _dense_tokens(module,layer):
    return module.effective_ema_tokens if layer == -1 else module.effective_ema_tokens[layer]


@contextmanager
def selected_weights(model,checkpoint):
    """Load selected raw/EMA adapter-head weights, then restore raw training state."""
    original=adapted_state(model)
    module=model.backbone.reins
    original_get_tokens=module.get_tokens
    state=dict(checkpoint["model"])
    tokens=state.pop("backbone.reins.effective_ema_tokens",None)
    missing=model.load_state_dict(state,strict=False)
    allowed={"backbone.reins.learnable_tokens_a","backbone.reins.learnable_tokens_b"} if tokens is not None else set()
    invalid=[n for n in missing.missing_keys if (not n.startswith("backbone.") or n.startswith("backbone.reins.")) and n not in allowed]
    if invalid or missing.unexpected_keys:
        raise ValueError(f"Selected checkpoint mismatch: {invalid}, {missing.unexpected_keys}")
    try:
        if tokens is not None:
            module.register_buffer("effective_ema_tokens",tokens.to(next(module.parameters()).device))
            module.get_tokens=MethodType(_dense_tokens,module)
        yield model
    finally:
        module.get_tokens=original_get_tokens
        if tokens is not None:del module.effective_ema_tokens
        model.load_state_dict(original,strict=False)
