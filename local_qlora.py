"""Matched laptop QLoRA runs. Preparing this module does not start an experiment."""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import config
import train

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs/laptop_qlora.json"


def read_config(path):
    recipe = json.loads(Path(path).read_text())
    required = {"dataset", "model", "lora", "train", "augmentation", "sampling", "eva", "runtime"}
    if set(recipe) != required:
        raise ValueError(f"Config sections must be {sorted(required)}")
    model, training, lora, eva = (recipe[key] for key in ("model", "train", "lora", "eva"))
    policy = train.dataset_config(recipe['dataset']['annotation_version'])
    if recipe['dataset']['id'] != policy['id'] or recipe['dataset']['classes'] != train.CLASSES:
        raise ValueError("Dataset identity/classes must match the verified health-label policy.")
    if policy['revision'] is not None and recipe['dataset']['revision'] != policy['revision']:
        raise ValueError("Dataset revision must match the verified annotation policy.")
    runtime = {"optimizer": "adamw", "scheduler": "cosine", "loss": "uniform_cross_entropy",
               "validate_every_epochs": 1, "selection_metric": "validation_coral_miou",
               "automatic_test": False, "results_root": "runs", "checkpoints_root": "checkpoints/local",
               "metrics_file": "metrics.csv", "figure_file": "prediction.png", "save_best_only": True}
    if recipe['runtime'] != runtime:
        raise ValueError("The runtime section must retain the supported optimizer, evaluation and output contract.")
    if model["architecture"] != "dinov3_single" or model["decoder"] != "dpt":
        raise ValueError("The laptop comparison uses one DINOv3 encoder and a DPT decoder.")
    if model["num_classes"] != 4 or model["ignore_index"] != 255:
        raise ValueError("Use the four-class health mapping with ignore_index=255.")
    if not model.get("quantization") or model["quantization"]["quant_type"] != "nf4" or not model["quantization"]["load_in_4bit"]:
        raise ValueError("Both runs must use 4-bit NF4 quantization.")
    if training["epochs"] < 1 or training["max_updates"] or training["early_stopping_patience"]:
        raise ValueError("Use complete epochs, with max_updates=0 and early_stopping_patience=0.")
    if any(type(training[key]) is not int or training[key] < 1
           for key in ("batch_size", "accumulation_steps")):
        raise ValueError("Batch size and accumulation steps must be positive integers.")
    if training["height"] < 32 or training["height"] % model["patch_size"]:
        raise ValueError("Height must be at least 32 and divisible by the patch size.")
    if training['width'] != 2 * training['height']:
        raise ValueError("Input width must be twice its height.")
    if training["precision"] != "bf16" or model["quantization"]["compute_dtype"] != "bfloat16":
        raise ValueError("Use matching BF16 training and 4-bit compute dtypes.")
    if lora["init_lora_weights"] is not True:
        raise ValueError("Construct identical standard adapters before optional EVA initialization.")
    if eva["rho"] != 1.0 or eva["whiten"]:
        raise ValueError("This comparison uses fixed-rank EVA without whitening.")
    if not 0 < eva["tau"] <= 1 or min(eva[key] for key in
            ("calibration_images", "batch_size", "max_patch_tokens_per_batch", "max_forwards", "max_seconds")) < 1:
        raise ValueError("EVA needs positive calibration limits and 0 < tau <= 1.")
    if eva["max_patch_tokens_per_batch"] < lora["r"]:
        raise ValueError("EVA patch-token cap must be at least the adapter rank.")
    return recipe


@contextmanager
def recipe_settings(recipe):
    # Scoped overrides leave config.py and all Modal defaults unchanged.
    sections = {"DATASET": "dataset", "MODEL": "model", "TRAIN": "train", "LORA": "lora",
                "AUGMENT": "augmentation", "SAMPLING": "sampling"}
    previous = {name: getattr(config, name) for name in sections}
    try:
        for name, section in sections.items():
            setattr(config, name, deepcopy(recipe[section]))
        yield
    finally:
        for name, value in previous.items():
            setattr(config, name, value)


@contextmanager
def preserve_rng():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def patch_inputs(layer_input, model_input, layer_name, prefix, maximum):
    value = layer_input[0] if isinstance(layer_input, (tuple, list)) else layer_input
    if value.ndim != 3 or value.shape[1] <= prefix:
        raise ValueError(f"Expected spatial DINOv3 tokens at {layer_name}")
    value = value[:, prefix:, :].reshape(-1, value.shape[-1])
    if len(value) > maximum:
        indices = torch.linspace(0, len(value) - 1, maximum, device=value.device).long()
        value = value.index_select(0, indices)
    return value.float()


def decoder_hash(model):
    digest = hashlib.sha256()
    for name, value in model.decoder.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def initialize_eva(model, training_dataset, settings, device):
    from peft import EvaConfig, get_eva_state_dict, initialize_lora_eva_weights

    before_training = model.training
    ids = {name: id(parameter) for name, parameter in model.named_parameters()}
    selected = np.random.default_rng(settings["seed"]).permutation(len(training_dataset))[
        :settings["calibration_images"]].tolist()
    calibration = train.Corals(training_dataset.rows, training_dataset.size[1],
                               augment=False, label_map=training_dataset.label_map)
    loader = DataLoader(Subset(calibration, selected), batch_size=settings["batch_size"],
                        shuffle=False, num_workers=0,
                        generator=torch.Generator().manual_seed(settings["seed"]))
    peft_config = model.backbone.peft_config["default"]
    peft_config.eva_config = EvaConfig(rho=settings["rho"], tau=settings["tau"],
                                      whiten=settings["whiten"], use_label_mask=False)
    peft_config.init_lora_weights = "eva"
    expected = {name for name, module in model.backbone.named_modules() if hasattr(module, "lora_A")}
    started, forwards = time.monotonic(), 0

    def forward(encoder, batch):
        nonlocal forwards
        if forwards >= settings["max_forwards"] or time.monotonic() - started >= settings["max_seconds"]:
            raise RuntimeError("EVA did not converge within its calibration budget; training has not started. "
                               "Increase the EVA limits in the config before retrying.")
        forwards += 1
        with torch.autocast(device.type, dtype=train.amp_dtype(device), enabled=device.type == "cuda"):
            return encoder(pixel_values=batch[0].to(device))

    from functools import partial
    try:
        with preserve_rng():
            model.eval()
            state = get_eva_state_dict(
                model.backbone, loader, forward_fn=forward,
                prepare_model_inputs_fn=lambda inputs, unused: None,
                prepare_layer_inputs_fn=partial(patch_inputs, prefix=model.prefix_tokens,
                                                maximum=settings["max_patch_tokens_per_batch"]),
                gather_distributed_inputs=False, show_progress_bar=False)
            if set(state) != expected or any(value.shape[0] != model.lora_config["r"] or
                    not torch.isfinite(value).all() for value in state.values()):
                raise RuntimeError("EVA returned incomplete, nonfinite or redistributed adapters.")
            initialize_lora_eva_weights(model.backbone, eva_state_dict=dict(state))
            if ids != {name: id(parameter) for name, parameter in model.named_parameters()}:
                raise RuntimeError("Fixed-rank EVA unexpectedly replaced parameters.")
            if any(torch.count_nonzero(module.lora_B["default"].weight)
                   for module in model.backbone.modules() if hasattr(module, "lora_B")):
                raise RuntimeError("EVA must preserve the zero initial adapter update.")
    finally:
        for module in model.backbone.modules():
            for key, hook in list(module._forward_hooks.items()):
                if type(hook).__module__ == "peft.tuners.lora.eva":
                    del module._forward_hooks[key]
        model.train(before_training)
    return {"forwards": forwards, "seconds": time.monotonic() - started,
            "training_image_indices": selected, "rng_preserved": True, "rho": settings["rho"]}


def make_initializer(recipe, initialization):
    def initialize(model, training_dataset, device):
        from bitsandbytes.nn import Linear4bit
        count = sum(isinstance(module, Linear4bit) for module in model.backbone.modules())
        if not count:
            raise RuntimeError("No 4-bit layers loaded; refusing a silent full-precision baseline.")
        report = {"method": initialization, "quantized_linears": count,
                  "initial_decoder_sha256": decoder_hash(model)}
        if initialization == "eva":
            print(f"Initializing EVA from training images (rank remains {recipe['lora']['r']})...", flush=True)
            report["eva"] = initialize_eva(model, training_dataset, recipe["eva"], device)
            print(f"EVA ready: {report['eva']['forwards']} calibration forwards.", flush=True)
        return report
    return initialize


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--init", choices=["standard", "eva"], default="standard")
    parser.add_argument("--output", type=Path,
                        help="Fresh result directory; default name includes initialization, dataset, "
                             "resolution, batch/accumulation sizes, epochs and seed")
    parser.add_argument("--print-config", action="store_true", help="Print the complete recipe without loading a model")
    args = parser.parse_args(argv)
    recipe = read_config(args.config)
    settings = recipe["train"]
    name = (f"qlora-{args.init}-{recipe['dataset']['annotation_version'].lower()}-"
            f"{settings['height']}x{settings['width']}-"
            f"b{settings['batch_size']}a{settings['accumulation_steps']}-"
            f"{settings['epochs']}e-s{settings['seed']}")
    output = args.output or ROOT / "runs" / name
    if args.print_config:
        print(json.dumps({"initialization": args.init, **recipe,
                          "results": str(output), "checkpoint": str(ROOT / "checkpoints/local" / output.name / "head.pt"),
                          "loss": "uniform_cross_entropy", "validation": "every_epoch",
                          "selection": "validation_coral_miou", "automatic_test": False}, indent=2))
        return
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Run this command from your GPU-enabled local terminal.")
    if not torch.cuda.is_bf16_supported():
        parser.error("This recipe requires a BF16-capable GPU.")
    versions = {package: version(package) for package in ("torch", "transformers", "peft", "bitsandbytes", "accelerate")}
    metadata = {"initialization": args.init, "config": recipe, "versions": versions,
                "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                  for name in ("train.py", "dpt.py", "local_qlora.py", "coral_data.py", "local_output.py")}}
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    with recipe_settings(recipe):
        train.main(["--output", str(output)], compact_output=True,
                   initialize_model=make_initializer(recipe, args.init), run_metadata=metadata)


if __name__ == "__main__":
    main()
