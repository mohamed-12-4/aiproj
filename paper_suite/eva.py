"""Public PEFT EVA initialization, isolated from the matched training RNG/state."""

from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import random
import time

import numpy as np
import peft
import torch
from torch.utils.data import DataLoader, Dataset

from .recipe import DATA_REVISION, ENCODER_REVISION, write_json


@contextmanager
def preserve_rng():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def frozen_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        if not value.requires_grad:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def patch_inputs(layer_input, model_input, layer_name, prefix=5, maximum=16384):
    """Deterministic spatial subsampling, identical for modules sharing inputs."""
    value = layer_input[0] if isinstance(layer_input, (tuple, list)) else layer_input
    if value.ndim != 3 or value.shape[1] <= prefix:
        raise ValueError(f"Expected DINOv3 [batch, prefix+patches, channels]: {layer_name}")
    value = value[:, prefix:, :].reshape(-1, value.shape[-1])
    if len(value) > maximum:
        indices = torch.linspace(0, len(value) - 1, maximum, device=value.device).long()
        value = value.index_select(0, indices)
    return value


def apply_eva_state(encoder, state):
    """Load the official EVA result without replacing trainable parameters."""
    from peft import initialize_lora_eva_weights
    ids = {name: id(p) for name, p in encoder.named_parameters()}
    initialize_lora_eva_weights(encoder, eva_state_dict=dict(state))
    if ids != {name: id(p) for name, p in encoder.named_parameters()}:
        raise RuntimeError("Fixed-rank EVA must preserve optimizer parameter identities")


def assert_lora_match(candidate, reference):
    """Require exact equality of all shared training/model/data hyperparameters."""
    for section in ("data", "augment", "model", "train"):
        left, right = dict(candidate[section]), dict(reference[section])
        if section == "train":
            left.pop("output_dir", None)
            right.pop("output_dir", None)
        if left != right:
            raise ValueError(f"EVA differs from saved LoRA in {section}")
    for key in ("precision", "gradient_checkpointing", "decoder_checkpointing", "ema_decay",
                "stop_after_epochs", "schedule_epochs", "ema_buffer_policy"):
        if candidate["experiment"][key] != reference["experiment"][key]:
            raise ValueError(f"EVA differs from saved LoRA runtime: {key}")
    return True


class CalibrationDataset(Dataset):
    def __init__(self, dataset, indices, augmentation, shape):
        self.dataset, self.indices = dataset, indices
        self.augmentation, self.shape = augmentation, shape

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        x, _ = self.augmentation(*self.dataset[self.indices[index]], target_size=self.shape)
        return x


def calibration_indices(dataset, count, seed):
    # Seeded sampling without replacement from the official training split.
    return np.random.default_rng(seed).permutation(len(dataset))[:count].tolist()


def initialize_eva(model, cfg, out, commit=lambda: None):
    from peft import get_eva_state_dict
    from .runner import atomic_save, author, base_dataset

    if peft.__version__ != "0.18.1":
        raise RuntimeError(f"Expected pinned public PEFT 0.18.1, got {peft.__version__}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "eva_initialization.pt"
    settings = cfg["experiment"]["eva"]
    identity = {"settings": settings, "encoder_revision": ENCODER_REVISION,
                "dataset_revision": DATA_REVISION, "lora": cfg["model"]["lora"]}
    if cache.exists():
        saved = torch.load(cache, map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise ValueError("Cached EVA initialization belongs to a different configuration")
        apply_eva_state(model.encoder, saved["state"])
        return dict(saved["report"], reused=True)

    started = time.monotonic()
    report = {}
    training = model.encoder.training
    before = frozen_fingerprint(model.encoder)
    try:
        with preserve_rng():
            author._set_seed(settings["calibration_seed"])
            dataset = base_dataset(cfg, "train")
            indices = calibration_indices(dataset, settings["calibration_images"], settings["calibration_seed"])
            augmentation = author.SegmentationTrainAugmentation(
                author.segmentation_augment_config_from_dict(cfg["augment"]),
                model.image_mean, model.image_std, ignore_index=0)
            calibration = CalibrationDataset(dataset, indices, augmentation, tuple(cfg["augment"]["val_size"]))
            loader = DataLoader(calibration, batch_size=settings["calibration_batch_size"],
                                shuffle=False, num_workers=0)
            forwards = 0

            def forward(encoder, x):
                nonlocal forwards
                if forwards >= settings["max_forward_batches"] or time.monotonic() - started > settings["max_initialization_seconds"]:
                    raise TimeoutError("EVA convergence budget exceeded; refusing an unconverged initialization")
                forwards += 1
                # FP32, no gradients, encoder only. DPT/BatchNorm never execute.
                with torch.no_grad(), torch.autocast(x.device.type, enabled=False):
                    result = encoder(pixel_values=x, output_hidden_states=False)
                if forwards == 1 or forwards % 5 == 0:
                    converged, total = 0, 0
                    for module in encoder.modules():
                        for hook in module._forward_hooks.values():
                            if hasattr(hook, "converged"):
                                total += 1
                                converged += int(bool(hook.converged.all()))
                    progress = {"status": "initializing", "forward_batches": forwards,
                                "active_svd_hooks": total, "active_hooks_converged": converged,
                                "elapsed_seconds": time.monotonic() - started}
                    write_json(out / "eva_progress.json", progress)
                    print("EVA_INIT " + json.dumps(progress), flush=True)
                    commit()
                return result

            prefix = 1 + int(model.encoder.config.num_register_tokens)
            state = get_eva_state_dict(
                model.encoder, loader, forward_fn=forward,
                prepare_model_inputs_fn=lambda inputs, config: None,
                prepare_layer_inputs_fn=lambda inputs, unused, name: patch_inputs(
                    inputs, unused, name, prefix=prefix, maximum=settings["max_patch_tokens_per_batch"]),
                show_progress_bar=False)
            if not state or any(w.shape[0] != cfg["model"]["lora"]["r"] or not torch.isfinite(w).all() for w in state.values()):
                raise RuntimeError("EVA returned missing, nonfinite or redistributed adapters")
            apply_eva_state(model.encoder, state)
            if any(torch.count_nonzero(module.lora_B["default"].weight) for module in model.encoder.modules() if hasattr(module, "lora_B")):
                raise RuntimeError("EVA B must remain zero before fine-tuning")
            if frozen_fingerprint(model.encoder) != before:
                raise RuntimeError("EVA changed frozen pretrained weights")
            source = Path(inspect.getfile(get_eva_state_dict))
            report = {"status": "converged", "peft_version": peft.__version__,
                      "implementation_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                      "public_implementation": "https://github.com/huggingface/peft/blob/v0.18.1/src/peft/tuners/lora/eva.py",
                      "identity": identity, "forward_batches": forwards,
                      "initialization_seconds": time.monotonic() - started,
                      "calibration_indices": indices,
                      "calibration_images": [str(dataset.pairs[i][0]) for i in indices],
                      "adapted_linear_count": len(state), "base_weights_unchanged": True,
                      "B_zero": True, "training_rng_preserved": True, "reused": False}
            atomic_save({"identity": identity, "state": {k: v.cpu() for k, v in state.items()},
                         "report": report}, cache)
            write_json(out / "eva_initialization.json", report)
            write_json(out / "eva_progress.json", {"status": "converged", "forward_batches": forwards,
                                                    "elapsed_seconds": report["initialization_seconds"]})
            commit()
    finally:
        # PEFT's initialization hooks must also be removed on a bounded failure.
        for module in model.encoder.modules():
            for key, hook in list(module._forward_hooks.items()):
                if type(hook).__module__ == "peft.tuners.lora.eva":
                    del module._forward_hooks[key]
        model.encoder.train(training)
    print(f"EVA_READY {len(state)} adapters in {report['initialization_seconds']:.1f}s", flush=True)
    return report
