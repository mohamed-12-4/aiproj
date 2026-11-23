from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]
AUTHOR = PROJECT / "third_party/coralscapes_v2_author"
ENCODER_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
ENCODER_REVISION = "ea8dc2863c51be0a264bab82070e3e8836b02d51"
DATA_REVISION = "8a4567b2fbec14882e09aebff63a7a1ca5e24dca"
METHODS = ("lora", "decoder_ft", "soma", "rein_g", "eva")
REIN_IMPLEMENTATION = "block_forward_v2"
COMMITS = {
    "coralscapes_v2_author": "2679068cd4c64d0c1a80600e41119a7fe522a28d",
    "soma_author": "7ea8c0e7c2ac64afe84a7db52d41e69bed5d5764",
    "rein_author": "dc063429c4dadc0da9c6252b3db22fc55a9882ab",
}


def root():
    return Path(os.environ.get("PAPER_ROOT", PROJECT))


def encoder_directory():
    return root() / ".cache/huggingface/hub" / ("models--" + ENCODER_ID.replace("/", "--")) / "snapshots" / ENCODER_REVISION


def data_directory():
    return root() / "data" / ("coralscapes_v2_paper_" + DATA_REVISION[:12])


def make_config(method, run_group="paper-v2-s42", smoke=False):
    if method not in METHODS:
        raise ValueError(method)
    cfg = yaml.safe_load((AUTHOR / "configs/vit_l.yaml").read_text())
    data = data_directory()
    cfg["data"].update(images_root=str(data / "leftImg8bit"), labels_root=str(data / "gtFine"),
                       num_classes=96, ignore_index=0)
    cfg["model"]["encoder_id"] = str(encoder_directory())
    cfg["train"].update(output_dir=str(root() / "checkpoints" / run_group / method),
                        wandb_enabled=False, log_training_batch=False, log_loss_plot=False)
    cfg["experiment"] = {
        "method": method, "decoder_warmup_epochs": 15, "total_epochs": 150,
        "ema_decay": 0.999, "ema_kind": "effective_weights_for_lora_and_soma",
        "ema_buffer_policy": "copy_live_buffers; recalibrate_batchnorm_on_training_images_before_final_test",
        "gradient_checkpointing": True,
        "precision": os.environ.get("PAPER_PRECISION", "fp16"),
        "decoder_checkpointing": os.environ.get("PAPER_DECODER_CHECKPOINTING", "0") == "1",
        "soma": {"rank": 16, "alpha": 32, "dropout": 0.05, "start_layer": 0,
                 "minor_components": True, "extra_weight_decay_annealing": False,
                 "note": "Matched adapter ablation; retain paper optimizer and adaptation locations rather than importing the full SoMA domain-generalization recipe."},
        "rein_g": {"token_length": 100, "scale_init": 0.001, "factorized_tokens": False,
                   "implementation": REIN_IMPLEMENTATION,
                   "note": "Official Rein-G feature adapter; DINOv3 prefix tokens excluded. No Rein-A, SAM or target-domain adaptation."},
    }
    if method == "eva":
        cfg["experiment"]["eva"] = {
            "implementation": "peft==0.18.1:initialize_lora_eva_weights",
            "author_repository": "https://github.com/ml-jku/EVA",
            "visual_code_revision": "007fd7b418bdeed98634f98363b7584487ca064c",
            "rho": 1.0, "tau": 0.99, "whiten": False,
            "adjust_scaling_factors": True,
            "calibration_images": 256, "calibration_batch_size": 4,
            "max_patch_tokens_per_batch": 16384,
            "max_forward_batches": 256, "max_initialization_seconds": 2700,
            "calibration_seed": 42,
            "note": "Fixed-rank EVA initialization ablation, matching author EVA-no-redistribution variant; all Coralscapes LoRA training hyperparameters retained. Training images only, author augmentation at fixed validation size, CLS/register tokens excluded from PCA.",
        }
    if smoke:
        cfg["train"]["output_dir"] += "-smoke"
    return cfg


def provenance(cfg):
    original = (AUTHOR / "configs/vit_l.yaml").read_bytes()
    return {"author_commits": COMMITS, "author_config_sha256": hashlib.sha256(original).hexdigest(),
            "encoder_id": ENCODER_ID, "encoder_revision": ENCODER_REVISION,
            "dataset_id": "josauder/coralscapesV2", "dataset_revision": DATA_REVISION,
            "label_policy": "Native IDs 1..95; output 0..95; ignore_index=0 (void, not the supervised background category)",
            "split_policy": "Released author config: train for training, official validation for selection, test only after training. DINO train+val combination is not explicitly specified in the paper; no assumption that SegFormer's combined split applies to DINO.",
            "evaluation": {"single_pass": [768, 1376], "also_report": "two native 1024x1024 tiles, non-overlapping, matching the released model card"},
            "comparison": "Same pretrained encoder and DPT initialization, author augmentations/optimizer/schedule/batch/epochs; adaptation method is varied. EMA is an additional parallel weight trajectory, not fed back into raw training.",
            "runtime_changes": ["Pinned offline encoder", "activation checkpointing with RNG preservation", "W&B and example plots disabled", "atomic checkpoints and complete EMA/scaler/RNG resume state"],
            "execution": {"gpu_request": os.environ.get("PAPER_GPU", "A100-80GB"),
                          "cpu_request": os.environ.get("PAPER_CPU", "16"),
                          "memory_MB": os.environ.get("PAPER_MEMORY_MB", "65536"),
                          "precision": cfg["experiment"]["precision"],
                          "decoder_checkpointing": cfg["experiment"]["decoder_checkpointing"],
                          "decoder_recompute_bn": "cloned buffers; one persistent update per batch"},
            "config": deepcopy(cfg)}


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)
