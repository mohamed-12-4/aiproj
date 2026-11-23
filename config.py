"""Shared settings for local and Modal training. Edit these before a new run."""

RUN_NAME = "coral-dinov3b-dpt-p1-s42"
DATASET = {"annotation_version": "V1"}  # --annotation-version V2 selects a separate run/cache.

TRAIN = {
    "height": 512,  # Width is 2 * height; use a multiple of the backbone patch size.
    "epochs": 20,  # Screening ceiling; validation early stopping can finish sooner.
    "batch_size": 1,  # Microbatch; four microbatches per optimizer update.
    "accumulation_steps": 4,
    "max_updates": 4000,  # Hard optimizer-update ceiling, including partial epochs.
    "early_stopping_patience": 5,
    "early_stopping_min_epochs": 5,
    "early_stopping_min_delta": 0.001,  # Absolute validation coral mIoU.
    "lr": 1e-4,
    "head_lr": 1e-4,
    "warmup_epochs": 3,
    "warmup_steps": 500,  # Uniform-CE scheduler steps after each optimizer update.
    "min_lr": 1e-6,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "seed": 42,
    "precision": "bf16",  # CUDA: bf16/fp16/fp32; CPU uses fp32.
    "num_workers": 2,
}

AUGMENT = {
    "crop_probability": 0.8,
    "crop_scale": [0.6, 1.0],  # Fraction of image area; preserve aspect ratio.
    "horizontal_flip": 0.5,
    "vertical_flip": 0.2,
    "brightness": 0.15,
    "contrast": 0.15,
    "saturation": 0.08,
}

SAMPLING = {
    # P1 retains image weighting; subtype-aware/native-resolution crops start in P2.
    "enabled": True,
    "dead_min_fraction": 0.05,  # Dead fraction among valid pixels in an image.
    "dead_image_weight": 3.0,  # Other images retain weight 1; epoch length is unchanged.
}

MODEL = {
    "id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "revision": "5931719e67bbdb9737e363e781fb0c67687896bc",
    "architecture": "dinov3_single",
    "decoder": "dpt",  # "shallow" provides a matched single-model control.
    "num_classes": 4,
    "ignore_index": 255,  # Background (0) remains supervised.
    "patch_size": 16,
    "head_channels": 128,
    "feature_layers": [3, 6, 9, -1],
    "dpt_neck_channels": [96, 192, 384, 768],
    "dpt_fusion_channels": 256,
    "dpt_reassemble_factors": [4, 2, 1, 0.5],
    "dpt_dropout": 0.1,
}

LORA = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}

# Legacy ensemble evaluation settings. P1 trains one encoder and one decoder.
ENSEMBLE = {"members": 3, "ece_bins": 15}

LOSS = {
    # Used only by the legacy hierarchical DINOv2 path; P1 uses uniform CE.
    "coral_gamma": 1.0,
    "health_gamma": 1.0,
    "health_weights": [1.0, 1.5, 2.0],  # Healthy, Bleached, Dead.
}

MODAL = {
    "gpu": "A10",
    "cpu": 4,
    "memory": 16384,  # MiB of CPU RAM.
    "timeout": 86400,  # Seconds.
    "volume": "coral-training",
}
