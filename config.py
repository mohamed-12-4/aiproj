"""Shared settings for local and Modal training. Edit these before a new run."""

RUN_NAME = "coral-dinov3b-lora-ensemble-3m"

TRAIN = {
    "height": 512,  # Width is 2 * height; use a multiple of the backbone patch size.
    "epochs": 60,
    "batch_size": 4,
    "lr": 1e-4,
    "head_lr": 1e-4,
    "warmup_epochs": 3,
    "warmup_steps": 500,  # Ensemble scheduler steps after each optimizer update.
    "min_lr": 1e-6,
    "weight_decay": 0.0,  # Official LoRA-Ensemble vision configuration.
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
    "enabled": True,
    "dead_min_fraction": 0.05,  # Dead fraction among valid pixels in an image.
    "dead_image_weight": 3.0,  # Other images retain weight 1; epoch length is unchanged.
}

MODEL = {
    "id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "revision": "5931719e67bbdb9737e363e781fb0c67687896bc",
    "architecture": "lora_ensemble",
    "patch_size": 16,
    "head_channels": 128,
    "feature_layers": [3, 6, 9, -1],
}

LORA = {
    "r": 8,
    "init_gain": 10.0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}

# Each member has its own adapters/head, sharing one frozen pretrained backbone.
# All members see the same augmented batch; probabilities are averaged at inference.
ENSEMBLE = {"members": 3, "ece_bins": 15}

LOSS = {
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
