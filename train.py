"""DINOv3 LoRA-Ensemble for coral-health segmentation (legacy checkpoints supported)."""

import argparse
import io
import json
import math
import os
from pathlib import Path

import config

ROOT = Path(os.environ.get("CORAL_ROOT", Path(__file__).resolve().parent))
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / ".cache/huggingface/hub"))
os.environ.setdefault("HF_XET_CACHE", str(ROOT / ".cache/huggingface/xet"))
os.environ.setdefault("USE_TF", "0")  # This project uses PyTorch only.

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image, ImageEnhance
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import AutoModel
from ensemble import CoralEnsemble

MODEL_ID = config.MODEL["id"]
OUTPUT = ROOT / "checkpoints" / config.RUN_NAME
CLASSES = ["Background", "Healthy", "Bleached", "Dead"]
CORAL_IDS = {
    1: [6, 17, 22, 25, 28, 31, 34, 36],  # Explicitly labeled alive.
    2: [4, 16, 19, 33],
    3: [3, 20, 23, 32, 37],
}
IGNORE = 255
# 0 is unlabeled; millepora (21) and turbinaria (27) have no health label.
LABEL_MAP = np.full(256, IGNORE, dtype=np.uint8)
LABEL_MAP[1:40] = 0
LABEL_MAP[[21, 27]] = IGNORE
for target, source_ids in CORAL_IDS.items():
    LABEL_MAP[source_ids] = target


def download_data():
    return Path(snapshot_download(
        "EPFL-ECEO/coralscapes", repo_type="dataset",
        local_dir=ROOT / "data/coralscapes",
        allow_patterns=["data/*.parquet", "*.json", "README.md"], max_workers=4,
    ))


def load_data(directory, height=None, prepare=True):
    """Cache small PNGs once, without loading the entire dataset into RAM."""
    height = config.TRAIN["height"] if height is None else height
    result = {}
    for split in ("train", "validation", "test"):
        output = directory / f"resized_{height}" / split
        output.mkdir(parents=True, exist_ok=True)
        shards = sorted((directory / "data").glob(f"{split}-*.parquet"))
        total = sum(pq.read_metadata(path).num_rows for path in shards)
        if not total:
            raise ValueError(f"No downloaded Parquet data found for {split}.")
        paths = [(output / f"{i:04d}_image.png", output / f"{i:04d}_mask.png")
                 for i in range(total)]
        if not (output / ".complete").exists():
            if not prepare:
                raise ValueError(f"Missing prepared {split} cache at {output}; run CPU preparation first.")
            index = 0
            with tqdm(total=total, desc=f"Preparing {split}") as progress:
                for shard in shards:
                    for batch in pq.ParquetFile(shard).iter_batches(batch_size=1, use_threads=False):
                        sample = batch.to_pylist()[0]
                        for key, path in zip(("image", "label"), paths[index]):
                            if path.exists():
                                continue
                            with Image.open(io.BytesIO(sample[key]["bytes"])) as image:
                                method = Image.Resampling.BICUBIC if key == "image" else Image.Resampling.NEAREST
                                image = image.resize((height * 2, height), method)
                                temporary = path.with_suffix(".tmp")
                                image.save(temporary, format="PNG")
                                temporary.replace(path)
                        index += 1
                        progress.update(1)
            (output / ".complete").touch()
        result[split] = paths
    return result


def sampling_weights(rows, directory, height, prepare=True):
    """Cache training-only Dead fractions on CPU; reuse them during GPU training."""
    settings = config.SAMPLING
    if not settings["enabled"]:
        return None, {"enabled": False}
    threshold, boost = settings["dead_min_fraction"], settings["dead_image_weight"]
    if not 0 < threshold <= 1 or not math.isfinite(boost) or boost < 1:
        raise ValueError("Use 0 < dead_min_fraction <= 1 and finite dead_image_weight >= 1.")
    cache = Path(directory) / f"resized_{height}" / "train_sampling.json"
    metadata = {"masks": [mask.name for _, mask in rows], "label_map": LABEL_MAP.tolist()}
    cached = json.loads(cache.read_text()) if cache.exists() else {}
    fractions = cached.get("dead_fractions", [])
    if (any(cached.get(key) != value for key, value in metadata.items())
            or len(fractions) != len(rows)):
        if not prepare:
            raise ValueError("Missing or stale training sampling statistics; run CPU preparation first.")
        fractions = []
        for _, mask_path in tqdm(rows, desc="Counting training Dead pixels (CPU)"):
            with Image.open(mask_path) as mask:
                target = LABEL_MAP[np.asarray(mask)]
            valid_pixels = np.count_nonzero(target != IGNORE)
            fractions.append(np.count_nonzero(target == 3) / max(1, valid_pixels))
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(".tmp")
        temporary.write_text(json.dumps({**metadata, "dead_fractions": fractions}) + "\n")
        temporary.replace(cache)
    fractions = np.asarray(fractions, dtype=np.float64)
    if not len(fractions) or not np.isfinite(fractions).all() or ((fractions < 0) | (fractions > 1)).any():
        raise ValueError("Training sampling statistics must contain valid fractions for every image.")
    rich = fractions >= threshold
    weights = np.where(rich, boost, 1.0)
    summary = {**settings, "train_images": len(rows), "dead_rich_images": int(rich.sum()),
               "uniform_dead_rich_share": float(rich.mean()),
               "expected_dead_rich_share": float(weights[rich].sum() / weights.sum()),
               "samples_per_epoch": len(rows), "replacement": True}
    return torch.as_tensor(weights, dtype=torch.double), summary


def image_tensor(image, size):
    image = image.convert("RGB").resize(size, Image.Resampling.BICUBIC)
    pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (pixels - mean) / std


def remap_mask(mask, size):
    ids = np.asarray(mask.resize(size, Image.Resampling.NEAREST))
    return torch.from_numpy(LABEL_MAP[ids].astype(np.int64))


def augment_pair(image, mask):
    """Apply the same geometry to image/mask, and mild color changes to image only."""
    settings = config.AUGMENT
    if torch.rand(()).item() < settings["crop_probability"]:
        fraction = math.sqrt(torch.empty(()).uniform_(*settings["crop_scale"]).item())
        width, height = image.size
        crop_width, crop_height = max(1, round(width * fraction)), max(1, round(height * fraction))
        left = torch.randint(width - crop_width + 1, ()).item()
        top = torch.randint(height - crop_height + 1, ()).item()
        box = (left, top, left + crop_width, top + crop_height)
        image, mask = image.crop(box), mask.crop(box)
    for probability, operation in ((settings["horizontal_flip"], Image.Transpose.FLIP_LEFT_RIGHT),
                                   (settings["vertical_flip"], Image.Transpose.FLIP_TOP_BOTTOM)):
        if torch.rand(()).item() < probability:
            image, mask = image.transpose(operation), mask.transpose(operation)
    for name, enhancer in (("brightness", ImageEnhance.Brightness),
                           ("contrast", ImageEnhance.Contrast), ("saturation", ImageEnhance.Color)):
        amount = settings[name]
        image = enhancer(image).enhance(torch.empty(()).uniform_(1 - amount, 1 + amount).item())
    return image, mask


class Corals(Dataset):
    def __init__(self, rows, height, augment=False):
        self.rows, self.size, self.augment = rows, (height * 2, height), augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image_path, mask_path = self.rows[index]
        with Image.open(image_path) as source, Image.open(mask_path) as mask:
            image = source.convert("RGB")
            if self.augment:
                image, mask = augment_pair(image, mask)
            return image_tensor(image, self.size), remap_mask(mask, self.size)


class CoralSegmenter(nn.Module):
    def __init__(self, adapter_dir=None, model_config=None):
        super().__init__()
        self.model_config = dict(config.MODEL if model_config is None else model_config)
        # Enable output_hidden_states to grab intermediate multi-scale features
        backbone = AutoModel.from_pretrained(
            self.model_config["id"], attn_implementation="sdpa", output_hidden_states=True,
            cache_dir=ROOT / ".cache/huggingface/hub",
        )
        self.patch_size = backbone.config.patch_size
        hidden_size = backbone.config.hidden_size
        head_channels = self.model_config["head_channels"]
        
        if adapter_dir is None:
            self.backbone = get_peft_model(backbone, LoraConfig(**config.LORA))
        else:
            self.backbone = PeftModel.from_pretrained(backbone, adapter_dir)
            
        # Fuse the transformer layers selected in config.py.
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_size * len(self.model_config["feature_layers"]), hidden_size, 1),
            nn.BatchNorm2d(hidden_size),
            nn.GELU()
        )
        
        self.head = nn.ModuleDict({
            "coral": nn.Sequential(
                nn.Conv2d(hidden_size, head_channels, 3, padding=1), nn.GELU(),
                nn.Conv2d(head_channels, 2, 1),
            ),
            "health": nn.Sequential(
                nn.Conv2d(hidden_size, head_channels, 3, padding=1), nn.GELU(),
                nn.Conv2d(head_channels, len(CLASSES) - 1, 1),
            ),
        })

    def features(self, images):
        batch, _, height, width = images.shape
        outputs = self.backbone(pixel_values=images)
        
        hidden_states = outputs.hidden_states
        selected_layers = [hidden_states[i] for i in self.model_config["feature_layers"]]
        
        # Drop CLS token, reshape, and concatenate along channel dimension
        multi_scale_features = []
        for state in selected_layers:
            tokens = state[:, 1:].transpose(1, 2).reshape(
                batch, -1, height // self.patch_size, width // self.patch_size
            )
            multi_scale_features.append(tokens)
            
        fused = torch.cat(multi_scale_features, dim=1)
        return self.fusion(fused)
    
    def hierarchical_logits(self, images):
        features = self.features(images)
        coral = self.head["coral"](features)
        health = self.head["health"](features)
        coral = F.interpolate(coral, images.shape[-2:], mode="bilinear", align_corners=False)
        health = F.interpolate(health, images.shape[-2:], mode="bilinear", align_corners=False)

        # Combine log probabilities so argmax still gives the four public classes.
        coral_log_prob = F.log_softmax(coral, dim=1)
        health_log_prob = F.log_softmax(health, dim=1)
        return coral, health, torch.cat(
            [coral_log_prob[:, :1], coral_log_prob[:, 1:] + health_log_prob], dim=1,
        )

    def forward(self, images):
        return self.hierarchical_logits(images)[2]


def focal_loss(logits, target, gamma, alpha=None):
    if target.numel() == 0:
        return logits.sum() * 0  # Keep background-only batches finite and differentiable.
    ce_loss = F.cross_entropy(logits, target, reduction="none")
    pt = torch.exp(-ce_loss)
    loss = (1 - pt) ** gamma * ce_loss
    if alpha is not None:
        loss = loss * alpha[target]
    return loss.mean()

def hierarchical_loss(coral_logits, health_logits, target):
    valid = target != IGNORE
    coral_target = (target > 0).long()
    
    # Standard Focal Loss for Coral vs Background
    coral_loss = focal_loss(coral_logits.permute(0, 2, 3, 1)[valid], coral_target[valid],
                            gamma=config.LOSS["coral_gamma"])
    
    health_valid = (target != IGNORE) & (target > 0)
    health_target = (target[health_valid] - 1).long()
    
    device = health_logits.device
    health_weights = torch.tensor(config.LOSS["health_weights"], device=device, dtype=torch.float)
    
    health_loss = focal_loss(
        health_logits.permute(0, 2, 3, 1)[health_valid], 
        health_target,
        gamma=config.LOSS["health_gamma"],
        alpha=health_weights
    )
    
    return coral_loss + health_loss

def class_weights(rows, size):
    counts = torch.zeros(len(CLASSES), dtype=torch.float64)
    for _, mask_path in tqdm(rows, desc="Counting training pixels"):
        with Image.open(mask_path) as mask:
            target = remap_mask(mask, size)
        counts += torch.bincount(target[target != IGNORE], minlength=len(CLASSES))
    # Reduce class imbalance without giving rare pixels extreme weights.
    weights = (counts.sum() / counts.clamp_min(1)).sqrt()
    return (weights / weights.mean()).float()


def scores(confusion):
    """Dataset-wide pixel metrics; rows are truth and columns are predictions."""
    confusion = confusion.double()
    true, predicted, correct = confusion.sum(1), confusion.sum(0), confusion.diag()

    def divide(numerator, denominator):
        return torch.where(denominator > 0, numerator / denominator, float("nan"))

    def number(value):
        return value.item() if torch.isfinite(value) else None

    iou = divide(correct, true + predicted - correct)
    f1 = divide(2 * correct, true + predicted)  # Pixel F1 is also Dice.
    result = {
        "pixel_accuracy": number(divide(correct.sum(), true.sum())),
        "pixel_f1": number(f1.nanmean()),
        "miou": number(iou.nanmean()),
        "coral_miou": number(iou[1:].nanmean()),
        "macro_f1": number(f1.nanmean()),
        "coral_f1": number(f1[1:].nanmean()),
    }
    for name, values in {"iou": iou, "f1": f1,
                         "precision": divide(correct, predicted),
                         "recall": divide(correct, true)}.items():
        result[name] = {label: number(value) for label, value in zip(CLASSES, values)}
    result["support"] = {label: int(value) for label, value in zip(CLASSES, true)}
    return result


def amp_dtype(device):
    precision = config.TRAIN["precision"]
    if precision not in ("bf16", "fp16", "fp32"):
        raise ValueError("TRAIN['precision'] must be bf16, fp16 or fp32.")
    if device.type != "cuda" or precision == "fp32":
        return None
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support BF16; choose a supported GPU or change precision.")
    return torch.bfloat16 if precision == "bf16" else torch.float16


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    confusion = torch.zeros((len(CLASSES), len(CLASSES)), dtype=torch.int64, device=device)
    ensemble = isinstance(model, CoralEnsemble)
    member_confusions = [torch.zeros_like(confusion) for _ in range(model.n_members)] if ensemble else []
    bins = config.ENSEMBLE["ece_bins"]
    bin_count = torch.zeros(bins, dtype=torch.float64, device=device)
    bin_correct, bin_confidence = bin_count.clone(), bin_count.clone()
    nll = brier = entropy = disagreement = count = 0.0
    for images, target in tqdm(loader, desc="Evaluating"):
        images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
        dtype = amp_dtype(device)
        valid = target != IGNORE
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            if ensemble:
                stats = model.prediction_stats(images, with_members=True)
                probability = stats["probabilities"]
                prediction = probability.argmax(1)
            else:
                prediction = model(images).argmax(1)
        pairs = len(CLASSES) * target[valid] + prediction[valid]
        confusion += torch.bincount(pairs, minlength=len(CLASSES) ** 2).reshape_as(confusion)
        if ensemble and valid.any():
            for matrix, member in zip(member_confusions, stats["members"]):
                pairs = len(CLASSES) * target[valid] + member.argmax(1)[valid]
                matrix += torch.bincount(pairs, minlength=16).reshape(4, 4)
            prob = probability.permute(0, 2, 3, 1)[valid]
            truth = target[valid]
            true_prob = prob.gather(1, truth[:, None]).squeeze(1)
            nll += -true_prob.clamp_min(1e-8).log().double().sum().item()
            brier += (prob.square().sum(1) - 2 * true_prob + 1).double().sum().item()
            entropy += stats["entropy"][valid].double().sum().item()
            disagreement += stats["mutual_information"][valid].double().sum().item()
            confidence, predicted = prob.max(1)
            indices = (confidence * bins).long().clamp_max(bins - 1)
            bin_count += torch.bincount(indices, minlength=bins)
            bin_confidence.scatter_add_(0, indices, confidence.double())
            bin_correct.scatter_add_(0, indices, (predicted == truth).double())
            count += truth.numel()
    result = scores(confusion.cpu())
    if ensemble:
        result["members"] = [scores(matrix.cpu()) for matrix in member_confusions]
        result["uncertainty"] = {
            "nll": nll / max(1, count), "brier": brier / max(1, count),
            "ece": (bin_correct - bin_confidence).abs().sum().item() / max(1, count),
            "ece_bins": bins, "predictive_entropy": entropy / max(1, count),
            "mutual_information": disagreement / max(1, count),
        }
    return result


def save_checkpoint(model, height, epoch, validation, directory=OUTPUT):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if isinstance(model, CoralEnsemble):
        temporary = directory / "head.tmp"
        torch.save({"format": "lora_ensemble_v1", "weights": model.trainable_state(),
                    "height": height, "classes": CLASSES, "epoch": epoch,
                    "validation": validation, "model_config": model.model_config,
                    "lora_config": model.lora_config,
                    "ensemble_config": model.ensemble_config}, temporary)
        temporary.replace(directory / "head.pt")
        return
    model.backbone.save_pretrained(directory / "adapter", save_embedding_layers=False)
    torch.save({"head": model.head.state_dict(), "fusion": model.fusion.state_dict(),
                "height": height, "classes": CLASSES, "epoch": epoch,
                "validation": validation, "model_config": model.model_config}, directory / "head.pt")


def load_checkpoint(directory=OUTPUT, local_files_only=False):
    state = torch.load(Path(directory) / "head.pt", map_location="cpu", weights_only=True)
    if state.get("format") == "lora_ensemble_v1":
        model = CoralEnsemble(state["model_config"], state["lora_config"], state["ensemble_config"],
                              ROOT / ".cache/huggingface/hub", local_files_only=local_files_only)
        model.load_trainable_state(state["weights"])
        return model, state["height"]
    if "fusion" not in state:
        raise ValueError("Checkpoint is missing fusion weights. Retrain with the updated train.py.")
    model_config = state.get("model_config")
    if model_config is None:
        # Existing checkpoints predate config.py and used this fixed architecture.
        adapter_config = json.loads((Path(directory) / "adapter/adapter_config.json").read_text())
        model_config = {"id": adapter_config["base_model_name_or_path"],
                        "head_channels": state["head"]["coral.0.weight"].shape[0],
                        "feature_layers": [3, 6, 9, -1]}
    model = CoralSegmenter(adapter_dir=Path(directory) / "adapter", model_config=model_config)
    model.head.load_state_dict(state["head"])
    model.fusion.load_state_dict(state["fusion"])
    return model, state["height"]


def make_scheduler(optimizer, epochs, warmup_epochs, min_lr):
    """Linear warmup then cosine decay; arguments can count epochs or updates."""
    def multiplier(epoch, minimum):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = min(1.0, (epoch - warmup_epochs) / max(1, epochs - warmup_epochs - 1))
        return minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2

    multipliers = [lambda epoch, minimum=min_lr / group["lr"]: multiplier(epoch, minimum)
                   for group in optimizer.param_groups]
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multipliers)


def main(argv=None, on_epoch_end=None, prepared_data=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=config.TRAIN["epochs"])
    parser.add_argument("--batch-size", type=int, default=config.TRAIN["batch_size"])
    parser.add_argument("--height", type=int, default=config.TRAIN["height"], help="Multiple of backbone patch size; width is twice this")
    parser.add_argument("--lr", type=float, default=config.TRAIN["lr"])
    parser.add_argument("--warmup-epochs", type=int, default=config.TRAIN["warmup_epochs"], help="Linear warmup before cosine decay")
    parser.add_argument("--warmup-steps", type=int, default=config.TRAIN["warmup_steps"], help="Ensemble warmup in optimizer updates; overrides warmup-epochs")
    parser.add_argument("--min-lr", type=float, default=config.TRAIN["min_lr"], help="Final learning rate after cosine decay")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="Directory for this run's checkpoint and metrics")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args(argv)
    patch = config.MODEL.get("patch_size", 14)
    if args.height < patch or args.height % patch or args.epochs < 1 or args.batch_size < 1:
        parser.error(f"Use positive epochs/batch size and a height divisible by {patch}.")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be nonnegative.")
    if not 0 <= args.warmup_epochs < args.epochs:
        parser.error("--warmup-epochs must be between 0 and epochs - 1.")
    if not (math.isfinite(args.lr) and math.isfinite(args.min_lr) and 0 < args.min_lr <= args.lr):
        parser.error("Use finite learning rates with 0 < min-lr <= lr.")

    directory = ROOT / "data/coralscapes" if prepared_data else download_data()
    if args.download_only:
        print(f"Dataset downloaded to {directory}")
        return
    torch.manual_seed(config.TRAIN["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = amp_dtype(device)
    print(f"Device: {device}; precision: {dtype or torch.float32}", flush=True)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)
    data = load_data(directory, args.height, prepare=not prepared_data)
    weights, sampling_summary = sampling_weights(
        data["train"], directory, args.height, prepare=not prepared_data,
    )
    print(f"Training sampling: {json.dumps(sampling_summary)}", flush=True)
    if on_epoch_end:
        on_epoch_end()  # Persist downloaded data and resized masks on Modal.
    train_set = Corals(data["train"], args.height, augment=True)
    loader_options = {"batch_size": args.batch_size, "num_workers": config.TRAIN["num_workers"],
                      "pin_memory": device.type == "cuda",
                      "persistent_workers": config.TRAIN["num_workers"] > 0}
    sampler = None if weights is None else WeightedRandomSampler(
        weights, num_samples=len(train_set), replacement=True,
        generator=torch.Generator().manual_seed(config.TRAIN["seed"]),
    )
    train_loader = DataLoader(train_set, sampler=sampler, shuffle=sampler is None, **loader_options)
    val_loader = DataLoader(Corals(data["validation"], args.height), **loader_options)
    ensemble = config.MODEL.get("architecture") == "lora_ensemble"
    model = (CoralEnsemble(config.MODEL, config.LORA, config.ENSEMBLE,
                          ROOT / ".cache/huggingface/hub", local_files_only=prepared_data)
             if ensemble else CoralSegmenter()).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}", flush=True)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": args.lr},
        {"params": (list(model.decoders.parameters()) if ensemble else
                    list(model.fusion.parameters()) + list(model.head.parameters())),
         "lr": config.TRAIN["head_lr"]},
    ], weight_decay=config.TRAIN["weight_decay"])
    total_updates = args.epochs * len(train_loader)
    if ensemble and args.warmup_steps >= total_updates:
        raise ValueError("warmup_steps must be smaller than total optimizer updates.")
    scheduler = make_scheduler(optimizer, total_updates if ensemble else args.epochs,
                               args.warmup_steps if ensemble else args.warmup_epochs, args.min_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    best = -1.0
    history = []
    args.output.mkdir(parents=True, exist_ok=True)

    def write_metrics(test=None):
        temporary = args.output / "metrics.tmp"
        temporary.write_text(json.dumps(
            {"settings": {**config.TRAIN, **vars(args), "model": config.MODEL,
                          "lora": config.LORA, "loss": config.LOSS,
                          "ensemble": config.ENSEMBLE if ensemble else None,
                          "effective_loss": "uniform_cross_entropy" if ensemble else "hierarchical_focal",
                          "upstream_commit": "812ced225717fdcce3c1e236e5dd17d14c2b1b2b" if ensemble else None,
                          "augmentation": config.AUGMENT,
                          "sampling": config.SAMPLING,
                          "effective_precision": str(dtype or torch.float32)},
             "sampling_summary": sampling_summary, "history": history, "test": test},
            indent=2, default=str, allow_nan=False,
        ) + "\n")
        temporary.replace(args.output / "metrics.json")

    for epoch in range(1, args.epochs + 1):
        model.train()
        learning_rate = optimizer.param_groups[0]["lr"]
        head_learning_rate = optimizer.param_groups[1]["lr"]
        total_loss, steps = 0.0, 0
        for images, target in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
            if not (target != IGNORE).any():
                continue
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            # Same images/augmentation and one optimizer update for all members.
            # Backpropagate each member immediately: equivalent mean loss, less activation RAM.
            for member in range(model.n_members if ensemble else 1):
                with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                    if ensemble:
                        logits = model.member_logits(images, member)
                        loss = F.cross_entropy(logits.float(), target, ignore_index=IGNORE) / model.n_members
                    else:
                        coral_logits, health_logits, _ = model.hierarchical_logits(images)
                        loss = hierarchical_loss(coral_logits, health_logits, target)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite training loss in epoch {epoch}, member {member}.")
                scaler.scale(loss).backward()
                batch_loss += loss.item()
                if ensemble:
                    del logits, loss  # Release this member's graph before the next forward.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, config.TRAIN["gradient_clip"], error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            if ensemble:
                scheduler.step()
            total_loss += batch_loss
            steps += 1
            if epoch == 1 and steps == 1:
                memory = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0
                print(f"First optimizer update passed; loss={batch_loss:.4f}; "
                      f"peak allocated GPU memory={memory:.2f} GiB", flush=True)
        validation = evaluate(model, val_loader, device)
        record = {"epoch": epoch, "lr": learning_rate, "head_lr": head_learning_rate,
                  "loss": total_loss / max(steps, 1), "validation": validation}
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation["coral_miou"] is not None and validation["coral_miou"] > best:
            best = validation["coral_miou"]
            save_checkpoint(model, args.height, epoch, validation, args.output)
        if steps and not ensemble:
            scheduler.step()
        write_metrics()
        if on_epoch_end:
            on_epoch_end()

    # Select using validation only, then evaluate the selected checkpoint on test once.
    del model, optimizer, scheduler, trainable
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model, _ = load_checkpoint(args.output, local_files_only=prepared_data)
    model.to(device)
    test_loader = DataLoader(Corals(data["test"], args.height), **loader_options)
    test = evaluate(model, test_loader, device)
    write_metrics(test)
    print(f"Test: {json.dumps(test)}\nSaved model: {args.output}")


if __name__ == "__main__":
    main()
