"""Evaluate a saved checkpoint and plot predictions beside ground truth."""

import argparse
import csv
import io
import json
import os
from pathlib import Path

from train import (CLASSES, IGNORE, LABEL_MAP, OUTPUT, ROOT,
                   image_tensor, load_checkpoint, remap_mask, scores)
from coral_data import checkpoint_dataset, data_directory

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")  # Save figures without opening a window (works in WSL).
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.nn import functional as F
from tqdm.auto import tqdm

COLORS = np.full((256, 3), 145, dtype=np.uint8)  # Gray = ignored pixels.
COLORS[:4] = [[0, 0, 0], [40, 190, 80], [255, 240, 130], [220, 65, 60]]


def samples_from(shards):
    """Read local Parquet files one image at a time; no new dataset cache."""
    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(
                batch_size=1, columns=["image", "label"], use_threads=False):
            sample = batch.to_pylist()[0]
            with Image.open(io.BytesIO(sample["image"]["bytes"])) as image:
                image = image.convert("RGB")
            with Image.open(io.BytesIO(sample["label"]["bytes"])) as mask:
                mask = mask.copy()
            yield image, mask, sample["image"].get("path")


def save_sample(image, target, prediction, path):
    # Ground-truth ignore regions are gray in both masks for a fair comparison.
    visible = prediction.copy()
    visible[target == IGNORE] = IGNORE
    display_image = image.resize((target.shape[1], target.shape[0]), Image.Resampling.BICUBIC)
    overlay = np.asarray(display_image).copy()
    coral = (visible != 0) & (visible != IGNORE)
    overlay[coral] = (0.55 * overlay[coral] + 0.45 * COLORS[visible[coral]]).astype(np.uint8)
    panels = [display_image, COLORS[target], COLORS[visible], overlay]
    titles = ["Image", "Ground truth", "Prediction", "Prediction overlay"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.1))
    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    legend = [Patch(color=COLORS[i] / 255, label=name) for i, name in enumerate(CLASSES)]
    legend.append(Patch(color=COLORS[IGNORE] / 255, label="Ignored (GT)"))
    fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_confusion(confusion, path):
    counts = confusion.numpy()
    totals = counts.sum(1, keepdims=True)
    percentages = np.divide(100.0 * counts, totals,
                            out=np.zeros_like(counts, dtype=float), where=totals > 0)
    fig, ax = plt.subplots(figsize=(6, 5))
    plot = ax.imshow(percentages, vmin=0, vmax=100, cmap="Blues")
    for row in range(len(CLASSES)):
        for col in range(len(CLASSES)):
            label = f"{percentages[row, col]:.1f}%" if totals[row, 0] else "N/A"
            ax.text(col, row, label, ha="center", va="center",
                    color="white" if percentages[row, col] > 50 else "black")
    ax.set(xticks=range(4), yticks=range(4), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicted class", ylabel="Ground truth", title="Confusion matrix (row %)")
    fig.colorbar(plot, ax=ax, label="% of ground-truth class pixels")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def percent(value):
    return "N/A" if value is None else f"{100 * value:.2f}%"


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=OUTPUT, help="Saved adapter and head directory")
    parser.add_argument("--split", choices=["test", "validation"], default="test")
    parser.add_argument("--samples", type=int, default=6, help="Number of randomly selected comparison figures")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for selecting figures")
    parser.add_argument("--resolution", choices=["original", "training"], default="original",
                        help="Mask resolution for scoring; model input always uses checkpoint height")
    parser.add_argument("--output", type=Path, help="Default: CHECKPOINT/evaluation/SPLIT_RESOLUTION")
    args = parser.parse_args()
    if args.samples < 0 or args.seed < 0:
        parser.error("--samples and --seed must be nonnegative.")
    if not (args.checkpoint / "head.pt").is_file():
        parser.error(f"No head.pt in {args.checkpoint}; finish training first.")
    annotation = checkpoint_dataset(args.checkpoint)
    label_map = np.asarray(annotation["label_map"], dtype=np.uint8)
    shards = sorted((data_directory(ROOT, annotation) / "data").glob(f"{args.split}-*.parquet"))
    total = sum(pq.read_metadata(path).num_rows for path in shards)
    if not total:
        parser.error("Local dataset missing. Run python train.py --download-only first.")

    torch.set_num_threads(4)
    model, height = load_checkpoint(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    output = args.output or args.checkpoint / "evaluation" / f"{args.split}_{args.resolution}"
    output.mkdir(parents=True, exist_ok=True)
    selected = set(np.random.default_rng(args.seed).choice(total, min(args.samples, total), replace=False).tolist())
    confusion = torch.zeros((len(CLASSES), len(CLASSES)), dtype=torch.int64)
    ignored_pixels, examples = 0, []
    print(f"Device: {device}; input: {height}x{height * 2}; masks: {args.resolution}; images: {total}", flush=True)

    for index, (image, mask, source) in enumerate(tqdm(samples_from(shards), total=total, desc=args.split)):
        size = mask.size if args.resolution == "original" else (height * 2, height)
        target = remap_mask(mask, size, label_map).numpy()
        pixels = image_tensor(image, (height * 2, height)).unsqueeze(0).to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(pixels)
        if hasattr(model, "n_members"):
            logits = logits.float().exp()  # Resize the mean probabilities for ensembles.
        # Interpolate logits before argmax; never interpolate class IDs bilinearly.
        logits = F.interpolate(logits.float(), size=target.shape, mode="bilinear", align_corners=False)
        prediction = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        valid = target != IGNORE
        pairs = len(CLASSES) * target[valid] + prediction[valid]
        confusion += torch.from_numpy(np.bincount(pairs, minlength=len(CLASSES) ** 2).reshape(len(CLASSES), len(CLASSES)))
        ignored_pixels += int((~valid).sum())
        if index in selected:
            filename = f"sample_{index:04d}.png"
            save_sample(image, target, prediction, output / filename)
            examples.append({"index": index, "source": source, "figure": filename})

    metrics = scores(confusion)
    report = {
        "checkpoint": str(args.checkpoint.resolve()), "model": model.model_config["id"],
        "split": args.split, "num_images": total, "classes": CLASSES,
        "input_size_hw": [height, height * 2], "mask_resolution": args.resolution,
        "valid_pixels": int(confusion.sum()), "ignored_pixels": ignored_pixels,
        "annotation_version": annotation["annotation_version"], "dataset": annotation,
        "ignored_source_ids": np.flatnonzero(label_map[:annotation["num_source_classes"] + 1] == IGNORE).tolist(),
        "source_id_to_class_id": label_map[:annotation["num_source_classes"] + 1].tolist(),
        "seed": args.seed, "samples": examples,
        "metrics": metrics, "confusion_matrix": confusion.tolist(),
        "confusion_matrix_axes": "rows=ground truth, columns=prediction",
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (output / "per_class.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["class", "precision", "recall", "f1_dice", "iou", "support_pixels"])
        for name in CLASSES:
            writer.writerow([name] + [metrics[key][name] for key in ("precision", "recall", "f1", "iou", "support")])
    save_confusion(confusion, output / "confusion_matrix.png")
    print("\n" + "\n".join(f"{key}: {percent(metrics[key])}" for key in
                          ("pixel_accuracy", "miou", "coral_miou", "macro_f1", "coral_f1")))
    print(f"\n{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1/Dice':>10} {'IoU':>10}")
    for name in CLASSES:
        print(f"{name:<12} " + " ".join(f"{percent(metrics[key][name]):>10}" for key in ("precision", "recall", "f1", "iou")))
    print(f"\nSaved metrics and figures to {output}")


if __name__ == "__main__":
    main()
