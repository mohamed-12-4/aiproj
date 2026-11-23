"""Measure class coverage and diagnose missed Dead pixels in a saved model."""

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from train import (CLASSES, CORAL_IDS, IGNORE, LABEL_MAP, OUTPUT, ROOT,
                   amp_dtype, image_tensor, load_checkpoint, scores)
from coral_data import checkpoint_dataset, data_directory, dataset_config


def labels():
    path = Path(__file__).parent / "coralscapes_metadata/id2label.json"
    return {0: "unmapped", **{int(k): v for k, v in json.loads(path.read_text()).items()}}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ratio(a, b):
    return float(a / b) if b else None


def class_counts(original):
    target = LABEL_MAP[original]
    return np.array([np.count_nonzero(target == c) for c in [0, 1, 2, 3, IGNORE]], dtype=np.int64)


def coverage(directory, height, output, split="validation", annotation=None):
    """Decode/count masks only on CPU. Preserve source filenames for GPU inference."""
    directory, output = Path(directory), Path(output)
    annotation = dataset_config("V1") if annotation is None else annotation
    legacy_lookup = np.asarray(annotation["source_to_legacy_id"], dtype=np.uint8)
    output.mkdir(parents=True, exist_ok=True)
    original_counts, resized_counts, original_fine = [], [], []
    rows = []
    for shard in sorted((directory / "data").glob("train-*.parquet")):
        for batch in pq.ParquetFile(shard).iter_batches(
                batch_size=1, columns=["label", "image.path"], use_threads=False):
            sample = batch.to_pylist()[0]
            with Image.open(io.BytesIO(sample["label"]["bytes"])) as mask:
                original = legacy_lookup[np.array(mask)]
                resized = legacy_lookup[np.array(mask.resize((height * 2, height), Image.Resampling.NEAREST))]
            counts, small = class_counts(original), class_counts(resized)
            fine = np.bincount(original.ravel(), minlength=256)
            original_counts.append(counts)
            resized_counts.append(small)
            original_fine.append(fine)
            row = {"index": len(rows), "image": sample["image"]["path"],
                   "mask": sample["label"]["path"]}
            for name, values in (("original", counts), ("training", small)):
                valid = int(values[:4].sum())
                for c, label in enumerate(CLASSES + ["Ignored"]):
                    row[f"{name}_{label}_pixels"] = int(values[c])
                    if c < 4:
                        row[f"{name}_{label}_fraction_valid"] = ratio(values[c], valid)
            rows.append(row)
    if not rows:
        raise ValueError("No training masks found.")
    original_counts, resized_counts = np.stack(original_counts), np.stack(resized_counts)
    original_fine = np.stack(original_fine)
    summary = {"images": len(rows), "training_size_hw": [height, height * 2],
               "annotation_version": annotation["annotation_version"],
               "category_taxonomy": annotation["dead_category_taxonomy"],
               "coverage_definition": "images containing at least one pixel of a class",
               "counts_are_before_augmentation_and_oversampling": True}
    coverage_rows = []
    for name, values in (("original", original_counts), ("training", resized_counts)):
        totals = values.sum(0)
        valid = int(totals[:4].sum())
        fractions = values[:, :4] / np.maximum(1, values[:, :4].sum(1))[:, None]
        table = []
        for c, label in enumerate(CLASSES):
            row = {"resolution": name, "class": label, "pixels": int(totals[c]),
                   "pixel_share_valid": ratio(totals[c], valid),
                   "images_present": int((values[:, c] > 0).sum()),
                   "image_coverage": float((values[:, c] > 0).mean()),
                   "images_at_least_1pct_valid": int((fractions[:, c] >= .01).sum()),
                   "images_at_least_5pct_valid": int((fractions[:, c] >= .05).sum())}
            table.append(row)
            coverage_rows.append(row)
        summary[name] = {"valid_pixels": valid, "ignored_pixels": int(totals[4]), "classes": table}
    names = labels()
    fine_rows = [{"original_id": sid, "original_label": names.get(sid, str(sid)),
                  "mapped_class": CLASSES[int(LABEL_MAP[sid])] if LABEL_MAP[sid] != IGNORE else "Ignored",
                  "pixels": int(original_fine[:, sid].sum()),
                  "images_present": int((original_fine[:, sid] > 0).sum())}
                 for sid in range(40)]
    # Exact expected pre-augmentation exposure under the configured image sampler.
    import config
    settings = config.SAMPLING
    fractions = resized_counts[:, 3] / np.maximum(1, resized_counts[:, :4].sum(1))
    weights = np.where(fractions >= settings["dead_min_fraction"], settings["dead_image_weight"], 1.)
    if not settings["enabled"]:
        weights[:] = 1
    weighted = (weights[:, None] * resized_counts[:, :4]).sum(0)
    summary["sampler"] = {"settings": settings,
                          "expected_unique_images_per_epoch": float((1 - (1 - weights / weights.sum()) ** len(rows)).sum()),
                          "expected_pixel_shares_before_augmentation": dict(zip(CLASSES, (weighted / weighted.sum()).tolist()))}
    write_csv(output / "training_class_coverage.csv", coverage_rows)
    write_csv(output / "training_per_image.csv", rows)
    write_csv(output / "training_original_categories.csv", fine_rows)
    write_json(output / "training_coverage.json", summary)

    sources = []
    for shard in sorted((directory / "data").glob(f"{split}-*.parquet")):
        for batch in pq.ParquetFile(shard).iter_batches(
                batch_size=64, columns=["image.path", "label.path"], use_threads=False):
            for sample in batch.to_pylist():
                sources.append({"index": len(sources), "image": sample["image"]["path"],
                                "mask": sample["label"]["path"]})
    if not sources:
        raise ValueError(f"No {split} images found.")
    write_json(output / f"{split}_sources.json", sources)
    print(json.dumps(summary), flush=True)
    return summary


def source_confusion(original, prediction):
    """Rows: original source IDs; columns: four predicted classes."""
    if original.shape != prediction.shape:
        raise ValueError("Mask and prediction shapes differ.")
    if original.max(initial=0) > 39 or prediction.max(initial=0) > 3:
        raise ValueError("Unexpected source or prediction class ID.")
    return np.bincount(4 * original.astype(np.int64).ravel() + prediction.ravel(),
                       minlength=40 * 4).reshape(40, 4)


def dead_row(matrix):
    counts = matrix[CORAL_IDS[3]].sum(0)
    support = int(counts.sum())
    missed = int(counts[:3].sum())
    result = {"dead_pixels": support, "dead_correct": int(counts[3]),
              "dead_recall": ratio(counts[3], support), "dead_missed": missed}
    for c, label in enumerate(CLASSES[:3]):
        result[f"dead_to_{label}_pixels"] = int(counts[c])
        result[f"dead_to_{label}_fraction_all_dead"] = ratio(counts[c], support)
        result[f"dead_to_{label}_fraction_missed_dead"] = ratio(counts[c], missed)
    return result


class DiagnosticImages(Dataset):
    def __init__(self, directory, height, split, sources):
        self.directory = Path(directory) / f"resized_{height}" / split
        self.size = (height * 2, height)
        self.sources = sources
        if not (self.directory / ".complete").exists():
            raise ValueError("Prepared data cache is missing; run CPU preparation first.")

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, index):
        with Image.open(self.directory / f"{index:04d}_image.png") as image:
            pixels = image_tensor(image, self.size)
        with Image.open(self.directory / f"{index:04d}_mask.png") as mask:
            original = torch.from_numpy(np.array(mask).astype(np.int64))
        return pixels, original, index


@torch.inference_mode()
def diagnose(checkpoint, directory, output, split="validation", batch_size=4):
    checkpoint, output = Path(checkpoint), Path(output)
    state = torch.load(checkpoint / "head.pt", map_location="cpu", weights_only=True)
    epoch = state["epoch"]
    reference_validation = state["validation"]
    del state
    model, height = load_checkpoint(checkpoint, local_files_only=True)
    annotation = model.dataset_config
    legacy_lookup = np.asarray(annotation["source_to_legacy_id"], dtype=np.uint8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    model.to(device).eval()
    sources = json.loads((output / f"{split}_sources.json").read_text())
    dataset = DiagnosticImages(directory, height, split, sources)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=2, pin_memory=device.type == "cuda")
    total = np.zeros((40, 4), dtype=np.int64)
    per_image, per_category_image = [], []
    dtype = amp_dtype(device)
    print(f"Diagnosing saved epoch {epoch}; {split}; {height}x{height * 2}; {device}; {dtype}", flush=True)
    names = labels()
    for pixels, original, indices in tqdm(loader, desc="Diagnosing"):
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            prediction = model(pixels.to(device)).argmax(1).cpu().numpy()
        for raw, pred, index in zip(original.numpy(), prediction, indices.tolist()):
            matrix = source_confusion(legacy_lookup[raw], pred)
            total += matrix
            per_image.append({**sources[index], **dead_row(matrix)})
            for sid in CORAL_IDS[3]:
                support = int(matrix[sid].sum())
                if support:
                    per_category_image.append({**sources[index], "original_id": sid,
                                               "original_label": names[sid], "pixels": support,
                                               "correct": int(matrix[sid, 3]),
                                               "recall": ratio(matrix[sid, 3], support)})
    confusion = np.stack([total[np.flatnonzero(LABEL_MAP[:40] == c)].sum(0) for c in range(4)])
    categories = []
    for sid in CORAL_IDS[3]:
        values = total[sid]
        support = int(values.sum())
        row = {"original_id": sid, "original_label": names[sid], "dead_pixels": support,
               "images_present": sum(x["original_id"] == sid for x in per_category_image),
               "dead_correct": int(values[3]), "dead_recall": ratio(values[3], support)}
        row.update({f"predicted_{label}_pixels": int(values[c]) for c, label in enumerate(CLASSES)})
        categories.append(row)
    assert sum(r["dead_pixels"] for r in per_image) == sum(r["dead_pixels"] for r in categories)
    assert sum(r["dead_correct"] for r in per_image) == sum(r["dead_correct"] for r in categories)
    positive = [r for r in per_image if r["dead_pixels"] > 0]
    substantial = [r for r in positive if r["dead_pixels"] >= 1000]
    summary = {"checkpoint": str(checkpoint), "checkpoint_epoch": epoch, "split": split,
               "annotation_version": annotation["annotation_version"],
               "category_taxonomy": annotation["dead_category_taxonomy"],
               "mask_resolution": "training", "height": height, "width": height * 2,
               "images": len(sources), "metrics": scores(torch.from_numpy(confusion)),
               "confusion_matrix": confusion.tolist(), "confusion_axes": "rows=truth; columns=prediction",
               "dead": dead_row(total), "dead_categories": categories,
               "images_with_dead": len(positive),
               "mean_per_image_dead_recall": float(np.mean([r["dead_recall"] for r in positive])) if positive else None,
               "median_per_image_dead_recall": float(np.median([r["dead_recall"] for r in positive])) if positive else None,
               "worst_images_minimum_1000_dead_pixels": sorted(substantial, key=lambda r: r["dead_recall"])[:12]}
    if split == "validation":
        summary["difference_from_saved_validation"] = {
            key: summary["metrics"][key] - reference_validation[key]
            for key in ("pixel_accuracy", "miou", "coral_miou")}
    write_csv(output / "dead_per_image.csv", per_image)
    write_csv(output / "dead_per_image_category.csv", per_category_image)
    write_csv(output / "dead_by_original_category.csv", categories)
    write_json(output / "validation_diagnostics.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=OUTPUT)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--stage", choices=["coverage", "predict", "all"], default="all")
    parser.add_argument("--height", type=int, default=512)
    args = parser.parse_args()
    annotation = checkpoint_dataset(args.checkpoint) if (args.checkpoint / "head.pt").exists() else dataset_config("V1")
    args.data = args.data or data_directory(ROOT, annotation)
    output = args.output or args.checkpoint / "diagnostics" / args.split
    if args.stage in ("coverage", "all"):
        coverage(args.data, args.height, output, args.split, annotation)
    if args.stage in ("predict", "all"):
        diagnose(args.checkpoint, args.data, output, args.split)


if __name__ == "__main__":
    main()
