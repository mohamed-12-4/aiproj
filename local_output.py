"""Compact local run output: one epoch table and one fixed validation montage."""

import csv
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


PALETTE = np.array([[25, 25, 25], [40, 190, 80], [255, 240, 130], [220, 65, 60]], dtype=np.uint8)
FIELDS = ("epoch", "optimizer_updates", "train_loss", "val_loss", "val_miou", "val_coral_miou",
          "background_iou", "healthy_iou", "bleached_iou", "dead_iou", "dead_precision", "dead_recall",
          "pixel_accuracy", "lr", "head_lr", "seconds", "peak_vram_gib", "is_best")


@contextmanager
def quiet_downloads():
    from huggingface_hub.utils import are_progress_bars_disabled, disable_progress_bars, enable_progress_bars
    from transformers.utils import logging
    hub_disabled = are_progress_bars_disabled()
    transformers_enabled = logging.is_progress_bar_enabled()
    disable_progress_bars()
    logging.disable_progress_bar()
    try:
        yield
    finally:
        if not hub_disabled:
            enable_progress_bars()
        if transformers_enabled:
            logging.enable_progress_bar()


def percentage(value):
    return "n/a" if value is None else f"{value:.1%}"


def color_mask(ids):
    colors = np.full((*ids.shape, 3), 110, dtype=np.uint8)
    valid = (ids >= 0) & (ids < len(PALETTE))
    colors[valid] = PALETTE[ids[valid]]
    return Image.fromarray(colors)


class LocalOutput:
    def __init__(self, directory):
        self.directory = Path(directory)

    def append(self, record, best_epoch):
        validation = record["validation"]
        row = {"epoch": record["epoch"], "optimizer_updates": record["optimizer_updates"],
               "train_loss": record["loss"], "val_loss": validation.get("loss"),
               "val_miou": validation["miou"], "val_coral_miou": validation["coral_miou"],
               **{f"{name.lower()}_iou": value for name, value in validation["iou"].items()},
               "dead_precision": validation["precision"]["Dead"], "dead_recall": validation["recall"]["Dead"],
               "pixel_accuracy": validation["pixel_accuracy"], "lr": record["lr"],
               "head_lr": record["head_lr"], "seconds": record["seconds"],
               "peak_vram_gib": record["peak_vram_gib"], "is_best": record["epoch"] == best_epoch}
        path = self.directory / "metrics.csv"
        with path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            if stream.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
        print(f"Epoch {record['epoch']:02d} | loss {record['loss']:.4f} | "
              f"val coral mIoU {percentage(validation['coral_miou'])} | "
              f"Dead IoU {percentage(validation['iou']['Dead'])} | "
              f"{record['seconds']:.0f}s{' | best' if row['is_best'] else ''}", flush=True)

    @torch.inference_mode()
    def figure(self, model, dataset, device, dtype, epoch):
        # The same evenly spaced, unaugmented validation examples for every run.
        indices = sorted(set(np.linspace(0, len(dataset) - 1, min(3, len(dataset)), dtype=int).tolist()))
        panel_width, panel_height = 320, 160
        row_height, header = panel_height + 25, 60
        canvas = Image.new("RGB", (panel_width * 4, header + row_height * len(indices)), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 6), f"Best validation checkpoint: epoch {epoch}", fill="black")
        draw.text((8, 25), "Background: dark | Healthy: green | Bleached: yellow | Dead: red | Ignored labels: gray", fill="black")
        for row, index in enumerate(indices):
            pixels, target = dataset[index][:2]
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                prediction = model(pixels.unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy()
            mean = pixels.new_tensor([.485, .456, .406])[:, None, None]
            std = pixels.new_tensor([.229, .224, .225])[:, None, None]
            rgb = (pixels * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
            image = Image.fromarray(np.rint(rgb * 255).astype(np.uint8))
            predicted = color_mask(prediction)
            overlay = np.asarray(Image.blend(image, predicted, .45)).copy()
            overlay[prediction == 0] = np.asarray(image)[prediction == 0]
            panels = (image, color_mask(target.numpy()), predicted, Image.fromarray(overlay))
            top = header + row * row_height
            for column, (panel, title) in enumerate(zip(panels, ("Input", "Ground truth", "Prediction", "Overlay"))):
                left = column * panel_width
                draw.text((left + 8, top + 5), f"{title} (validation #{index})", fill="black")
                nearest = column in (1, 2)
                canvas.paste(panel.resize((panel_width, panel_height),
                             Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS), (left, top + 25))
        temporary = self.directory / ".prediction.tmp"
        canvas.save(temporary, format="PNG")
        temporary.replace(self.directory / "prediction.png")
