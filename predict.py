"""Predict coral health and save class IDs, colors, and an overlay."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from train import OUTPUT, image_tensor, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=OUTPUT)
    args = parser.parse_args()
    model, height = load_checkpoint(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    image = Image.open(args.image).convert("RGB")
    pixels = image_tensor(image, (height * 2, height)).unsqueeze(0).to(device)
    with torch.inference_mode():
        stats = model.prediction_stats(pixels) if hasattr(model, "prediction_stats") else None
        logits = stats["probabilities"] if stats else model(pixels)
        logits = F.interpolate(logits, size=(image.height, image.width),
                               mode="bilinear", align_corners=False)
        ids = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)

    # Background, Healthy, Bleached, Dead.
    palette = np.array([[0, 0, 0], [40, 190, 80], [255, 240, 130], [220, 65, 60]], dtype=np.uint8)
    colors = Image.fromarray(palette[ids])
    blended = np.array(Image.blend(image, colors, 0.45))
    blended[ids == 0] = np.asarray(image)[ids == 0]  # Leave background unchanged.
    directory = Path("predictions")
    directory.mkdir(exist_ok=True)
    prefix = directory / args.image.stem
    Image.fromarray(ids).save(f"{prefix}_ids.png")
    colors.save(f"{prefix}_colors.png")
    Image.fromarray(blended).save(f"{prefix}_overlay.png")
    if stats:
        from matplotlib import colormaps
        maps = {}
        for key in ("entropy", "mutual_information"):
            values = F.interpolate(stats[key][:, None], size=(image.height, image.width),
                                   mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
            maps[key] = values.astype(np.float16)
            colors = colormaps["magma"](np.clip(values / np.log(4), 0, 1))[:, :, :3]
            Image.fromarray((255 * colors).astype(np.uint8)).save(f"{prefix}_{key}.png")
        np.savez_compressed(f"{prefix}_uncertainty.npz", **maps)
    print(f"Saved predictions to {directory}; IDs: 0=Background, 1=Healthy, 2=Bleached, 3=Dead")


if __name__ == "__main__":
    main()
