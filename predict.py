"""Apply a saved model and save one input/prediction/overlay figure."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PIL import ImageDraw
from torch.nn import functional as F

from train import OUTPUT, ROOT, image_tensor, load_checkpoint
from local_output import PALETTE, quiet_downloads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=OUTPUT)
    parser.add_argument("--output", type=Path, help="One PNG figure; defaults to runs/predictions/IMAGE.png")
    args = parser.parse_args()
    with quiet_downloads():
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
    colors = Image.fromarray(PALETTE[ids])
    blended = np.array(Image.blend(image, colors, 0.45))
    blended[ids == 0] = np.asarray(image)[ids == 0]  # Leave background unchanged.
    width = min(640, image.width)
    height = max(1, round(image.height * width / image.width))
    figure = Image.new("RGB", (3 * width, height + 65), "white")
    draw = ImageDraw.Draw(figure)
    draw.text((8, 6), "Healthy: green | Bleached: yellow | Dead: red", fill="black")
    for column, (panel, title) in enumerate(((image, "Input"), (colors, "Prediction"),
                                            (Image.fromarray(blended), "Overlay"))):
        draw.text((column * width + 8, 30), title, fill="black")
        figure.paste(panel.resize((width, height), Image.Resampling.NEAREST if column == 1
                                 else Image.Resampling.LANCZOS), (column * width, 65))
    output = args.output or ROOT / "runs" / "predictions" / f"{args.image.stem}.png"
    if output.suffix.lower() != ".png":
        parser.error("--output must be a PNG file.")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.save(output)
    print(f"Prediction figure: {output}")


if __name__ == "__main__":
    main()
