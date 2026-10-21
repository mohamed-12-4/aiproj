"""Color a saved Coralscapes mask and create an overlay with a class legend.

Run: python3 visualize_mask.py --index 0
Mappings: https://huggingface.co/datasets/EPFL-ECEO/coralscapes/tree/main
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures without opening a desktop viewer.
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent


def load_mappings():
    metadata = ROOT / "coralscapes_metadata"
    id2label = {
        int(key): value
        for key, value in json.loads((metadata / "id2label.json").read_text()).items()
    }
    label2color = json.loads((metadata / "label2color.json").read_text())
    return id2label, label2color


def visualize_mask(image, mask, output_prefix):
    """Accept PIL images, save visualizations, and return IDs and label names."""
    id2label, label2color = load_mappings()
    ids = np.asarray(mask)  # Preserve the original class IDs; do not convert to RGB.
    if ids.ndim != 2 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("Expected a single-channel mask containing integer class IDs.")
    if ids.min() < 0 or ids.max() > 255:
        raise ValueError("Expected Coralscapes mask IDs between 0 and 255.")
    if image.size != mask.size:
        raise ValueError("The image and mask must have matching dimensions.")

    # Unmapped IDs, including 0, are shown in black.
    palette = np.zeros((256, 3), dtype=np.uint8)
    for class_id, label in id2label.items():
        palette[class_id] = label2color[label]
    colored = Image.fromarray(palette[ids])
    image = image.convert("RGB")
    overlay = Image.blend(image, colored, alpha=0.45)

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    colored.save(f"{output_prefix}_colors.png")
    overlay.save(f"{output_prefix}_overlay.png")

    present_ids, counts = np.unique(ids, return_counts=True)
    labels = {
        int(class_id): id2label.get(int(class_id), "unmapped")
        for class_id in present_ids
    }
    legend = []
    for class_id, count in zip(present_ids, counts):
        label = labels[int(class_id)]
        print(f"{class_id:2d}  {label:30s}  {count:8d} pixels ({count / ids.size:.2%})")
        legend.append(Patch(
            facecolor=palette[class_id] / 255,
            edgecolor="gray",
            label=f"{class_id}: {label}",
        ))

    legend_height = 0.30 * ((len(legend) + 2) // 3) + 0.2
    height = 4.2 + legend_height
    fig, axes = plt.subplots(1, 3, figsize=(18, height))
    for axis, picture, title in zip(
        axes, (image, colored, overlay), ("Original image", "Class colors", "Overlay")
    ):
        axis.imshow(picture, interpolation="nearest")
        axis.set_title(title)
        axis.axis("off")
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, legend_height / height, 1, 1))
    fig.savefig(f"{output_prefix}_preview.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return ids, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=0, help="Saved image/mask number")
    args = parser.parse_args()
    prefix = ROOT / "previews" / f"mask_{args.index}"
    with Image.open(ROOT / f"image_{args.index}.png") as image:
        with Image.open(ROOT / f"mask_{args.index}.png") as mask:
            visualize_mask(image, mask, prefix)
    print(f"Saved colored mask, overlay, and preview under {prefix.parent}")


if __name__ == "__main__":
    main()
