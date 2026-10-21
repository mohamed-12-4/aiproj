from datasets import load_dataset
from PIL import Image
train = load_dataset(
    "EPFL-ECEO/coralscapes",
    split="train",             # or "validation" / "test"
    streaming=True,
)

# Fetch and process just five examples
for i, sample in enumerate(train.take(5)):
    sample["image"].save(f"image_{i}.png")
    sample["label"].save(f"mask_{i}.png")