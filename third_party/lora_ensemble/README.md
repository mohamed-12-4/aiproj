# Upstream LoRA implementation

Source: https://github.com/prs-eth/LoRA-Ensemble
Commit: `812ced225717fdcce3c1e236e5dd17d14c2b1b2b`
Original file: `models/lora.py`

`Init_Weight` and `LoRA` are copied without changing their class definitions.
Only imports needed by those two classes are retained. The unused torchvision,
vmap/AST wrappers and global device singleton are omitted. The MIT license is
preserved here. Our Hugging Face DINOv3 routing and segmentation adaptation live
in `ensemble.py`, rather than changing the upstream classification architecture.
