# Coral health segmentation with DINOv3 LoRA-Ensemble

The current experiment uses `facebook/dinov3-vitb16-pretrain-lvd1689m` with three
rank-8 LoRA members sharing one frozen ViT-B/16 backbone. Each member has its own
segmentation decoder. Predictions are the arithmetic mean of member softmax
probabilities, not the mean of logits or a merge of adapter weights.

## Run

Edit `config.py`; both local training and Modal read it:

```bash
source .venv/bin/activate
python train.py
modal run --detach --name coral-dinov3b-lora-ensemble-3m modal_train.py
```

Install `requirements.txt` in an environment with the appropriate CUDA PyTorch
build. DINOv3 requires approved Hugging Face access and a local login. The Modal
launcher downloads the pinned model locally if necessary and uploads only
`config.json` and `model.safetensors` to the private `coral-training` Volume.
It does not upload Hugging Face credentials. Subsequent runs reuse these files.

After the launcher prints a workflow ID and dashboard URL, the remote controller
handles CPU preparation followed by GPU training. `--detach` lets it continue
when the terminal closes or the computer shuts down. Dataset download, resizing,
sampling statistics and model staging use CPU only. The A10 starts afterward;
it loads the prepared data and model without network access to Hugging Face.
`modal app logs APP_ID` follows progress; closing the log viewer does not stop training.

Defaults: 512×1024 images, batch 4, 60 epochs, BF16 on an A10, AdamW with learning
rate 1e-4 for adapters and heads, zero weight decay, 500 optimizer-update warmup,
cosine decay to 1e-6, and global gradient clipping at 1.0. Batch 4 means four
images seen by each of the three members before one optimizer update.
The 24-hour timeout includes the entire workflow. Increasing members increases
training/inference work; sharing the backbone saves weight storage, not all computation.
Choose a new `RUN_NAME` for each experiment; existing output directories are rejected.

| Setting | Controls |
| --- | --- |
| `RUN_NAME` | `checkpoints/RUN_NAME` output directory |
| `TRAIN` | Resolution, epochs, batch size, learning rates, precision, clipping, scheduler, workers, seed |
| `MODEL` | Pinned backbone ID/revision, patch size, feature layers, decoder channels |
| `LORA` | Rank, Xavier initialization gain, Q/K/V/output projection placement |
| `ENSEMBLE` | Member count and evaluation ECE bins |
| `AUGMENT` | Paired crops/flips and image-only color augmentation |
| `SAMPLING` | Dead-image oversampling threshold and weight |
| `LOSS` | Historical hierarchical-loss settings, used only by the legacy DINOv2 path |
| `MODAL` | GPU, CPU, memory, timeout and persistent Volume |

The ensemble uses uniform four-class cross-entropy, not the legacy hierarchical
focal loss. `warmup_steps` overrides `warmup_epochs` for this architecture.
Local CLI overrides include `--epochs`, `--height`, `--batch-size`, `--lr`,
`--warmup-steps`, `--min-lr`, and `--output`. Modal accepts `--epochs`, `--height`,
`--batch-size`, `--lr`, `--min-lr`, `--gpu`, and `--run-name`; other values come
from the uploaded configuration. CPU execution uses FP32.

## Relation to the paper and public code

Paper: [LoRA-Ensemble](https://arxiv.org/html/2405.14438v5).
Code: [prs-eth/LoRA-Ensemble](https://github.com/prs-eth/LoRA-Ensemble), cloned at
commit `812ced225717fdcce3c1e236e5dd17d14c2b1b2b`.
The MIT-licensed `LoRA` and `Init_Weight` classes from `models/lora.py` are vendored
with their class definitions unchanged in `third_party/lora_ensemble/`.
The complete clone used for inspection is `/tmp/coral-lora-ensemble-upstream`.

Matching the public method/configuration:

- Shared frozen pretrained backbone; separate A/B matrices and heads per member.
- LoRA in Q, K, V and attention output projections in every transformer block.
- Rank 8, Xavier-uniform A initialization with gain 10, zero-initialized B.
  The upstream module has no alpha/r scaling and no LoRA dropout.
- Same augmented minibatch for every member, independent adapter/head initializations,
  mean member cross-entropy, one AdamW update, gradient clipping 1.0.
- Learning rate 1e-4, weight decay zero, 500-update warmup, per-update cosine schedule.
- Arithmetic mean of softmax probabilities at inference; per-member scores and
  ensemble NLL, Brier and ECE reported alongside segmentation scores.

Adaptations, rather than an exact reproduction of its classification benchmark:

- DINOv3 self-supervised ViT-B replaces the public supervised torchvision ViT.
  CLS plus all four register tokens are removed before reshaping patch features.
- Three members for the first experiment; members run sequentially inside each
  minibatch instead of upstream vmap. Gradients are divided by three and accumulated
  before one update: the mean-loss objective is unchanged, with less activation memory.
- Each decoder fuses four intermediate layers and predicts four pixel classes.
  Frozen DINOv3 stays in evaluation mode to disable its random RoPE coordinate
  rescaling; each trainable decoder has its own BatchNorm state.
- Coral-specific resolution, augmentations, oversampling, 60 epochs, BF16, batch 4,
  nonzero final LR floor, and validation coral-mIoU checkpoint selection.
- No enhanced-diversity regularizer or post-hoc temperature fitting in this first run.
  Pixel ECE uses 15 equal-width confidence bins; Brier is the sum over four classes
  averaged over valid pixels. These are pixelwise adaptations of classification metrics.

These changes also make this a multi-factor experiment relative to the previous
DINOv2 hierarchical model. Its results cannot isolate the benefit of ensembling;
individual member metrics provide a within-run comparison. Better OOD behavior
must be tested on held-out locations/datasets. Entropy or disagreement alone
is not evidence of improved OOD performance.

## Data, sampling and augmentation

The official Coralscapes train/validation/test splits are retained (1517/166/392).
Parquet files live in `data/coralscapes`; resized images/masks are cached under
`resized_HEIGHT`. Masks use nearest-neighbor resizing. Original files are retained.

Images with at least 5% Dead pixels among valid labeled pixels get sampling
weight 3; other images get weight 1. Sampling is with replacement, with exactly
1517 draws per epoch. Each draw gets fresh augmentation. This boosts Dead-rich
images; it does not guarantee Dead-centered crops or equal pixel-class frequencies.
Only training is oversampled. CPU preparation caches mask fractions in
`train_sampling.json`; GPU training refuses to generate a missing cache.

Paired crops use 60–100% of image area with 80% probability and preserve the
2:1 aspect ratio. Horizontal/vertical flip probabilities are 50%/20%.
Brightness and contrast vary ±15%, saturation ±8%; validation/test are unaugmented.
Normalization uses ImageNet RGB mean/std.

| Output ID | Class | Original Coralscapes IDs |
| --- | --- | --- |
| 0 | Background | Other labeled pixels, except the ignored IDs below |
| 1 | Healthy | 6, 17, 22, 25, 28, 31, 34, 36 |
| 2 | Bleached | 4, 16, 19, 33 |
| 3 | Dead | 3, 20, 23, 32, 37 |
| 255 | Ignore in loss/metrics | 0 (unmapped), 21 (millepora), 27 (turbinaria) |

Healthy means the dataset's alive labels. Fish, sand, other species and dead
clams become Background. We do not infer health for species without a health label.

## Metrics, checkpoints and visualization

Each epoch records pixel accuracy, IoU, F1/Dice, precision, recall and support,
plus macro and coral-only means. Counts accumulate over the full split and
exclude ignored pixels. Coral scores still penalize false positives on Background.
Validation additionally records individual member scores and ensemble NLL,
Brier, ECE, mean predictive entropy and mutual information (member disagreement).
Best weights are selected by validation coral mIoU; test is evaluated once at the end.
These training reports use resized-mask resolution.

`head.pt` stores all members' adapters, decoders, BatchNorm buffers and architecture
settings; the shared pretrained weights are loaded from the pinned HF revision.
`metrics.json` stores effective configuration, sampling summary and results.
Files are committed to the Modal Volume each epoch. Existing DINOv2 checkpoints
with fusion weights remain loadable. No full optimizer/resume checkpoint is saved.

```bash
modal volume get coral-training /checkpoints/coral-dinov3b-lora-ensemble-3m ./checkpoints/coral-dinov3b-lora-ensemble-3m
python evaluate.py --checkpoint checkpoints/coral-dinov3b-lora-ensemble-3m --samples 6
python predict.py path/to/image.png --checkpoint checkpoints/coral-dinov3b-lora-ensemble-3m
```

`evaluate.py` defaults to original-size masks, saving JSON/CSV metrics, a confusion
matrix and image/ground-truth/prediction/overlay comparisons. Use
`--resolution training` to match the training report's mask resolution. Ensemble
probabilities are interpolated before argmax. This four-class task is not directly
comparable to the Coralscapes paper's 39-class benchmark.

`predict.py` saves class IDs, colors, overlay, predictive-entropy and
mutual-information heatmaps, plus raw uncertainty maps in a compressed NPZ file.
Uncertainty is measured in nats; heatmaps use a fixed 0–ln(4) scale.
There is no learned OOD threshold or post-hoc probability calibration.

Checks:

```bash
HF_HUB_OFFLINE=1 USE_TF=0 python -m unittest test_ensemble -v
```

These test member isolation, frozen weights, chunked/full mean-loss gradient
equivalence, probability averaging, uncertainty metrics and checkpoint fidelity
using a small real DINOv3 architecture on CPU.

## Diagnose Dead errors and training coverage

```bash
modal run --detach --name coral-dead-diagnostics modal_diagnose.py
```

This evaluates the existing best checkpoint on validation at its training
resolution; it does not train or change model weights. A CPU stage counts original
and resized training masks and records source filenames. A GPU stage then measures
the full confusion matrix, where missed Dead pixels go, Dead recall for each of
the five original Dead annotation categories, and recall for every validation
image. Source-category IDs are retained in the prepared masks.

Reports live in `checkpoints/RUN_NAME/diagnostics/validation/` on the Volume:
`training_coverage.json`, `training_class_coverage.csv`, `training_per_image.csv`,
`training_original_categories.csv`, `validation_diagnostics.json`,
`dead_by_original_category.csv`, `dead_per_image.csv`, and
`dead_per_image_category.csv`. Coverage means at least one class pixel in an image;
additional columns count images with at least 1% and 5% class coverage. Counts
precede augmentation and oversampling. Undefined recalls are empty in CSV/null
in JSON. Error-destination fractions distinguish all Dead pixels from missed
Dead pixels. Per-image mean recall excludes images without Dead pixels.

With local data and checkpoint weights available, run `python diagnose.py`.
Use `--stage coverage` for CPU counts only, or `--stage predict` to reuse them.
`python -m unittest test_diagnose -v` checks the counters and nested Parquet input.
