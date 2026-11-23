"""Run frozen-model selection on V2 validation, then compare on untouched inputs.

Historical test scores have already been seen. New fusion weights are selected
only on validation; every test pixel is evaluated with frozen selected weights.
"""

import csv
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from .core import (Candidate, ConfusionCounter, FAMILIES, GRID, candidates, confusion, health_lookup,
                   predict_all, predict_selected, select_indices, summarize)

PUBLIC_REVISION = "545e9339f7b77267ffab412dfabec5624f38dce0"
PUBLIC_ID = "EPFL-ECEO/coralscapesv2-dinov3-vitl-lora-dpt-95_class"
LORA_REMOTE = "checkpoints/paper-v2-fresh40-s42-20261101/lora/best_ema_calibrated.pt"
SCORES = ("native_miou", "coral_miou", "health_miou", "dead_iou", "dead_precision",
          "dead_recall", "healthy_iou", "bleached_iou", "background_iou",
          "native_pixel_accuracy", "health_pixel_accuracy")
COLORS = np.array([[25, 25, 25], [40, 190, 80], [255, 240, 130],
                   [220, 65, 60], [130, 130, 130]], dtype=np.uint8)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def save_bundle(path, bundle):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.pt")
    torch.save(bundle, tmp)
    tmp.replace(path)


def construct_models(workspace, public_folder):
    from paper_suite.adapters import source_model_class
    from paper_suite.recipe import ENCODER_ID, ENCODER_REVISION
    from paper_suite.runner import model_kwargs
    from transformers import AutoImageProcessor
    from safetensors.torch import load_file

    encoder = workspace / ".cache/huggingface/hub" / ("models--" + ENCODER_ID.replace("/", "--")) / "snapshots" / ENCODER_REVISION
    if not (encoder / "model.safetensors").is_file():
        raise FileNotFoundError("The existing pinned pretrained encoder is missing")
    source = workspace / LORA_REMOTE
    checkpoint = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    if checkpoint.get("epoch") != 39 or checkpoint.get("checkpoint_kind") != "effective_ema_inference_bn_calibrated":
        raise ValueError("Expected the selected epoch-40, BatchNorm-calibrated LoRA EMA")
    cfg = checkpoint["config"]
    if cfg["data"]["num_classes"] != 96 or cfg["data"]["ignore_index"] != 0:
        raise ValueError("LoRA checkpoint uses a different native-label policy")
    kwargs = model_kwargs(cfg)
    kwargs["encoder_id"] = str(encoder)
    lora = source_model_class()(**kwargs)
    lora.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    lora.eval().requires_grad_(False).cuda()

    # Use the released inference blueprint and the unchanged released tensors.
    spec = importlib.util.spec_from_file_location("fusion_released_dpt", public_folder / "coralscapes_hub_model.py")
    blueprint = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = blueprint
    spec.loader.exec_module(blueprint)
    hub_cfg = json.loads((public_folder / "config.json").read_text())
    if hub_cfg["num_classes"] != 96 or hub_cfg["ignore_index"] != 0:
        raise ValueError("Public model uses a different native-label policy")
    public_kwargs = blueprint._segmenter_kwargs_from_hub(hub_cfg)
    public_kwargs["encoder_id"] = str(encoder)
    processor = AutoImageProcessor.from_pretrained(str(encoder), local_files_only=True, use_fast=True, do_resize=False)
    expected_processor = json.loads((public_folder / "preprocessor_config.json").read_text())
    if processor.image_mean != expected_processor["image_mean"] or processor.image_std != expected_processor["image_std"]:
        raise ValueError("Public model image normalization mismatch")
    public = blueprint.Dinov3DPTSegmenter(**public_kwargs, image_processor=processor)
    state = load_file(public_folder / "model.safetensors")
    state = {key.replace("encoder.model.layer.", "encoder.layer.", 1): value for key, value in state.items()}
    public.load_state_dict(state, strict=True)
    del state
    public.eval().requires_grad_(False).cuda()
    gc.collect()
    provenance = dict(lora=dict(profile="m12mm-4m2005", volume="coral-training", path=LORA_REMOTE,
                                sha256=sha256(source), epoch_completed=40,
                                kind="effective_ema_inference_bn_calibrated"),
                      public=dict(repo=PUBLIC_ID, revision=PUBLIC_REVISION,
                                  sha256=sha256(public_folder / "model.safetensors")),
                      encoder_revision=ENCODER_REVISION)
    return {"lora": lora, "public": public}, provenance


def make_dataset(workspace, split):
    from paper_suite.adapters import source_model_class
    source_model_class()
    from dinov3_dpt_seg.dataset import CityscapesLikeDataset, CityscapesLikePaths
    from paper_suite.recipe import DATA_REVISION
    root = workspace / "data" / ("coralscapes_v2_paper_" + DATA_REVISION[:12])
    ready = json.loads((root / "ready.json").read_text())
    if ready["revision"] != DATA_REVISION or ready["counts"] != {"train": 1790, "val": 170, "test": 473}:
        raise ValueError("Dataset revision or official split sizes changed")
    ds = CityscapesLikeDataset(CityscapesLikePaths(root / "leftImg8bit", root / "gtFine", split,
                                                "_leftImg8bit.png", "_gtFine.png"))
    if len(ds) != {"val": 170, "test": 473}[split]:
        raise ValueError("Unexpected number of evaluation images")
    return ds


@torch.inference_mode()
def image_tensor(model, image):
    # The released blueprint exposes processor.image_mean/std, while the
    # training wrapper additionally offers convenience properties.
    return TF.normalize(TF.to_tensor(image), model.processor.image_mean,
                        model.processor.image_std).unsqueeze(0)


@torch.inference_mode()
def tile_output(model, image):
    x = image_tensor(model, image).cuda()
    torch.cuda.synchronize()
    started = time.monotonic()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x).float()[0]
    prediction = logits.argmax(0)
    probabilities = logits.softmax(0)
    torch.cuda.synchronize()
    return probabilities, prediction, time.monotonic() - started


def small_prediction(prediction):
    return F.interpolate(prediction[None, None].float(), size=(160, 160), mode="nearest")[0, 0].byte().cpu().numpy()


@torch.inference_mode()
def evaluate_split(models, dataset, split, groups, records, status_hook, deadline):
    count = len(records) + 2
    counts = torch.zeros(count, 96, 96, dtype=torch.long, device="cuda")
    by_site, previews, per_image = {}, {}, []
    examples = set(np.linspace(0, len(dataset) - 1, 3, dtype=int).tolist()) if split == "val" else set()
    model_seconds = {name: 0.0 for name in models}
    fusion_seconds = [0.0] * len(records)
    started = time.monotonic()
    previous_log = started
    for image_index, (image, label) in enumerate(dataset):
        if time.monotonic() >= deadline:
            raise TimeoutError("The evaluation's bounded GPU runtime has expired")
        if image.size != (2048, 1024) or label.size != image.size:
            raise ValueError("Expected lossless native 1024x2048 V2 image/mask pairs")
        site = dataset.pairs[image_index][0].parent.name
        if site not in by_site:
            by_site[site] = torch.zeros_like(counts)
        image_counts = torch.zeros_like(counts) if split == "test" else None
        if image_index in examples:
            previews[image_index] = dict(image=image.resize((320, 160)),
                target=np.asarray(label.resize((320, 160), Image.Resampling.NEAREST)).copy(),
                predictions=np.zeros((count, 160, 320), dtype=np.uint8))
        native_target = np.asarray(label, dtype=np.int64)
        for tile, left in enumerate((0, 1024)):
            crop = image.crop((left, 0, left + 1024, 1024))
            target = torch.as_tensor(native_target[:, left:left + 1024].copy(), device="cuda")
            count_prediction = ConfusionCounter(target)
            first, first_pred, seconds = tile_output(models["lora"], crop)
            model_seconds["lora"] += seconds
            second, second_pred, seconds = tile_output(models["public"], crop)
            model_seconds["public"] += seconds

            def add(index, prediction):
                matrix = count_prediction(prediction)
                counts[index] += matrix
                by_site[site][index] += matrix
                if image_counts is not None:
                    image_counts[index] += matrix
                if image_index in examples:
                    previews[image_index]["predictions"][index, :, tile * 160:(tile + 1) * 160] = small_prediction(prediction)

            add(0, first_pred)
            add(1, second_pred)
            if split == "val":
                generated = predict_all(first, second, groups)
                for j, (candidate, prediction) in enumerate(generated):
                    if candidate != records[j]:
                        raise RuntimeError("Grid order changed during validation")
                    add(j + 2, prediction)
            else:
                for j, candidate in enumerate(records):
                    torch.cuda.synchronize()
                    tick = time.monotonic()
                    predicted = predict_selected(first, second, groups, candidate)
                    torch.cuda.synchronize()
                    fusion_seconds[j] += time.monotonic() - tick
                    add(j + 2, predicted)
            del first, second, first_pred, second_pred, target
        if image_counts is not None:
            c = image_counts.cpu()
            # Sufficient per-image native statistics for paired resampling, not dense maps.
            per_image.append(dict(site=site, stats=torch.stack((c.diagonal(dim1=1, dim2=2), c.sum(1), c.sum(2)), dim=1)))
        if (image_index + 1) % 25 == 0 or image_index + 1 == len(dataset):
            status_hook(dict(phase=split, images_completed=image_index + 1, images_total=len(dataset),
                             seconds=round(time.monotonic() - started, 1)))
        if time.monotonic() - previous_log > 180:
            # A bounded heartbeat, not a per-batch progress bar.
            print(f"Evaluation {split}: {image_index+1}/{len(dataset)} images completed", flush=True)
            previous_log = time.monotonic()
    torch.cuda.synchronize()
    return dict(counts=counts.cpu(), by_site={s: c.cpu() for s, c in by_site.items()},
                previews=previews, model_seconds=model_seconds, fusion_seconds=fusion_seconds,
                per_image=per_image, seconds=time.monotonic() - started)


def scalar_metrics(metrics):
    return {key: metrics[key] for key in SCORES}


def write_csv(directory, row):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "metrics.csv"
    tmp = directory / ".metrics.tmp"
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    tmp.replace(path)


def figure(directory, previews, prediction_index, groups, title):
    lookup = groups.cpu().numpy()
    canvas = Image.new("RGB", (1280, 60 + 185 * len(previews)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), title, fill="black")
    draw.text((8, 24), "Fixed validation examples; native predictions mapped to health. Green Healthy / yellow Bleached / red Dead / gray ignored.", fill="black")
    for row, (index, item) in enumerate(sorted(previews.items())):
        rgb = item["image"]
        truth = Image.fromarray(COLORS[lookup[item["target"]]])
        pred = Image.fromarray(COLORS[lookup[item["predictions"][prediction_index]]])
        overlay = Image.blend(rgb, pred, .45)
        top = 60 + row * 185
        for column, (panel, label) in enumerate(zip((rgb, truth, pred, overlay), ("Input", "Ground truth", "Prediction", "Overlay"))):
            draw.text((column * 320 + 5, top + 2), f"{label}: validation #{index}", fill="black")
            canvas.paste(panel, (column * 320, top + 22))
    temporary = directory / ".prediction.tmp"
    canvas.save(temporary, format="PNG")
    temporary.replace(directory / "prediction.png")


def run_suite(workspace, public_folder, suite, commit=lambda: None, max_seconds=3500):
    from paper_suite.recipe import DATA_REVISION
    workspace, public_folder = Path(workspace), Path(public_folder)
    started = time.monotonic()
    deadline = started + max_seconds
    status_path = workspace / "checkpoints/local" / f"{suite}-status.json"

    def status_hook(state):
        atomic_json(status_path, dict(suite=suite, **state))
        commit()

    names = [f"{suite}-{suffix}" for suffix in ("lora", "public", "e1-global", "e2-dead", "e3-hierarchical")]
    if any((workspace / "runs" / name / "metrics.csv").exists() for name in names):
        raise FileExistsError("This suite already has result files; use a fresh suite name")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    status_hook(dict(phase="preparing_models", images_completed=0))
    print("Preparing existing epoch-40 LoRA EMA and pinned released DPT; frozen two-tile inference", flush=True)
    models, sources = construct_models(workspace, public_folder)
    groups_cpu = health_lookup()
    groups = groups_cpu.cuda()
    grid = candidates()
    provenance = dict(sources=sources, dataset_revision=DATA_REVISION,
                      inference="two non-overlapping native 1024x1024 tiles; BF16 forward; FP32 softmax/fusion",
                      selection="highest native validation mIoU; ties prefer shared weights, then LoRA",
                      health_policy="native argmax then fixed V2-to-legacy-to-health mapping; unknown predictions remain errors",
                      grid=list(GRID), trainable_parameters_updated=0,
                      software=dict(torch=str(torch.__version__)),
                      source_sha256={str(p.relative_to(Path(__file__).resolve().parents[1])): sha256(p)
                                     for p in Path(__file__).resolve().parent.glob('*.py')})
    bn_before = {name: {k: v.detach().cpu().clone() for k, v in model.named_buffers()
                        if 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k}
                 for name, model in models.items()}
    validation = evaluate_split(models, make_dataset(workspace, "val"), "val", groups, grid, status_hook, deadline)
    selected, selection_metrics = select_indices(grid, validation["counts"][2:], groups_cpu)
    winners = [grid[selected[f]] for f in FAMILIES]
    val_indices = [0, 1] + [selected[f] + 2 for f in FAMILIES]
    bundles, rows = [], []
    for i, name in enumerate(names):
        candidate = None if i < 2 else winners[i - 2]
        vi = val_indices[i]
        val_metrics = summarize(validation['counts'][vi], groups_cpu)
        bundle = dict(checkpoint_kind="frozen_fusion_inference_recipe", format_version=1,
                      name=name, candidate=candidate.record() if candidate else None,
                      individual_model=("lora", "public")[i] if i < 2 else None,
                      provenance=provenance, validation=dict(metrics=val_metrics,
                      counts=validation['counts'][vi],
                      sites={s: summarize(c[vi], groups_cpu) for s, c in validation['by_site'].items()}))
        if i >= 2:
            indices = [j for j, c in enumerate(grid) if c.family == candidate.family]
            bundle['validation']['search'] = [dict(candidate=grid[j].record(), **scalar_metrics(selection_metrics[j])) for j in indices]
            # These include the equal-weight hierarchical ablation, selected on validation only.
            bundle['validation']['site_grid_counts'] = {s: c[torch.tensor(indices) + 2] for s, c in validation['by_site'].items()}
        row = dict(epoch=1, evaluation_only=True, phase="validation_selected", model=name,
                   family=candidate.family if candidate else "individual",
                   a=candidate.a if candidate else '', b=candidate.b if candidate else '',
                   validation_images=170, test_images=0,
                   **{f"val_{k}": val_metrics[k] for k in SCORES},
                   **{f"test_{k}": '' for k in SCORES}, inference_seconds_per_test_image='',
                   suite_seconds=round(time.monotonic() - started, 3), peak_vram_gib='')
        directory = workspace / "runs" / name
        write_csv(directory, row)
        figure(directory, validation['previews'], vi, groups_cpu, f"{name} | selected on native validation mIoU")
        save_bundle(workspace / "checkpoints/local" / name / "head.pt", bundle)
        bundles.append(bundle)
        rows.append(row)
        print(f"Validation {name}: native={val_metrics['native_miou']:.4f} coral={val_metrics['coral_miou']:.4f} Dead={val_metrics['dead_iou']:.4f}" + (f" a={candidate.a} b={candidate.b}" if candidate else ''), flush=True)
    commit()
    del validation
    gc.collect()
    torch.cuda.empty_cache()
    status_hook(dict(phase="test", images_completed=0, images_total=473,
                     selected={f: winners[i].record() for i, f in enumerate(FAMILIES)}))
    # There is deliberately no candidate-selection call after reading test labels.
    test = evaluate_split(models, make_dataset(workspace, "test"), "test", groups, winners, status_hook, deadline)
    output = []
    for i, name in enumerate(names):
        metrics = summarize(test['counts'][i], groups_cpu)
        inference_seconds = (test['model_seconds'][('lora', 'public')[i]] if i < 2
                             else sum(test['model_seconds'].values()) + test['fusion_seconds'][i - 2])
        bundles[i]['test'] = dict(metrics=metrics, counts=test['counts'][i],
            sites={s: summarize(c[i], groups_cpu) for s, c in test['by_site'].items()},
            per_image_native_stats=torch.stack([r['stats'][i] for r in test['per_image']]),
            per_image_sites=[r['site'] for r in test['per_image']],
            inference_seconds=inference_seconds)
        rows[i].update(phase="complete", test_images=473,
                       **{f"test_{k}": metrics[k] for k in SCORES},
                       inference_seconds_per_test_image=inference_seconds / 473,
                       suite_seconds=round(time.monotonic() - started, 3),
                       peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30)
        write_csv(workspace / "runs" / name, rows[i])
        save_bundle(workspace / "checkpoints/local" / name / "head.pt", bundles[i])
        output.append(dict(name=name, candidate=bundles[i]['candidate'], validation=scalar_metrics(bundles[i]['validation']['metrics']),
                           test=scalar_metrics(metrics), seconds_per_image=inference_seconds / 473))
        print(f"Test {name}: native={metrics['native_miou']:.4f} coral={metrics['coral_miou']:.4f} Dead={metrics['dead_iou']:.4f}", flush=True)
    for name, model in models.items():
        buffers = dict(model.named_buffers())
        if any(not torch.equal(value, buffers[key].cpu()) for key, value in bn_before[name].items()):
            raise RuntimeError("BatchNorm buffers changed during frozen evaluation")
    status_hook(dict(phase="complete", images_completed=473, images_total=473,
                     seconds=round(time.monotonic() - started, 1), results=output))
    print("Final paths: " + ', '.join('runs/' + name for name in names), flush=True)
    return dict(suite=suite, results=output, seconds=time.monotonic() - started)
