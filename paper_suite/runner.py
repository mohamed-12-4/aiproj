"""Run the unmodified author training loop with adapter and EMA instrumentation."""

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml

from .adapters import build_model, set_epoch_phase, source_model_class
from .ema import EffectiveEMA
from .execution import author_torch_proxy, precision_dtype
from .recipe import AUTHOR, METHODS, PROJECT, REIN_IMPLEMENTATION, data_directory, make_config, provenance, root, write_json

source_model_class()  # Make the pinned author package importable, without editing it.
from dinov3_dpt_seg import train as author
from dinov3_dpt_seg.augmentations import (AugmentedSegDataset, SegmentationTrainAugmentation,
                                         SegmentationValTransform, segmentation_augment_config_from_dict)
from dinov3_dpt_seg.dataset import CityscapesLikeDataset, CityscapesLikePaths

TORCH_SAVE = torch.save


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    TORCH_SAVE(value, temporary)
    temporary.replace(path)


def model_kwargs(cfg):
    m, lora = cfg["model"], cfg["model"]["lora"]
    return dict(encoder_id=m["encoder_id"], num_classes=cfg["data"]["num_classes"],
                freeze_encoder=m["freeze_encoder"], attn_implementation=m["attn_implementation"],
                fusion_hidden_size=m["fusion_hidden_size"], neck_hidden_sizes=tuple(m["neck_hidden_sizes"]),
                reassemble_factors=tuple(m["reassemble_factors"]), lora_enabled=lora["enabled"],
                lora_r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                lora_target_modules=lora["target_modules"])


def base_dataset(cfg, split):
    d = cfg["data"]
    return CityscapesLikeDataset(CityscapesLikePaths(
        images_root=Path(d["images_root"]), labels_root=Path(d["labels_root"]), split=split,
        image_suffix=d["image_suffix"], label_suffix=d["label_suffix"]))


def state_fingerprint(model, prefix):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name.startswith(prefix):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_smoke(method, group):
    cfg = make_config(method, group, smoke=True)
    out = Path(cfg["train"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    author._set_seed(cfg["train"]["seed"])
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for the physical-batch smoke test")
    started = time.time()
    model = build_model(model_kwargs(cfg), cfg["experiment"], device=device)
    decoder_hash = state_fingerprint(model, "neck.") + state_fingerprint(model, "seg_head.")
    ema = EffectiveEMA(model, cfg["experiment"]["ema_decay"])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=precision_dtype(cfg) == torch.float16)
    augmentation = SegmentationTrainAugmentation(segmentation_augment_config_from_dict(cfg["augment"]),
                                                model.image_mean, model.image_std, ignore_index=0)
    dataset = base_dataset(cfg, "train")
    worst = max(cfg["augment"]["train_sizes"], key=lambda hw: hw[0] * hw[1])
    xs, ys = [], []
    for index in range(8):
        x, y = augmentation(*dataset[index], target_size=tuple(worst))
        xs.append(x)
        ys.append(y)
    x, y = torch.stack(xs).to(device), torch.stack(ys).to(device)
    phases = [0, cfg["experiment"]["decoder_warmup_epochs"]] if method == "decoder_ft" else [0]
    for epoch in phases:
        phase = set_epoch_phase(model, cfg["experiment"], epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=precision_dtype(cfg)):
            logits = model(x)
            loss = F.cross_entropy(logits, y, ignore_index=0)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not torch.isfinite(loss) or not grads or any(not torch.isfinite(g).all() for g in grads):
            raise RuntimeError(f"Non-finite or absent smoke gradients for {method}/{phase}")
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < previous_scale:
            raise RuntimeError("AMP skipped the smoke optimizer step")
        ema.update(model)
        print(f"SMOKE {method}/{phase}: shape={list(x.shape)}, loss={float(loss.detach()):.5f}, "
              f"allocated_GiB={torch.cuda.max_memory_allocated()/2**30:.2f}", flush=True)
        del logits, loss, grads
    # Use a small subset for the transactional swap; training was tested at full batch.
    model.eval()
    with torch.no_grad(), ema.apply(model), torch.autocast("cuda", dtype=precision_dtype(cfg)):
        test_logits = model(x[:1])
        if test_logits.shape != (1, 96, *worst) or not torch.isfinite(test_logits).all():
            raise RuntimeError("EMA smoke inference is invalid")
    result = {"method": method, "status": "passed", "physical_batch": 8,
              "shape": list(x.shape), "phase_epochs_tested": phases,
              "decoder_initialization_sha256": decoder_hash,
              "peak_allocated_GiB": torch.cuda.max_memory_allocated()/2**30,
              "gpu": torch.cuda.get_device_name(), "seconds": time.time() - started}
    write_json(out / "smoke.json", result)
    del test_logits, x, y, xs, ys, model, optimizer, scaler, ema
    gc.collect()
    torch.cuda.empty_cache()
    return result


def rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": (state[0], state[1].tolist(), state[2], state[3], state[4]),
            "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state["python"])
    a = state["numpy"]
    np.random.set_state((a[0], np.asarray(a[1], dtype=np.uint32), a[2], a[3], a[4]))
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run(method, group, commit=lambda: None):
    cfg = make_config(method, group)
    out = Path(cfg["train"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "completed.json").exists():
        return json.loads((out / "completed.json").read_text())
    last = out / "last.pt"
    resume = torch.load(last, map_location="cpu", weights_only=True) if last.exists() else None
    if resume is not None and resume.get("checkpoint_kind") != "raw_training_resume":
        raise ValueError("Resume requires raw weights, optimizer, EMA and scaler state")
    if resume is not None and method == "rein_g" and resume.get("experiment", {}).get("rein_g", {}).get("implementation") != REIN_IMPLEMENTATION:
        raise ValueError("Legacy Rein checkpoints require an explicit weights-only warm start; they cannot resume corrected Rein training")
    start_epoch = int(resume["epoch"]) + 1 if resume else 0
    if start_epoch < 150 and resume:
        cfg["train"].update(resume_from=str(last), resume_optimizer=True, epochs=150-start_epoch)
    config_path = out / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    write_json(out / "provenance.json", provenance(make_config(method, group)))
    state = {"epoch": start_epoch-1, "model": None, "ema": None, "scaler": None,
             "global_iter": None, "best_ema": -1.0, "best_raw": -1.0, "raw_metrics": None}
    if (out / "best_ema_metrics.json").exists():
        state["best_ema"] = json.loads((out / "best_ema_metrics.json").read_text())["miou"]
    if (out / "best_raw_metrics.json").exists():
        state["best_raw"] = json.loads((out / "best_raw_metrics.json").read_text())["miou"]
    original_model, original_epoch, original_validation = author.Dinov3DPTSegmenter, author.train_epoch, author.validate_resized
    original_save, original_argv = torch.save, sys.argv[:]
    original_author_torch = author.torch

    def factory(**kwargs):
        model = build_model(kwargs, cfg["experiment"], device=torch.device("cuda"))
        state["model"] = model
        print(f"MODEL {method}: {sum(p.numel() for p in model.parameters()):,} total; "
              f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} optimizer parameters", flush=True)
        return model

    def train_epoch(model, loader, optimizer, scaler, *args, **kwargs):
        state["epoch"] += 1
        state["scaler"] = scaler
        state["global_iter"] = kwargs.get("global_iter_ref")
        if state["ema"] is None:
            # LP-FT encoder parameters still require gradients here and are in
            # the optimizer, so their EMA shadows exist before warmup freezing.
            state["ema"] = EffectiveEMA(model, cfg["experiment"]["ema_decay"])
            if resume:
                state["ema"].load_state_dict(resume["ema"])
                if scaler.is_enabled() and resume.get("scaler"):
                    scaler.load_state_dict(resume["scaler"])
                restore_rng(resume["rng"])
                state["global_iter"][0] = resume["global_iter"]
            optimizer.register_step_post_hook(lambda *_: state["ema"].update(model))
        phase = set_epoch_phase(model, cfg["experiment"], state["epoch"])
        print(f"EPOCH {state['epoch']+1}/150 method={method} phase={phase} "
              f"precision={cfg['experiment']['precision']} scaler_enabled={scaler.is_enabled()} "
              f"ema_updates={state['ema'].num_updates}", flush=True)
        return original_epoch(model, loader, optimizer, scaler, *args, **kwargs)

    def validate(model, *args, **kwargs):
        raw_result = original_validation(model, *args, **kwargs)
        state["raw_metrics"] = {"epoch": state["epoch"], "loss": raw_result[0],
                                "miou": raw_result[1], "accuracy": raw_result[2]}
        ema = state["ema"]
        with ema.apply(model):
            ema_result = original_validation(model, *args, **kwargs)
        loss, miou, accuracy, _ = ema_result
        row = {"epoch": state["epoch"], "loss": loss, "miou": miou, "accuracy": accuracy,
               "optimizer_updates": ema.num_updates, "buffers": "copied_live; final BN recalibration pending"}
        with (out / "ema_metrics.jsonl").open("a") as f:
            f.write(json.dumps(finite_json(row)) + "\n")
        if math.isfinite(miou) and miou > state["best_ema"]:
            state["best_ema"] = miou
            atomic_save({"model": ema.inference_state(model), "epoch": state["epoch"], "config": cfg,
                         "checkpoint_kind": "effective_ema_inference_adapter_B_zeroed",
                         "metrics": row}, out / "best_ema.pt")
            write_json(out / "best_ema_metrics.json", row)
        print(f"EMA epoch={state['epoch']} mIoU={miou:.6f} updates={ema.num_updates}", flush=True)
        return raw_result

    def save(value, filename, *args, **kwargs):
        if isinstance(value, dict) and "model" in value and Path(filename).parent == out:
            if Path(filename).name == "best.pt":
                row = state["raw_metrics"]
                if row is None or row["miou"] <= state["best_raw"]:
                    return  # Preserve the pre-interruption best on a resumed run.
                state["best_raw"] = row["miou"]
                write_json(out / "best_raw_metrics.json", row)
            value = dict(value)
            value.update(ema=state["ema"].state_dict(), scaler=state["scaler"].state_dict(), rng=rng_state(),
                         global_iter=state["global_iter"][0], checkpoint_kind="raw_training_resume",
                         experiment=cfg["experiment"])
            atomic_save(value, filename)
            write_json(out / "progress.json", {"epoch_completed": state["epoch"]+1,
                       "optimizer_updates": state["ema"].num_updates, "method": method,
                       "phase": set_epoch_phase(state["model"], cfg["experiment"], state["epoch"]),
                       "last_checkpoint": Path(filename).name})
            commit()
        else:
            original_save(value, filename, *args, **kwargs)

    try:
        author.Dinov3DPTSegmenter, author.train_epoch, author.validate_resized = factory, train_epoch, validate
        author.torch = author_torch_proxy(cfg["experiment"].get("precision", "fp16"))
        torch.save = save
        if start_epoch < 150:
            sys.argv = [str(AUTHOR / "dinov3_dpt_seg/train.py"), "--config", str(config_path), "--device", "cuda"]
            author.main()
        else:
            state["model"] = build_model(model_kwargs(cfg), cfg["experiment"], device=torch.device("cuda"))
        # No further optimizer updates: restore the unmodified entry points
        # before loading selected weights and running final evaluations.
    except BaseException as error:
        write_json(out / "failure.json", {"error_type": type(error).__name__, "message": str(error)[:2000],
                   "epoch": state["epoch"]})
        commit()
        raise
    finally:
        author.Dinov3DPTSegmenter, author.train_epoch, author.validate_resized = original_model, original_epoch, original_validation
        author.torch = original_author_torch
        torch.save, sys.argv = original_save, original_argv
    del resume
    results = final_evaluation(state["model"], cfg, out, commit)
    write_json(out / "completed.json", results)
    commit()
    return results


@torch.no_grad()
def recalibrate_bn(model, cfg):
    model.eval()
    norms = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not norms:
        return
    previous = [m.momentum for m in norms]
    for m in norms:
        m.reset_running_stats()
        m.momentum = None
        m.train()
    transform = SegmentationValTransform(tuple(cfg["augment"]["val_size"]), model.image_mean, model.image_std)
    loader = DataLoader(AugmentedSegDataset(base_dataset(cfg, "train"), transform), batch_size=8,
                        num_workers=cfg["train"]["num_workers"], shuffle=False, drop_last=False)
    for x, _ in loader:
        with torch.autocast("cuda", dtype=precision_dtype(cfg)):
            model(x.cuda(non_blocking=True))
    for m, momentum in zip(norms, previous):
        m.momentum = momentum
        m.eval()


def confusion_metrics(confusion, ignore=0):
    c = confusion.double()
    tp = c.diag()
    union = c.sum(0) + c.sum(1) - tp
    present = union > 0
    if ignore is not None:
        present[ignore] = False
    iou = torch.where(union > 0, tp/union.clamp_min(1), torch.nan)
    return finite_json({"pixel_accuracy": float(tp.sum()/c.sum().clamp_min(1)),
                        "miou_present_classes": float(iou[present].mean()),
                        "per_class_iou": iou.tolist(), "confusion": confusion.long().tolist()})


def mapped_metrics(confusion):
    """Map native top-1 predictions, retaining unknown predictions as errors."""
    metadata = json.loads((PROJECT / "coralscapes_metadata/v2.json").read_text())
    legacy = torch.zeros(96, dtype=torch.long, device=confusion.device)
    for source, target in metadata["fine_to_legacy_id"].items():
        legacy[int(source)] = int(target)
    positions = (legacy[:,None]*40 + legacy[None,:]).flatten()
    coarse = torch.zeros(1600, dtype=torch.long, device=confusion.device)
    coarse.scatter_add_(0, positions, confusion.flatten())
    lookup = torch.zeros(40, dtype=torch.long, device=confusion.device)
    lookup[0] = lookup[21] = lookup[27] = 4  # ignored/unmapped prediction bucket
    for target, ids in ((1,[6,17,22,25,28,31,34,36]),(2,[4,16,19,33]),(3,[3,20,23,32,37])):
        lookup[ids] = target
    health = lookup[legacy]
    valid_targets = health != 4
    positions = (health[valid_targets,None]*5 + health[None,:]).flatten()
    hc = torch.zeros(25, dtype=torch.long, device=confusion.device)
    hc.scatter_add_(0, positions, confusion[valid_targets].flatten())
    hc = hc.reshape(5,5)
    hm = confusion_metrics(hc, ignore=4)
    hm["coral_miou"] = finite_json(float(torch.tensor([
        x if x is not None else float('nan') for x in hm['per_class_iou'][1:4]]).nanmean()))
    dead_tp = hc[3,3].double()
    hm["dead"] = {"precision": float(dead_tp/hc[:,3].sum().clamp_min(1)),
                  "recall": float(dead_tp/hc[3].sum().clamp_min(1)),
                  "iou": hm["per_class_iou"][3]}
    hm["fine_dead_recalls"] = {}
    for source, name in metadata["id2label"].items():
        i = int(source)
        if int(health[i]) == 3:
            total = int(confusion[i].sum())
            hm["fine_dead_recalls"][name] = float(confusion[i,health==3].sum()/total) if total else None
    hm["legacy_dead_recalls"] = {}
    cm = coarse.reshape(40,40)
    for i in (3,20,23,32,37):
        total = int(cm[i].sum())
        hm["legacy_dead_recalls"][str(i)] = float(cm[i,lookup==3].sum()/total) if total else None
    return {"legacy_39": confusion_metrics(cm), "health_4": hm,
            "mapping_protocol": "Map 95-class top-1 predictions; ignored predictions remain an error for valid targets."}


@torch.no_grad()
def evaluate_protocol(model, cfg, tiled=False):
    from torchvision.transforms import functional as TF
    model.eval()
    confusion = torch.zeros((96, 96), device="cuda", dtype=torch.long)
    by_site = {}
    dataset = base_dataset(cfg, "test")
    for index, (image, label) in enumerate(dataset):
        image = image.convert("RGB")
        if tiled:
            logits = []
            for left in (0, 1024):
                x = TF.normalize(TF.to_tensor(image.crop((left, 0, left+1024, 1024))), model.image_mean, model.image_std)
                with torch.autocast("cuda", dtype=precision_dtype(cfg)):
                    logits.append(model(x.unsqueeze(0).cuda()).float())
            prediction = torch.cat(logits, dim=-1).argmax(1).squeeze(0)
        else:
            resized = TF.resize(image, cfg["augment"]["val_size"], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
            x = TF.normalize(TF.to_tensor(resized), model.image_mean, model.image_std)
            with torch.autocast("cuda", dtype=precision_dtype(cfg)):
                logits = model(x.unsqueeze(0).cuda())
            prediction = F.interpolate(logits.float(), size=(1024, 2048), mode="bilinear", align_corners=False).argmax(1).squeeze(0)
        target = torch.as_tensor(np.array(label), device="cuda", dtype=torch.long)
        valid = target != 0
        image_confusion = torch.bincount(target[valid]*96+prediction[valid], minlength=96*96).reshape(96, 96)
        confusion += image_confusion
        site = dataset.pairs[index][0].parent.name
        if site not in by_site:
            by_site[site] = torch.zeros_like(confusion)
        by_site[site] += image_confusion
        if (index+1) % 100 == 0:
            print(f"TEST tiled={tiled}: {index+1}/473", flush=True)
    result = confusion_metrics(confusion)
    # Include the author's torchmetrics macro convention as a separate number.
    from torchmetrics.functional.classification.jaccard import _jaccard_index_reduce
    result["miou_author_torchmetrics"] = float(_jaccard_index_reduce(confusion, average="macro", ignore_index=0))
    result["inference_hw"] = [1024, 1024] if tiled else [768, 1376]
    result.update(mapped_metrics(confusion))
    result["by_site"] = {}
    for site, counts in by_site.items():
        h = mapped_metrics(counts)["health_4"]
        result["by_site"][site] = {"coral_miou": h["coral_miou"], "dead": h["dead"],
                                   "health_per_class_iou": h["per_class_iou"][:4]}
    return result


def final_evaluation(model, cfg, out, commit):
    results = {}
    for name, filename in (("raw", "best.pt"), ("ema", "best_ema.pt")):
        path = out / filename
        if not path.exists():
            raise FileNotFoundError(f"No selected {name} checkpoint")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"], strict=True)
        epoch = checkpoint["epoch"]
        del checkpoint
        if name == "ema":
            recalibrate_bn(model, cfg)
            atomic_save({"model": {k: v.cpu() for k,v in model.state_dict().items()}, "config": cfg,
                         "epoch": epoch, "checkpoint_kind": "effective_ema_inference_bn_calibrated"},
                        out / "best_ema_calibrated.pt")
            commit()
        results[name] = {"selected_epoch": epoch, "single_pass": evaluate_protocol(model, cfg),
                         "two_tiles": evaluate_protocol(model, cfg, tiled=True)}
        write_json(out / "test_results.json", results)
        commit()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--group", default="paper-v2-s42")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    print(run_smoke(args.method, args.group) if args.smoke_only else run(args.method, args.group))
