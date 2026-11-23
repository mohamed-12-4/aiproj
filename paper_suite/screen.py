"""Bounded, fresh V2 adapter comparison using the pinned authors' training step.

Forty completed epochs are an early stop along the original 150-epoch schedule.
Native classes, physical batch, augmentation, loss and optimizer recipe are kept.
"""

from contextlib import contextmanager
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml

from .adapters import build_model
from .ema import EffectiveEMA
from .execution import author_torch_proxy, precision_dtype
from .recipe import DATA_REVISION, REIN_IMPLEMENTATION, data_directory, make_config, provenance, write_json
from .runner import (atomic_save, author, base_dataset, confusion_metrics, final_evaluation,
                     finite_json, mapped_metrics, model_kwargs, restore_rng, rng_state, state_fingerprint)

SCREEN_VERSION = "fresh_v2_40_v1"


def screen_config(method, group, run_id, decoder_checkpointing=False):
    if method not in ("lora", "rein_g", "eva"):
        raise ValueError("This comparison contains LoRA, corrected Rein-G and fixed-rank EVA")
    cfg = make_config(method, group)
    cfg["train"]["num_workers"] = 8
    cfg["experiment"].update(precision="bf16", decoder_checkpointing=decoder_checkpointing,
                             screen_version=SCREEN_VERSION, stop_after_epochs=40,
                             schedule_epochs=150, fresh_initialization=True, run_id=run_id)
    return cfg


def check_resume(checkpoint, cfg):
    experiment = checkpoint.get("experiment", {})
    for key in ("screen_version", "run_id", "method", "stop_after_epochs", "schedule_epochs"):
        if experiment.get(key) != cfg["experiment"].get(key):
            raise ValueError(f"Checkpoint belongs to a different experiment: {key}")
    if checkpoint.get("checkpoint_kind") != "raw_training_resume":
        raise ValueError("Only complete raw training state can resume")
    if cfg["experiment"]["method"] == "rein_g" and experiment.get("rein_g", {}).get("implementation") != REIN_IMPLEMENTATION:
        raise ValueError("The legacy Rein feature-capture implementation cannot resume here")
    if cfg["experiment"]["method"] == "eva" and experiment.get("eva") != cfg["experiment"]["eva"]:
        raise ValueError("EVA initialization settings differ from the resume checkpoint")
    return int(checkpoint["epoch"]) + 1


def validation_epoch(epoch, stop=40):
    # Keep the authors' zero-based validation cadence, and always evaluate the endpoint.
    return epoch <= 5 or epoch % 5 == 0 or epoch + 1 == stop


def cache_native_train_val(cfg):
    """Copy verified bytes to the container disk once; no image/label changes."""
    destination = Path("/tmp/coralscapes-v2-native") / DATA_REVISION[:12]
    for folder, key in (("leftImg8bit", "images_root"), ("gtFine", "labels_root")):
        source = Path(cfg["data"][key])
        for split in ("train", "val"):
            target = destination / folder / split
            if not target.exists():
                shutil.copytree(source / split, target)
        cfg["data"][key] = str(destination / folder)


@contextmanager
def author_runtime(precision):
    original_torch = author.torch
    original_print = getattr(author, "print", None)
    import builtins
    logged = [0]
    def limited_print(*args, **kwargs):
        if args and str(args[0]).startswith("[train/epoch]"):
            logged[0] += 1
            if logged[0] % 20:
                return
        builtins.print(*args, **kwargs)
    author.torch, author.print = author_torch_proxy(precision), limited_print
    try:
        yield
    finally:
        author.torch = original_torch
        if original_print is None:
            del author.print
        else:
            author.print = original_print


@torch.no_grad()
def validate_with_health(model, dataset, cfg, device):
    """Author resize/upsample protocol; derive all metrics from the same predictions."""
    from torchvision.transforms import functional as TF
    from torchmetrics.functional.classification.jaccard import _jaccard_index_reduce
    model.eval()
    classes = cfg["data"]["num_classes"]
    counts = torch.zeros((classes, classes), dtype=torch.long, device=device)
    by_site, total_loss = {}, torch.zeros((), device=device)
    for index in range(len(dataset)):
        image, label = dataset[index]
        target = torch.as_tensor(np.array(label, dtype=np.int64), device=device).unsqueeze(0)
        resized = TF.resize(image.convert("RGB"), cfg["augment"]["val_size"],
                            interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        x = TF.normalize(TF.to_tensor(resized), model.image_mean, model.image_std).unsqueeze(0).to(device)
        with torch.autocast(device.type, dtype=precision_dtype(cfg), enabled=device.type == "cuda"):
            logits = model(x)
        up = F.interpolate(logits.float(), size=target.shape[-2:], mode="bilinear", align_corners=False)
        total_loss += F.cross_entropy(up, target, ignore_index=0)
        predicted = up.argmax(1)
        valid = target != 0
        cm = torch.bincount(target[valid]*classes + predicted[valid], minlength=classes*classes).reshape(classes, classes)
        counts += cm
        site = dataset.pairs[index][0].parent.name
        if site not in by_site:
            by_site[site] = torch.zeros_like(counts)
        by_site[site] += cm
        if (index + 1) % 50 == 0:
            print(f"VALIDATION {index+1}/{len(dataset)}", flush=True)
    cpu_counts = counts.cpu()
    result = {"loss": float(total_loss / max(1, len(dataset))),
              "miou": float(_jaccard_index_reduce(cpu_counts, average="macro", ignore_index=0)),
              "accuracy": float(cpu_counts.diag().sum() / cpu_counts.sum().clamp_min(1)),
              "native": confusion_metrics(cpu_counts), "health_4": mapped_metrics(cpu_counts)["health_4"],
              "by_site": {site: mapped_metrics(cm.cpu())["health_4"] for site, cm in by_site.items()},
              "mapping_protocol": "Native top-1 to fixed V2-to-legacy-to-health mapping; unknown predictions remain errors"}
    return finite_json(result)


def run_screen(method, group, run_id, deadline, decoder_checkpointing, commit=lambda: None):
    cfg = screen_config(method, group, run_id, decoder_checkpointing)
    out = Path(cfg["train"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "completed.json").exists():
        return json.loads((out / "completed.json").read_text())
    checkpoint = torch.load(out / "last.pt", map_location="cpu", weights_only=True) if (out / "last.pt").exists() else None
    start = check_resume(checkpoint, cfg) if checkpoint else 0
    if time.time() >= deadline - 900:
        raise RuntimeError("Submission budget deadline expired before training could start")
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    write_json(out / "provenance.json", provenance(cfg))
    author._set_seed(cfg["train"]["seed"])
    device = torch.device("cuda")
    model = build_model(model_kwargs(cfg), cfg["experiment"], device=device)
    initial = {"decoder_sha256": state_fingerprint(model, "neck.") + state_fingerprint(model, "seg_head."),
               "fresh_start": checkpoint is None, "encoder_revision": provenance(cfg)["encoder_revision"],
               "method": method, "run_id": run_id}
    if checkpoint is None:
        write_json(out / "initialization.json", initial)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    cache_native_train_val(cfg)
    augmentation = author.SegmentationTrainAugmentation(
        author.segmentation_augment_config_from_dict(cfg["augment"]), model.image_mean, model.image_std, ignore_index=0)
    loader = DataLoader(base_dataset(cfg, "train"), batch_size=8, shuffle=True, num_workers=8,
                        pin_memory=True, drop_last=True, worker_init_fn=author._worker_init_fn,
                        collate_fn=author._make_train_collate(augmentation), prefetch_factor=2)
    val_dataset = base_dataset(cfg, "val")
    if method == "eva" and checkpoint is None:
        from .eva import initialize_eva
        initial["eva"] = initialize_eva(model, cfg, out, commit=commit)
        write_json(out / "initialization.json", initial)
        commit()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step:
        author._warmup_cosine_lr_multiplier(step, cfg["train"]["warmup_steps"], 150 * len(loader)))
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    ema = EffectiveEMA(model, cfg["experiment"]["ema_decay"])
    global_iter, best = [0], {"raw": -1., "ema": -1.}
    if checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["lr_scheduler"])
        ema.load_state_dict(checkpoint["ema"])
        global_iter[0], best = checkpoint["global_iter"], checkpoint["best"]
        restore_rng(checkpoint["rng"])
    del checkpoint
    optimizer.register_step_post_hook(lambda *_: ema.update(model))
    stop, previous_seconds = 40, None
    completed = start
    try:
        with author_runtime("bf16"):
            for epoch in range(start, stop):
                if previous_seconds and time.time() + previous_seconds * 1.4 + 900 >= deadline:
                    write_json(out / "budget_stopped.json", {"epoch_completed": completed, "deadline": deadline})
                    commit()
                    return {"status": "budget_stopped", "epoch_completed": completed}
                started = time.time()
                print(f"EPOCH {epoch+1}/40 method={method} BF16 batch=8 schedule_horizon=150 updates={ema.num_updates}", flush=True)
                train_loss, train_accuracy, train_miou = author.train_epoch(
                    model, loader, optimizer, scaler, device, 96, 0, True,
                    lr_scheduler=scheduler, global_iter_ref=global_iter)
                row = {"epoch": epoch, "epoch_completed": epoch+1, "optimizer_updates": ema.num_updates,
                       "train_loss": train_loss, "train_accuracy": train_accuracy, "train_miou": train_miou,
                       "lr": optimizer.param_groups[0]["lr"]}
                if validation_epoch(epoch):
                    raw = validate_with_health(model, val_dataset, cfg, device)
                    with ema.apply(model):
                        averaged = validate_with_health(model, val_dataset, cfg, device)
                    row.update(raw=raw, ema=averaged)
                    for name, metrics in (("raw", raw), ("ema", averaged)):
                        score = metrics["miou"]
                        if score is not None and score > best[name]:
                            best[name] = score
                            weights = ema.inference_state(model) if name == "ema" else {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                            atomic_save({"model": weights, "epoch": epoch, "config": cfg,
                                         "checkpoint_kind": "selected_inference", "metrics": metrics},
                                        out / ("best_ema.pt" if name == "ema" else "best.pt"))
                            write_json(out / f"best_{name}_metrics.json", dict(metrics, epoch=epoch))
                            del weights
                    print(f"VALIDATED epoch={epoch+1} raw_mIoU={raw['miou']:.6f} ema_mIoU={averaged['miou']:.6f}", flush=True)
                row["seconds_before_checkpoint"] = time.time() - started
                atomic_save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                             "lr_scheduler": scheduler.state_dict(), "ema": ema.state_dict(), "scaler": scaler.state_dict(),
                             "rng": rng_state(), "global_iter": global_iter[0], "best": best,
                             "experiment": cfg["experiment"], "config": cfg, "checkpoint_kind": "raw_training_resume"}, out / "last.pt")
                completed = epoch + 1
                previous_seconds = time.time() - started
                row["seconds_with_checkpoint"] = previous_seconds
                with (out / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(finite_json(row)) + "\n")
                write_json(out / "progress.json", {"epoch_completed": completed, "optimizer_updates": ema.num_updates,
                           "method": method, "last_checkpoint": "last.pt", "stop_after_epochs": 40,
                           "last_epoch_seconds": previous_seconds})
                commit()
    except BaseException as error:
        write_json(out / "failure.json", {"error_type": type(error).__name__, "message": str(error)[:2000],
                                          "epoch_completed": completed})
        commit()
        raise
    result = {"status": "completed_40_epoch_screen", "epoch_completed": completed,
              "optimizer_updates": ema.num_updates, "schedule_horizon_epochs": 150, "best_validation": best}
    # Final testing uses the original volume paths, which include the test split.
    eval_cfg = screen_config(method, group, run_id, decoder_checkpointing)
    if time.time() + 1800 < deadline:
        del optimizer, scheduler, ema, loader
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        with author_runtime("bf16"):
            result["test"] = final_evaluation(model, eval_cfg, out, commit)
    else:
        result["test_evaluation_pending"] = True
    write_json(out / "completed.json", result)
    commit()
    return result
