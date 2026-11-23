"""Inference-only evaluation of frozen models at original public NEOM points."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_pilot(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    rows = []
    for image in manifest["images"]:
        for point in image["points"]:
            path = root / point["crop"]
            if sha256(path) != point["crop_sha256"]:
                raise RuntimeError("An evaluation input changed after selection")
            with Image.open(path) as crop:
                if crop.size != (512, 512):
                    raise RuntimeError("Pilot crops must retain the fixed native 512-square protocol")
            rows.append(dict(point, metadata=image["metadata"], crop_path=str(path)))
    n = manifest["points_per_class"]
    if n != 12 or len(rows) != 3*n or Counter(p["target"] for p in rows) != {1:n, 2:n, 3:n}:
        raise RuntimeError("The frozen stratified pilot is incomplete")
    return manifest, rows


def health_lookup(path):
    meta = json.loads(Path(path).read_text())
    legacy_to_health = [0] * 40
    for index in (0, 21, 27):
        legacy_to_health[index] = 4
    for value, labels in [(1, [6,17,22,25,28,31,34,36]), (2, [4,16,19,33]), (3, [3,20,23,32,37])]:
        for label in labels:
            legacy_to_health[label] = value
    return [legacy_to_health[int(meta["fine_to_legacy_id"].get(str(i), 0))] for i in range(96)]


def summarize(rows):
    counts = np.zeros((5, 5), dtype=np.int64)
    for row in rows:
        counts[row["target"], row["predicted_health"]] += 1
    classes = {}
    for i, name in [(1, "Alive"), (2, "Bleached"), (3, "Dead")]:
        tp = int(counts[i, i]); actual = int(counts[i].sum()); predicted = int(counts[:, i].sum())
        classes[name] = dict(support=actual, precision=tp / predicted if predicted else None,
                             recall=tp / actual if actual else None,
                             f1=2*tp / (actual+predicted) if actual+predicted else None)
    recalls = [v["recall"] for v in classes.values() if v["recall"] is not None]
    f1s = [v["f1"] for v in classes.values() if v["f1"] is not None]
    return dict(points=len(rows), accuracy=float(counts.trace()/max(len(rows), 1)),
                macro_recall=sum(recalls)/len(recalls) if recalls else None,
                macro_f1=sum(f1s)/len(f1s) if f1s else None, classes=classes,
                confusion=counts.tolist(), order=["Background", "Alive", "Bleached", "Dead", "Unmapped"])


def evaluate(method, root, checkpoint_path):
    import torch
    started = time.monotonic()
    manifest, rows = read_pilot(root)
    lookup = health_lookup("/app/coralscapes_metadata/v2.json")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", mmap=True, weights_only=True)
    if checkpoint["epoch"] + 1 != 40:
        raise RuntimeError("The pilot must use the selected epoch40 inference checkpoint")
    predictions = []
    if method == "reinpp":
        from mmseg.datasets.transforms import PackSegInputs
        from reinpp_training.model import build_model, selected_weights
        model = build_model(); model.eval(); model.requires_grad_(False)
        pack = PackSegInputs()
        with torch.inference_mode(), selected_weights(model, checkpoint):
            for start in range(0, len(rows), 4):
                chunk = rows[start:start+4]
                packed = [pack(dict(img=np.array(Image.open(p["crop_path"]).convert("RGB"))[:, :, ::-1].copy(),
                                    img_path=p["crop_path"], ori_shape=(512,512), img_shape=(512,512)))
                          for p in chunk]
                outputs = model.test_step(dict(inputs=[p["inputs"] for p in packed],
                                               data_samples=[p["data_samples"] for p in packed]))
                for point, output in zip(chunk, outputs):
                    x, y = point["crop_xy"]
                    predictions.append(int(output.pred_sem_seg.data[0, y, x]))
                print(f"NEOM {method}: {len(predictions)}/{len(rows)} points", flush=True)
    elif method == "lora":
        from torchvision.transforms import functional as TF
        from paper_suite.adapters import build_model
        from paper_suite.runner import model_kwargs
        cfg = checkpoint["config"]
        experiment = dict(cfg["experiment"], gradient_checkpointing=False, decoder_checkpointing=False)
        model = build_model(model_kwargs(cfg), experiment, device=torch.device("cuda"))
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval(); model.requires_grad_(False)
        with torch.inference_mode():
            for start in range(0, len(rows), 4):
                chunk = rows[start:start+4]
                batch = torch.stack([TF.normalize(TF.to_tensor(Image.open(p["crop_path"]).convert("RGB")),
                                                   model.image_mean, model.image_std) for p in chunk]).cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(batch)
                for index, point in enumerate(chunk):
                    x, y = point["crop_xy"]
                    predictions.append(int(logits[index, :, y, x].argmax()))
                print(f"NEOM {method}: {len(predictions)}/{len(rows)} points", flush=True)
    else:
        raise ValueError(method)
    for point, prediction in zip(rows, predictions):
        if not 0 <= prediction < 96:
            raise RuntimeError("Invalid native V2 prediction")
        point.update(predicted_native=prediction, predicted_health=lookup[prediction])
        point.pop("crop_path")
    groups = defaultdict(list)
    for point in rows:
        groups[point["metadata"].get("Reef_name", "unknown")].append(point)
    return dict(status="completed", method=method, checkpoint=str(checkpoint_path), selected_epoch=40,
                checkpoint_sha256=sha256(checkpoint_path), manifest_sha256=sha256(Path(root)/"manifest.json"),
                gpu=torch.cuda.get_device_name(), seconds=time.monotonic()-started,
                input_protocol=manifest["input_protocol"], warnings=manifest["warnings"],
                metrics=summarize(rows), by_reef={k:summarize(v) for k,v in groups.items()}, predictions=rows)
