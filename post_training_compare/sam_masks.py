"""Precompute SAM2 masks once; SAM is never resident during Rein-A training."""
import argparse
import json
from pathlib import Path
import time
import os

os.environ.setdefault("USE_TF","0")
os.environ.setdefault("USE_FLAX","0")

import numpy as np
from PIL import Image
import torch

from .assets import OUT, write_json, sha256


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--limit",type=int,default=0)
    args=parser.parse_args()
    from transformers import Sam2Model, Sam2Processor, pipeline
    torch.set_num_threads(4)
    assets=json.loads((OUT/"assets.json").read_text())["sam"]
    directory=Path(assets["local"]["config.json"]).parent
    model=Sam2Model.from_pretrained(directory,local_files_only=True)
    processor=Sam2Processor.from_pretrained(directory,local_files_only=True)
    generator=pipeline("mask-generation",model=model,image_processor=processor.image_processor,device=0)
    source=OUT/("source512.json" if (OUT/"source512.json").exists() else "source.json")
    rows=json.loads(source.read_text())+json.loads((OUT/"adaptation.json").read_text())
    if args.limit:rows=rows[:args.limit]
    directory=OUT/"sam_masks";directory.mkdir(exist_ok=True)
    records=[]
    for i,row in enumerate(rows):
        path=directory/(sha256(row["path"])+"_amg080.npz")
        started=time.monotonic()
        if not path.exists():
            image=Image.open(row["path"]).convert("RGB")
            with torch.inference_mode():
                result=generator(image,points_per_batch=64,points_per_crop=32,
                                 pred_iou_thresh=.8,stability_score_thresh=.95)
            masks=np.asarray(result["masks"],dtype=np.uint8)
            if not len(masks):masks=np.zeros((0,image.height,image.width),dtype=np.uint8)
            if masks.shape[1:]!=(image.height,image.width):raise ValueError("SAM masks do not match native patch dimensions")
            order=np.argsort(np.asarray(result["scores"]))[::-1][:100]
            masks=masks[order]
            np.savez_compressed(path,packed=np.packbits(masks,axis=-1),shape=np.asarray(masks.shape))
        with np.load(path) as value:n=int(value["shape"][0])
        records.append(dict(path=row["path"],mask_path=str(path),masks=n,sha256=sha256(path)))
        print(f"SAM2 masks {i+1}/{len(rows)}: {n} regions, {time.monotonic()-started:.2f}s",flush=True)
        write_json(OUT/("sam_probe.json" if args.limit else "sam_manifest.json"),
            dict(model=assets["repo"],revision=assets["revision"],points_per_side=32,pred_iou_thresh=.8,
                 records=records,empty_images=sum(r["masks"]==0 for r in records),
                 generation="Official HF SAM2 automatic mask-generation pipeline; masks cached offline"))


def load(path,device="cuda"):
    cached=OUT/"sam_masks"/(sha256(path)+"_amg080.npz")
    with np.load(cached) as value:
        shape=tuple(value["shape"].tolist())
        masks=np.unpackbits(value["packed"],axis=-1)[...,:shape[-1]].reshape(shape)
    return torch.from_numpy(masks.copy()).float().to(device)


if __name__=="__main__":main()
