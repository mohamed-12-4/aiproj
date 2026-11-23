"""Fetch only the released model, selected Rein++ inference weights and metadata.

Uses Modal's volume file API, without creating a cloud function or GPU job.
The original DINOv3 backbone is reused from the existing local HF cache.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/external_eval/alwajh_compare"
CACHE = ROOT / ".cache/huggingface/hub"
MODEL = "EPFL-ECEO/coralscapesv2-dinov3-vitl-lora-dpt-95_class"
DATASET = "ReefNet/ReefNet-1.0"
PROFILE = "mohamed-elashmony05"
GROUP = "reinpp-v2-mhgelu-m2f-s42-40e-20261101"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def metadata():
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=False)
    manifest_path = OUT / "assets.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    result = dict(previous)
    for kind, repo in [("model", MODEL), ("dataset", DATASET)]:
        old = previous.get(kind, {})
        info = (api.model_info(repo, revision=old.get("revision"), files_metadata=True) if kind == "model"
                else api.dataset_info(repo, revision=old.get("revision"), files_metadata=True))
        files = {v.rfilename: dict(bytes=v.size, sha256=getattr(v.lfs, "sha256", None))
                 for v in info.siblings}
        result[kind] = dict(repo=repo, revision=info.sha, files=files, local=old.get("local", {}))
        wanted = (["config.json", "preprocessor_config.json", "README.md"] if kind == "model" else
                  ["ReefNet_RSG-TestDataset_annotations.csv", "ReefNet_RSG-TestDataset_labelmapping.xlsx"])
        for filename in wanted:
            if filename not in files:
                raise RuntimeError(f"Required published file missing: {repo}/{filename}")
            path = hf_hub_download(repo, filename, repo_type=kind, revision=info.sha,
                                   cache_dir=str(CACHE), token=False)
            result[kind]["local"][filename] = path
        write_json(manifest_path, result)
        print(json.dumps(dict(kind=kind, repo=repo, revision=info.sha, files=files)), flush=True)
    return result


def weights():
    from huggingface_hub import hf_hub_download
    manifest_path = OUT / "assets.json"
    result = json.loads(manifest_path.read_text())
    entry = result["model"]
    filename = "model.safetensors"
    print("Downloading only the published dense inference weights", flush=True)
    path = Path(hf_hub_download(MODEL, filename, revision=entry["revision"],
                                cache_dir=str(CACHE), token=False))
    expected = entry["files"][filename]
    if expected["bytes"] and path.stat().st_size != expected["bytes"]:
        raise RuntimeError("Published weight size mismatch")
    digest = sha256(path)
    if expected["sha256"] and digest != expected["sha256"]:
        raise RuntimeError("Published weight SHA-256 mismatch")
    entry["local"][filename] = str(path)
    entry["verified_sha256"] = digest
    write_json(manifest_path, result)
    print(json.dumps(dict(model=str(path), bytes=path.stat().st_size, sha256=digest)), flush=True)


def rein():
    # Set the account before importing Modal. No credentials are printed.
    os.environ["MODAL_PROFILE"] = PROFILE
    import modal
    volume = modal.Volume.from_name("coral-training", create_if_missing=False)
    remote = f"/checkpoints/{GROUP}/best_ema.pt"
    entries = list(volume.iterdir(f"/checkpoints/{GROUP}"))
    chosen = next((v for v in entries if str(v.path).endswith("/best_ema.pt")), None)
    if chosen is None:
        raise RuntimeError("The requested 40-epoch Rein++ inference checkpoint was not found")
    target = ROOT / "checkpoints/external_compare/reinpp40_best_ema.pt"
    record_path = OUT / "rein_checkpoint.json"
    expected = chosen.size
    if target.exists() and record_path.exists():
        record = json.loads(record_path.read_text())
        if target.stat().st_size == expected and sha256(target) == record["sha256"]:
            print("Reusing verified Rein++ inference checkpoint", flush=True)
            return
        raise RuntimeError("Existing Rein++ checkpoint does not match its transfer record")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".download")
    print(json.dumps(dict(source="Modal volume API", profile=PROFILE, remote=remote, bytes=expected)), flush=True)
    with tmp.open("wb") as f:
        volume.read_file_into_fileobj(remote, f)
    if tmp.stat().st_size != expected:
        raise RuntimeError("Incomplete Modal checkpoint transfer")
    digest = sha256(tmp)
    tmp.replace(target)
    record = dict(profile=PROFILE, group=GROUP, remote=remote, path=str(target), bytes=expected,
                  sha256=digest, fetched_at=time.time(), optimizer_downloaded=False,
                  cloud_compute_started=False)
    write_json(record_path, record)
    print(json.dumps(record), flush=True)


def sam():
    from huggingface_hub import HfApi, hf_hub_download
    path = OUT / "assets.json"
    result = json.loads(path.read_text())
    repo = "facebook/sam2-hiera-tiny"
    old = result.get("sam", {})
    info = HfApi(token=False).model_info(repo, revision=old.get("revision"), files_metadata=True)
    wanted = [v for v in info.siblings if v.rfilename in ["model.safetensors", "config.json",
                "preprocessor_config.json", "processor_config.json", "README.md"]]
    entry = dict(repo=repo, revision=info.sha, local={})
    for item in wanted:
        local = hf_hub_download(repo, item.rfilename, revision=info.sha, cache_dir=str(CACHE), token=False)
        if item.lfs and sha256(local) != item.lfs.sha256:
            raise RuntimeError("SAM2 checksum mismatch")
        entry["local"][item.rfilename] = local
        print(json.dumps(dict(sam_file=item.rfilename, bytes=item.size)), flush=True)
    result["sam"] = entry
    write_json(path, result)


def code():
    from huggingface_hub import hf_hub_download
    path=OUT/"assets.json"
    result=json.loads(path.read_text())
    entry=result["model"]
    name="coralscapes_hub_model.py"
    entry["local"][name]=hf_hub_download(MODEL,name,revision=entry["revision"],cache_dir=str(CACHE),token=False)
    write_json(path,result)
    print(entry["local"][name],flush=True)


def source(size=224):
    """Read a bounded source replay/calibration subset; keep only native crops."""
    import io
    import random
    import numpy as np
    from PIL import Image
    os.environ["MODAL_PROFILE"]=PROFILE
    import modal
    volume=modal.Volume.from_name("coral-training",create_if_missing=False)
    prefix="/data/coralscapes_v2_paper_8a4567b2fbec"
    destination=OUT/("source" if size==224 else "source512")
    destination.mkdir(exist_ok=True)
    record_path=OUT/("source.json" if size==224 else "source512.json")
    if record_path.exists():
        print("Reusing frozen source replay subset",flush=True);return
    entries=sorted([v.path for v in volume.iterdir(prefix+"/leftImg8bit/train",recursive=True)
                    if v.path.endswith("_leftImg8bit.png")])
    if len(entries)!=1790:raise RuntimeError("Unexpected source-training split")
    chosen=random.Random(42).sample(entries,16)
    rows=[]
    for i,remote in enumerate(chosen):
        mask_remote=remote.replace("/leftImg8bit/","/gtFine/").replace("_leftImg8bit.png","_gtFine.png")
        image_bytes,mask_bytes=io.BytesIO(),io.BytesIO()
        volume.read_file_into_fileobj(remote,image_bytes)
        volume.read_file_into_fileobj(mask_remote,mask_bytes)
        image=Image.open(io.BytesIO(image_bytes.getvalue())).convert("RGB")
        mask=Image.open(io.BytesIO(mask_bytes.getvalue()))
        rng=random.Random(42+i)
        for attempt in range(30):
            left=rng.randrange(image.width-size+1);top=rng.randrange(image.height-size+1)
            box=(left,top,left+size,top+size)
            y=np.array(mask.crop(box))
            if (y>0).mean()>=.6:break
        if y.max()>95:raise ValueError("Source labels must retain native V2 IDs")
        x_path=destination/f"source_{i:02d}.png";y_path=destination/f"source_{i:02d}_mask.png"
        image.crop(box).save(x_path);Image.fromarray(y).save(y_path)
        rows.append(dict(path=str(x_path),mask=str(y_path),remote=remote,bbox=list(box),
            sha256=sha256(x_path),mask_sha256=sha256(y_path),
            original_sha256=hashlib.sha256(image_bytes.getvalue()).hexdigest(),
            classes=[int(v) for v in np.unique(y)],profile=PROFILE))
        print(f"Source native crops {i+1}/16",flush=True)
    write_json(record_path,rows)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=["metadata", "weights", "rein", "sam", "code", "source", "source512"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"metadata": metadata, "weights": weights, "rein": rein, "sam": sam, "code":code,
     "source":source,"source512":lambda:source(512)}[args.action]()


if __name__ == "__main__":
    main()
