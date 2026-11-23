"""Bounded external inference; never trains, retries, or transfers checkpoints.

MODAL_PROFILE=mohamed-elashmony05 .venv/bin/modal run modal_neom_pilot.py --prepare-only
MODAL_PROFILE=mohamed-elashmony05 .venv/bin/modal run modal_neom_pilot.py
MODAL_PROFILE=m12mm-4m2005 NEOM_MODEL=lora .venv/bin/modal run modal_neom_pilot.py
"""
import hashlib
import io
import json
import os
from pathlib import Path
import time

import modal

LOCAL = Path(__file__).resolve().parent
OUTPUT = LOCAL / "analysis/external_eval/neom_pilot"
GROUP = "neom-confirmed36-s42-20261102"
METHOD = os.environ.get("NEOM_MODEL", "reinpp")
PROFILE = {"reinpp": "mohamed-elashmony05", "lora": "m12mm-4m2005"}
volume = modal.Volume.from_name("coral-training")
app = modal.App("coralscapes-external-neom-pilot")
leases = modal.Dict.from_name("coralscapes-external-pilot-owners", create_if_missing=True)
if modal.is_local():
    if METHOD == "reinpp":
        from modal_reinpp import image
    elif METHOD == "lora":
        os.environ.setdefault("PAPER_GPU", "L40S")
        os.environ.setdefault("PAPER_CPU", "2")
        os.environ.setdefault("PAPER_MEMORY_MB", "16384")
        from modal_paper_suite import image
    else:
        raise ValueError(METHOD)
    image = (image.add_local_file(LOCAL / "external_pilot_runtime.py", "/app/external_pilot_runtime.py")
             .add_local_dir(OUTPUT / "data", "/pilot"))
else:
    image = modal.Image.debian_slim()


@app.function(image=image, cpu=(2,2), memory=(8192,8192), timeout=600,
              retries=0, volumes={"/workspace":volume})
def audit_remote():
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    from PIL import Image
    from scipy.fft import dctn
    from external_pilot_runtime import read_pilot, sha256
    volume.reload()
    manifest, rows = read_pilot("/pilot")
    hashes = {v["image_id"]:int(v["perceptual_hash"],16) for v in manifest["images"]}
    root = Path("/workspace/data/coralscapes_v2_paper_8a4567b2fbec/leftImg8bit/train")
    paths = sorted(root.rglob("*_leftImg8bit.png"))
    assert len(paths)==1790
    best = {key:dict(distance=65) for key in hashes}
    def image_hash(path):
        with Image.open(path) as image:
            a=np.asarray(image.convert("L").resize((32,32),Image.Resampling.LANCZOS),dtype=float)
        values=dctn(a,norm="ortho")[:8,:8].flatten()
        bits=values>np.median(values[1:]);bits[0]=False
        number=sum(int(v)<<i for i,v in enumerate(bits))
        return path,number
    # Overlap cloud-file reads; decoding remains within the two-CPU resource cap.
    with ThreadPoolExecutor(max_workers=16) as pool:
        for index,(path,number) in enumerate(pool.map(image_hash,paths)):
            for key,value in hashes.items():
                distance=(number^value).bit_count()
                if distance<best[key]["distance"]:best[key]=dict(distance=distance,train_image=str(path.relative_to(root)))
            if (index+1)%300==0:print(f"Duplicate audit {index+1}/1790",flush=True)
    suspicious={str(k):v for k,v in best.items() if v["distance"]<=6}
    result=dict(status="passed" if not suspicious else "needs_review",training_images=1790,
                external_images=len(hashes),external_points=len(rows),near_duplicate_threshold_hamming=6,
                nearest_training_images=best,suspicious=suspicious,manifest_sha256=sha256("/pilot/manifest.json"),
                limitations="Perceptual duplicate screening does not prove complete survey/provenance independence.")
    return result


@app.function(image=image, gpu="L40S", cpu=(2,2), memory=(16384,16384), timeout=240,
              retries=0, max_containers=1, single_use_containers=True, scaledown_window=2,
              volumes={"/workspace":volume})
def evaluate_remote(method: str, audit: dict):
    from external_pilot_runtime import evaluate, sha256
    if audit["status"]!="passed" or audit["manifest_sha256"]!=sha256("/pilot/manifest.json"):
        raise RuntimeError("The frozen evaluation data must pass its training-duplicate audit first")
    key=GROUP+":"+method;owner=modal.current_function_call_id()
    if not leases.put(key,owner,skip_if_exists=True) and leases.get(key)!=owner:
        raise RuntimeError("This pilot was already submitted; refusing a second GPU run")
    os.environ["HF_HUB_OFFLINE"]=os.environ["TRANSFORMERS_OFFLINE"]="1"
    volume.reload()
    destination=Path("/workspace/external_evaluations")/GROUP/(method+".json")
    if destination.exists():return json.loads(destination.read_text())
    checkpoint=(Path("/workspace/checkpoints/reinpp-v2-mhgelu-m2f-s42-40e-20261101/best_ema.pt")
                if method=="reinpp" else Path("/workspace/checkpoints/paper-v2-fresh40-s42-20261101/lora/best_ema_calibrated.pt"))
    result=evaluate(method,"/pilot",checkpoint)
    result.update(duplicate_audit=audit,group=GROUP)
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix(".tmp");temporary.write_text(json.dumps(result,indent=2)+"\n");temporary.replace(destination)
    volume.commit()
    return result


@app.local_entrypoint()
def main(prepare_only: bool=False):
    if os.environ.get("MODAL_PROFILE")!=PROFILE.get(METHOD):
        raise ValueError("Each model must use its existing account; no checkpoint transfer is performed")
    audit_path=OUTPUT/"duplicate_audit.json"
    if prepare_only:
        result=audit_remote.remote();audit_path.write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result,indent=2));return
    audit=json.loads(audit_path.read_text())
    journal=OUTPUT/(METHOD+"_submission.json")
    if journal.exists():
        print("Already submitted; no new GPU run:",journal.read_text());return
    call=evaluate_remote.spawn(METHOD,audit)
    record=dict(group=GROUP,method=METHOD,profile=PROFILE[METHOD],call_id=call.object_id,
                gpu="L40S",gpu_timeout_seconds=240,retries=0,submitted_at=time.time(),dashboard=app.get_dashboard_url())
    journal.write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record,indent=2),flush=True)
    result=call.get()
    (OUTPUT/(METHOD+"_results.json")).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ["predictions","duplicate_audit"]},indent=2))
