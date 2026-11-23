"""One isolated 40-epoch V2 experiment per Modal profile; no GPU launchers."""
from pathlib import Path
import io
import json
import os
import time
import uuid

import modal

os.environ.setdefault("PAPER_GPU", "L40S")
os.environ.setdefault("PAPER_CPU", "4")
os.environ.setdefault("PAPER_MEMORY_MB", "24576")
os.environ.setdefault("PAPER_PRECISION", "bf16")
from modal_paper_suite import image, volume

LOCAL = Path(__file__).resolve().parent
if modal.is_local():
    image = image.add_local_file(LOCAL / "modal_paper_suite.py", "/app/modal_paper_suite.py")
app = modal.App("coralscapes-v2-fresh40")
leases = modal.Dict.from_name("coralscapes-v2-fresh40-owners", create_if_missing=True)
PROFILE_METHODS = {"mohamed-elashmony05": "rein_g", "m12mm-4m2005": "lora"}


def claim(key):
    owner = modal.current_function_call_id()
    inserted = leases.put(key, owner, skip_if_exists=True)
    if not inserted and leases.get(key) != owner:
        raise RuntimeError(f"Another function call already owns {key}; refusing a duplicate GPU run")


@app.function(image=image, gpu="L40S", cpu=(4,4), memory=(24576,24576), timeout=1200,
              retries=0, max_containers=1, scaledown_window=2, single_use_containers=True,
              volumes={"/workspace": volume})
def probe_remote(method: str, group: str, deadline: float):
    claim(f"probe:{group}:{method}")
    if time.time() >= deadline:
        raise RuntimeError("The original probe deadline expired")
    import subprocess
    import sys
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    volume.reload()
    from paper_suite.recipe import root, write_json
    target = root() / "checkpoints" / group / method / "efficiency_probe.json"
    if target.exists():
        return json.loads(target.read_text())
    results = {}
    for checkpointing in (False, True):
        if time.time() >= deadline - 60:
            break
        subgroup = f"{group}-probe-{method}-gc{int(checkpointing)}"
        path = root() / "checkpoints" / subgroup / "bf16.json"
        command = [sys.executable, "-m", "paper_suite.benchmark", "--group", subgroup,
                   "--precision", "bf16", "--method", method,
                   "--decoder-checkpointing", str(int(checkpointing))]
        process = subprocess.run(command, check=False, timeout=max(1, int(deadline-time.time()-30)))
        result = json.loads(path.read_text()) if path.exists() else {"status":"failed", "exit_code": process.returncode}
        results[str(int(checkpointing))] = result
        volume.commit()
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, results)
    volume.commit()
    return results


@app.function(image=image, gpu="L40S", cpu=(4,4), memory=(24576,24576), timeout=40000,
              retries=0, max_containers=1, scaledown_window=2, single_use_containers=True,
              volumes={"/workspace": volume})
def train_remote(method: str, group: str, run_id: str, deadline: float, decoder_checkpointing: bool):
    claim(f"train:{group}:{method}")
    if time.time() >= deadline - 900:
        raise RuntimeError("The original training budget deadline expired")
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    volume.reload()
    from paper_suite.screen import run_screen
    try:
        return run_screen(method, group, run_id, deadline, decoder_checkpointing, commit=volume.commit)
    finally:
        volume.commit()


@app.function(image=image, cpu=(.125,.125), memory=(256,256), timeout=43200, retries=0)
def wait_remote(call_id: str):
    """A passive keeper: never submits or restarts training."""
    return modal.FunctionCall.from_id(call_id).get()


@app.local_entrypoint()
def main(method: str, group: str = "paper-v2-fresh40-s42-20261101", probe_only: bool = False,
         decoder_checkpointing: bool = False):
    import re
    from paper_suite.recipe import ENCODER_ID, ENCODER_REVISION, DATA_REVISION, write_json
    profile = os.environ.get("MODAL_PROFILE", "")
    if PROFILE_METHODS.get(profile) != method:
        raise ValueError("Use mohamed-elashmony05 for Rein and m12mm-4m2005 for LoRA")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", group):
        raise ValueError("Invalid output group")
    manifest_path = LOCAL / "analysis/paper_v2_sources" / profile / (group + "-submission.json")
    state = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "profile": profile, "method": method, "group": group, "run_id": uuid.uuid4().hex,
        "fresh_start": True, "stop_after_epochs": 40, "schedule_horizon_epochs": 150,
        "gpu": "L40S", "precision": "bf16", "cpu_limit": 4, "memory_limit_MiB": 24576,
        "gpu_cpu_ram_hourly_usd": 2.331648, "account_credit_usd": 30}
    phase = "probe" if probe_only else "train"
    if state.get(phase + "_call_id"):
        print(f"Already submitted {phase}: {state[phase + '_call_id']}")
        print(f"Manifest: {manifest_path}")
        return
    ready_path = f"/data/coralscapes_v2_paper_{DATA_REVISION[:12]}/ready.json"
    metadata = io.BytesIO()
    volume.read_file_into_fileobj(ready_path, metadata)
    ready = json.loads(metadata.getvalue())
    if ready.get("revision") != DATA_REVISION or ready.get("counts") != {"train":1790,"val":170,"test":473}:
        raise ValueError("The verified native V2 dataset must already be prepared")
    encoder = "/.cache/huggingface/hub/models--" + ENCODER_ID.replace("/", "--") + "/snapshots/" + ENCODER_REVISION
    names = {Path(entry.path).name for entry in volume.listdir(encoder)}
    if not {"model.safetensors", "config.json", "preprocessor_config.json"} <= names:
        raise ValueError("The pinned original pretrained encoder must already exist")
    if probe_only:
        deadline = time.time() + 1200
        call = probe_remote.spawn(method, group, deadline)
    else:
        # A brand-new directory is required for the first submission. Only a
        # replay of the same function call can resume its own complete state.
        try:
            existing = volume.listdir(f"/checkpoints/{group}/{method}")
        except modal.exception.NotFoundError:
            existing = []
        if any(Path(entry.path).name in {"last.pt", "progress.json", "completed.json"} for entry in existing):
            raise ValueError("Training state already exists; refusing to treat it as a fresh run")
        probe = io.BytesIO()
        volume.read_file_into_fileobj(f"/checkpoints/{group}/{method}/efficiency_probe.json", probe)
        results = json.loads(probe.getvalue())
        selected = results.get(str(int(decoder_checkpointing)), {})
        if selected.get("status") != "passed":
            raise ValueError("The selected configuration has not passed its full-batch GPU probe")
        if method == "rein_g":
            norms = selected.get("rein_token_gradient_norm_by_layer", [])
            if len(norms) != 24 or not all(value > 0 for value in norms) or selected.get("rein_readout_max_abs") != [0.,0.,0.,0.]:
                raise ValueError("Corrected Rein must pass every full-backbone gradient and DPT readout check")
        deadline = time.time() + 40000
        state["decoder_checkpointing"] = decoder_checkpointing
        state["maximum_training_compute_usd"] = 40000 * 2.331648 / 3600
        call = train_remote.spawn(method, group, state["run_id"], deadline, decoder_checkpointing)
    state.update({phase + "_call_id": call.object_id, phase + "_deadline_utc_timestamp": deadline,
                  phase + "_dashboard": app.get_dashboard_url()})
    write_json(manifest_path, state)
    keeper = wait_remote.spawn(call.object_id)
    state[phase + "_keeper_call_id"] = keeper.object_id
    write_json(manifest_path, state)
    print(f"Submitted {phase} {method} on {profile}: {call.object_id}")
    print(f"Manifest: {manifest_path}")
    print(f"Dashboard: {app.get_dashboard_url()}")
