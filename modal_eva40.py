"""One fresh, matched fixed-rank EVA run on the existing LoRA account."""
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
os.environ.setdefault("PAPER_ROOT", "/workspace")
from modal_paper_suite import image, volume

app = modal.App("coralscapes-v2-eva40")
owners = modal.Dict.from_name("coralscapes-v2-eva40-owners", create_if_missing=True)
LOCAL = Path(__file__).resolve().parent
if modal.is_local():
    image = image.add_local_file(LOCAL / "modal_paper_suite.py", "/app/modal_paper_suite.py")
REFERENCE = "paper-v2-fresh40-s42-20261101/lora"
PROFILE = "m12mm-4m2005"


@app.function(image=image, gpu="L40S", cpu=(4, 4), memory=(24576, 24576), timeout=86400,
              retries=0, max_containers=1, scaledown_window=2, single_use_containers=True,
              volumes={"/workspace": volume})
def train_remote(group: str, run_id: str, deadline: float):
    key, owner = f"{group}:eva", modal.current_function_call_id()
    if not owners.put(key, owner, skip_if_exists=True) and owners.get(key) != owner:
        raise RuntimeError("Another function call already owns this EVA run")
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import gc
    import torch
    import yaml
    from paper_suite.benchmark import benchmark
    from paper_suite.eva import assert_lora_match
    from paper_suite.recipe import root, write_json
    from paper_suite.screen import run_screen, screen_config

    volume.reload()
    cfg = screen_config("eva", group, run_id, False)
    out = Path(cfg["train"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    reference = root() / "checkpoints" / REFERENCE
    saved_cfg = yaml.safe_load((reference / "config.yaml").read_text())
    assert_lora_match(cfg, saved_cfg)
    reference_init = json.loads((reference / "initialization.json").read_text())
    write_json(out / "comparison_control.json", {
        "reference": REFERENCE, "shared_hyperparameters_equal": True,
        "reference_decoder_sha256": reference_init["decoder_sha256"],
        "comparison": "Only A initialization differs; fixed ranks, no LoRA-FA, no block freezing",
        "paper_departures": ["40-epoch screen on the 150-epoch schedule", "BF16 L40S runtime and effective-weight EMA match the saved LoRA control"],
    })
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    volume.commit()
    try:
        probe_file = out / "efficiency_probe.json"
        if not probe_file.exists():
            print("EVA_STAGE initialization_and_full_batch_probe", flush=True)
            result = benchmark(group, "bf16", method="eva", decoder_checkpointing=False,
                               commit=volume.commit)
            if result.get("status") != "passed" or result["decoder_initialization_sha256"] != reference_init["decoder_sha256"]:
                raise RuntimeError("EVA full-batch probe failed or decoder differs from saved LoRA")
            write_json(probe_file, {"0": result})
            volume.commit()
            gc.collect()
            torch.cuda.empty_cache()
        print("EVA_STAGE training_40_epochs", flush=True)
        return run_screen("eva", group, run_id, deadline, False, commit=volume.commit)
    except BaseException as error:
        write_json(out / "failure.json", {"error_type": type(error).__name__, "message": str(error)[:2000]})
        raise
    finally:
        volume.commit()


@app.function(image=image, cpu=(.125, .125), memory=(256, 256), timeout=86400, retries=0)
def wait_remote(call_id: str):
    """Passive keeper; never restarts or submits GPU work."""
    return modal.FunctionCall.from_id(call_id).get()


@app.local_entrypoint()
def main(group: str = "paper-v2-eva40-s42-20261102"):
    import re
    import yaml
    from paper_suite.eva import assert_lora_match
    from paper_suite.recipe import DATA_REVISION, ENCODER_ID, ENCODER_REVISION, write_json
    from paper_suite.screen import screen_config
    if os.environ.get("MODAL_PROFILE") != PROFILE:
        raise ValueError(f"EVA must use LoRA's existing account: {PROFILE}")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", group):
        raise ValueError("Invalid output group")
    manifest_path = LOCAL / "analysis/paper_v2_sources" / PROFILE / (group + "-submission.json")
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("train_call_id"):
            print(f"Already submitted: {previous['train_call_id']}; refusing a duplicate")
            return

    def read_json(path):
        buffer = io.BytesIO()
        volume.read_file_into_fileobj(path, buffer)
        return json.loads(buffer.getvalue())

    ready = read_json(f"/data/coralscapes_v2_paper_{DATA_REVISION[:12]}/ready.json")
    if ready.get("revision") != DATA_REVISION or ready.get("counts") != {"train": 1790, "val": 170, "test": 473}:
        raise ValueError("Expected the existing verified native V2 dataset")
    encoder = "/.cache/huggingface/hub/models--" + ENCODER_ID.replace("/", "--") + "/snapshots/" + ENCODER_REVISION
    if not {"model.safetensors", "config.json", "preprocessor_config.json"} <= {Path(x.path).name for x in volume.listdir(encoder)}:
        raise ValueError("Pinned original pretrained encoder is missing")
    run_id = uuid.uuid4().hex
    # Assert locally too, before allocating a GPU.
    buffer = io.BytesIO()
    volume.read_file_into_fileobj(f"/checkpoints/{REFERENCE}/config.yaml", buffer)
    assert_lora_match(screen_config("eva", group, run_id), yaml.safe_load(buffer.getvalue()))
    try:
        existing = volume.listdir(f"/checkpoints/{group}/eva")
    except modal.exception.NotFoundError:
        existing = []
    if existing:
        raise ValueError("EVA output directory already exists; refusing a fresh overwrite")
    deadline = time.time() + 86400
    call = train_remote.spawn(group, run_id, deadline)
    state = {"profile": PROFILE, "method": "eva", "group": group, "run_id": run_id,
             "reference": REFERENCE, "fresh_start": True, "stop_after_epochs": 40,
             "schedule_horizon_epochs": 150, "expected_optimizer_updates": 8920,
             "gpu": "L40S", "precision": "bf16", "physical_batch": 8,
             "decoder_checkpointing": False, "eva_rho": 1.0,
             "train_call_id": call.object_id, "deadline_utc_timestamp": deadline,
             "dashboard": app.get_dashboard_url(), "automatic_retries": 0}
    write_json(manifest_path, state)
    keeper = wait_remote.spawn(call.object_id)
    state["keeper_call_id"] = keeper.object_id
    write_json(manifest_path, state)
    print(json.dumps(state, indent=2), flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
