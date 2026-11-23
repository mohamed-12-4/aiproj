"""Submit matched paper-recipe runs; only local code constructs the Modal image."""

from pathlib import Path
import json
import os

import modal

LOCAL = Path(__file__).resolve().parent
app = modal.App("coralscapes-v2-paper-suite")
volume = modal.Volume.from_name("coral-training")
GPU = os.environ.get("PAPER_GPU", "A100-80GB")
CPU = int(os.environ.get("PAPER_CPU", "16"))
MEMORY = int(os.environ.get("PAPER_MEMORY_MB", "65536"))
if modal.is_local():
    image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install("torch==2.8.0", "torchvision==0.23.0", "transformers==4.57.6", "peft==0.18.1",
                          "pyarrow==21.0.0", "huggingface_hub==0.36.0", "torchmetrics==1.8.2",
                          "PyYAML==6.0.3", "Pillow>=10", "numpy>=1.26", "tqdm>=4.66", "matplotlib>=3.8")
             .env({"PAPER_ROOT": "/workspace", "HF_HOME": "/workspace/.cache/huggingface",
                   "HF_HUB_DISABLE_TELEMETRY": "1", "USE_TF": "0", "USE_FLAX": "0",
                   "PAPER_GPU": GPU, "PAPER_CPU": str(CPU), "PAPER_MEMORY_MB": str(MEMORY),
                   "PAPER_PRECISION": os.environ.get("PAPER_PRECISION", "fp16"),
                   "PAPER_DECODER_CHECKPOINTING": os.environ.get("PAPER_DECODER_CHECKPOINTING", "0"),
                   "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                   "PYTHONPATH": "/app", "PYTHONUNBUFFERED": "1", "MPLCONFIGDIR": "/tmp/matplotlib"})
             .add_local_dir(LOCAL / "paper_suite", "/app/paper_suite", ignore=["**/__pycache__/**"])
             .add_local_dir(LOCAL / "coralscapes_metadata", "/app/coralscapes_metadata")
             .add_local_dir(LOCAL / "third_party/coralscapes_v2_author", "/app/third_party/coralscapes_v2_author",
                            ignore=[".git/**", "**/__pycache__/**"])
             .add_local_file(LOCAL / "third_party/soma_author/rein/peft/sora.py", "/app/third_party/soma_author/rein/peft/sora.py")
             .add_local_file(LOCAL / "third_party/rein_author/simple_reins/reins.py", "/app/third_party/rein_author/simple_reins/reins.py"))
else:
    image = modal.Image.debian_slim()


@app.function(image=image, cpu=8, memory=32768, timeout=7200, retries=0, volumes={"/workspace": volume})
def prepare_remote():
    from paper_suite.data import prepare
    result = prepare()
    volume.commit()
    return result


@app.function(image=image, gpu=GPU, cpu=CPU, memory=MEMORY, timeout=86400,
              retries=0, single_use_containers=True, volumes={"/workspace": volume})
def train_remote(method: str, group: str, smoke_only: bool = False):
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from paper_suite.runner import run_smoke, run
    volume.reload()
    try:
        result = run_smoke(method, group)
        volume.commit()
        if smoke_only:
            return result
        return run(method, group, commit=volume.commit)
    finally:
        volume.commit()


@app.function(image=image, gpu="A10", cpu=4, memory=24576, timeout=3600,
              retries=0, single_use_containers=True, volumes={"/workspace": volume})
def benchmark_remote(group: str):
    import subprocess
    import sys
    from paper_suite.recipe import root
    volume.reload()
    results = {}
    for precision in ("bf16", "fp16"):
        command = [sys.executable, "-m", "paper_suite.benchmark", "--group", group, "--precision", precision]
        result = subprocess.run(command, check=False)
        path = root() / "checkpoints" / group / (precision + ".json")
        results[precision] = json.loads(path.read_text()) if path.exists() else {"exit_code": result.returncode}
        volume.commit()
    print(json.dumps(results), flush=True)
    return results


@app.function(image=image, cpu=0.125, memory=1024, timeout=86400, retries=0)
def wait_remote(calls: dict[str, str]):
    """Keep a detached App alive; replaying this waiter never submits GPUs."""
    results = {}
    for method, call_id in calls.items():
        try:
            modal.FunctionCall.from_id(call_id).get()
            results[method] = "completed"
        except Exception as error:
            results[method] = type(error).__name__
    return results


@app.function(image=image, cpu=0.25, memory=1024, timeout=86400, retries=0, volumes={"/workspace": volume})
def supervise_remote(group: str, calls: dict[str, str], continuations: dict[str, int] | None = None):
    """Continue from committed state only if Modal's hard function deadline fires.

    Never retry model errors or explicit cancellations. Rotate the CPU observer
    before its own deadline; active training is not interrupted by this rotation.
    """
    import time
    from paper_suite.recipe import root, write_json
    started = time.monotonic()
    continuations = continuations or {}
    finished = {}
    target = root() / "checkpoints" / group / "continuation_monitor.json"
    while True:
        volume.reload()
        control_path = root() / "checkpoints" / group / "run_control.json"
        paused = set(json.loads(control_path.read_text()).get("paused_methods", [])) if control_path.exists() else set()
        for method, call_id in list(calls.items()):
            if method in finished:
                continue
            if method in paused:
                finished[method] = "paused_by_user"
                continue
            completed = root() / "checkpoints" / group / method / "completed.json"
            if completed.exists():
                finished[method] = "completed"
                continue
            try:
                modal.FunctionCall.from_id(call_id).get(timeout=0)
                finished[method] = "completed"
            except modal.exception.FunctionTimeoutError:
                checkpoint = root() / "checkpoints" / group / method / "last.pt"
                count = continuations.get(method, 0)
                if not checkpoint.exists() or count >= 2:
                    finished[method] = "hard_deadline_without_continuation"
                    continue
                continuation = train_remote.spawn(method, group, False)
                calls[method] = continuation.object_id
                continuations[method] = count + 1
                print(f"Continuing {method} from its committed checkpoint after the Modal hard deadline: {continuation.object_id}", flush=True)
            except TimeoutError:
                pass  # Non-blocking get: the training input is still running.
            except Exception as error:
                finished[method] = "failed_without_retry:" + type(error).__name__
        state = {"group": group, "calls": calls, "deadline_continuations": continuations,
                 "terminal_status": finished, "policy": "Only hard function deadlines resume from raw checkpoints; model errors and cancellations are not retried."}
        write_json(target, state)
        volume.commit()
        if len(finished) == len(calls):
            return state
        if time.monotonic() - started > 18 * 3600:
            remaining = {m:c for m,c in calls.items() if m not in finished}
            next_observer = supervise_remote.spawn(group, remaining, continuations)
            print(f"Observer handoff without interrupting training: {next_observer.object_id}", flush=True)
            return {"next_observer": next_observer.object_id}
        time.sleep(60)


@app.local_entrypoint()
def main(group: str = "paper-v2-s42", methods: str = "lora,decoder_ft,soma,rein_g",
         prepare_only: bool = False, smoke_only: bool = False, supervise_only: bool = False,
         benchmark_only: bool = False, benchmark_gpu: str = "A10", submission_file: str = ""):
    import re
    from paper_suite.recipe import METHODS, ENCODER_ID, ENCODER_REVISION, PROJECT
    from huggingface_hub import snapshot_download
    if not re.fullmatch(r"[A-Za-z0-9_-]+", group):
        raise ValueError("Invalid group name")
    chosen = methods.split(",")
    if len(set(chosen)) != len(chosen) or any(m not in METHODS for m in chosen):
        raise ValueError("Invalid or repeated method")
    if benchmark_only:
        if benchmark_gpu not in ("A10", "L40S"):
            raise ValueError("Cost screening is restricted to A10 and L40S")
        call = benchmark_remote.with_options(gpu=benchmark_gpu).spawn(group)
        print(f"Submitted cost benchmark on {benchmark_gpu}: {call.object_id}")
        print(f"Dashboard: {app.get_dashboard_url()}")
        return
    if supervise_only:
        path = Path(submission_file) if submission_file else LOCAL / "analysis/paper_v2_sources/submission.json"
        submitted = json.loads(path.read_text())
        if submitted["group"] != group:
            raise ValueError("Submission manifest belongs to a different group")
        selected_calls = {method: call_id for method, call_id in submitted["calls"].items() if method in chosen}
        call = supervise_remote.spawn(group, selected_calls)
        print(f"Submitted continuation observer: {call.object_id}")
        print(f"Dashboard: {app.get_dashboard_url()}")
        return
    remote = "/.cache/huggingface/hub/models--" + ENCODER_ID.replace("/", "--") + "/snapshots/" + ENCODER_REVISION
    needed = {"config.json", "model.safetensors", "preprocessor_config.json"}
    try:
        existing = {Path(x.path).name for x in volume.listdir(remote)}
    except modal.exception.NotFoundError:
        existing = set()
    if not needed <= existing:
        source = Path(snapshot_download(ENCODER_ID, revision=ENCODER_REVISION,
                      cache_dir=PROJECT / ".cache/huggingface/hub", allow_patterns=sorted(needed)))
        with volume.batch_upload(force=True) as upload:
            for name in sorted(needed):
                upload.put_file(source / name, remote + "/" + name)
        print("Staged pinned DINOv3-L weights; credentials remain local.")
    if prepare_only:
        call = prepare_remote.spawn()
        try:
            print(call.get(timeout=7200))
        except BaseException:
            call.cancel(terminate_containers=True)
            raise
    else:
        # Submit from this local process. A remote launcher can be preempted and
        # replayed after spawning children, creating independent duplicate GPUs.
        # Persist each call immediately so a repeated CLI invocation can reuse it.
        from paper_suite.recipe import write_json
        import io
        profile = os.environ.get("MODAL_PROFILE", "default")
        suffix = "-smoke-submission.json" if smoke_only else "-submission.json"
        manifest = LOCAL / "analysis/paper_v2_sources" / profile / (group + suffix)
        previous = json.loads(manifest.read_text()) if manifest.exists() else {}
        state = {"group": group, "profile": profile, "smoke_only": smoke_only,
                 "calls": dict(previous.get("calls", {}))}
        prepare_remote.remote()
        for method in chosen:
            old_id = state["calls"].get(method)
            if old_id:
                try:
                    modal.FunctionCall.from_id(old_id).get(timeout=0)
                except modal.exception.FunctionTimeoutError:
                    pass  # A hard deadline may resume the raw checkpoint.
                except TimeoutError:
                    print(f"Already submitted {method}: {old_id}")
                    continue
                except Exception as error:
                    raise RuntimeError(f"Inspect the previous {method} call {old_id} before resubmitting") from error
                else:
                    print(f"Already completed {method}: {old_id}")
                    continue
            call = train_remote.spawn(method, group, smoke_only)
            state["calls"][method] = call.object_id
            write_json(manifest, state)
            print(f"Submitted {method}: {call.object_id}")
        with volume.batch_upload(force=True) as upload:
            upload.put_file(io.BytesIO(json.dumps(state).encode()), f"/checkpoints/{group}/submission.json")
        # Modal keeps the last triggered input alive in detached mode. This
        # passive waiter holds all selected calls without owning their launches.
        keeper = wait_remote.spawn({method: state["calls"][method] for method in chosen})
        print(f"Submitted passive waiter: {keeper.object_id}")
        print(f"Submission manifest: {manifest}")
        print(f"Dashboard: {app.get_dashboard_url()}")
        print("Use modal run --detach so training continues after this CLI exits.")
