"""Run coral LoRA training on Modal and persist data/checkpoints in a Volume."""

from pathlib import Path
import math
import re

import modal
import config

LOCAL = Path(__file__).resolve().parent
app = modal.App("coral-lora-training")
volume = modal.Volume.from_name(config.MODAL["volume"], create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(LOCAL / "requirements.txt"))
    .env({"CORAL_ROOT": "/workspace", "USE_TF": "0", "PYTHONUNBUFFERED": "1",
          "PYTHONPATH": "/app"})
    .add_local_file(LOCAL / "train.py", "/app/train.py")
    .add_local_file(LOCAL / "config.py", "/app/config.py")
    .add_local_file(LOCAL / "ensemble.py", "/app/ensemble.py")
    .add_local_file(LOCAL / "dpt.py", "/app/dpt.py")
    .add_local_file(LOCAL / "coral_data.py", "/app/coral_data.py")
    .add_local_dir(LOCAL / "coralscapes_metadata", "/app/coralscapes_metadata")
    .add_local_dir(LOCAL / "third_party", "/app/third_party", ignore=["**/__pycache__/**"])
)


@app.function(image=image, cpu=config.MODAL["cpu"], memory=config.MODAL["memory"],
              timeout=config.MODAL["timeout"], volumes={"/workspace": volume})
def prepare_remote(height: int, annotation_version: str = "V1"):
    from train import download_data, load_data, sampling_weights
    from coral_data import dataset_config
    from huggingface_hub import snapshot_download

    print(f"Preparing {height}x{height * 2} data on CPU (no GPU allocated).", flush=True)
    snapshot_download(config.MODEL["id"], revision=config.MODEL["revision"],
                      cache_dir="/workspace/.cache/huggingface/hub",
                      local_files_only=True)
    print("Pretrained DINOv3 cached on CPU.", flush=True)
    annotation = dataset_config(annotation_version)
    print(f"Annotation source: {annotation['id']} @ {annotation['revision']}; {annotation_version}", flush=True)
    directory = download_data(annotation)
    data = load_data(directory, height, settings=annotation)
    _, summary = sampling_weights(data["train"], directory, height, label_map=annotation["label_map"])
    print(f"Prepared training sampling: {summary}", flush=True)
    volume.commit()


@app.function(image=image, gpu=config.MODAL["gpu"], cpu=config.MODAL["cpu"],
              memory=config.MODAL["memory"], timeout=config.MODAL["timeout"],
              volumes={"/workspace": volume})
def train_remote(epochs: int, height: int, batch_size: int, lr: float,
                 warmup_epochs: int, min_lr: float, run_name: str,
                 accumulation_steps: int, max_updates: int, patience: int,
                 min_epochs: int, min_delta: float, warmup_steps: int, seed: int,
                 decoder: str, annotation_version: str = "V1"):
    from train import main

    volume.reload()  # Pick up the cache committed by the CPU preparation function.
    output = Path("/workspace/checkpoints") / run_name
    if output.exists():
        raise ValueError(f"Run {run_name!r} already exists; choose a new --run-name.")
    try:
        main([
            "--epochs", str(epochs), "--height", str(height),
            "--batch-size", str(batch_size), "--lr", str(lr),
            "--warmup-epochs", str(warmup_epochs), "--min-lr", str(min_lr),
            "--output", str(output),
            "--accumulation-steps", str(accumulation_steps), "--max-updates", str(max_updates),
            "--patience", str(patience), "--min-epochs", str(min_epochs),
            "--min-delta", str(min_delta), "--warmup-steps", str(warmup_steps),
            "--seed", str(seed), "--decoder", decoder,
            "--annotation-version", annotation_version,
        ], on_epoch_end=volume.commit, prepared_data=True)
    finally:
        volume.commit()
    return f"Saved to Modal Volume {config.MODAL['volume']}: /checkpoints/{run_name}"


@app.function(image=image, cpu=0.125, memory=256, timeout=config.MODAL["timeout"])
def run_remote(epochs: int, height: int, batch_size: int, lr: float,
               warmup_epochs: int, min_lr: float, gpu: str, run_name: str,
               accumulation_steps: int, max_updates: int, patience: int,
               min_epochs: int, min_delta: float, warmup_steps: int, seed: int,
               decoder: str, annotation_version: str = "V1"):
    """Coordinate both stages in the cloud so local shutdown cannot interrupt them."""
    prepare_remote.remote(height, annotation_version)
    return train_remote.with_options(gpu=gpu).remote(
        epochs, height, batch_size, lr, warmup_epochs, min_lr, run_name,
        accumulation_steps, max_updates, patience, min_epochs, min_delta, warmup_steps, seed, decoder,
        annotation_version,
    )


@app.local_entrypoint()
def main(epochs: int = config.TRAIN["epochs"], height: int = config.TRAIN["height"],
         batch_size: int = config.TRAIN["batch_size"], lr: float = config.TRAIN["lr"],
         warmup_epochs: int = config.TRAIN["warmup_epochs"], min_lr: float = config.TRAIN["min_lr"],
         gpu: str = config.MODAL["gpu"], run_name: str = "",
         accumulation_steps: int = config.TRAIN["accumulation_steps"],
         max_updates: int = config.TRAIN["max_updates"],
         patience: int = config.TRAIN["early_stopping_patience"],
         min_epochs: int = config.TRAIN["early_stopping_min_epochs"],
         min_delta: float = config.TRAIN["early_stopping_min_delta"],
         warmup_steps: int = config.TRAIN["warmup_steps"], seed: int = config.TRAIN["seed"],
         decoder: str = config.MODEL["decoder"],
         annotation_version: str = config.DATASET["annotation_version"]):
    if annotation_version not in ("V1", "V2"):
        raise ValueError("--annotation-version must be V1 or V2.")
    run_name = run_name or (config.RUN_NAME if annotation_version == "V1" else config.RUN_NAME + "-v2")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("--run-name must contain only letters, digits, '-' or '_'.")
    patch = config.MODEL.get("patch_size", 14)
    if height < patch or height % patch or epochs < 1 or batch_size < 1:
        raise ValueError(f"Use positive epochs/batch size and a height divisible by {patch}.")
    if not (math.isfinite(lr) and math.isfinite(min_lr) and 0 < min_lr <= lr):
        raise ValueError("Use finite learning rates with 0 < min-lr <= lr.")
    if accumulation_steps < 1 or max_updates < 0 or patience < 0 or min_epochs < 1 or warmup_steps < 0 or seed < 0:
        raise ValueError("Invalid accumulation, update ceiling, early stopping, warmup or seed.")
    if not math.isfinite(min_delta) or min_delta < 0 or decoder not in ("dpt", "shallow"):
        raise ValueError("Use nonnegative min-delta and a dpt or shallow decoder.")
    # Fetch gated weights locally, then upload only the two model files, never a token.
    # The pinned snapshot directory is sufficient for HF's local-only loader.
    from huggingface_hub import snapshot_download
    remote = ("/.cache/huggingface/hub/models--" + config.MODEL["id"].replace("/", "--")
              + "/snapshots/" + config.MODEL["revision"])
    try:
        present = {Path(entry.path).name for entry in volume.listdir(remote)}
    except modal.exception.NotFoundError:
        present = set()
    if not {"config.json", "model.safetensors"} <= present:
        source = Path(snapshot_download(
            config.MODEL["id"], revision=config.MODEL["revision"],
            cache_dir=LOCAL / ".cache/huggingface/hub",
            allow_patterns=["config.json", "model.safetensors"],
        ))
        with volume.batch_upload(force=True) as upload:
            for name in ("config.json", "model.safetensors"):
                upload.put_file(source / name, remote + "/" + name)
        print("Uploaded pretrained weights; Hugging Face credentials stay local.")
    call = run_remote.spawn(
        epochs, height, batch_size, lr, warmup_epochs, min_lr, gpu, run_name,
        accumulation_steps, max_updates, patience, min_epochs, min_delta, warmup_steps, seed, decoder,
        annotation_version,
    )
    print(f"Submitted workflow: {call.object_id}")
    print(f"Dashboard: {app.get_dashboard_url()}")
    print("Use --detach; after submission you can close the terminal or shut down.")
