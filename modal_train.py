"""Run coral LoRA training on Modal and persist data/checkpoints in a Volume."""

from pathlib import Path
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
    .add_local_dir(LOCAL / "third_party", "/app/third_party", ignore=["**/__pycache__/**"])
)


@app.function(image=image, cpu=config.MODAL["cpu"], memory=config.MODAL["memory"],
              timeout=config.MODAL["timeout"], volumes={"/workspace": volume})
def prepare_remote(height: int):
    from train import download_data, load_data, sampling_weights
    from huggingface_hub import snapshot_download

    print(f"Preparing {height}x{height * 2} data on CPU (no GPU allocated).", flush=True)
    snapshot_download(config.MODEL["id"], revision=config.MODEL["revision"],
                      cache_dir="/workspace/.cache/huggingface/hub",
                      local_files_only=True)
    print("Pretrained DINOv3 cached on CPU.", flush=True)
    directory = download_data()
    data = load_data(directory, height)
    _, summary = sampling_weights(data["train"], directory, height)
    print(f"Prepared training sampling: {summary}", flush=True)
    volume.commit()


@app.function(image=image, gpu=config.MODAL["gpu"], cpu=config.MODAL["cpu"],
              memory=config.MODAL["memory"], timeout=config.MODAL["timeout"],
              volumes={"/workspace": volume})
def train_remote(epochs: int, height: int, batch_size: int, lr: float,
                 warmup_epochs: int, min_lr: float, run_name: str):
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
        ], on_epoch_end=volume.commit, prepared_data=True)
    finally:
        volume.commit()
    return f"Saved to Modal Volume {config.MODAL['volume']}: /checkpoints/{run_name}"


@app.function(image=image, cpu=0.125, memory=256, timeout=config.MODAL["timeout"])
def run_remote(epochs: int, height: int, batch_size: int, lr: float,
               warmup_epochs: int, min_lr: float, gpu: str, run_name: str):
    """Coordinate both stages in the cloud so local shutdown cannot interrupt them."""
    prepare_remote.remote(height)
    return train_remote.with_options(gpu=gpu).remote(
        epochs, height, batch_size, lr, warmup_epochs, min_lr, run_name,
    )


@app.local_entrypoint()
def main(epochs: int = config.TRAIN["epochs"], height: int = config.TRAIN["height"],
         batch_size: int = config.TRAIN["batch_size"], lr: float = config.TRAIN["lr"],
         warmup_epochs: int = config.TRAIN["warmup_epochs"], min_lr: float = config.TRAIN["min_lr"],
         gpu: str = config.MODAL["gpu"], run_name: str = config.RUN_NAME):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("--run-name must contain only letters, digits, '-' or '_'.")
    patch = config.MODEL.get("patch_size", 14)
    if height < patch or height % patch or epochs < 1 or batch_size < 1:
        raise ValueError(f"Use positive epochs/batch size and a height divisible by {patch}.")
    if not 0 <= warmup_epochs < epochs or not 0 < min_lr <= lr:
        raise ValueError("Use 0 <= warmup-epochs < epochs and 0 < min-lr <= lr.")
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
    )
    print(f"Submitted workflow: {call.object_id}")
    print(f"Dashboard: {app.get_dashboard_url()}")
    print("Use --detach; after submission you can close the terminal or shut down.")
