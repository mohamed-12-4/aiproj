"""CPU class counts followed by GPU validation diagnostics; no training."""
from pathlib import Path

import modal
import config

app = modal.App("coral-diagnostics")
LOCAL = Path(__file__).resolve().parent
volume = modal.Volume.from_name(config.MODAL["volume"])
if modal.is_local():
    from modal_train import image as training_image
    image = (training_image.add_local_file(LOCAL / "diagnose.py", "/app/diagnose.py")
             .add_local_dir(LOCAL / "coralscapes_metadata", "/app/coralscapes_metadata"))
else:
    image = modal.Image.debian_slim()  # Decorators are already configured by the client.


@app.function(image=image, cpu=4, memory=16384, timeout=3600, volumes={"/workspace": volume})
def count_remote(run_name: str):
    import torch
    from diagnose import coverage
    from coral_data import data_directory, dataset_config
    checkpoint = Path("/workspace/checkpoints") / run_name
    state = torch.load(checkpoint / "head.pt", map_location="cpu", weights_only=True)
    height, epoch = state["height"], state["epoch"]
    annotation = state.get("dataset_config") or dataset_config("V1")
    del state
    print(f"Using saved epoch {epoch}. Counting masks on CPU; no GPU allocated.", flush=True)
    coverage(data_directory("/workspace", annotation), height, checkpoint / "diagnostics/validation", annotation=annotation)
    volume.commit()


@app.function(image=image, gpu=config.MODAL["gpu"], cpu=4, memory=16384,
              timeout=3600, volumes={"/workspace": volume})
def predict_remote(run_name: str):
    from diagnose import diagnose
    from coral_data import checkpoint_dataset, data_directory
    volume.reload()
    checkpoint = Path("/workspace/checkpoints") / run_name
    annotation = checkpoint_dataset(checkpoint)
    result = diagnose(checkpoint, data_directory("/workspace", annotation), checkpoint / "diagnostics/validation")
    volume.commit()
    return {"epoch": result["checkpoint_epoch"], "dead": result["dead"]}


@app.function(image=image, cpu=.125, memory=256, timeout=7200)
def workflow(run_name: str):
    count_remote.remote(run_name)
    return predict_remote.remote(run_name)


@app.local_entrypoint()
def main(run_name: str = config.RUN_NAME):
    call = workflow.spawn(run_name)
    print(f"Diagnostics workflow: {call.object_id}")
    print(f"Dashboard: {app.get_dashboard_url()}")
