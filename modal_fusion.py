"""One bounded, isolated Modal inference job for three V2 ensemble experiments."""

from pathlib import Path
import io
import json
import os
import re
import time

import modal

# Reuse the existing dependency image; leave all original launchers and records alone.
os.environ.setdefault("PAPER_GPU", "L40S")
os.environ.setdefault("PAPER_CPU", "4")
os.environ.setdefault("PAPER_MEMORY_MB", "24576")
os.environ.setdefault("PAPER_PRECISION", "bf16")
ROOT = Path(__file__).resolve().parent
PUBLIC_REVISION = "545e9339f7b77267ffab412dfabec5624f38dce0"
PUBLIC = ROOT / ".cache/huggingface/hub/models--EPFL-ECEO--coralscapesv2-dinov3-vitl-lora-dpt-95_class/snapshots" / PUBLIC_REVISION
if modal.is_local():
    from modal_paper_suite import image, volume
    image = image.add_local_dir(ROOT / "fusion_eval", "/app/fusion_eval", ignore=["**/__pycache__/**"])
    # Stage the already verified public release, without downloading another model.
    for name in ("model.safetensors", "config.json", "preprocessor_config.json", "coralscapes_hub_model.py"):
        image = image.add_local_file((PUBLIC / name).resolve(), "/app/public_model/" + name)
else:
    image = modal.Image.debian_slim()
    volume = modal.Volume.from_name("coral-training")
app = modal.App("coralscapes-v2-three-fusion-experiments")


@app.function(image=image, gpu="L40S", cpu=(4, 4), memory=(24576, 24576),
              timeout=3600, retries=0, max_containers=1, scaledown_window=2,
              single_use_containers=True, volumes={"/workspace": volume})
def evaluate_remote(suite: str):
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from fusion_eval.runner import run_suite
    volume.reload()
    return run_suite('/workspace', '/app/public_model', suite, commit=volume.commit, max_seconds=3500)


def download_outputs(suite):
    names = [f"{suite}-{suffix}" for suffix in ("lora", "public", "e1-global", "e2-dead", "e3-hierarchical")]
    for name in names:
        for relative in (f"runs/{name}/metrics.csv", f"runs/{name}/prediction.png", f"checkpoints/local/{name}/head.pt"):
            target = ROOT / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + '.tmp')
            with temporary.open('wb') as stream:
                volume.read_file_into_fileobj('/' + relative, stream)
            temporary.replace(target)


@app.local_entrypoint()
def main(suite: str = "fusion-v2-20261122", fetch_only: bool = False):
    if os.environ.get('MODAL_PROFILE') != 'm12mm-4m2005':
        raise ValueError('Use the existing LoRA workspace: MODAL_PROFILE=m12mm-4m2005')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', suite):
        raise ValueError('Invalid suite name')
    record_path = ROOT / 'checkpoints/local' / f'{suite}-submission.json'
    record_path.parent.mkdir(parents=True, exist_ok=True)
    if fetch_only:
        download_outputs(suite)
        print('Downloaded the five compact result sets and selected inference recipes.')
        return
    if record_path.exists():
        raise FileExistsError('A submission record exists; inspect or fetch that job instead of submitting another')
    # This read-only preflight happens before any GPU function is submitted.
    buffer = io.BytesIO()
    volume.read_file_into_fileobj('/checkpoints/paper-v2-fresh40-s42-20261101/lora/progress.json', buffer)
    if json.loads(buffer.getvalue()).get('epoch_completed') != 40:
        raise ValueError('The completed 40-epoch LoRA run is unavailable')
    if not any(Path(x.path).name == 'best_ema_calibrated.pt' for x in volume.listdir('/checkpoints/paper-v2-fresh40-s42-20261101/lora')):
        raise FileNotFoundError('The selected calibrated LoRA EMA checkpoint is missing')
    call = evaluate_remote.spawn(suite)
    record = dict(suite=suite, call_id=call.object_id, profile='m12mm-4m2005',
                  dashboard=app.get_dashboard_url(), submitted_at=time.time(),
                  kind='frozen_model_inference_only', max_gpu_seconds=3600)
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    print(f"Submitted one L40S inference job: {call.object_id}", flush=True)
    print(f"Dashboard: {record['dashboard']}", flush=True)
    result = call.get()
    download_outputs(suite)
    record['completed_at'] = time.time()
    record['results'] = result['results']
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    print('Completed and downloaded all three experiments plus both individual-model baselines.', flush=True)
    for row in result['results']:
        m = row['test']
        print(f"{row['name']}: native={m['native_miou']:.4f} coral={m['coral_miou']:.4f} Dead={m['dead_iou']:.4f}", flush=True)
