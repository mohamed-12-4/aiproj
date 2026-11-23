"""Run frozen LoRA/Rein++ mixtures, selective routing and reliability analysis."""

import io
import json
import os
from pathlib import Path
import re
import time

import modal

ROOT = Path(__file__).resolve().parent
REIN = ROOT / 'checkpoints/external_compare/reinpp40_best_ema.pt'
volume = modal.Volume.from_name('coral-training')
app = modal.App('coralscapes-v2-spatial-fusion')
if modal.is_local():
    # The first install exactly matches the cached DPT runtime; the extra
    # pure-Python MMCV layer avoids a second incompatible Torch installation.
    image = (modal.Image.debian_slim(python_version='3.12')
        .pip_install('torch==2.8.0', 'torchvision==0.23.0', 'transformers==4.57.6', 'peft==0.18.1',
                     'pyarrow==21.0.0', 'huggingface_hub==0.36.0', 'torchmetrics==1.8.2',
                     'PyYAML==6.0.3', 'Pillow>=10', 'numpy>=1.26', 'tqdm>=4.66', 'matplotlib>=3.8')
        .pip_install('numpy==1.26.4', 'setuptools<81', 'mmcv-lite==2.1.0', 'mmengine==0.10.7',
                     'mmsegmentation==1.2.2', 'mmdet==3.3.0', 'timm==0.9.16', 'einops==0.8.0',
                     'ftfy==6.3.1', 'regex==2024.11.6', 'cityscapesscripts==2.3.0',
                     'opencv-python==4.10.0.84', 'scipy==1.13.1')
        .apt_install('libgl1', 'libglib2.0-0', 'libgomp1')
        .env({'PAPER_ROOT': '/workspace', 'HF_HOME': '/workspace/.cache/huggingface',
              'HF_HUB_DISABLE_TELEMETRY': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
              'USE_TF': '0', 'USE_FLAX': '0', 'PYTHONPATH': '/app', 'PYTHONUNBUFFERED': '1',
              'MPLCONFIGDIR': '/tmp/matplotlib', 'OMP_NUM_THREADS': '4',
              'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}))
    for package in ('spatial_fusion', 'fusion_eval', 'paper_suite', 'reinpp_training',
                    'post_training_compare', 'coralscapes_metadata'):
        image = image.add_local_dir(ROOT / package, '/app/' + package, ignore=['**/__pycache__/**'])
    image = (image.add_local_dir(ROOT / 'third_party/coralscapes_v2_author',
                    '/app/third_party/coralscapes_v2_author', ignore=['.git/**', '**/__pycache__/**'])
        .add_local_file(ROOT / 'third_party/soma_author/rein/peft/sora.py', '/app/third_party/soma_author/rein/peft/sora.py')
        .add_local_file(ROOT / 'third_party/rein_author/simple_reins/reins.py', '/app/third_party/rein_author/simple_reins/reins.py')
        .add_local_dir(ROOT / 'third_party/rein_plus_author/projects/rein++/reinpp', '/app/reinpp', ignore=['**/__pycache__/**'])
        .add_local_dir(ROOT / 'third_party/rein_plus_author/projects/rein++/configs', '/app/author_configs')
        .add_local_file(ROOT / 'third_party/rein_plus_author/configs/_base_/models/rein_dinov2_mask2former.py', '/app/author_train_head.py')
        .add_local_file(REIN, '/app/selected_reinpp_ema.pt'))
else:
    image = modal.Image.debian_slim()


@app.function(image=image, gpu='L40S', cpu=(4, 4), memory=(24576, 24576),
              timeout=5400, retries=0, max_containers=1, scaledown_window=2,
              single_use_containers=True, volumes={'/workspace': volume})
def evaluate_remote(suite: str):
    from spatial_fusion.runner import run_suite
    volume.reload()
    return run_suite('/workspace', '/app/selected_reinpp_ema.pt', suite,
                     commit=volume.commit, max_seconds=5300)


def download(suite):
    for suffix in ('lora', 'reinpp', 'e1-raw', 'e1-calibrated', 'e2-selective'):
        name = f'{suite}-{suffix}'
        for relative in (f'runs/{name}/metrics.csv', f'runs/{name}/prediction.png',
                         f'checkpoints/local/{name}/head.pt'):
            target = ROOT / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + '.tmp')
            with temporary.open('wb') as stream:
                volume.read_file_into_fileobj('/' + relative, stream)
            temporary.replace(target)
            if target.name == 'head.pt':
                from spatial_fusion.artifacts import make_portable
                make_portable(target)


@app.local_entrypoint()
def main(suite: str = 'spatial-v2-20261122', fetch_only: bool = False):
    if os.environ.get('MODAL_PROFILE') != 'm12mm-4m2005':
        raise ValueError('Use MODAL_PROFILE=m12mm-4m2005, where the LoRA and native V2 data reside')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', suite):
        raise ValueError('Invalid suite name')
    record_path = ROOT / 'checkpoints/local' / f'{suite}-submission.json'
    record_path.parent.mkdir(parents=True, exist_ok=True)
    if fetch_only:
        download(suite)
        print('Downloaded the five compact results and inference recipes.')
        return
    if record_path.exists():
        raise FileExistsError('This submission record already exists; fetch its results instead')
    buffer = io.BytesIO()
    volume.read_file_into_fileobj('/checkpoints/paper-v2-fresh40-s42-20261101/lora/progress.json', buffer)
    if json.loads(buffer.getvalue()).get('epoch_completed') != 40:
        raise ValueError('The completed LoRA checkpoint is unavailable')
    if not REIN.is_file() or REIN.stat().st_size != 102609220:
        raise ValueError('The existing selected Rein++ EMA transfer is missing or incomplete')
    call = evaluate_remote.spawn(suite)
    record = dict(suite=suite, call_id=call.object_id, profile='m12mm-4m2005',
                  dashboard=app.get_dashboard_url(), submitted_at=time.time(),
                  kind='frozen_models_calibration_fusion_and_reliability', max_gpu_seconds=5400,
                  augmentation=False)
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    print(f'Submitted frozen-model fusion: {call.object_id}', flush=True)
    print(f'Dashboard: {record["dashboard"]}', flush=True)
    result = call.get()
    download(suite)
    record.update(completed_at=time.time(), results=result['results'])
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    print('Completed both fusion experiments and uncertainty/Dead-cover analysis; downloaded all outputs.', flush=True)

