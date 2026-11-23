"""Publishable moving-footage model demonstration; no training or test tuning."""
import json
import os
from pathlib import Path
import re
import modal

ROOT = Path(__file__).resolve().parent
volume = modal.Volume.from_name('coral-training')
app = modal.App('coralscapes-moving-reef-demo')
if modal.is_local():
    import imageio_ffmpeg
    from modal_spatial_fusion import image
    from modal_fusion import PUBLIC
    image = (image.add_local_file(ROOT / 'reef_video_demo.py', '/app/reef_video_demo.py')
        .add_local_file(ROOT / 'deliverables/coralscapes-explained-20261123/source_clip.mp4', '/app/reef_source.mp4')
        .add_local_file(imageio_ffmpeg.get_ffmpeg_exe(), '/app/ffmpeg_binary'))
    for name in ('model.safetensors', 'config.json', 'preprocessor_config.json', 'coralscapes_hub_model.py'):
        image = image.add_local_file((PUBLIC / name).resolve(), '/app/public_model/' + name)
else:
    image = modal.Image.debian_slim()


@app.function(image=image, gpu='L40S', cpu=4, memory=24576, timeout=1200, retries=0,
              single_use_containers=True, scaledown_window=2, volumes={'/workspace': volume})
def demonstrate(name):
    from reef_video_demo import run
    volume.reload()
    return run('/workspace', '/app/public_model', '/app/reef_source.mp4',
               '/workspace/deliverables/' + name, '/app/ffmpeg_binary', commit=volume.commit)


@app.local_entrypoint()
def main(name: str = 'coralscapes-explained-20261123'):
    if os.environ.get('MODAL_PROFILE') != 'm12mm-4m2005' or not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('Use the LoRA Modal profile and a valid output name')
    out = ROOT / 'deliverables' / name
    out.mkdir(exist_ok=True, parents=True)
    record = out / 'video_submission.json'
    if record.exists():
        raise FileExistsError('A submission record exists; do not submit a duplicate video job')
    call = demonstrate.spawn(name)
    record.write_text(json.dumps(dict(call_id=call.object_id, dashboard=app.get_dashboard_url()), indent=2))
    print(f'Moving-video inference: {call.object_id}', flush=True)
    files = call.get()
    for filename in files:
        with (out / filename).open('wb') as stream:
            volume.read_file_into_fileobj('/deliverables/' + name + '/' + filename, stream)
    print(f'Downloaded moving-video demonstration to {out}', flush=True)
