"""Generate five genuine held-out comparisons without changing benchmark results."""

import json
import os
from pathlib import Path
import re
import modal

ROOT = Path(__file__).resolve().parent
volume = modal.Volume.from_name('coral-training')
app = modal.App('coralscapes-v2-test-demonstration')
if modal.is_local():
    from modal_spatial_fusion import image
    from modal_fusion import PUBLIC
    image = image.add_local_file(ROOT / 'model_demo.py', '/app/model_demo.py')
    for name in ('model.safetensors', 'config.json', 'preprocessor_config.json', 'coralscapes_hub_model.py'):
        image = image.add_local_file((PUBLIC / name).resolve(), '/app/public_model/' + name)
else:
    image = modal.Image.debian_slim()


@app.function(image=image, gpu='L40S', cpu=4, memory=24576, timeout=900, retries=0,
              single_use_containers=True, scaledown_window=2, volumes={'/workspace': volume})
def demonstrate(name):
    from model_demo import infer
    volume.reload()
    result = infer('/workspace', '/app/public_model', '/app/selected_reinpp_ema.pt',
                   '/workspace/deliverables/' + name)
    volume.commit()
    return result


@app.local_entrypoint()
def main(name: str = 'coralscapes-current-models-20261123'):
    if os.environ.get('MODAL_PROFILE') != 'm12mm-4m2005' or not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('Use the LoRA Modal profile and a valid demonstration name')
    output = ROOT / 'deliverables' / name
    output.mkdir(parents=True, exist_ok=True)
    record = output / 'demo_submission.json'
    if record.exists():
        raise FileExistsError('This demonstration was already submitted; fetch its files instead')
    call = demonstrate.spawn(name)
    record.write_text(json.dumps(dict(call_id=call.object_id, dashboard=app.get_dashboard_url()), indent=2) + '\n')
    print(f'Fresh five-image test demonstration: {call.object_id}', flush=True)
    result = call.get()
    for filename in result['files']:
        with (output / filename).open('wb') as stream:
            volume.read_file_into_fileobj(f'/deliverables/{name}/{filename}', stream)
    print(f'Downloaded actual test comparisons to {output}', flush=True)
