"""Fresh model inference on moving reef footage, with synchronized overlays."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

SOURCE_URL = 'https://josauder.github.io/coralscapes/static/videos/trimmed_trimmed_smaller.mp4'
MODEL_NAMES = ('Public model', 'Our LoRA', 'Our Rein++', '50/50 ensemble', 'Calibrated ensemble')


def font(size):
    return ImageFont.load_default(size=size)


def display_metadata(root):
    root = Path(root)
    meta = json.loads((root / 'coralscapes_metadata/v2.json').read_text())
    labels = json.loads((root / 'coralscapes_metadata/id2label.json').read_text())
    colors = json.loads((root / 'coralscapes_metadata/label2color.json').read_text())
    lookup = np.zeros(96, dtype=np.uint8)
    palette = np.full((40, 3), 140, dtype=np.uint8)
    for native, legacy in meta['fine_to_legacy_id'].items():
        lookup[int(native)] = int(legacy)
    for key, label in labels.items():
        palette[int(key)] = colors.get(label, [140, 140, 140])
    return lookup, palette, labels


def pad_frame(image):
    x = np.asarray(image)
    bottom, right = (-image.height) % 16, (-image.width) % 16
    return Image.fromarray(np.pad(x, ((0, bottom), (0, right), (0, 0)), mode='edge'))


class VideoWriter:
    def __init__(self, executable, path, size, fps=15):
        self.path = Path(path)
        self.log = (Path('/tmp') / (self.path.stem + '-ffmpeg.log')).open('wb')
        self.process = subprocess.Popen([str(executable), '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{size[0]}x{size[1]}', '-r', str(fps),
            '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '21',
            '-threads', '1', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)],
            stdin=subprocess.PIPE, stderr=self.log)

    def write(self, frame):
        self.process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())

    def close(self):
        self.process.stdin.close()
        status = self.process.wait(timeout=90)
        self.log.close()
        if status:
            raise RuntimeError(f'Video encoding failed for {self.path.name}: {status}')


def overlay(image, native, lookup, palette):
    color = Image.fromarray(palette[lookup[native]])
    return Image.blend(image, color, .52)


def legend(draw, y, legend_ids, palette, labels, size=19):
    for i, label in enumerate(legend_ids):
        x = 24 + i * 376
        color = tuple(int(v) for v in palette[label])
        draw.rectangle((x, y, x + 20, y + 20), fill=color)
        text = labels.get(str(label), 'unknown')
        draw.text((x + 29, y - 2), text, font=font(size), fill='white')


def render(image, predictions, lookup, palette, labels, legend_ids, seconds):
    hero = Image.new('RGB', (1920, 720), '#101a28')
    d = ImageDraw.Draw(hero)
    d.text((20, 12), 'Original reef footage', font=font(25), fill='white')
    d.text((980, 12), 'Our 50/50 ensemble | segmentation overlay', font=font(25), fill='white')
    hero.paste(image, (0, 52))
    hero.paste(overlay(image, predictions[3], lookup, palette), (960, 52))
    legend(d, 612, legend_ids, palette, labels)
    d.text((24, 652), 'Footage: Sauder et al., CoralScapes / EPFL-ECEO | josauder.github.io/coralscapes/', font=font(18), fill='#bbcadd')
    d.text((24, 683), f'Frame-by-frame predictions | original playback speed | {seconds:04.1f} s | qualitative demonstration', font=font(18), fill='#bbcadd')

    comparison = Image.new('RGB', (1920, 1080), '#101a28')
    d = ImageDraw.Draw(comparison)
    d.text((24, 18), 'Moving reef footage | same frames, five frozen models', font=font(33), fill='white')
    d.text((24, 67), 'Colors show the same scene categories in every panel. Each displayed frame has fresh predictions.', font=font(22), fill='#bbcadd')
    panels = [('Original footage', image)] + [(name, overlay(image, pred, lookup, palette))
                                             for name, pred in zip(MODEL_NAMES, predictions)]
    for i, (name, frame) in enumerate(panels):
        x, y = 24 + (i % 3) * 632, 124 + (i // 3) * 394
        d.rounded_rectangle((x - 5, y - 5, x + 613, y + 380), radius=8, fill='#1c2b40')
        d.text((x + 4, y + 2), name, font=font(23), fill='white')
        comparison.paste(frame.resize((608, 342), Image.Resampling.LANCZOS), (x, y + 34))
    legend(d, 930, legend_ids, palette, labels, size=18)
    d.text((24, 980), f'{seconds:04.1f} s | Native predictions mapped to the legacy scene palette for display | no temporal smoothing', font=font(20), fill='#bbcadd')
    d.text((24, 1022), 'Source: CoralScapes project website, Sauder et al. / EPFL-ECEO | video has no frame-wise ground-truth scores', font=font(19), fill='#bbcadd')
    return hero, comparison


@torch.inference_mode()
def run(workspace, public_folder, source, destination, executable, commit=lambda: None):
    import cv2
    from fusion_eval.runner import construct_models, image_tensor, sha256
    from spatial_fusion.runtime import build_models, rein_batch
    from spatial_fusion.core import normalize, temperature
    workspace, destination = Path(workspace), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'video_manifest.json').exists():
        raise FileExistsError('This moving-video demonstration already exists')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    print('Preparing frozen public / LoRA / Rein++ models for moving-video inference', flush=True)
    models, provenance = build_models(workspace, '/app/selected_reinpp_ema.pt')
    release, public_provenance = construct_models(workspace, Path(public_folder))
    public = release.pop('public')
    del release
    lookup, palette, labels = display_metadata('/app')
    shutil.copyfile(executable, '/tmp/reef-demo-ffmpeg')
    os.chmod('/tmp/reef-demo-ffmpeg', 0o755)
    cap = cv2.VideoCapture(str(source))
    frames, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps != 15 or frames != 300:
        raise ValueError('Expected the fixed 20-second, 15-fps clean source clip')
    encoded = Path('/tmp/reef-encoded')
    encoded.mkdir(exist_ok=True)
    hero = VideoWriter('/tmp/reef-demo-ffmpeg', encoded / 'reef_overlay.mp4', (1920, 720))
    comparison = VideoWriter('/tmp/reef-demo-ffmpeg', encoded / 'models_comparison.mp4', (1920, 1080))
    started = time.monotonic()
    for i in range(frames):
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f'Source decode stopped at frame {i}')
        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        padded = pad_frame(image)
        x = image_tensor(models['lora'], padded).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            first = models['lora'](x)[0].float().softmax(0)
            pub = public(x)[0].float().argmax(0)
        batch = models['reinpp'].data_preprocessor(rein_batch(models['reinpp'], padded), training=False)
        second = normalize(models['reinpp'].inference(batch['inputs'], [s.metainfo for s in batch['data_samples']])[0].float())
        predicted = [pub, first.argmax(0), second.argmax(0), (first * .5 + second * .5).argmax(0),
                     (temperature(first, 1.5) * .5 + second * .5).argmax(0)]
        predicted = [p[:image.height, :image.width].byte().cpu().numpy() for p in predicted]
        if i == 0:
            c = np.bincount(lookup[predicted[3]].flatten(), minlength=40)
            legend_ids = [int(v) for v in np.argsort(c)[::-1] if v != 0][:5]
        a, b = render(image, predicted, lookup, palette, labels, legend_ids, i / fps)
        hero.write(a)
        comparison.write(b)
        if i == 150:
            a.save(destination / 'overlay_preview.png')
            b.save(destination / 'comparison_preview.png')
        if (i + 1) % 50 == 0:
            status = dict(completed=i + 1, total=frames, seconds=round(time.monotonic() - started, 1))
            (destination / 'progress.json').write_text(json.dumps(status))
            commit()
            print(f'Moving-video predictions: {i+1}/{frames}', flush=True)
        del first, second, pub, x, predicted, batch
    cap.release()
    hero.close()
    comparison.close()
    for filename in ('reef_overlay.mp4', 'models_comparison.mp4'):
        shutil.copyfile(encoded / filename, destination / filename)
    manifest = dict(source_url=SOURCE_URL, source_sha256=sha256(source),
        attribution='CoralScapes project website; Jonathan Sauder et al., EPFL-ECEO',
        source_preparation='First20 seconds; clean left half only; 30fps sampled to15fps at original playback speed',
        inference='Fresh prediction for every displayed frame; DPT full960x544 padded frame; Rein++ native512 sliding; clip padding after inference',
        seconds=20, fps=15, frames=300, temporal_smoothing=False, test_time_augmentation=False,
        quantitative_test=False, model_sources=provenance, public_source=public_provenance['public'],
        palette='Native95 argmax mapped to39 legacy scene categories for shared display',
        models=list(MODEL_NAMES), processing_seconds=time.monotonic() - started)
    (destination / 'video_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    commit()
    print('Finished moving reef overlays and synchronized model comparison', flush=True)
    return ['reef_overlay.mp4', 'models_comparison.mp4', 'overlay_preview.png', 'comparison_preview.png', 'video_manifest.json']
