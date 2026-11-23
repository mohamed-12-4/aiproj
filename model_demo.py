"""Fresh, deterministic held-out inference and an honest comparison video.

The video presents real test images; its timeline is presentation time, not
measured inference latency or a continuous underwater recording.
"""

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


def select_examples(dataset):
    sites = sorted({pair[0].parent.name for pair in dataset.pairs})
    return [indices[len(indices) // 2] for site in sites
            for indices in [[i for i, pair in enumerate(dataset.pairs) if pair[0].parent.name == site]]]


def font(size):
    return ImageFont.load_default(size=size)


def comparison_frame(image, target, predictions, groups, site, filename, metrics):
    colors = np.array([[28, 34, 41], [40, 190, 80], [255, 240, 130],
                       [220, 65, 60], [140, 140, 140]], dtype=np.uint8)
    canvas = Image.new('RGB', (1920, 1080), '#0b1422')
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 20), 'CoralScapesV2 | Actual test-image inference', font=font(34), fill='white')
    draw.text((24, 70), f'{site}  /  {filename}', font=font(22), fill='#b7c8dd')
    panels = [('Input image', image, None),
              ('Ground truth: health groups', Image.fromarray(colors[groups[target]]), None)]
    for name, native in predictions.items():
        mask = Image.fromarray(colors[groups[native]])
        overlay = Image.blend(image, mask, .48)
        panels.append((name, overlay, metrics[name]))
    for index, (title, panel, score) in enumerate(panels):
        x, y = 24 + (index % 3) * 632, 126 + (index // 3) * 400
        draw.rounded_rectangle((x - 5, y - 5, x + 613, y + 381), radius=10, fill='#18263a')
        draw.text((x + 5, y + 3), title, font=font(23), fill='white')
        if score:
            text = f'Image native mIoU {100*score["native_miou"]:.1f}%'
            if score['dead_iou'] is not None:
                text += f'  |  Dead IoU {100*score["dead_iou"]:.1f}%'
            draw.text((x + 5, y + 35), text, font=font(17), fill='#b7c8dd')
        canvas.paste(panel.resize((608, 304), Image.Resampling.LANCZOS), (x, y + 70))
    draw.text((24, 944), 'HEALTH:  green = Healthy     yellow = Bleached     red = Dead     dark = Background     gray = unknown / ignored GT',
              font=font(20), fill='#dce6f3')
    draw.text((24, 983), 'Native prediction first, then the fixed health mapping. All models frozen; no test-time augmentation.',
              font=font(20), fill='#b7c8dd')
    draw.text((24, 1022), 'Five deterministic test examples, one per site. Video playback speed is independent of inference speed.',
              font=font(19), fill='#93abc6')
    return canvas


@torch.inference_mode()
def infer(workspace, public_folder, rein_checkpoint, output):
    from fusion_eval.core import ConfusionCounter, health_lookup, summarize
    from fusion_eval.runner import construct_models, make_dataset, sha256, tile_output
    from spatial_fusion.runtime import build_models, outputs
    workspace, output = Path(workspace), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    print('Preparing the frozen public, LoRA and Rein++ models for five test examples', flush=True)
    models, sources = build_models(workspace, rein_checkpoint)
    released, released_sources = construct_models(workspace, Path(public_folder))
    public = released.pop('public')
    del released
    gc.collect()
    dataset = make_dataset(workspace, 'test')
    lookup = health_lookup()
    device_lookup = lookup.cuda()
    manifest = dict(kind='actual_test_image_comparison', continuous_video=False,
        selection='Middle filename-sorted image within each official test site; independent of model scores',
        augmentation=False, sources=sources, public_source=released_sources['public'],
        source_sha256=sha256(Path(__file__)), examples=[])
    for n, index in enumerate(select_examples(dataset)):
        image, label = dataset[index]
        first, second, fp, sp = outputs(models, image)
        ensemble = (first * .5 + second * .5).argmax(0)
        published = torch.empty_like(fp)
        for left in (0, 1024):
            probabilities, predicted, _ = tile_output(public, image.crop((left, 0, left + 1024, 1024)))
            published[:, left:left + 1024] = predicted
            del probabilities, predicted
        target = torch.as_tensor(np.asarray(label, dtype=np.int64).copy(), device='cuda')
        counter = ConfusionCounter(target)
        predictions = {'Public LoRA + DPT': published, 'Our LoRA EMA + DPT': fp,
                       'Our Rein++ EMA + Mask2Former': sp, '50/50 LoRA + Rein++ ensemble': ensemble}
        metrics = {name: {k: v for k, v in summarize(counter(pred), device_lookup).items()
                          if k in ('native_miou', 'coral_miou', 'dead_iou')}
                   for name, pred in predictions.items()}
        image_path, label_path = dataset.pairs[index]
        frame = comparison_frame(image, np.asarray(label),
            {name: pred.cpu().numpy() for name, pred in predictions.items()}, lookup.numpy(),
            image_path.parent.name, image_path.name, metrics)
        filename = f'test_comparison_{n + 1}.png'
        frame.save(output / filename)
        manifest['examples'].append(dict(site=image_path.parent.name, image=image_path.name,
            image_sha256=sha256(image_path), target_sha256=sha256(label_path), frame=filename, metrics=metrics))
        print(f'Test comparison {n+1}/5: {image_path.parent.name}', flush=True)
        del first, second, fp, sp, ensemble, published, target, counter, predictions
    (output / 'demo_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return dict(files=[r['frame'] for r in manifest['examples']] + ['demo_manifest.json'])


def encode(folder):
    import imageio_ffmpeg
    folder = Path(folder)
    manifest = json.loads((folder / 'demo_manifest.json').read_text())
    fps = 12
    target = folder / 'model_comparison.mp4'
    writer = imageio_ffmpeg.write_frames(str(target), (1920, 1080), fps=fps,
        codec='libx264', pix_fmt_out='yuv420p', quality=7,
        output_params=['-movflags', '+faststart'], macro_block_size=2)
    writer.send(None)
    def hold(frame, seconds):
        for _ in range(int(seconds * fps)):
            writer.send(np.asarray(frame, dtype=np.uint8))
    def card(title, lines):
        frame = Image.new('RGB', (1920, 1080), '#0b1422')
        d = ImageDraw.Draw(frame)
        d.text((90, 150), title, font=font(48), fill='white')
        for i, line in enumerate(lines):
            d.text((90, 280 + i * 80), line, font=font(30), fill='#c7d7eb')
        return frame
    hold(card('CoralScapesV2 | Model comparison', [
        'Fresh inference on five official test images, one per test site.',
        'Public model / Our LoRA / Our Rein++ / Raw 50:50 ensemble',
        'Health overlays: Healthy / Bleached / Dead / Background',
        'Animated test-image demonstration; presentation time is not inference latency.']), 4)
    for example in manifest['examples']:
        with Image.open(folder / example['frame']) as frame:
            hold(frame.convert('RGB'), 6)
    hold(card('Full test set | 473 images', [
        '                                     Native mIoU     Coral mIoU     Dead IoU',
        'Public model                    40.19              68.69           41.24',
        'Our LoRA EMA                 42.40              69.09           43.07',
        'Our Rein++ EMA              40.80              66.05           41.96',
        'Raw 50:50 ensemble        44.86              70.64           45.10',
        'Calibrated 50:50               44.97              70.01           44.97']), 6)
    writer.close()
    print(f'Video: {target} (40 seconds, 1920x1080, H.264)', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--encode', type=Path, required=True)
    encode(parser.parse_args().encode)
