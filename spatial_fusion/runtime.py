"""Preserve each selected model's native inference in one frozen runtime."""

import gc
import logging
from pathlib import Path
from types import MethodType
import warnings

import numpy as np
import torch

from fusion_eval.runner import LORA_REMOTE, image_tensor, sha256
from .core import normalize


def build_models(workspace, rein_checkpoint):
    from paper_suite.adapters import source_model_class
    from paper_suite.recipe import ENCODER_ID, ENCODER_REVISION, DATA_REVISION
    from paper_suite.runner import model_kwargs
    workspace, rein_checkpoint = Path(workspace), Path(rein_checkpoint)
    encoder = workspace / '.cache/huggingface/hub' / ('models--' + ENCODER_ID.replace('/', '--')) / 'snapshots' / ENCODER_REVISION
    checkpoint = torch.load(workspace / LORA_REMOTE, map_location='cpu', weights_only=True, mmap=True)
    if checkpoint.get('epoch') != 39 or checkpoint.get('checkpoint_kind') != 'effective_ema_inference_bn_calibrated':
        raise ValueError('Expected the epoch-40 calibrated effective LoRA EMA')
    cfg = checkpoint['config']
    if cfg['data']['train_split'] != 'train' or cfg['data']['val_split'] != 'val':
        raise ValueError('LoRA was not trained with the expected held-out validation split')
    kwargs = model_kwargs(cfg)
    kwargs['encoder_id'] = str(encoder)
    lora = source_model_class()(**kwargs)
    lora.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    lora.eval().requires_grad_(False).cuda()

    # This already-validated helper uses MMCV's own PyTorch reference deformable
    # attention, keeping the old native architecture compatible with Torch 2.8.
    import post_training_compare.runtime as native
    native.BASE = encoder
    native.configure()
    from reinpp_training.model import register, _dense_tokens, as_float
    from reinpp_training.config import model_config
    from mmengine.config import ConfigDict
    from mmseg.registry import MODELS
    warnings.filterwarnings('ignore', message='.*MultiScaleDeformableAttention.*')
    register()
    for name in ('mmengine', 'mmseg', 'mmdet'):
        logging.getLogger(name).setLevel(logging.ERROR)
    rein = MODELS.build(ConfigDict(model_config()))
    native.hf_to_native(rein.backbone)
    checkpoint = torch.load(rein_checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if checkpoint.get('epoch') != 39 or checkpoint.get('kind') != 'reinpp_selected_ema':
        raise ValueError('Expected the original selected epoch-40 Rein++ effective-token EMA')
    if checkpoint['config']['dataset_revision'] != DATA_REVISION or checkpoint['config']['model'] != model_config():
        raise ValueError('Rein++ checkpoint architecture or dataset revision differs')
    state = dict(checkpoint['model'])
    tokens = state.pop('backbone.reins.effective_ema_tokens')
    loaded = rein.load_state_dict(state, strict=False)
    allowed = {'backbone.reins.learnable_tokens_a', 'backbone.reins.learnable_tokens_b'}
    invalid = [k for k in loaded.missing_keys if (not k.startswith('backbone.') or k.startswith('backbone.reins.')) and k not in allowed]
    if invalid or loaded.unexpected_keys:
        raise ValueError(f'Rein++ selected inference state mismatch: {invalid}, {loaded.unexpected_keys}')
    rein.backbone.reins.register_buffer('effective_ema_tokens', tokens)
    rein.backbone.reins.get_tokens = MethodType(_dense_tokens, rein.backbone.reins)
    rein.eval().requires_grad_(False).cuda()
    original = rein.backbone.forward

    def backbone(*args, **kwargs):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            values = original(*args, **kwargs)
        return as_float(values)

    rein.backbone.forward = backbone
    del checkpoint, state, tokens
    gc.collect()
    provenance = dict(lora=dict(path=LORA_REMOTE, sha256=sha256(workspace / LORA_REMOTE),
                               profile='m12mm-4m2005', epoch_completed=40),
        reinpp=dict(path='/checkpoints/reinpp-v2-mhgelu-m2f-s42-40e-20261101/best_ema.pt',
                    profile='mohamed-elashmony05', sha256=sha256(rein_checkpoint), epoch_completed=40),
        encoder_revision=ENCODER_REVISION,
        inference=dict(lora='two native 1024x1024 non-overlapping tiles; BF16 forward',
                       reinpp='native MMSeg sliding 512x512 stride341; BF16 backbone, FP32 Mask2Former; MMCV reference deformable attention'),
        split_policy='Both training drivers use train only, official val for checkpoint selection; no gradient updates here')
    return dict(lora=lora, reinpp=rein), provenance


def rein_batch(model, image):
    from mmseg.structures import SegDataSample
    rgb = np.asarray(image, dtype=np.uint8)
    inputs = torch.from_numpy(rgb[..., ::-1].copy()).permute(2, 0, 1)
    shape = (image.height, image.width)
    sample = SegDataSample(metainfo=dict(ori_shape=shape, img_shape=shape,
                                        scale_factor=(1., 1.), flip=False))
    return dict(inputs=[inputs], data_samples=[sample])


@torch.inference_mode()
def outputs(models, image):
    if image.size != (2048, 1024):
        raise ValueError('Expected a native 2048x1024 image')
    first = torch.empty(96, 1024, 2048, device='cuda', dtype=torch.float32)
    first_prediction = torch.empty(1024, 2048, device='cuda', dtype=torch.long)
    for left in (0, 1024):
        x = image_tensor(models['lora'], image.crop((left, 0, left + 1024, 1024))).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = models['lora'](x)
        first[:, :, left:left + 1024] = logits[0].float().softmax(0)
        first_prediction[:, left:left + 1024] = logits[0].float().argmax(0)
        del x, logits
    rein = models['reinpp']
    batch = rein.data_preprocessor(rein_batch(rein, image), training=False)
    metadata = [x.metainfo for x in batch['data_samples']]
    # The official head upsamples query mask logits BEFORE sigmoid; using its
    # own inference avoids changing that order when forming semantic scores.
    scores = rein.inference(batch['inputs'], metadata)[0].float()
    if scores.shape != first.shape or not bool(torch.isfinite(scores).all()):
        raise ValueError('Nonfinite or spatially misaligned native Rein++ scores')
    second_prediction = scores.argmax(0)
    second = normalize(scores)
    return first, second, first_prediction, second_prediction


@torch.inference_mode()
def check_native_protocol(models, image, actual):
    expected = models['reinpp'].test_step(rein_batch(models['reinpp'], image))[0].pred_sem_seg.data[0]
    differences = int((actual != expected).sum())
    if differences:
        raise RuntimeError(f'The Rein++ probability adapter changed {differences} native predictions')
    return dict(reinpp_native_prediction_differences=differences,
                geometry=[image.height, image.width], augmentation=False)
