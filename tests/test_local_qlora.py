"""CPU fixtures for matched initialization, checkpointing and local run setup."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from transformers import DINOv3ViTConfig, DINOv3ViTModel

import config
from dpt import CoralDINO
import local_qlora
from train import Corals


class LocalRecipeTests(unittest.TestCase):
    def test_config_uses_twenty_complete_epochs_and_printing_is_read_only(self):
        recipe = local_qlora.read_config(local_qlora.DEFAULT_CONFIG)
        self.assertEqual(recipe['train']['epochs'], 20)
        self.assertEqual(recipe['train']['max_updates'], 0)
        self.assertEqual(recipe['train']['early_stopping_patience'], 0)
        self.assertEqual(recipe['model']['quantization']['quant_type'], 'nf4')
        policy = local_qlora.train.dataset_config('V2')
        self.assertEqual(recipe['dataset']['annotation_version'], 'V2')
        self.assertEqual(recipe['dataset']['id'], policy['id'])
        self.assertEqual(recipe['dataset']['revision'], policy['revision'])
        terminal = io.StringIO()
        with patch('torch.cuda.is_available', side_effect=AssertionError('GPU accessed')), redirect_stdout(terminal):
            local_qlora.main(['--init', 'eva', '--print-config'])
        actual = json.loads(terminal.getvalue())
        self.assertEqual(actual['initialization'], 'eva')
        self.assertFalse(actual['automatic_test'])
        self.assertEqual(Path(actual['results']).name,
                         f"qlora-eva-v2-{recipe['train']['height']}x{recipe['train']['width']}-"
                         f"b{recipe['train']['batch_size']}a{recipe['train']['accumulation_steps']}-20e-s42")
        self.assertNotEqual(Path(actual['results']).name, 'qlora-eva-v1-20e-s42')

    def test_settings_are_scoped_and_shared_between_run_modes(self):
        recipe = local_qlora.read_config(local_qlora.DEFAULT_CONFIG)
        original = config.MODEL
        def check(*args, **kwargs):
            self.assertEqual(config.DATASET, recipe['dataset'])
            self.assertEqual(config.TRAIN, recipe['train'])
            self.assertEqual(config.LORA, recipe['lora'])
            self.assertEqual(kwargs['run_metadata']['config'], recipe)
            self.assertEqual(args[0][0], '--output')
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.is_bf16_supported', return_value=True), \
             patch('train.main', side_effect=check) as main:
            local_qlora.main(['--init', 'standard'])
            local_qlora.main(['--init', 'eva'])
        self.assertEqual(main.call_count, 2)
        self.assertIs(config.MODEL, original)

    def test_v2_rejects_a_revision_from_the_other_annotation_release(self):
        recipe = local_qlora.read_config(local_qlora.DEFAULT_CONFIG)
        recipe['dataset']['revision'] = 'acd72c757c59db055b800dc054a2291487e4566f'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'recipe.json'
            path.write_text(json.dumps(recipe))
            with self.assertRaisesRegex(ValueError, 'revision'):
                local_qlora.read_config(path)


class AdaptationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.backbone = DINOv3ViTModel(DINOv3ViTConfig(
            hidden_size=16, intermediate_size=32, num_hidden_layers=4,
            num_attention_heads=4, patch_size=16, num_register_tokens=4,
            output_hidden_states=True))
        self.settings = {**config.MODEL, 'feature_layers': [1, 2, 3, 4],
                         'dpt_neck_channels': [4, 8, 16, 32], 'dpt_fusion_channels': 8,
                         'encoder_checkpointing': True, 'decoder_checkpointing': True}
        self.lora = {'r': 2, 'lora_alpha': 4, 'lora_dropout': .05,
                     'target_modules': ['q_proj', 'v_proj'], 'init_lora_weights': True}

    def model(self):
        with patch('dpt.AutoModel.from_pretrained', side_effect=lambda *a, **k: deepcopy(self.backbone)):
            return CoralDINO(self.settings, self.lora, '/tmp')

    def test_nf4_loader_options_and_explicit_no_quantization_fallback(self):
        settings = {**self.settings, 'quantization':
                    local_qlora.read_config(local_qlora.DEFAULT_CONFIG)['model']['quantization']}
        backbone = deepcopy(self.backbone)
        with patch('dpt.torch.cuda.is_available', return_value=True), \
             patch('dpt.torch.cuda.current_device', return_value=0), \
             patch('dpt.AutoModel.from_pretrained', return_value=backbone) as load, \
             patch('dpt.prepare_model_for_kbit_training', return_value=backbone) as prepare:
            model = CoralDINO(settings, self.lora, '/tmp')
        options = load.call_args.kwargs
        self.assertTrue(options['quantization_config'].load_in_4bit)
        self.assertEqual(options['quantization_config'].bnb_4bit_quant_type, 'nf4')
        self.assertEqual(options['quantization_config'].bnb_4bit_compute_dtype, torch.bfloat16)
        self.assertTrue(options['quantization_config'].bnb_4bit_use_double_quant)
        self.assertEqual(options['device_map'], {'': 0})
        prepare.assert_called_once_with(backbone, use_gradient_checkpointing=False)
        recipe = local_qlora.read_config(local_qlora.DEFAULT_CONFIG)
        with self.assertRaisesRegex(RuntimeError, 'No 4-bit layers'):
            local_qlora.make_initializer(recipe, 'standard')(model, None, torch.device('cpu'))

    def test_checkpointing_retains_rope_eval_and_adapter_gradients(self):
        model = self.model()
        base = model.backbone.get_base_model()
        self.assertFalse(base.training)
        self.assertFalse(base.rope_embeddings.training)
        self.assertTrue(all(layer.training and layer.gradient_checkpointing for layer in base.layer))
        x = torch.randn(2, 3, 32, 64)
        target = torch.randint(4, (2, 32, 64))
        settings = {**self.settings, 'encoder_checkpointing': False, 'decoder_checkpointing': False}
        with patch('dpt.AutoModel.from_pretrained', side_effect=lambda *a, **k: deepcopy(self.backbone)):
            reference = CoralDINO(settings, self.lora, '/tmp')
        reference.load_trainable_state(model.trainable_state())
        torch.manual_seed(7)
        F.cross_entropy(model(x), target).backward()
        torch.manual_seed(7)
        F.cross_entropy(reference(x), target).backward()
        gradients = [parameter.grad for name, parameter in model.named_parameters() if 'lora_B' in name]
        self.assertTrue(all(value is not None and torch.isfinite(value).all() and value.abs().sum() > 0
                            for value in gradients))
        reference_buffers = dict(reference.decoder.named_buffers())
        for name, value in model.decoder.named_buffers():
            torch.testing.assert_close(value, reference_buffers[name], rtol=0, atol=0)
        reference_parameters = dict(reference.named_parameters())
        for name, value in model.named_parameters():
            if value.grad is not None:
                torch.testing.assert_close(value.grad, reference_parameters[name].grad)
        model.eval()
        self.assertTrue(all(not layer.training for layer in base.layer))

    def test_eva_preserves_initial_function_decoder_frozen_weights_and_rng(self):
        model = self.model()
        x = torch.randn(1, 3, 32, 64)
        model.eval()
        with torch.no_grad(): expected = model(x)
        model.train()
        frozen = {name: value.clone() for name, value in model.named_parameters() if not value.requires_grad}
        decoder_before = local_qlora.decoder_hash(model)
        settings = {**local_qlora.read_config(local_qlora.DEFAULT_CONFIG)['eva'],
                    'calibration_images': 4, 'max_patch_tokens_per_batch': 8, 'tau': .001, 'max_forwards': 32}
        with tempfile.TemporaryDirectory() as directory:
            rows = []
            rng = np.random.default_rng(10)
            for i in range(4):
                image, mask = Path(directory)/f'{i}.png', Path(directory)/f'{i}-mask.png'
                Image.fromarray(rng.integers(0, 256, (32, 64, 3), dtype=np.uint8)).save(image)
                Image.fromarray(np.full((32, 64), 3, dtype=np.uint8)).save(mask)
                rows.append((image, mask))
            dataset = Corals(rows, 32)
            torch.manual_seed(50)
            state = torch.get_rng_state().clone()
            report = local_qlora.initialize_eva(model, dataset, settings, torch.device('cpu'))
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            self.assertLessEqual(report['forwards'], 32)
            self.assertEqual(local_qlora.decoder_hash(model), decoder_before)
            self.assertTrue(model.training)
            for name, value in model.named_parameters():
                if name in frozen: self.assertTrue(torch.equal(value, frozen[name]), name)
            with torch.no_grad(): actual = model.eval()(x)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for module in model.backbone.modules():
                self.assertFalse(any(type(h).__module__ == 'peft.tuners.lora.eva' for h in module._forward_hooks.values()))
            # A bounded failure must clean up hooks and restore RNG/modes too.
            model.train()
            state = torch.get_rng_state().clone()
            with self.assertRaisesRegex(RuntimeError, 'calibration budget'):
                local_qlora.initialize_eva(model, dataset, {**settings, 'max_forwards': 1}, torch.device('cpu'))
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            self.assertTrue(model.training)
            for module in model.backbone.modules():
                self.assertFalse(any(type(h).__module__ == 'peft.tuners.lora.eva' for h in module._forward_hooks.values()))


if __name__ == '__main__':
    unittest.main()
