"""Tests of matched initialization, adapter gradients and effective-weight EMA."""

import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from copy import deepcopy
from inspect import unwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers.models.dinov3_vit.configuration_dinov3_vit import DINOv3ViTConfig
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTModel

from paper_suite.adapters import build_model, set_epoch_phase, source_model_class
from paper_suite.ema import EffectiveEMA
from paper_suite.execution import author_torch_proxy, checkpoint_decoder_module
from paper_suite.recipe import make_config

source_model_class()
from dinov3_dpt_seg import model as author_model

torch.set_num_threads(2)


class FakeLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_layer = nn.Linear(1, 1, bias=False)
        self.base_layer.weight.requires_grad_(False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(1, 1, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(1, 1, bias=False)})
        self.scaling, self.merged = {"default": 1.0}, False
        self.register_buffer("counter", torch.tensor(1))

    def forward(self, x):
        return self.base_layer(x) + self.lora_B["default"](self.lora_A["default"](x))


class EMATests(unittest.TestCase):
    def setUp(self):
        self.model = nn.Sequential(FakeLoRA())
        with torch.no_grad():
            self.model[0].base_layer.weight.fill_(4)
            self.model[0].lora_A["default"].weight.fill_(2)
            self.model[0].lora_B["default"].weight.fill_(3)
        self.ema = EffectiveEMA(self.model, decay=0.5)

    def change_and_update(self):
        with torch.no_grad():
            self.model[0].lora_A["default"].weight.fill_(4)
            self.model[0].lora_B["default"].weight.fill_(5)
            self.model[0].counter.fill_(7)
        self.ema.update(self.model)

    def test_ema_averages_effective_weights_not_separate_factors(self):
        self.change_and_update()
        # mean(4+2*3, 4+4*5)=17; 4+mean(2,4)*mean(3,5)=16 is wrong.
        with self.ema.apply(self.model):
            self.assertEqual(float(self.model(torch.ones(1, 1))), 17)
            self.assertEqual(float(self.model[0].lora_B["default"].weight), 0)
            self.assertEqual(int(self.model[0].counter), 7)
        self.assertEqual(float(self.model(torch.ones(1, 1))), 24)

    def test_apply_restores_raw_state_even_after_exception(self):
        self.change_and_update()
        before = deepcopy(self.model.state_dict())
        with self.assertRaises(RuntimeError):
            with self.ema.apply(self.model):
                raise RuntimeError("evaluation failed")
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_export_loads_without_double_applying_adapter(self):
        self.change_and_update()
        clone = nn.Sequential(FakeLoRA())
        clone.load_state_dict(self.ema.inference_state(self.model))
        self.assertEqual(float(clone(torch.ones(1, 1))), 17)

    def test_resume_preserves_ema_trajectory(self):
        self.change_and_update()
        clone = EffectiveEMA(self.model, decay=0.1)
        clone.load_state_dict(deepcopy(self.ema.state_dict()))
        self.ema.update(self.model)
        clone.update(self.model)
        self.assertEqual(clone.num_updates, self.ema.num_updates)
        for name in self.ema.names:
            torch.testing.assert_close(clone.shadow[name], self.ema.shadow[name], rtol=0, atol=0)

    def test_ema_hook_does_not_update_when_amp_skips_an_overflow(self):
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=1e-4)
        optimizer.register_step_post_hook(lambda *_: self.ema.update(self.model))
        scaler = torch.amp.GradScaler("cpu", init_scale=1.0)
        loss = self.model(torch.ones(1,1)).square().mean()
        scaler.scale(loss).backward()
        self.model[0].lora_B["default"].weight.grad.fill_(float("inf"))
        scaler.step(optimizer)
        scaler.update()
        self.assertEqual(self.ema.num_updates,0)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(self.model(torch.ones(1,1)).square().mean()).backward()
        scaler.step(optimizer)
        scaler.update()
        self.assertEqual(self.ema.num_updates,1)


class MetricTests(unittest.TestCase):
    def test_unknown_predictions_and_background_false_positives_are_counted(self):
        from paper_suite.runner import mapped_metrics
        c = torch.zeros((96,96),dtype=torch.long)
        c[4,4],c[4,0],c[5,4] = 10,2,3
        dead = mapped_metrics(c)["health_4"]["dead"]
        self.assertAlmostEqual(dead["precision"],10/13)
        self.assertAlmostEqual(dead["recall"],10/12)
        self.assertAlmostEqual(dead["iou"],10/15)


class ExecutionTests(unittest.TestCase):
    def test_decoder_recompute_preserves_gradients_rng_and_batchnorm_buffers(self):
        torch.manual_seed(5)
        reference = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.Dropout(.2), nn.Conv2d(4, 2, 1))
        recomputed = deepcopy(reference)
        checkpoint_decoder_module(recomputed)
        x = torch.randn(2, 3, 8, 8)
        torch.manual_seed(19)
        expected = reference(x)
        expected.square().mean().backward()
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(19)
        actual = recomputed(x)
        actual.square().mean().backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        for (name, param), (_, other) in zip(reference.named_parameters(), recomputed.named_parameters()):
            torch.testing.assert_close(param.grad, other.grad, rtol=0, atol=0, msg=name)
        for name, buffer in reference.named_buffers():
            torch.testing.assert_close(buffer, dict(recomputed.named_buffers())[name], rtol=0, atol=0)

    def test_bf16_proxy_is_local_and_disables_loss_scaling(self):
        original_autocast, original_scaler = torch.autocast, torch.cuda.amp.GradScaler
        with patch.object(torch, "autocast") as autocast:
            author_torch_proxy("bf16").autocast("cuda", dtype=torch.float16)
            self.assertEqual(autocast.call_args.kwargs["dtype"], torch.bfloat16)
        proxy = author_torch_proxy("bf16")
        self.assertFalse(proxy.cuda.amp.GradScaler().is_enabled())
        self.assertIs(torch.autocast, original_autocast)
        self.assertIs(torch.cuda.amp.GradScaler, original_scaler)


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(13)
        cls.pretrained = DINOv3ViTModel(DINOv3ViTConfig(
            hidden_size=32, num_hidden_layers=4, num_attention_heads=4, intermediate_size=64,
            image_size=64, patch_size=16, num_register_tokens=4))
        cls.processor = SimpleNamespace(image_mean=[.485,.456,.406], image_std=[.229,.224,.225])

    def build(self, method):
        cfg = make_config(method)
        cfg["experiment"]["gradient_checkpointing"] = False
        kwargs = dict(encoder_id="offline-tiny-test", num_classes=96, freeze_encoder=True,
                      lora_enabled=True, lora_r=16, lora_alpha=32, lora_dropout=.05,
                      lora_target_modules="all-linear", neck_hidden_sizes=(8,16,16,32),
                      fusion_hidden_size=16, image_processor=self.processor)
        torch.manual_seed(42)
        with patch.object(author_model.AutoModel, "from_pretrained", side_effect=lambda *a,**kw: deepcopy(self.pretrained)):
            model = build_model(kwargs, cfg["experiment"])
        return model, cfg

    def test_dpt_initialization_is_identical_for_all_variants(self):
        reference, _ = self.build("lora")
        for method in ("decoder_ft", "soma", "rein_g"):
            model, _ = self.build(method)
            for name, value in reference.state_dict().items():
                if name.startswith(("neck.", "seg_head.")):
                    torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)

    def test_soma_initial_effective_encoder_equals_pretrained(self):
        model, _ = self.build("soma")
        model.encoder.eval()
        self.pretrained.eval()
        x = torch.randn(2,3,64,96)
        with torch.no_grad():
            actual = model.encoder(x).last_hidden_state
            expected = self.pretrained(x).last_hidden_state
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        self.assertEqual(model.adapter_locations, 24)

    def test_all_adapters_receive_gradients_and_keep_base_frozen(self):
        for method, token in (("lora", "lora_B"),("soma", "lora_B"),("rein_g", "reins")):
            model, _ = self.build(method)
            model.train()
            logits = model(torch.randn(2,3,64,96))
            self.assertEqual(logits.shape, (2,96,64,96))
            logits.square().mean().backward()
            grads = [p.grad for n,p in model.encoder.named_parameters() if token in n and p.grad is not None]
            self.assertTrue(grads, method)
            self.assertTrue(any(torch.count_nonzero(g)>0 for g in grads), method)
            for n,p in model.encoder.named_parameters():
                if token == "reins" and not n.startswith("reins."):
                    self.assertIsNone(p.grad, n)
                elif token != "reins" and "lora_" not in n:
                    self.assertIsNone(p.grad, n)

    def test_decoder_warmup_then_unfreeze_is_tracked_by_ema(self):
        model, cfg = self.build("decoder_ft")
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        ema = EffectiveEMA(model)
        self.assertTrue(any(n.startswith("encoder.") for n in ema.names))
        self.assertEqual(set_epoch_phase(model,cfg["experiment"],0), "decoder_warmup")
        self.assertFalse(any(p.requires_grad for p in model.encoder.parameters()))
        self.assertEqual(set_epoch_phase(model,cfg["experiment"],15), "full_finetuning")
        self.assertTrue(all(p.requires_grad for p in model.encoder.parameters()))
        optimizer_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
        self.assertTrue(all(id(p) in optimizer_ids for p in model.encoder.parameters()))

    def test_rein_does_not_modify_cls_or_register_tokens(self):
        model, _ = self.build("rein_g")
        checked = []
        def check(layer, args, kwargs, output):
            original = unwrap(layer.forward)(*args, **kwargs)
            torch.testing.assert_close(output[:,:5],original[:,:5],rtol=0,atol=0)
            self.assertFalse(torch.equal(output[:,5:],original[:,5:]))
            checked.append(True)
        handle = model.encoder.layer[0].register_forward_hook(check, with_kwargs=True)
        model.encoder(torch.randn(2,3,64,96), output_hidden_states=True)
        handle.remove()
        self.assertTrue(checked)

    def test_rein_dpt_readouts_and_every_layer_gradient_with_checkpointing(self):
        for checkpointing in (False, True):
            model, _ = self.build("rein_g")
            model.train()
            if checkpointing:
                model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                checkpoint_decoder_module(model.neck)
                checkpoint_decoder_module(model.seg_head)
            captured = {}
            handles = []
            for index, layer in enumerate(model.encoder.layer):
                def capture(module, args, output, i=index):
                    captured[i] = output.detach().clone()
                handles.append(layer.register_forward_hook(capture))
            x = torch.randn(2,3,64,96)
            states = model.encoder(x, output_hidden_states=True).hidden_states
            for index, feature in zip(model.backbone_layer_indices, model._select_hidden_states(states)):
                expected = torch.cat((captured[index][:,:1], captured[index][:,5:]), dim=1)
                torch.testing.assert_close(feature, expected, rtol=0, atol=0)
            for handle in handles:
                handle.remove()
            model(x).square().mean().backward()
            norms = model.encoder.reins.learnable_tokens.grad.flatten(1).norm(dim=1)
            self.assertTrue(torch.isfinite(norms).all())
            self.assertTrue((norms > 0).all(), (checkpointing, norms))


if __name__ == "__main__":
    unittest.main()
