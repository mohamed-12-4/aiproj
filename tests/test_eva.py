"""EVA initialization invariants and exact matching to the saved LoRA control."""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

from copy import deepcopy
from functools import partial
import random
import unittest

import numpy as np
import torch
from peft import get_eva_state_dict

from paper_suite.eva import apply_eva_state, assert_lora_match, frozen_fingerprint, patch_inputs, preserve_rng
from paper_suite.ema import EffectiveEMA
from paper_suite.screen import check_resume, screen_config
from tests.test_paper_suite import AdapterTests


class EvaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        AdapterTests.setUpClass()

    def test_shared_recipe_and_resume_guard(self):
        eva = screen_config("eva", "test", "eva-id")
        lora = screen_config("lora", "test", "lora-id")
        self.assertTrue(assert_lora_match(eva, lora))
        changed = deepcopy(eva)
        changed["train"]["lr"] *= 2
        with self.assertRaisesRegex(ValueError, "train"):
            assert_lora_match(changed, lora)
        checkpoint = {"epoch": 0, "checkpoint_kind": "raw_training_resume", "experiment": deepcopy(eva["experiment"])}
        self.assertEqual(check_resume(checkpoint, eva), 1)
        checkpoint["experiment"]["eva"]["rho"] = 2
        with self.assertRaisesRegex(ValueError, "EVA"):
            check_resume(checkpoint, eva)

    def test_patch_sampling_excludes_prefix_and_preserves_shared_inputs(self):
        x = torch.arange(2*20*4).reshape(2,20,4)
        actual = patch_inputs((x,), None, "linear", prefix=5, maximum=7)
        expected = x[:,5:].reshape(-1,4)
        expected = expected[torch.linspace(0,len(expected)-1,7).long()]
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual, patch_inputs((x.clone(),), None, "other", prefix=5, maximum=7))

    def test_calibration_rng_is_isolated_even_on_failure(self):
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        expected = (torch.rand(4), random.random(), np.random.rand())
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        with self.assertRaises(ValueError):
            with preserve_rng():
                torch.rand(50)
                random.random()
                np.random.rand()
                raise ValueError("failure")
        torch.testing.assert_close(torch.rand(4), expected[0], rtol=0, atol=0)
        self.assertEqual(random.random(), expected[1])
        self.assertEqual(np.random.rand(), expected[2])

    def test_public_eva_preserves_function_decoder_parameters_and_ema(self):
        builder = AdapterTests()
        reference, _ = builder.build("lora")
        model, _ = builder.build("eva")
        reference.eval()
        model.eval()
        for name, p in reference.state_dict().items():
            if name.startswith(("neck.", "seg_head.")):
                torch.testing.assert_close(p, model.state_dict()[name], rtol=0, atol=0)
        before = frozen_fingerprint(model.encoder)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)
        ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
        model.encoder.peft_config["default"].eva_config.tau = 0.0
        images = [torch.randn(2,3,64,96) for _ in range(3)]
        with preserve_rng():
            state = get_eva_state_dict(model.encoder, images,
                forward_fn=lambda encoder,x: encoder(pixel_values=x),
                prepare_model_inputs_fn=lambda inputs,config: None,
                prepare_layer_inputs_fn=partial(patch_inputs, prefix=5, maximum=48),
                show_progress_bar=False)
            apply_eva_state(model.encoder, state)
        self.assertEqual(len(state),24)
        self.assertTrue(all(w.shape[0] == 16 for w in state.values()))
        self.assertEqual(before, frozen_fingerprint(model.encoder))
        self.assertEqual(ids, [id(p) for group in optimizer.param_groups for p in group["params"]])
        with torch.no_grad():
            torch.testing.assert_close(model(images[0]), reference(images[0]), rtol=0, atol=0)
        model.train()
        model(images[0]).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model(images[1]).square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for n,p in model.named_parameters() if "lora_A" in n))
        ema = EffectiveEMA(model)
        ema.update(model)
        raw = deepcopy(model.state_dict())
        with ema.apply(model):
            self.assertTrue(torch.isfinite(model(images[0])).all())
        for name,p in model.state_dict().items():
            if name in dict(model.named_parameters()):
                torch.testing.assert_close(p, raw[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
