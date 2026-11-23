"""Small CPU checks for member isolation, loss equivalence and checkpoint fidelity."""
import copy
import os
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("USE_TF", "0")
import torch
from torch import nn
from torch.nn import functional as F
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from ensemble import CoralEnsemble, MemberProjection
from train import save_checkpoint, load_checkpoint, evaluate


class EnsembleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.backbone = DINOv3ViTModel(DINOv3ViTConfig(
            hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, patch_size=16, num_register_tokens=4,
            output_hidden_states=True,
        ))
        self.settings = {"id": "tiny-test", "head_channels": 8, "feature_layers": [1, -1]}
        self.lora = {"r": 2, "init_gain": 10, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}
        with patch("ensemble.AutoModel.from_pretrained", return_value=self.backbone):
            self.model = CoralEnsemble(self.settings, self.lora, {"members": 3}, "/tmp")
        self.x = torch.randn(2, 3, 32, 64)
        self.target = torch.randint(4, (2, 32, 64))
        self.target[:, :2] = 255

    def test_projection_zero_update_and_isolated_gradients(self):
        linear = nn.Linear(32, 32).requires_grad_(False)
        router = MemberProjection(linear, 3, self.lora)
        x = torch.randn(2, 4, 32)
        self.assertTrue(torch.equal(router(x), linear(x)))
        self.assertFalse(torch.equal(router.adapters[0].w_a.weight, router.adapters[1].w_a.weight))
        router.member = 1
        router(x).square().mean().backward()
        self.assertIsNone(router.adapters[0].w_b.weight.grad)
        self.assertIsNotNone(router.adapters[1].w_b.weight.grad)
        self.assertIsNone(linear.weight.grad)

    def test_chunked_gradients_match_repeated_batch_mean_ce(self):
        full = copy.deepcopy(self.model)
        for member in range(3):
            loss = F.cross_entropy(self.model.member_logits(self.x, member), self.target, ignore_index=255) / 3
            loss.backward()
        logits = torch.cat([full.member_logits(self.x, member) for member in range(3)])
        F.cross_entropy(logits, self.target.repeat(3, 1, 1), ignore_index=255).backward()
        for (name, a), (_, b) in zip(self.model.named_parameters(), full.named_parameters()):
            if a.requires_grad:
                self.assertTrue(torch.allclose(a.grad, b.grad, atol=2e-6, rtol=2e-5), name)
            else:
                self.assertIsNone(a.grad, name)

    def test_checkpoint_roundtrip_and_frozen_backbone(self):
        frozen = {n: p.clone() for n, p in self.model.named_parameters() if not p.requires_grad}
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=1e-4)
        for member in range(3):
            (F.cross_entropy(self.model.member_logits(self.x, member), self.target, ignore_index=255) / 3).backward()
        optimizer.step()
        for name, p in self.model.named_parameters():
            if name in frozen:
                self.assertTrue(torch.equal(p, frozen[name]), name)
        self.model.eval()
        with torch.no_grad():
            expected = self.model(self.x)
        with tempfile.TemporaryDirectory() as directory:
            save_checkpoint(self.model, 32, 1, {}, directory)
            # The pretrained backbone is exactly the original frozen one.
            with patch("ensemble.AutoModel.from_pretrained", return_value=self.backbone):
                # Strip adapters first: simulate loading pristine pretrained weights.
                for module in list(self.backbone.modules()):
                    for name, child in list(module.named_children()):
                        if isinstance(child, MemberProjection):
                            setattr(module, name, child.adapters[0].w)
                restored, height = load_checkpoint(directory)
            restored.eval()
            with torch.no_grad():
                self.assertTrue(torch.equal(expected, restored(self.x)))
            self.assertEqual(height, 32)

    def test_probability_mean_uncertainty_and_metrics(self):
        self.model.eval()
        with torch.no_grad():
            stats = self.model.prediction_stats(self.x, with_members=True)
        self.assertTrue(torch.allclose(stats["probabilities"], torch.stack(stats["members"]).mean(0), atol=1e-7))
        self.assertTrue(torch.allclose(stats["probabilities"].sum(1), torch.ones(2, 32, 64), atol=1e-6))
        self.assertTrue((stats["mutual_information"] >= 0).all())
        report = evaluate(self.model, [(self.x, self.target)], torch.device("cpu"))
        self.assertEqual(len(report["members"]), 3)
        self.assertEqual(sum(report["support"].values()), 2 * 30 * 64)
        self.assertGreaterEqual(report["uncertainty"]["nll"], 0)
        self.assertTrue(0 <= report["uncertainty"]["ece"] <= 1)


if __name__ == "__main__":
    unittest.main()
