"""Endpoint, provenance and author-validation parity for bounded V2 runs."""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn

from paper_suite.screen import check_resume, screen_config, validate_with_health, validation_epoch
from paper_suite.runner import author


class ScreeningTests(unittest.TestCase):
    def test_forty_epoch_endpoint_retains_original_schedule_and_native_labels(self):
        cfg = screen_config("rein_g", "test", "unique")
        self.assertEqual(cfg["train"]["epochs"], 150)
        self.assertEqual(cfg["experiment"]["stop_after_epochs"], 40)
        self.assertEqual(cfg["data"]["num_classes"], 96)
        self.assertEqual(cfg["data"]["ignore_index"], 0)
        self.assertTrue(validation_epoch(39))

    def test_resume_rejects_old_rein_and_other_run_initializations(self):
        cfg = screen_config("rein_g", "test", "unique")
        state = {"epoch": 12, "checkpoint_kind": "raw_training_resume", "experiment": deepcopy(cfg["experiment"])}
        self.assertEqual(check_resume(state, cfg), 13)
        state["experiment"]["rein_g"].pop("implementation")
        with self.assertRaisesRegex(ValueError, "legacy Rein"):
            check_resume(state, cfg)
        state["experiment"] = deepcopy(cfg["experiment"])
        state["experiment"]["run_id"] = "older-run"
        with self.assertRaisesRegex(ValueError, "run_id"):
            check_resume(state, cfg)

    def test_health_validation_keeps_author_native_metrics_and_void_policy(self):
        class Model(nn.Module):
            image_mean, image_std = (.5,.5,.5), (.25,.25,.25)
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 96, 1)
            def forward(self, x):
                return self.conv(x)
        class Dataset:
            pairs = [(Path("/images/site01/a.png"), Path("/labels/site01/a.png"))]
            def __len__(self):
                return 1
            def __getitem__(self, index):
                image = np.arange(24*32*3, dtype=np.uint8).reshape(24,32,3)
                labels = np.full((24,32), 14, dtype=np.uint8)
                labels[:8] = 0
                labels[8:16] = 4
                return Image.fromarray(image), Image.fromarray(labels)
        torch.manual_seed(10)
        model, dataset = Model(), Dataset()
        cfg = screen_config("lora", "test", "unique")
        cfg["augment"]["val_size"] = [16,24]
        device = torch.device("cpu")
        expected = author.validate_resized(model, dataset, device, 96, 0, False,
                                           (16,24), model.image_mean, model.image_std)
        actual = validate_with_health(model, dataset, cfg, device)
        self.assertAlmostEqual(actual["loss"], expected[0], places=6)
        self.assertAlmostEqual(actual["miou"], expected[1], places=7)
        self.assertAlmostEqual(actual["accuracy"], expected[2], places=7)
        self.assertEqual(sum(map(sum, actual["native"]["confusion"])), 16*32)
        self.assertIn("site01", actual["by_site"])


if __name__ == "__main__":
    unittest.main()
