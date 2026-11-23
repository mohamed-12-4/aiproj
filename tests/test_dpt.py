"""CPU integration checks for the P1 model, training budget and evaluation contract."""

import copy
import csv
from contextlib import redirect_stdout
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from transformers import DINOv3ViTConfig, DINOv3ViTModel

import config
from dpt import CoralDINO
from train import EarlyStopping, evaluate, load_checkpoint, load_data, main, save_checkpoint, train_uniform_epoch


class DPTTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.backbone = DINOv3ViTModel(DINOv3ViTConfig(
            hidden_size=32, intermediate_size=64, num_hidden_layers=4,
            num_attention_heads=4, patch_size=16, num_register_tokens=4,
            output_hidden_states=True,
        ))
        self.settings = {**config.MODEL, "id": "tiny-test", "revision": "test-revision",
                         "feature_layers": [1, 2, 3, 4], "head_channels": 8,
                         "dpt_neck_channels": [4, 8, 16, 32], "dpt_fusion_channels": 8}
        self.lora = {**config.LORA, "r": 2, "lora_alpha": 4}
        self.images = torch.randn(2, 3, 32, 64)
        self.target = torch.randint(4, (2, 32, 64))
        self.target[:, :2] = 255

    def new_backbone(self, *args, **kwargs):
        return copy.deepcopy(self.backbone)

    def new_model(self, decoder="dpt"):
        with patch("dpt.AutoModel.from_pretrained", side_effect=self.new_backbone) as load:
            model = CoralDINO({**self.settings, "decoder": decoder}, self.lora, "/tmp")
        self.assertEqual(load.call_args.kwargs["revision"], "test-revision")
        return model

    def test_dpt_shapes_modes_frozen_weights_and_checkpoint(self):
        model = self.new_model()
        frozen = {name: p.clone() for name, p in model.named_parameters() if not p.requires_grad}
        self.assertFalse(model.backbone.base_model.model.training)
        dropouts = [m for m in model.backbone.modules() if isinstance(m, nn.Dropout) and m.p == .05]
        self.assertTrue(dropouts)
        self.assertTrue(all(m.training for m in dropouts))
        self.assertEqual(model.decoder.config.num_labels, 4)
        self.assertEqual(model.decoder.config.semantic_loss_ignore_index, 255)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        logits = model(self.images)
        self.assertEqual(logits.shape, (2, 4, 32, 64))
        F.cross_entropy(logits, self.target, ignore_index=255).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for n, p in model.named_parameters() if "lora_B" in n))
        optimizer.step()
        for name, p in model.named_parameters():
            if name in frozen:
                self.assertIsNone(p.grad)
                self.assertTrue(torch.equal(p, frozen[name]), name)
        model.eval()
        self.assertTrue(all(not m.training for m in dropouts))
        with torch.no_grad():
            expected = model(self.images)
        with tempfile.TemporaryDirectory() as directory:
            save_checkpoint(model, 32, 1, {"coral_miou": .2}, directory)
            with patch("dpt.AutoModel.from_pretrained", side_effect=self.new_backbone):
                restored, height = load_checkpoint(directory)
            with torch.no_grad():
                actual = restored.eval()(self.images)
            self.assertTrue(torch.equal(expected, actual))
            self.assertEqual(height, 32)

    def test_matched_shallow_interface_and_grid_validation(self):
        model = self.new_model("shallow")
        self.assertEqual(model(self.images).shape, (2, 4, 32, 64))
        with self.assertRaisesRegex(ValueError, "patch size"):
            model(torch.randn(1, 3, 33, 64))

    def test_accumulation_matches_valid_pixel_ce_and_short_final_window(self):
        model = nn.Conv2d(3, 4, 1)
        reference = copy.deepcopy(model)
        images = torch.randn(5, 3, 4, 8)
        target = torch.zeros(5, 4, 8, dtype=torch.long)  # Background must train.
        target[1, :3] = 255
        target[3, :1] = 255
        batches = [(x[None], y[None]) for x, y in zip(images, target)]
        # A fully ignored microbatch must not create an update or affect normalization.
        batches.insert(2, (images[:1], torch.full_like(target[:1], 255)))
        optimizer = torch.optim.SGD(model.parameters(), lr=.05)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
        loss, updates = train_uniform_epoch(
            model, batches, optimizer, torch.amp.GradScaler("cuda", enabled=False),
            scheduler, torch.device("cpu"), 2,
        )
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.05)
        for start in range(0, 5, 2):
            reference_optimizer.zero_grad()
            F.cross_entropy(reference(images[start:start + 2]), target[start:start + 2], ignore_index=255).backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), config.TRAIN["gradient_clip"])
            reference_optimizer.step()
        self.assertEqual(updates, 3)
        self.assertEqual(scheduler.last_epoch, 3)
        self.assertTrue(np.isfinite(loss))
        for actual, expected in zip(model.parameters(), reference.parameters()):
            self.assertTrue(torch.allclose(actual, expected, atol=1e-7))

    def test_update_ceiling_stops_consuming_batches(self):
        model = nn.Conv2d(3, 4, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
        consumed = []

        def batches():
            for i in range(5):
                consumed.append(i)
                yield self.images[:1], self.target[:1]

        _, updates = train_uniform_epoch(
            model, batches(), optimizer, torch.amp.GradScaler("cuda", enabled=False),
            scheduler, torch.device("cpu"), 2, max_updates=1,
        )
        self.assertEqual(updates, 1)
        self.assertEqual(consumed, [0, 1])

    def test_subtype_and_site_metrics_exclude_ignored_pixels(self):
        class FixedModel(nn.Module):
            def forward(self, images):
                ids = torch.tensor([[[3, 0, 3, 1, 3, 0, 1]], [[3, 3, 2, 3, 0, 0, 1]]])
                return F.one_hot(ids, 4).permute(0, 3, 1, 2).float() * 10

        original = torch.tensor([[[3, 20, 23, 32, 37, 5, 0]], [[3, 20, 23, 32, 37, 5, 21]]])
        target = torch.tensor([[[3, 3, 3, 3, 3, 0, 255]], [[3, 3, 3, 3, 3, 0, 255]]])
        report = evaluate(FixedModel(), [(torch.zeros(2, 3, 1, 7), target, original, ["site1", "site2"])], torch.device("cpu"))
        self.assertEqual(report["dead_original_categories"]["20"]["recall"], .5)
        self.assertEqual(len(report["dead_original_categories"]), 5)
        self.assertEqual(set(report["by_site"]), {"site1", "site2"})
        self.assertEqual(sum(report["support"].values()), 12)
        self.assertEqual(report["precision"]["Dead"], 1)
        self.assertEqual(report["recall"]["Dead"], .6)

    def test_early_stopping_meaningful_improvement_and_minimum_epochs(self):
        stopper = EarlyStopping(patience=2, min_epochs=4, min_delta=.01)
        self.assertFalse(stopper.update(.2, 1))
        self.assertFalse(stopper.update(.205, 2))
        self.assertFalse(stopper.update(.209, 3))
        self.assertTrue(stopper.update(.209, 4))
        self.assertFalse(stopper.update(.22, 5))
        self.assertFalse(EarlyStopping(0, 1, 0).update(None, 10))

    def test_full_training_saves_best_and_obeys_update_budget(self):
        def png(array):
            stream = io.BytesIO()
            Image.fromarray(array).save(stream, format="PNG")
            return stream.getvalue()

        original = np.tile(np.array([5, 6, 4, 3, 20, 23, 32, 37], dtype=np.uint8), (32, 8))
        image = np.full((32, 64, 3), 128, dtype=np.uint8)
        settings = {**config.TRAIN, "epochs": 3, "batch_size": 1, "accumulation_steps": 2,
                    "max_updates": 3, "warmup_steps": 0, "num_workers": 0,
                    "early_stopping_patience": 0, "height": 32}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "data").mkdir()
            for site, (split, count) in enumerate((("train", 5), ("validation", 2), ("test", 2)), 1):
                rows = [{"image": {"bytes": png(image), "path": f"site{site}_{i}_image.png"},
                         "label": {"bytes": png(original), "path": f"site{site}_{i}_mask.png"}}
                        for i in range(count)]
                pq.write_table(pa.Table.from_pylist(rows), directory / "data" / f"{split}-000.parquet")
            output = directory / "run"
            with patch.dict(config.TRAIN, settings), patch.dict(config.MODEL, self.settings), \
                 patch.dict(config.LORA, self.lora), \
                 patch("train.download_data", return_value=directory), \
                 patch("dpt.AutoModel.from_pretrained", side_effect=self.new_backbone):
                main(["--output", str(output)], compact_output=False)
                with self.assertRaises(SystemExit):
                    main(["--output", str(output)], compact_output=False)
            metrics = json.loads((output / "metrics.json").read_text())
            self.assertEqual(metrics["optimizer_updates"], 3)
            self.assertEqual(metrics["stop_reason"], "max_updates")
            self.assertEqual(len(metrics["history"]), 1)
            self.assertEqual(metrics["settings"]["effective_batch_size"], 2)
            self.assertEqual(metrics["settings"]["effective_loss"], "uniform_cross_entropy")
            self.assertEqual(metrics["settings"]["loss"]["ignore_index"], 255)
            self.assertEqual(metrics["settings"]["early_stopping_patience"], 0)
            self.assertEqual(set(metrics["test"]["by_site"]), {"site3"})
            self.assertEqual(len(metrics["test"]["dead_original_categories"]), 5)
            self.assertTrue((output / "head.pt").exists())

            # The local entrypoint must keep its two-file contract and avoid test evaluation.
            local = directory / "compact-run"
            terminal = io.StringIO()
            with patch.dict(config.TRAIN, settings), patch.dict(config.MODEL, self.settings), \
                 patch.dict(config.LORA, self.lora), patch("train.ROOT", directory), \
                 patch("train.download_data", return_value=directory), \
                 patch("dpt.AutoModel.from_pretrained", side_effect=self.new_backbone), \
                 patch("train.load_data", wraps=load_data) as prepare, \
                 patch("train.evaluate", wraps=evaluate) as validate, redirect_stdout(terminal):
                main(["--output", str(local)])
            self.assertEqual(prepare.call_args.kwargs["splits"], ("train", "validation"))
            self.assertEqual(validate.call_count, 1)
            self.assertEqual({p.name for p in local.iterdir()}, {"metrics.csv", "prediction.png"})
            with (local / "metrics.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["optimizer_updates"], "3")
            self.assertEqual(rows[0]["is_best"], "True")
            self.assertGreater(float(rows[0]["val_loss"]), 0)
            self.assertAlmostEqual(float(rows[0]["val_coral_miou"]), metrics["history"][0]["validation"]["coral_miou"])
            checkpoint = directory / "checkpoints/local/compact-run/head.pt"
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(state["training_settings"]["seed"], 42)
            self.assertEqual(state["dataset_config"]["annotation_version"], "V1")
            with Image.open(local / "prediction.png") as figure:
                self.assertEqual(figure.size, (1280, 430))
            self.assertEqual(terminal.getvalue().count("Epoch 01 |"), 1)
            self.assertNotIn('"by_site"', terminal.getvalue())


if __name__ == "__main__":
    unittest.main()
