"""Checks that V2 masks, sampling, provenance and evaluation use the V2 taxonomy."""

import copy
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
from coral_data import checkpoint_dataset, data_directory, dataset_config
from train import Corals, evaluate, main, sampling_weights


class V2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.policy = dataset_config("V2")

    def test_v2_mapping_covers_every_label_and_retains_context(self):
        mapping = self.policy["label_map"]
        self.assertEqual(len(self.policy["source_labels"]), 95)
        self.assertEqual(set(mapping[1:96]), {0, 1, 2, 3, 255})
        cases = {0: 255, 2: 1, 3: 2, 4: 3, 5: 0, 12: 0, 22: 0,
                 23: 0, 51: 255, 52: 2, 53: 3, 68: 0, 91: 255, 92: 3, 95: 1}
        for source, target in cases.items():
            self.assertEqual(mapping[source], target, self.policy["source_labels"].get(str(source)))
        self.assertNotEqual(data_directory("/tmp", self.policy), data_directory("/tmp", dataset_config("V1")))

    def test_sampler_and_dataset_use_the_v2_mask_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            rows = []
            for i, source in enumerate((4, 23)):
                image, mask = directory / f"{i}_image.png", directory / f"{i}_mask.png"
                Image.new("RGB", (64, 32)).save(image)
                Image.fromarray(np.full((32, 64), source, dtype=np.uint8)).save(mask)
                rows.append((image, mask))
            weights, summary = sampling_weights(rows, directory, 32, label_map=self.policy["label_map"])
            self.assertEqual(weights.tolist(), [3, 1])
            self.assertEqual(summary["dead_rich_images"], 1)
            dataset = Corals(rows, 32, label_map=self.policy["label_map"])
            self.assertTrue((dataset[0][1] == 3).all())
            self.assertTrue((dataset[1][1] == 0).all())

    def test_v2_reports_fine_dead_categories_and_legacy_groups(self):
        policy = self.policy

        class FixedModel(nn.Module):
            dataset_config = policy

            def forward(self, images):
                ids = torch.tensor([[[3, 3, 3, 3, 0, 3, 2]]])
                return F.one_hot(ids, 4).permute(0, 3, 1, 2).float()

        original = torch.tensor([[[4, 17, 28, 84, 23, 92, 52]]])
        target = torch.tensor([[[3, 3, 3, 3, 0, 3, 2]]])
        report = evaluate(FixedModel(), [(torch.zeros(1, 3, 1, 7), target, original, ["site10"])], torch.device("cpu"))
        self.assertEqual(report["recall"]["Dead"], 1)
        self.assertEqual(report["dead_original_categories"]["3"]["support"], 2)
        self.assertEqual(report["dead_original_categories"]["20"]["support"], 1)
        self.assertEqual(report["dead_source_categories"]["92"]["label"], "turbinaria dead")
        self.assertEqual(report["dead_source_categories"]["92"]["recall"], 1)
        self.assertIsNone(report["dead_source_categories"]["47"]["recall"])

    def test_v2_training_reload_and_split_provenance(self):
        def png(array):
            stream = io.BytesIO()
            Image.fromarray(array).save(stream, format="PNG")
            return stream.getvalue()

        torch.manual_seed(42)
        backbone = DINOv3ViTModel(DINOv3ViTConfig(
            hidden_size=32, intermediate_size=64, num_hidden_layers=4,
            num_attention_heads=4, patch_size=16, num_register_tokens=4,
            output_hidden_states=True,
        ))
        model_settings = {**config.MODEL, "id": "tiny-test", "feature_layers": [1, 2, 3, 4],
                          "dpt_neck_channels": [4, 8, 16, 32], "dpt_fusion_channels": 8}
        train_settings = {**config.TRAIN, "height": 32, "num_workers": 0,
                          "warmup_steps": 0, "accumulation_steps": 2, "max_updates": 2}
        original = np.tile(np.array([2, 3, 4, 17, 28, 84, 92, 23], dtype=np.uint8), (32, 8))
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "data").mkdir()
            for site, (split, count) in enumerate((("train", 4), ("validation", 1), ("test", 1)), 1):
                rows = [{"image": {"path": f"site{site}_{i}_image.png", "bytes": png(np.full((32, 64, 3), 128, np.uint8))},
                         "label": {"path": f"site{site}_{i}_mask.png", "bytes": png(original)}}
                        for i in range(count)]
                pq.write_table(pa.Table.from_pylist(rows), directory / "data" / f"{split}-000.parquet")
            output = directory / "v2-run"
            with patch.dict(config.TRAIN, train_settings), patch.dict(config.MODEL, model_settings), \
                 patch("train.download_data", return_value=directory), \
                 patch("dpt.AutoModel.from_pretrained", side_effect=lambda *a, **k: copy.deepcopy(backbone)):
                main(["--annotation-version", "V2", "--output", str(output)], compact_output=False)
            metrics = json.loads((output / "metrics.json").read_text())
            self.assertEqual(metrics["settings"]["annotation_version"], "V2")
            self.assertEqual(metrics["settings"]["dataset"]["id"], "josauder/coralscapesV2")
            self.assertEqual(metrics["settings"]["split_sizes"], {"train": 4, "validation": 1, "test": 1})
            self.assertEqual(metrics["optimizer_updates"], 2)
            self.assertEqual(checkpoint_dataset(output), self.policy)
            # Support is counted in the raw V2 taxonomy after reload for final test.
            self.assertEqual(metrics["test"]["dead_source_categories"]["4"]["support"], 256)
            self.assertEqual(metrics["test"]["support"]["Dead"], 1280)


if __name__ == "__main__":
    unittest.main()
