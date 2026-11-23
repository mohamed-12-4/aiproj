"""CPU fixtures for exact fusion, endpoint handling, and label-policy integrity."""
import unittest

import torch

from fusion_eval.core import (Candidate, ConfusionCounter, candidates, confusion, fuse, health_lookup,
                              predict_all, predict_selected, select_indices, summarize)

torch.set_num_threads(2)


class FusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        self.groups = torch.tensor([4, 0, 1, 1, 2, 2, 3, 3])
        self.first = torch.randn(8, 4, 5).softmax(0)
        self.second = torch.randn(8, 4, 5).softmax(0)

    def test_optimized_search_matches_full_probability_formulas(self):
        actual = list(predict_all(self.first, self.second, self.groups))
        self.assertEqual([c for c, _ in actual], candidates())
        for candidate, prediction in actual:
            scores = fuse(self.first, self.second, self.groups, candidate)
            torch.testing.assert_close(scores.sum(0), torch.ones(4, 5))
            self.assertTrue(torch.equal(prediction, scores.argmax(0)), candidate)
            self.assertTrue(torch.equal(prediction, predict_selected(
                self.first, self.second, self.groups, candidate)), candidate)

    def test_both_models_are_exact_endpoints_of_every_family(self):
        for family in ("global", "dead", "hierarchical"):
            for weight, expected in ((0.0, self.second), (1.0, self.first)):
                actual = fuse(self.first, self.second, self.groups,
                              Candidate(family, weight, weight))
                torch.testing.assert_close(actual, expected)

    def test_global_is_diagonal_of_class_dependent_fusion(self):
        for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
            torch.testing.assert_close(
                fuse(self.first, self.second, self.groups, Candidate("global", weight, weight)),
                fuse(self.first, self.second, self.groups, Candidate("dead", weight, weight)))

    def test_ties_choose_lowest_native_id(self):
        maps = torch.full_like(self.first, 1 / 8)
        for candidate, predicted in predict_all(maps, maps, self.groups):
            self.assertTrue(torch.equal(predicted, torch.zeros_like(predicted)), candidate)

    def test_zero_mass_health_groups_remain_finite_and_normalized(self):
        first = torch.zeros_like(self.first)
        second = torch.zeros_like(self.second)
        first[2] = 1
        second[6] = 1
        for candidate, prediction in predict_all(first, second, self.groups):
            scores = fuse(first, second, self.groups, candidate)
            self.assertTrue(torch.isfinite(scores).all())
            torch.testing.assert_close(scores.sum(0), torch.ones(4, 5))
            self.assertTrue(torch.equal(prediction, scores.argmax(0)), candidate)

    def test_hierarchical_group_marginals_follow_health_weight(self):
        for a, b in ((0.25, 0.75), (0.75, 0.25)):
            actual = fuse(self.first, self.second, self.groups, Candidate("hierarchical", a, b))
            for group in range(5):
                mask = self.groups == group
                expected = a * self.first[mask].sum(0) + (1 - a) * self.second[mask].sum(0)
                torch.testing.assert_close(actual[mask].sum(0), expected)


class MetricTests(unittest.TestCase):
    def test_sharded_counts_match_independent_numpy_accumulation(self):
        import numpy as np
        target = torch.randint(0, 96, (33, 91))
        predicted = torch.randint(0, 96, target.shape)
        expected = np.zeros((96, 96), dtype=np.int64)
        valid = target.numpy() != 0
        np.add.at(expected, (target.numpy()[valid], predicted.numpy()[valid]), 1)
        for shards in (1, 7, 32):
            actual = ConfusionCounter(target, blocks=shards)(predicted)
            np.testing.assert_array_equal(actual.numpy(), expected)

    def test_metrics_match_existing_native_v2_evaluator(self):
        from reinpp_training.metrics import mapped_metrics, summarize as existing_summary
        counts = torch.randint(0, 500, (96, 96))
        counts[0] = 0
        expected_native = existing_summary(counts)
        expected_health = mapped_metrics(counts)
        actual = summarize(counts, health_lookup())
        self.assertAlmostEqual(actual['native_miou'], expected_native['miou'])
        self.assertAlmostEqual(actual['coral_miou'], expected_health['coral_miou'])
        for key in ('iou', 'precision', 'recall'):
            self.assertAlmostEqual(actual['dead_' + key], expected_health['dead'][key])

    def test_actual_mapping_and_unknown_prediction_errors(self):
        groups = health_lookup()
        self.assertEqual(int(groups[0]), 4)
        self.assertEqual(int(groups[4]), 3)
        self.assertEqual(int(groups[23]), 0)  # Dead giant clam is not dead coral.
        counts = torch.zeros(96, 96, dtype=torch.long)
        counts[4, 4], counts[4, 0], counts[5, 4] = 10, 2, 3
        metrics = summarize(counts, groups)
        self.assertAlmostEqual(metrics['dead_precision'], 10 / 13)
        self.assertAlmostEqual(metrics['dead_recall'], 10 / 12)
        self.assertAlmostEqual(metrics['dead_iou'], 10 / 15)

    def test_void_targets_are_ignored_but_void_predictions_are_errors(self):
        target = torch.tensor([[0, 4, 4]])
        predicted = torch.tensor([[4, 4, 0]])
        counts = confusion(target, predicted)
        self.assertEqual(int(counts.sum()), 2)
        self.assertEqual(int(counts[4, 0]), 1)
        self.assertEqual(int(counts[0].sum()), 0)

    def test_selection_uses_native_metric_and_keeps_family_separate(self):
        records = candidates()
        counts = torch.zeros(len(records), 96, 96, dtype=torch.long)
        counts[:, 4, 0] = 10
        desired = {}
        for family in ("global", "dead", "hierarchical"):
            index = next(i for i, c in enumerate(records) if c.family == family)
            counts[index, 4, 0] = 0
            counts[index, 4, 4] = 10
            desired[family] = index
        selected, _ = select_indices(records, counts, health_lookup())
        self.assertEqual(selected, desired)


class RuntimeInterfaceTests(unittest.TestCase):
    def test_released_model_processor_interface_is_sufficient(self):
        from types import SimpleNamespace
        from PIL import Image
        from fusion_eval.runner import image_tensor
        model = SimpleNamespace(processor=SimpleNamespace(image_mean=[.485, .456, .406],
                                                          image_std=[.229, .224, .225]))
        image = Image.new('RGB', (4, 2), (255, 0, 127))
        actual = image_tensor(model, image)
        expected = (torch.tensor([1., 0., 127 / 255]) - torch.tensor([.485, .456, .406])) / torch.tensor([.229, .224, .225])
        torch.testing.assert_close(actual, expected[None, :, None, None].expand(1, 3, 2, 4))

    def test_evaluation_outputs_keep_the_two_file_format(self):
        import tempfile
        from pathlib import Path
        import numpy as np
        from PIL import Image
        from fusion_eval.runner import figure, write_csv
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_csv(root, dict(epoch=1, evaluation_only=True, val_native_miou=.4))
            preview = {0: dict(image=Image.new('RGB', (320, 160)),
                               target=np.zeros((160, 320), dtype=np.uint8),
                               predictions=np.zeros((1, 160, 320), dtype=np.uint8))}
            figure(root, preview, 0, health_lookup(), 'CPU fixture')
            self.assertEqual({p.name for p in root.iterdir()}, {'metrics.csv', 'prediction.png'})
            with Image.open(root / 'prediction.png') as image:
                self.assertEqual(image.size, (1280, 245))


if __name__ == '__main__':
    unittest.main()
