"""Small CPU fixtures for routing safeguards, calibration and reliability metrics."""

import unittest

import numpy as np
import torch

from fusion_eval.core import health_lookup, summarize
from spatial_fusion.analysis import (auroc, correlation, cover_metrics, health_counts, joint_errors,
                                      pixel_histogram, reliability, retained_pixel_risk)
from spatial_fusion.core import (Gate, entropy, fit_temperature, gate_features, health_js,
    health_probabilities, mixture_prediction, normalize, patch_mean, route, temperature,
    uncertainty_features)

torch.set_num_threads(2)


class ProbabilityTests(unittest.TestCase):
    def test_positive_score_normalization_preserves_ratios(self):
        scores = torch.tensor([[[2., 0.]], [[1., 0.]]])
        actual = normalize(scores)
        torch.testing.assert_close(actual[:, 0, 0], torch.tensor([2/3, 1/3]))
        torch.testing.assert_close(actual[:, 0, 1], torch.tensor([.5, .5]))

    def test_temperature_changes_sharpness_and_preserves_prediction(self):
        p = torch.tensor([[[.8]], [[.2]]])
        self.assertIs(temperature(p, 1.), p)
        for t in (.25, .5, 2., 4.):
            calibrated = temperature(p, t)
            self.assertEqual(int(calibrated.argmax()), 0)
            self.assertAlmostEqual(float(calibrated.sum()), 1., places=6)
        self.assertGreater(float(entropy(temperature(p, 2.))), float(entropy(p)))
        with self.assertRaises(ValueError):
            temperature(p, 0.)

    def test_temperature_fit_softens_an_overconfident_inaccurate_model(self):
        logp = torch.tensor([[.999, .001]] * 10).log()
        target = torch.tensor([0] * 8 + [1] * 2)
        fitted = fit_temperature(logp, target)
        self.assertGreater(fitted['temperature'], 1.)
        self.assertLess(fitted['nll'], next(r['nll'] for r in fitted['search'] if r['temperature'] == 1.))

    def test_mixture_endpoints_keep_original_native_argmax(self):
        p = torch.ones(3, 2, 2) / 3
        first, second = torch.ones(2, 2, dtype=torch.long), torch.full((2, 2), 2, dtype=torch.long)
        self.assertIs(mixture_prediction(p, p, 1., first, second), first)
        self.assertIs(mixture_prediction(p, p, 0., first, second), second)

    def test_health_marginals_sum_native_classes(self):
        p = torch.tensor([[[.1]], [[.2]], [[.3]], [[.4]]])
        groups = torch.tensor([4, 1, 1, 3])
        h = health_probabilities(p, groups)
        torch.testing.assert_close(h[:, 0, 0], torch.tensor([0., .5, 0., .4, .1]))

    def test_js_is_zero_for_identical_and_one_for_disjoint(self):
        a = torch.tensor([[[1.]], [[0.]]])
        self.assertEqual(float(health_js(a, a)), 0.)
        self.assertAlmostEqual(float(health_js(a, 1 - a)), 1., places=6)

    def test_patch_means_preserve_edges_of_odd_images(self):
        x = torch.arange(15.).reshape(3, 5)
        actual = patch_mean(x, 2)
        self.assertEqual(actual.shape, x.shape)
        self.assertEqual(float(actual[0, 0]), float(x[:2, :2].mean()))
        self.assertEqual(float(actual[2, 4]), float(x[2, 4]))


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.groups = torch.tensor([4, 1, 1, 3])
        self.first = torch.full((4, 4), 1, dtype=torch.long)
        self.second = torch.full((4, 4), 3, dtype=torch.long)
        self.fh = torch.full((5, 4, 4), .125)
        self.fh[1] = .5
        self.sh = torch.full((5, 4, 4), .025)
        self.sh[3] = .9

    def features(self, second=None):
        return gate_features(self.fh, self.sh, self.first,
                             self.second if second is None else second, self.groups, patch=2)

    def test_confident_consistent_backup_can_override(self):
        result, selected = route(self.first, self.second, self.features(), Gate(.1, 0.))
        self.assertTrue(bool(selected.all()))
        self.assertTrue(torch.equal(result, self.second))

    def test_no_routing_control_and_threshold_safeguards(self):
        for candidate in (Gate(None), Gate(.5), Gate(.1, 1.)):
            result, selected = route(self.first, self.second, self.features(), candidate)
            self.assertFalse(bool(selected.any()))
            self.assertTrue(torch.equal(result, self.first))

    def test_taxonomic_disagreement_cannot_change_agreed_health(self):
        other_taxon = torch.full_like(self.first, 2)
        result, selected = route(self.first, other_taxon, self.features(other_taxon), Gate(0.))
        self.assertFalse(bool(selected.any()))
        self.assertTrue(torch.equal(result, self.first))

    def test_unknown_backup_health_is_never_routed(self):
        unknown = torch.zeros_like(self.first)
        _, selected = route(self.first, unknown, self.features(unknown), Gate(0.))
        self.assertFalse(bool(selected.any()))

    def test_inconsistent_backup_patches_are_protected(self):
        checker = self.second.clone()
        checker[::2, ::2] = 1
        checker[1::2, 1::2] = 1
        _, selected = route(self.first, checker, self.features(checker), Gate(0.))
        self.assertFalse(bool(selected.any()))

    def test_uncertainty_uses_predicted_roi_and_handles_no_predicted_coral(self):
        groups = torch.tensor([4, 0])
        pred = torch.ones(4, 4, dtype=torch.long)
        h = torch.zeros(5, 4, 4); h[0] = 1
        features, js = uncertainty_features(h, h, pred, pred, groups, patch=2)
        self.assertTrue(all(np.isfinite(v) for v in features.values()))
        self.assertEqual(features['health_disagreement'], 0.)
        self.assertEqual(float(js.sum()), 0.)


class AnalysisTests(unittest.TestCase):
    def test_recipe_scalar_metadata_is_portable_without_changing_tensors(self):
        import tempfile
        from pathlib import Path
        from spatial_fusion.artifacts import make_portable
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'head.pt'
            expected = torch.tensor([[1, 7], [3, 8]])
            torch.save(dict(counts=expected, metric=np.float64(.123456789)), path)
            self.assertEqual(make_portable(path), 1)
            loaded = torch.load(path, weights_only=True)
            self.assertTrue(torch.equal(loaded['counts'], expected))
            self.assertIs(type(loaded['metric']), float)
            self.assertEqual(loaded['metric'], .123456789)
            self.assertEqual(make_portable(path), 0)

    def test_native_to_health_counts_matches_existing_summary(self):
        c = torch.randint(0, 10, (2, 96, 96)); c[:, 0] = 0
        actual = health_counts(c, health_lookup())
        for i in range(2):
            self.assertTrue(torch.equal(actual[i], summarize(c[i], health_lookup())['health_confusion']))

    def test_cover_uses_same_annotated_area_and_keeps_unknown_prediction_errors(self):
        c = torch.zeros(5, 5, dtype=torch.long)
        c[3, 3], c[3, 4], c[1, 1] = 2, 1, 7
        m = cover_metrics(c)
        self.assertEqual(m['dead_true_cover_pp'], 30.)
        self.assertEqual(m['dead_pred_cover_pp'], 20.)
        self.assertEqual(m['dead_cover_error_pp'], 10.)
        self.assertAlmostEqual(m['health_pixel_error'], .1)

    def test_joint_error_categories_are_exhaustive_and_void_is_ignored(self):
        target = torch.tensor([[1, 1, 3, 3, 0]])
        first = torch.tensor([[1, 1, 1, 1, 0]])
        second = torch.tensor([[1, 3, 3, 0, 0]])
        r = joint_errors(target, first, second, torch.tensor([4, 1, 2, 3]))
        self.assertEqual(r['native'].tolist(), [1, 1, 1, 1])
        self.assertEqual(r['health'].tolist(), [1, 1, 1, 1])

    def test_auc_ties_and_uninformative_or_missing_classes(self):
        self.assertEqual(auroc([0, 1], [0., 1.]), 1.)
        self.assertEqual(auroc([0, 1], [1., 0.]), 0.)
        self.assertEqual(auroc([0, 1], [.5, .5]), .5)
        self.assertIsNone(auroc([0, 0], [0., 1.]))
        self.assertAlmostEqual(correlation([1, 2, 2], [1, 3, 3]), 1.)

    def test_pixel_referral_removes_high_scoring_errors(self):
        groups = torch.tensor([4, 1, 3])
        target = torch.tensor([[1, 1, 1, 1, 1, 0]])
        pred = torch.tensor([[1, 1, 1, 1, 2, 2]])
        score = torch.tensor([[0., 0., 0., 0., 1., 1.]])
        hist = pixel_histogram(score, target, pred, groups, bins=2)
        self.assertEqual(int(hist.sum()), 5)
        self.assertEqual(retained_pixel_risk(hist, .8), 0.)

    def test_image_reliability_reports_absolute_pp_errors(self):
        rows = [dict(site='a', dead_cover_error_pp=float(i), health_pixel_error=i / 10,
                     uncertainty={'health_js':float(i)}) for i in range(10)]
        r = reliability(rows)
        self.assertEqual(r['dead_cover_mae_pp'], 4.5)
        p = r['predictors']['health_js']
        self.assertEqual(p['retained80_dead_cover_mae_pp'], 3.5)
        self.assertEqual(p['auroc_cover_error_gt5pp'], 1.)

    def test_constant_uncertainty_cannot_gain_from_site_ordering(self):
        rows = [dict(site='a', dead_cover_error_pp=float(i), health_pixel_error=i / 10,
                     uncertainty={'health_js':0.}) for i in range(10)]
        r = reliability(rows)
        self.assertEqual(r['predictors']['health_js']['retained80_dead_cover_mae_pp'], 4.5)


if __name__ == '__main__':
    unittest.main()
