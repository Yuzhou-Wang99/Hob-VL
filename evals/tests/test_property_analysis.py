"""Small independent checks of the new descriptive-analysis denominators."""
import unittest

try:
    import numpy as np
    import pandas as pd
    from evals.analyze_properties import metrics, slopes
except ImportError:
    np = None


@unittest.skipIf(np is None, "Optional property-analysis dependencies are not installed")
class PropertyAnalysisTests(unittest.TestCase):
    def test_invalid_and_incomplete_pairs_and_label_imbalance(self):
        rows = [
            ("a", "Yes", 1, 1, "Yes", 0),
            ("a", "Yes", 0, 1, "No", 0),
            ("b", "No", 0, 0, None, 1),
            ("b", "No", 1, 1, "No", 0),
            ("c", "Yes", 1, 1, "Yes", 0),
        ]
        frame = pd.DataFrame(rows, columns=["pair_id", "target", "correct", "valid", "prediction", "token_limit"])
        frame["scene_id"] = frame.pair_id
        m = metrics(frame)
        self.assertEqual(m["accuracy"], 3 / 5)
        self.assertEqual(m["accuracy_on_valid"], 3 / 4)
        self.assertAlmostEqual(m["balanced_accuracy"], (2 / 3 + 1 / 2) / 2)
        self.assertEqual(m["majority_label_baseline"], 3 / 5)
        self.assertEqual(m["pairs"], 2)  # singleton c is not an equivalent pair
        self.assertEqual(m["valid_pairs"], 1)
        self.assertEqual(m["pair_conflict_rate"], 1)
        self.assertEqual(m["pairs_both_correct_accuracy"], 0)
        self.assertEqual(m["token_limit"], 1)
        self.assertIsNone(metrics(frame[frame.target == "Yes"])["balanced_accuracy"])

    def test_adjustment_recovers_known_slope_and_flags_no_within_family_variation(self):
        # A synthetic linear response tests the algebra, not model performance.
        rng = np.random.default_rng(71)
        n = 100
        family = np.repeat([0, 1], n // 2)
        x = rng.normal(size=n) + family
        frame = pd.DataFrame(dict(
            model="fixture", format="symbolic", family=family.astype(str), track="synthetic", target="Yes",
            prompt_words=rng.integers(10, 100, n), variant=np.where(np.arange(n) % 2, "base", "de_morgan"),
            correct=.4 + .1*x + .2*family, valid=1, token_limit=0,
            certificate_size=x, two_fact_predictability=.5+.1*family, total_influence=2+family,
        ))
        results = {r["property"]: r for r in slopes(frame)}
        self.assertAlmostEqual(results["certificate_size"]["adjusted_accuracy_pp_per_sd"], 10*x.std(), places=9)
        for name in ("two_fact_predictability", "total_influence"):
            self.assertLess(results[name]["residual_variance_fraction"], 1e-10)
            self.assertTrue(np.isnan(results[name]["adjusted_accuracy_pp_per_sd"]))


if __name__ == "__main__":
    unittest.main()
