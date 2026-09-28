"""Offline comparisons of real runner artifacts, including retry histories."""

import json
import unittest

from PIL import Image

from evals.compare import _load_rows, compare, main
from evals.providers import ProviderError
from evals.prompts import ANSWER_ONLY_PROMPT_POLICY
from evals.tests import test_runner as fixtures
from unittest.mock import patch


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.RunnerTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.symbolic = self.fixture.output
        self.nl = self.fixture.root / "runs" / "nl"

    def execute(self, output, data, answers=None, **kwargs):
        return self.fixture.execute(answers or [fixtures.response("Yes")], output=output,
                                    data=data, limit=1, **kwargs)

    def pair(self, **nl_settings):
        self.execute(self.symbolic, "symbolic")
        self.execute(self.nl, "natural-language", **nl_settings)

    def test_identification_label_only_comparison_requires_matching_prompt_policy(self):
        self.execute(self.symbolic, "identification-symbolic", [fixtures.response("A")],
                     prompt_policy=ANSWER_ONLY_PROMPT_POLICY)
        self.execute(self.nl, "identification", [fixtures.response("A")],
                     prompt_policy=ANSWER_ONLY_PROMPT_POLICY)
        report = compare(self.symbolic, self.nl)
        self.assertEqual(report["prompt_policy"], ANSWER_ONLY_PROMPT_POLICY)
        self.assertEqual(report["paired"]["counts"]["both_correct"], 1)
        original = self.fixture.root / "runs" / "original-identification"
        self.execute(original, "identification", [fixtures.response("A")])
        with self.assertRaisesRegex(ValueError, "prompt_preparation"):
            compare(self.symbolic, original)

    def test_successful_retry_compares_latest_state_without_double_counting(self):
        self.execute(self.symbolic, "symbolic", [ProviderError("offline rejection", uncertain=False)])
        self.execute(self.symbolic, "symbolic", resume=True)
        self.execute(self.nl, "natural-language")
        before = (self.symbolic / "predictions.jsonl").read_bytes()
        result = compare(self.nl, self.symbolic / "predictions.jsonl")
        self.assertEqual(result["symbolic"]["total"], 1)
        self.assertEqual(result["symbolic"]["errors"], 0)
        self.assertEqual(result["paired"]["counts"]["both_correct"], 1)
        self.assertEqual((self.symbolic / "predictions.jsonl").read_bytes(), before)
        output = self.fixture.root / "comparison"
        with patch("builtins.print"):
            self.assertEqual(main([str(self.symbolic), str(self.nl), "--output", str(output)]), 0)
        self.assertTrue((output / "comparison.md").is_file())

    def test_multiple_errors_use_latest_error_then_latest_answer(self):
        for attempt in range(2):
            self.execute(self.symbolic, "symbolic", [ProviderError("offline rejection", uncertain=False)],
                         resume=bool(attempt))
        self.execute(self.nl, "natural-language")
        result = compare(self.symbolic, self.nl)
        self.assertEqual(result["symbolic"]["total"], 1)
        self.assertEqual(result["symbolic"]["errors"], 1)
        self.assertFalse(result["symbolic"]["complete"])
        self.execute(self.symbolic, "symbolic", [fixtures.response("No")], resume=True)
        result = compare(self.symbolic, self.nl)
        self.assertEqual(result["symbolic"]["errors"], 0)
        self.assertEqual(result["paired"]["counts"]["natural_language_only"], 1)

    def test_settled_duplicates_including_invalid_answers_are_rejected(self):
        for index, answer in enumerate(("Yes", "No", "Yes because")):
            output = self.fixture.root / "runs" / f"duplicate-{index}"
            self.execute(output, "symbolic", [fixtures.response(answer)])
            path = output / "predictions.jsonl"
            path.write_bytes(path.read_bytes() * 2)
            with self.assertRaisesRegex(ValueError, "settled id"):
                _load_rows(path)

    def test_retry_requires_unchanged_question_evidence(self):
        self.execute(self.symbolic, "symbolic", [ProviderError("offline rejection", uncertain=False)])
        self.execute(self.symbolic, "symbolic", resume=True)
        path = self.symbolic / "predictions.jsonl"
        original = fixtures.read_jsonl(path)
        for key in ("target", "metadata", "prompt_sha256", "prepared_image_sha256", "image_path"):
            rows = json.loads(json.dumps(original))
            rows[0][key] = {"different": True} if key == "metadata" else "different"
            fixtures.write_jsonl(path, rows)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Retry evidence"):
                _load_rows(path)

    def test_unfinished_history_is_rejected(self):
        self.pair()
        path = self.symbolic / "predictions.jsonl"
        path.write_bytes(path.read_bytes().rstrip(b"\n"))
        with self.assertRaisesRegex(ValueError, "unfinished final line"):
            compare(self.symbolic, self.nl)

    def test_generation_and_model_mismatches_are_rejected(self):
        self.execute(self.symbolic, "symbolic")
        changes = ({"reasoning_effort": "high"}, {"image_detail": "low"},
                   {"max_output_tokens": 8}, {"temperature": 0.3},
                   {"model": "openai/different-model"},
                   {"base_url": "https://different.invalid/v1"},
                   {"model": "anthropic/test-model"})
        for index, settings in enumerate(changes):
            output = self.fixture.root / "runs" / f"settings-{index}"
            self.execute(output, "natural-language", **settings)
            with self.subTest(settings=settings), self.assertRaisesRegex(ValueError, "model setting"):
                compare(self.symbolic, output)

    def test_image_preparation_mismatch_is_rejected(self):
        self.pair(max_image_edge=1)
        with self.assertRaisesRegex(ValueError, "image_preparation"):
            compare(self.symbolic, self.nl)

    def test_zai_thinking_mode_must_match_across_formats(self):
        model = "zai/glm-4.6v-flashx"
        self.execute(self.symbolic, "symbolic", model=model, thinking_mode="enabled")
        for index, mode in enumerate((None, "disabled")):
            other = self.fixture.root / "runs" / f"nl-mode-{index}"
            self.execute(other, "natural-language", model=model, thinking_mode=mode)
            with self.assertRaisesRegex(ValueError, "thinking_mode"):
                compare(self.symbolic, other)
        self.execute(self.nl, "natural-language", model=model, thinking_mode="enabled")
        self.assertEqual(compare(self.symbolic, self.nl)["paired"]["paired_rows"], 1)

    def test_gemini_thinking_level_must_match_across_formats(self):
        model = "gemini/gemini-3.5-flash-lite"
        self.execute(self.symbolic, "symbolic", model=model, thinking_level="minimal")
        for index, level in enumerate((None, "low")):
            other = self.fixture.root / "runs" / f"nl-level-{index}"
            self.execute(other, "natural-language", model=model, thinking_level=level)
            with self.assertRaisesRegex(ValueError, "thinking_level"):
                compare(self.symbolic, other)
        self.execute(self.nl, "natural-language", model=model, thinking_level="minimal")
        self.assertEqual(compare(self.symbolic, self.nl)["paired"]["paired_rows"], 1)

    def test_gemini_thinking_budget_must_match_across_formats(self):
        model = "gemini/gemini-2.5-flash"
        self.execute(self.symbolic, "symbolic", model=model, thinking_budget=0)
        self.execute(self.nl, "natural-language", model=model)
        with self.assertRaisesRegex(ValueError, "thinking_budget"):
            compare(self.symbolic, self.nl)
        matched = self.fixture.root / "runs" / "nl-zero"
        self.execute(matched, "natural-language", model=model, thinking_budget=0)
        self.assertEqual(compare(self.symbolic, matched)["paired"]["paired_rows"], 1)

    def test_mixed_prompt_policies_are_rejected(self):
        self.pair(prompt_policy="original")
        with self.assertRaisesRegex(ValueError, "prompt_preparation"):
            compare(self.symbolic, self.nl)

    def test_original_prompt_runs_remain_comparable(self):
        self.execute(self.symbolic, "symbolic", prompt_policy="original")
        self.execute(self.nl, "natural-language", prompt_policy="original")
        report = compare(self.symbolic, self.nl)
        self.assertEqual(report["prompt_policy"], "original")
        self.assertEqual(report["paired"]["paired_rows"], 1)

    def test_same_preparation_but_different_image_bytes_are_rejected(self):
        self.execute(self.symbolic, "symbolic")
        image = self.fixture.root / self.fixture.inputs[0]["image_path"]
        Image.new("RGB", (2, 2), "white").save(image)
        self.execute(self.nl, "natural-language")
        with self.assertRaisesRegex(ValueError, "prepared_image_sha256"):
            compare(self.symbolic, self.nl)

    def test_record_image_must_match_its_own_manifest(self):
        self.pair()
        path = self.nl / "predictions.jsonl"
        rows = fixtures.read_jsonl(path)
        rows[0]["prepared_image_sha256"] = "different"
        fixtures.write_jsonl(path, rows)
        with self.assertRaisesRegex(ValueError, "image does not match its manifest"):
            compare(self.symbolic, self.nl)

    def test_different_answer_files_are_rejected(self):
        self.execute(self.symbolic, "symbolic")
        self.fixture.gold[0]["target"] = "No"
        fixtures.write_jsonl(self.fixture.gold_file, self.fixture.gold)
        self.execute(self.nl, "natural-language")
        with self.assertRaisesRegex(ValueError, "answers_sha256"):
            compare(self.symbolic, self.nl)

    def test_missing_manifest_cannot_silently_skip_settings_checks(self):
        self.pair()
        (self.nl / "manifest.json").unlink()
        with self.assertRaisesRegex(ValueError, "manifest"):
            compare(self.symbolic, self.nl)

    def test_summary_cannot_override_manifest_settings(self):
        self.pair(reasoning_effort="high")
        path = self.nl / "summary.json"
        summary = json.loads(path.read_text())
        summary["model"]["reasoning_effort"] = None
        path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "Summary and manifest disagree"):
            compare(self.symbolic, self.nl)

    def test_transport_and_key_variable_differences_are_allowed(self):
        self.pair(timeout=30, retries=1, api_key_env="OFFLINE_OTHER_KEY")
        self.assertEqual(compare(self.symbolic, self.nl)["paired"]["paired_rows"], 1)

    def test_identification_format_pair_is_supported(self):
        self.execute(self.symbolic, "identification-symbolic", [fixtures.response("A")])
        self.execute(self.nl, "identification", [fixtures.response("AF")])
        self.assertEqual(compare(self.symbolic, self.nl)["paired"]["counts"]["symbolic_only"], 1)


if __name__ == "__main__":
    unittest.main()
