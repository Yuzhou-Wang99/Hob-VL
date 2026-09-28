"""Offline integration tests: real image files, fake model responses, no API calls."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from evals.data import sha256
from evals.prompts import (ANSWER_ONLY_PROMPT_POLICY, BOOLEAN_OUTPUT_INSTRUCTION,
                          DEFAULT_PROMPT_POLICY, IDENTIFICATION_OUTPUT_INSTRUCTION)
from evals.providers import ProviderError, build_request
from evals.run import evaluate
from evals.rescore import rescore
from evals.scoring import parse_answer, parse_label, reported_cost, summarize


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def response(text, *, complete=True):
    return {
        "text": text,
        "usage": {"input_tokens": 7, "output_tokens": 1},
        "response_id": "offline-response",
        "finish_reason": "stop" if complete else "length",
        "complete": complete,
    }


class AnswerParsingTests(unittest.TestCase):
    def test_bold_policy_accepts_only_standalone_answers(self):
        for text, expected in [("**Yes**", "Yes"), (" __no__ \n", "No"),
                               ("**YES!**", "Yes"), ("__No__.", "No")]:
            with self.subTest(text=text):
                self.assertIsNone(parse_answer(text))
                self.assertEqual(parse_answer(text, answer_format="plain-or-bold"), expected)
        for text in ["*Yes*", "***Yes***", "**Yes__", "**Yes", "**Yes** **No**",
                     "**Yes.**!", "** Yes **", "**Yes, because blue.**", "Reasoning\n**Yes**",
                     "The answer is **No**", "```Yes```", '{"answer":"Yes"}', "__True__"]:
            with self.subTest(text=text):
                self.assertIsNone(parse_answer(text, answer_format="plain-or-bold"))

    def test_cost_subtotal_distinguishes_missing_and_zero(self):
        rows = [{"usage": {"cost_in_usd_ticks": 100_000_000}},
                {"usage": {"cost_in_usd_ticks": 0}}, {"usage": {}}, {"status": "error"}]
        billing = reported_cost(rows)
        self.assertEqual(billing["reported_cost_usd"], 0.01)
        self.assertEqual(billing["saved_attempts_with_cost"], 2)
        self.assertEqual(billing["saved_attempts_without_cost"], 2)
        self.assertIsNone(reported_cost([{"usage": {}}])["reported_cost_usd"])
        self.assertEqual(reported_cost([{"usage": {"cost_in_usd_ticks": 0}}])["reported_cost_usd"], 0)

    def test_accepts_only_whole_yes_or_no_answers(self):
        for text, expected in [
            ("Yes", "Yes"), (" No \n", "No"), ("yEs.", "Yes"), ("NO!", "No")
        ]:
            with self.subTest(text=text):
                self.assertEqual(parse_answer(text), expected)

    def test_rejects_prose_boolean_aliases_and_ambiguous_answers(self):
        for text in ["", " ", "True", "False", "1", "0", "Yes or No", "Yes, because blue.",
                     "The answer is Yes", "No. Actually, Yes.", "```Yes```", "yesterday", "nobody"]:
            with self.subTest(text=text):
                self.assertIsNone(parse_answer(text))

    def test_pair_metrics_do_not_credit_two_invalid_answers_as_agreement(self):
        rows = []
        for family, status, prediction, correct in [("good", "ok", "Yes", True),
                                                     ("invalid", "invalid", None, False)]:
            for variant in ["base", "de_morgan"]:
                rows.append({"id": f"{family}-{variant}", "target": "Yes", "prediction": prediction,
                             "correct": correct, "status": status, "usage": {},
                             "metadata": {"formula_family_id": family, "variant": variant,
                                          "track": "synthetic", "cohort": "test"}})
        rows.append({"id": "unpaired", "target": "No", "prediction": "No", "correct": True,
                     "status": "ok", "metadata": {"formula_family_id": "unpaired", "variant": "base"}})
        result = summarize(rows)
        self.assertEqual(result["overall"]["accuracy"], 3 / 5)
        pairs = result["equivalent_pairs"]
        self.assertEqual(pairs["complete_pairs"], 2)
        self.assertEqual(pairs["agreement"], 0.5)
        self.assertEqual(pairs["both_correct_accuracy"], 0.5)
        self.assertEqual(pairs["excluded_incomplete_or_nonpair_families"], 1)


class LabelParsingTests(unittest.TestCase):
    candidates = ("A", "B", "C", "AF")

    def test_bold_policy_still_checks_whole_candidate_labels(self):
        for text in ["**af**", "__AF!__", " **Af**. "]:
            self.assertEqual(parse_label(text, self.candidates, answer_format="plain-or-bold"), "AF")
            self.assertIsNone(parse_label(text, self.candidates))
        for text in ["**A F**", "**AB**", "**A or B**", "Answer: **AF**", "**A__"]:
            self.assertIsNone(parse_label(text, self.candidates, answer_format="plain-or-bold"))
        self.assertIsNone(parse_label("**A**", ("AF", "B"), answer_format="plain-or-bold"))

    def test_accepts_only_a_complete_candidate_label(self):
        for text, expected in [("A", "A"), (" b \n", "B"), ("AF.", "AF"), ("c!", "C")]:
            with self.subTest(text=text):
                self.assertEqual(parse_label(text, self.candidates), expected)

    def test_multi_letter_labels_are_not_split(self):
        self.assertEqual(parse_label("AF", self.candidates), "AF")
        self.assertIsNone(parse_label("A F", self.candidates))
        self.assertIsNone(parse_label("A", ("AF", "B")))

    def test_rejects_prose_multiple_and_unknown_labels(self):
        for text in ["", " ", "Yes", "No", "Label A", "The answer is B", "A or B",
                     "A, F", "Z", "AB", "AA", "```A```", "A1"]:
            with self.subTest(text=text):
                self.assertIsNone(parse_label(text, self.candidates))


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hob_vl-evals-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repo"
        self.dataset = self.root / "dataset" / "hob_vl_v1"
        self.dataset.mkdir(parents=True)
        self.inputs = []
        self.gold = []
        for index, (track, target) in enumerate([
            ("synthetic", "Yes"), ("photo", "No"), ("synthetic", "Yes"), ("photo", "No")
        ]):
            directory = ("images/synthetic" if track == "synthetic"
                         else "images/photos")
            image_path = f"{directory}/scene_{index}.png"
            image_file = self.root / image_path
            image_file.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (2, 2), color=(index * 30, 20, 40)).save(image_file)
            row = {"id": f"example-{index}", "image_path": image_path,
                   "prompt": f"Condition {index}. Answer Yes or No."}
            self.inputs.append(row)
            self.gold.append({
                "id": row["id"], "target": target, "track": track,
                "variant": "base" if index < 2 else "rewrite",
                "formula_family_id": f"formula-{index % 2}",
                "scene_family": f"scene-{index}",
                "input": {"image_path": image_path, "prompt": row["prompt"]},
                "audit": {"cohort": "test-cohort"},
            })
        self.input_file = self.dataset / "model_inputs.jsonl"
        self.gold_file = self.dataset / "dataset.jsonl"
        write_jsonl(self.input_file, self.inputs)
        # Deliberately out of input order: scoring must join on ID, not position.
        write_jsonl(self.gold_file, list(reversed(self.gold)))
        natural = [{**row, "prompt": f"Natural language condition {index}. Answer Yes or No."}
                   for index, row in enumerate(self.inputs)]
        write_jsonl(self.dataset / "model_inputs_natural_language.jsonl", natural)
        ident_image = "images/photos/ident_scene.png"
        Image.new("RGB", (2, 2), color=(9, 9, 9)).save(self.root / ident_image)
        self.ident_inputs = [
            {"id": f"ident-{index}", "image_path": ident_image,
             "candidate_labels": ["A", "B", "AF"],
             "prompt": f"Identify the object {index}. Answer with only its label."}
            for index in range(2)]
        self.ident_gold = [
            {"id": row["id"], "task": "single_object_identification",
             "image_index": 0, "scene_id": "photo_t", "scene_family": "ident-family",
             "split": "evaluation", "input": dict(row), "target": target, "audit": {}}
            for row, target in zip(self.ident_inputs, ["A", "AF"])]
        write_jsonl(self.dataset / "model_inputs_photo_identification_1000.jsonl", self.ident_inputs)
        write_jsonl(self.dataset / "model_inputs_photo_identification_1000_symbolic.jsonl",
                    [{**row, "formula": "p1", "atoms": []} for row in self.ident_inputs])
        (self.dataset / "photo_identification_1000.json").write_text(
            json.dumps(self.ident_gold), encoding="utf-8")
        self.output = self.root / "runs" / "test"
        self.key_check = self.enterContext(patch("evals.run.check_api_key", return_value="offline-test-key"))
        self.model_call = self.enterContext(patch("evals.run.call_model", side_effect=AssertionError("No model calls permitted")))

    def run_eval(self, **kwargs):
        options = {"model": "openai/test-model", "repo_root": self.root, "output": self.output}
        options.update(kwargs)
        return evaluate(**options)

    def execute(self, answers=None, **kwargs):
        self.model_call.side_effect = answers if answers is not None else [response("Yes"), response("No"),
                                                                          response("Yes"), response("No")]
        return self.run_eval(execute=True, **kwargs)

    def assert_preflight_failure(self, **kwargs):
        with self.assertRaises((ValueError, OSError)):
            self.run_eval(execute=True, **kwargs)
        self.key_check.assert_not_called()
        self.model_call.assert_not_called()

    def test_zai_thinking_truncation_saves_audit_continues_and_is_settled_on_resume(self):
        from evals.providers import call_model

        replies = []
        for answer, finish in (("Yes", "stop"), ("Yes", "length"), (None, "length"), ("No", "stop")):
            replies.append({"id": "offline-thinking-response", "choices": [{
                "message": {"content": answer, "reasoning_content": "Yes. Still considering the image."},
                "finish_reason": finish,
            }], "usage": {"prompt_tokens": 100, "completion_tokens": 4096,
                          "total_tokens": 4196,
                          "completion_tokens_details": {"reasoning_tokens": 4095}}})
        self.model_call.side_effect = call_model
        options = dict(model="zai/glm-4.6v-flashx", thinking_mode="enabled", max_output_tokens=4096)
        with patch("evals.providers._api_key", return_value="offline-test-key"), \
                patch("evals.providers._post_json", side_effect=replies) as transport:
            report = self.run_eval(execute=True, **options)
        self.assertEqual(transport.call_count, 4)
        for call in transport.call_args_list:
            self.assertEqual(call.args[1]["thinking"], {"type": "enabled"})
            self.assertEqual(call.args[1]["max_tokens"], 4096)
        self.assertTrue(report["complete"])
        self.assertFalse(report["stopped_on_error"])
        self.assertEqual(report["overall"]["invalid"], 2)
        self.assertEqual(report["overall"]["correct"], 2)
        self.assertEqual(report["overall"]["errors"], 0)
        self.assertEqual(report["usage"]["output_tokens"], 16384)
        rows = read_jsonl(self.output / "predictions.jsonl")
        self.assertEqual([row["status"] for row in rows], ["ok", "invalid", "invalid", "ok"])
        self.assertEqual(rows[2]["response"], "")
        for row in rows:
            self.assertEqual(row["reasoning_content"], "Yes. Still considering the image.")
        self.assertEqual(rows[2]["finish_reason"], "length")
        self.assertIsNone(rows[2]["prediction"])
        self.assertFalse((self.output / "inflight.json").exists())
        before = (self.output / "predictions.jsonl").read_bytes()
        self.model_call.reset_mock()
        self.key_check.reset_mock()
        self.run_eval(execute=True, resume=True, retry_pending=True, **options)
        self.assertEqual((self.output / "predictions.jsonl").read_bytes(), before)
        for mode in (None, "disabled"):
            with self.assertRaisesRegex(ValueError, "Resume mismatch"):
                self.run_eval(execute=True, resume=True, **{**options, "thinking_mode": mode})
        self.model_call.assert_not_called()
        self.key_check.assert_not_called()

    def test_zai_thinking_cli_prepares_all_formats_without_credentials(self):
        from evals.__main__ import main

        for fmt in ("symbolic", "natural-language", "identification-symbolic", "identification"):
            output = self.root / "runs" / fmt
            with patch("builtins.print"):
                code = main(["--model", "zai/glm-4.6v-flashx", "--thinking-mode", "enabled",
                             "--max-output-tokens", "4096", "--answer-format", "plain-or-bold",
                             "--prompt-policy", "answer-only-v1", "--data", fmt, "--limit", "1",
                             "--repo-root", str(self.root), "--output", str(output)])
            self.assertEqual(code, 0)
            report = json.loads((output / "dry_run.json").read_text())
            self.assertEqual(report["model"]["thinking_mode"], "enabled")
            preview = json.loads((output / "sample_requests.json").read_text())[0]
            self.assertEqual(preview["payload"]["thinking"], {"type": "enabled"})
        self.model_call.assert_not_called()
        self.key_check.assert_not_called()

    def test_hosted_pilot_settings_survive_scoring_and_resume(self):
        for provider, model in (("qwen", "qwen3-vl-8b-instruct"), ("zai", "glm-4.5v"),
                                ("zai", "glm-4.6v-flash"), ("zai", "glm-4.6v-flashx"),
                                ("deepseek", "deepseek-flash"),
                                ("deepinfra", "meta-llama/Llama-4-Scout-17B-16E-Instruct")):
            for fmt in ("symbolic", "natural-language", "identification-symbolic", "identification"):
                with self.subTest(provider=provider, model=model, format=fmt):
                    output = self.output.parent / f"{provider}-{model.replace('/', '-')}-{fmt}"
                    options = dict(model=f"{provider}/{model}", data=fmt, output=output,
                                   reasoning_effort="none", max_output_tokens=2048,
                                   answer_format="plain-or-bold", prompt_policy="answer-only-v1", limit=1)
                    if provider == "qwen":
                        options["base_url"] = "https://workspace.example/compatible-mode/v1"
                    label = "A" if fmt.startswith("identification") else "Yes"
                    self.model_call.reset_mock()
                    report = self.execute([response(label)], **options)
                    self.assertTrue(report["complete"])
                    self.assertEqual(report["overall"]["correct"], 1)
                    payload = self.model_call.call_args.args[1]
                    suffix = (IDENTIFICATION_OUTPUT_INSTRUCTION if fmt.startswith("identification")
                              else BOOLEAN_OUTPUT_INSTRUCTION)
                    self.assertTrue(payload["messages"][0]["content"][0]["text"].endswith(suffix))
                    self.assertNotIn("target", payload)
                    self.assertNotIn("reasoning_effort", payload)
                    if provider in {"zai", "deepseek"}:
                        self.assertEqual(payload["thinking"], {"type": "disabled"})
                    else:
                        self.assertNotIn("thinking", payload)
                    manifest = json.loads((output / "manifest.json").read_text())
                    self.assertEqual(manifest["identity"]["model"], report["model"])
                    self.assertEqual(report["model"]["reasoning_effort"], "none")
                    self.assertEqual(report["model"]["model"], model)
                    self.model_call.reset_mock()
                    resumed = self.run_eval(execute=True, resume=True, **options)
                    self.model_call.assert_not_called()
                    self.assertTrue(resumed["complete"])
                    with self.assertRaisesRegex(ValueError, "Resume mismatch"):
                        self.run_eval(execute=True, resume=True,
                                      **{**options, "reasoning_effort": None})
                    if model == "glm-4.6v-flashx":
                        with self.assertRaisesRegex(ValueError, "Resume mismatch"):
                            self.run_eval(execute=True, resume=True,
                                          **{**options, "model": "zai/glm-4.6v-flash"})

    def test_boolean_instruction_in_every_dry_and_live_request_for_every_provider(self):
        originals = {path: path.read_bytes() for path in self.dataset.glob("*.json*")}
        providers = ("openai", "anthropic", "gemini", "openai-compatible")
        for provider in providers:
            for fmt in ("symbolic", "natural-language"):
                filename = ("model_inputs.jsonl" if fmt == "symbolic" else
                            "model_inputs_natural_language.jsonl")
                inputs = read_jsonl(self.dataset / filename)
                expected = [row["prompt"] + "\n\n" + BOOLEAN_OUTPUT_INSTRUCTION for row in inputs]
                options = {"model": f"{provider}/test-model", "data": fmt}
                if provider == "openai-compatible":
                    options["base_url"] = "https://example.invalid/v1"
                for execute in (False, True):
                    with self.subTest(provider=provider, data=fmt, execute=execute):
                        output = self.output.parent / f"{provider}-{fmt}-{execute}"
                        with patch("evals.run.build_request", wraps=build_request) as builder:
                            result = (self.execute(output=output, **options) if execute else
                                      self.run_eval(output=output, **options))
                        actual = [call.args[1] for call in builder.call_args_list]
                        self.assertEqual(actual, expected)
                        self.assertTrue(all(text.count(BOOLEAN_OUTPUT_INSTRUCTION) == 1 for text in actual))
                        self.assertEqual(result["prompt_policy"], DEFAULT_PROMPT_POLICY)
                        manifest = json.loads((output / "manifest.json").read_text())
                        self.assertEqual(manifest["identity"]["prompt_preparation"],
                                         {"policy": DEFAULT_PROMPT_POLICY,
                                          "suffix": "\n\n" + BOOLEAN_OUTPUT_INSTRUCTION})
                        if execute:
                            rows = read_jsonl(output / "predictions.jsonl")
                            self.assertEqual([row["prompt_sha256"] for row in rows],
                                             [sha256(text.encode("utf-8")) for text in expected])
        self.assertEqual(originals, {path: path.read_bytes() for path in originals})

    def test_identification_prompts_and_hashes_are_unchanged(self):
        for fmt in ("identification-symbolic", "identification"):
            output = self.output.parent / fmt
            with patch("evals.run.build_request", wraps=build_request) as builder:
                result = self.execute([response("A"), response("AF")], data=fmt, output=output)
            expected = [row["prompt"] for row in self.ident_inputs]
            self.assertEqual([call.args[1] for call in builder.call_args_list], expected)
            self.assertEqual(result["prompt_policy"], "original")
            identity = json.loads((output / "manifest.json").read_text())["identity"]
            self.assertNotIn("prompt_preparation", identity)
            self.assertEqual([row["prompt_sha256"] for row in read_jsonl(output / "predictions.jsonl")],
                             [sha256(text.encode("utf-8")) for text in expected])

    def test_optional_label_instruction_is_sent_and_recorded_without_editing_inputs(self):
        originals = {path: path.read_bytes() for path in self.dataset.glob("*.json*")}
        for fmt in ("identification-symbolic", "identification"):
            for live in (False, True):
                with self.subTest(data=fmt, live=live):
                    output = self.output.parent / f"label-only-{fmt}-{live}"
                    options = {"data": fmt, "output": output,
                               "prompt_policy": ANSWER_ONLY_PROMPT_POLICY}
                    with patch("evals.run.build_request", wraps=build_request) as builder:
                        result = (self.execute([response("A"), response("AF")], **options)
                                  if live else self.run_eval(**options))
                    expected = [row["prompt"] + "\n\n" + IDENTIFICATION_OUTPUT_INSTRUCTION
                                for row in self.ident_inputs]
                    self.assertEqual([call.args[1] for call in builder.call_args_list], expected)
                    self.assertEqual(result["prompt_policy"], ANSWER_ONLY_PROMPT_POLICY)
                    identity = json.loads((output / "manifest.json").read_text())["identity"]
                    self.assertEqual(identity["prompt_preparation"],
                                     {"policy": ANSWER_ONLY_PROMPT_POLICY,
                                      "suffix": "\n\n" + IDENTIFICATION_OUTPUT_INSTRUCTION})
                    if live:
                        rows = read_jsonl(output / "predictions.jsonl")
                        self.assertEqual([row["prompt_sha256"] for row in rows],
                                         [sha256(text.encode("utf-8")) for text in expected])
        self.assertEqual(originals, {path: path.read_bytes() for path in originals})

    def test_answer_only_policy_preserves_boolean_run_identity(self):
        for fmt in ("symbolic", "natural-language"):
            manifests = []
            for policy in (DEFAULT_PROMPT_POLICY, ANSWER_ONLY_PROMPT_POLICY):
                output = self.output.parent / f"{fmt}-{policy}"
                self.execute(data=fmt, output=output, prompt_policy=policy)
                manifests.append(json.loads((output / "manifest.json").read_text()))
            self.assertEqual(manifests[0]["identity"], manifests[1]["identity"])
            self.assertEqual(manifests[0]["fingerprint"], manifests[1]["fingerprint"])

    def test_label_only_run_resumes_and_rescores_but_rejects_prompt_changes(self):
        options = {"data": "identification", "prompt_policy": ANSWER_ONLY_PROMPT_POLICY}
        self.execute([response("A"), response("The answer is AF")], **options)
        before = {path.name: path.read_bytes() for path in self.output.iterdir()}
        self.key_check.reset_mock()
        self.model_call.reset_mock()
        with self.assertRaisesRegex(ValueError, "prompt policy changed"):
            self.run_eval(data="identification", execute=True, resume=True)
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.output.iterdir()})
        report = rescore([self.output], output=self.output.parent / "label-only-rescore",
                         repo_root=self.root)
        self.assertEqual(report["runs"][0]["prompt_policy"], ANSWER_ONLY_PROMPT_POLICY)
        self.assertEqual(report["runs"][0]["original"]["overall"]["invalid"], 1)
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.output.iterdir()})
        result = self.run_eval(execute=True, resume=True, **options)
        self.assertTrue(result["complete"])
        self.assertEqual(result["overall"]["invalid"], 1)
        self.key_check.assert_not_called()
        self.model_call.assert_not_called()

    def test_legacy_boolean_run_requires_original_policy_and_remains_rescorable(self):
        self.execute(prompt_policy="original")
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertNotIn("prompt_preparation", manifest["identity"])
        rows = read_jsonl(self.output / "predictions.jsonl")
        self.assertEqual([row["prompt_sha256"] for row in rows],
                         [sha256(row["prompt"].encode("utf-8")) for row in self.inputs])
        original_files = {path.name: path.read_bytes() for path in self.output.iterdir()}
        self.key_check.reset_mock()
        self.model_call.reset_mock()
        with self.assertRaisesRegex(ValueError, "--prompt-policy original"):
            self.run_eval(execute=True, resume=True)
        self.assertEqual(original_files, {path.name: path.read_bytes() for path in self.output.iterdir()})
        result = self.run_eval(execute=True, resume=True, prompt_policy="original")
        self.assertTrue(result["complete"])
        report = rescore([self.output], output=self.output.parent / "rescore-legacy", repo_root=self.root)
        self.assertEqual(report["runs"][0]["prompt_policy"], "original")
        self.assertEqual(report["runs"][0]["original"]["overall"]["correct"], 4)
        self.key_check.assert_not_called()
        self.model_call.assert_not_called()

    def test_new_prompt_run_rejects_original_policy_on_resume(self):
        self.execute()
        self.key_check.reset_mock()
        self.model_call.reset_mock()
        with self.assertRaisesRegex(ValueError, "prompt policy changed"):
            self.run_eval(execute=True, resume=True, prompt_policy="original")
        self.key_check.assert_not_called()
        self.model_call.assert_not_called()

    def test_unknown_prompt_policy_rejected_before_credentials(self):
        self.assert_preflight_failure(prompt_policy="unknown-policy")

    def test_rescore_rejects_modified_prompt_instruction(self):
        from evals.run import _json_bytes

        self.execute()
        path = self.output / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["identity"]["prompt_preparation"]["suffix"] = "Different instruction"
        manifest["fingerprint"] = sha256(_json_bytes(manifest["identity"]))
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "prompt_preparation"):
            rescore([self.output], output=self.output.parent / "rescore-changed", repo_root=self.root)

    def test_bold_scoring_preserves_payloads_and_rejects_incomplete_answers(self):
        answers = [response("**Yes**"), response("__No__"),
                   response("**Yes**", complete=False), response("Explanation: **No**")]
        strict = self.execute(answers)
        payloads = self.model_call.call_args_list
        self.model_call.reset_mock()
        bold_output = self.output.parent / "bold"
        bold = self.execute(answers, output=bold_output, answer_format="plain-or-bold")
        self.assertEqual(self.model_call.call_args_list, payloads)
        self.assertEqual(strict["overall"]["invalid"], 4)
        self.assertEqual(bold["overall"]["invalid"], 2)
        self.assertEqual(bold["overall"]["correct"], 2)
        self.assertNotEqual(strict["parser"], bold["parser"])
        self.assertNotEqual(strict["fingerprint"], bold["fingerprint"])
        self.key_check.reset_mock()
        self.model_call.reset_mock()
        self.assert_preflight_failure(output=bold_output, resume=True)
        resumed = self.run_eval(output=bold_output, execute=True, resume=True, answer_format="plain-or-bold")
        self.assertTrue(resumed["complete"])
        self.model_call.assert_not_called()

    def test_offline_rescore_keeps_original_files_and_prose_invalid(self):
        self.execute([response("**Yes**"), response("__No__"),
                      response("**Yes**", complete=False), response("Explanation: **No**")])
        originals = {path.name: path.read_bytes() for path in self.output.iterdir()}
        self.key_check.reset_mock()
        self.model_call.reset_mock()
        report = rescore([self.output], output=self.output.parent / "rescored", repo_root=self.root)
        run = report["runs"][0]
        self.assertEqual(report["api_calls"], 0)
        self.assertEqual(run["original"]["overall"]["invalid"], 4)
        self.assertEqual(run["rescored"]["overall"]["invalid"], 2)
        self.assertEqual(run["rescored"]["overall"]["correct"], 2)
        self.assertEqual(len(run["changed_rows"]), 2)
        self.assertEqual(originals, {path.name: path.read_bytes() for path in self.output.iterdir()})
        self.key_check.assert_not_called()
        self.model_call.assert_not_called()

    def test_rescore_identification_and_partial_run(self):
        self.execute([response("**A**"), ProviderError("offline timeout", uncertain=True)], data="identification")
        report = rescore([self.output], output=self.output.parent / "rescored", repo_root=self.root)["runs"][0]
        self.assertFalse(report["complete"])
        self.assertEqual(report["pending_rows"], 1)
        self.assertEqual(report["rescored"]["overall"]["correct"], 1)
        self.assertEqual(report["rescored"]["overall"]["errors"], 1)
        self.assertEqual(report["parser"], "candidate-label-full-response-bold-v1")
        self.assertTrue((self.output / "inflight.json").exists())

    def test_rescore_uses_latest_attempt_and_tracks_missing_costs(self):
        first = response("**Yes**")
        first["usage"]["cost_in_usd_ticks"] = 100_000_000
        self.execute([first, ProviderError("offline rejection", uncertain=False)])
        result = self.execute([response("**No**"), response("Yes"), response("No")], resume=True)
        self.assertEqual(result["billing"]["reported_cost_usd"], 0.01)
        self.assertEqual(result["billing"]["saved_attempts_without_cost"], 4)
        report = rescore([self.output], output=self.output.parent / "rescored", repo_root=self.root)["runs"][0]
        self.assertTrue(report["complete"])
        self.assertEqual(report["rescored"]["overall"]["total"], 4)
        self.assertEqual(report["rescored"]["overall"]["correct"], 4)
        self.assertEqual(report["billing"], result["billing"])

    def test_rescore_refuses_active_runs_changed_data_and_modified_scores(self):
        self.execute()
        options = {"output": self.output.parent / "rescored", "repo_root": self.root}
        lock = self.output / ".lock"
        lock.touch()
        with self.assertRaisesRegex(ValueError, "locked"):
            rescore([self.output], **options)
        lock.unlink()
        original_data = self.input_file.read_bytes()
        self.input_file.write_bytes(original_data + b"\n")
        with self.assertRaisesRegex(ValueError, "no longer matches"):
            rescore([self.output], **options)
        self.input_file.write_bytes(original_data)
        rows = read_jsonl(self.output / "predictions.jsonl")
        rows[0].update(prediction="No", correct=False)
        write_jsonl(self.output / "predictions.jsonl", rows)
        with self.assertRaisesRegex(ValueError, "Saved scoring"):
            rescore([self.output], **options)
        self.assertFalse(options["output"].exists())

    def test_default_is_a_dry_run_without_credentials_or_predictions(self):
        result = self.run_eval()
        self.key_check.assert_not_called()
        self.model_call.assert_not_called()
        self.assertEqual(Path(result["output_dir"]), self.output.resolve())
        self.assertTrue((self.output / "manifest.json").is_file())
        self.assertTrue((self.output / "dry_run.json").is_file())
        self.assertFalse((self.output / "predictions.jsonl").exists())
        self.assertEqual(result["api_calls"], 0)
        self.assertFalse(result["credentials_read"])
        self.assertEqual(result["payloads_built"], 4)
        self.assertEqual(result["unique_images_decoded"], 4)
        self.assertEqual(result["counts"]["track"], {"photo": 2, "synthetic": 2})

    def test_explicit_dry_run_validates_natural_language_data(self):
        result = self.run_eval(data="natural-language", dry_run=True)
        self.key_check.assert_not_called()
        self.model_call.assert_not_called()
        self.assertTrue((Path(result["output_dir"]) / "dry_run.json").is_file())

    def test_anthropic_disabled_thinking_is_recorded_and_used_for_all_formats(self):
        for fmt in ("symbolic", "natural-language", "identification-symbolic", "identification"):
            with self.subTest(data=fmt):
                self.model_call.reset_mock()
                output = self.output.parent / f"haiku-{fmt}"
                answers = [response("A"), response("AF")] if fmt.startswith("identification") else None
                result = self.execute(answers, model="anthropic/claude-haiku-4-5-20251001", data=fmt,
                                      reasoning_effort="none", output=output)
                self.assertTrue(result["complete"])
                self.assertEqual(result["model"]["reasoning_effort"], "none")
                manifest = json.loads((output / "manifest.json").read_text())
                self.assertEqual(manifest["identity"]["model"]["reasoning_effort"], "none")
                for call in self.model_call.call_args_list:
                    self.assertEqual(call.args[1]["thinking"], {"type": "disabled"})
                    self.assertNotIn("target", json.dumps(call.args[1]))
                self.model_call.reset_mock()
                self.key_check.reset_mock()
                self.assert_preflight_failure(model="anthropic/claude-haiku-4-5-20251001", data=fmt,
                                              output=output, resume=True)
                result = self.run_eval(model="anthropic/claude-haiku-4-5-20251001", data=fmt,
                                       reasoning_effort="none", output=output, execute=True, resume=True)
                self.assertTrue(result["complete"])
                self.model_call.assert_not_called()
                self.key_check.assert_not_called()

    def test_gemini_minimal_thinking_is_recorded_without_claiming_zero_reasoning(self):
        for fmt in ("symbolic", "natural-language", "identification-symbolic", "identification"):
            with self.subTest(data=fmt):
                self.model_call.reset_mock()
                output = self.output.parent / f"gemini-minimal-{fmt}"
                texts = ["A", "AF"] if fmt.startswith("identification") else ["Yes", "No", "Yes", "No"]
                answers = [response(text) for text in texts]
                for answer in answers:
                    answer["usage"].update(reasoning_tokens=3, output_tokens=4, total_tokens=11)
                result = self.execute(answers, model="gemini/gemini-3.5-flash-lite", data=fmt,
                                      thinking_level="minimal", output=output)
                self.assertTrue(result["complete"])
                self.assertEqual(result["model"]["thinking_level"], "minimal")
                self.assertEqual(result["usage"]["reasoning_tokens"], 3 * len(answers))
                manifest = json.loads((output / "manifest.json").read_text())
                self.assertEqual(manifest["identity"]["model"]["thinking_level"], "minimal")
                self.assertNotIn("thinking_budget", manifest["identity"]["model"])
                for call in self.model_call.call_args_list:
                    self.assertEqual(call.args[1]["generationConfig"]["thinkingConfig"], {"thinkingLevel": "minimal"})
                    self.assertNotIn("target", json.dumps(call.args[1]))
                self.model_call.reset_mock()
                self.key_check.reset_mock()
                for level in (None, "low"):
                    self.assert_preflight_failure(model="gemini/gemini-3.5-flash-lite", data=fmt,
                                                  thinking_level=level, output=output, resume=True)
                result = self.run_eval(model="gemini/gemini-3.5-flash-lite", data=fmt,
                                       thinking_level="minimal", output=output, execute=True, resume=True)
                self.assertTrue(result["complete"])
                self.model_call.assert_not_called()
                self.key_check.assert_not_called()

    def test_gemma_pilot_all_formats_preserve_thinking_choice_on_resume(self):
        for fmt in ("symbolic", "natural-language", "identification-symbolic", "identification"):
            with self.subTest(format=fmt):
                output = self.output.parent / f"gemma-{fmt}"
                options = dict(model="gemini/gemma-4-26b-a4b-it", data=fmt, output=output,
                               thinking_level="minimal", max_output_tokens=2048, limit=1,
                               answer_format="plain-or-bold", prompt_policy="answer-only-v1")
                self.model_call.reset_mock()
                report = self.execute([response("A" if fmt.startswith("identification") else "Yes")],
                                      **options)
                self.assertTrue(report["complete"])
                self.assertEqual(report["overall"]["correct"], 1)
                payload = self.model_call.call_args.args[1]
                self.assertEqual(payload["generationConfig"]["thinkingConfig"],
                                 {"thinkingLevel": "minimal"})
                suffix = (IDENTIFICATION_OUTPUT_INSTRUCTION if fmt.startswith("identification")
                          else BOOLEAN_OUTPUT_INSTRUCTION)
                self.assertTrue(payload["contents"][0]["parts"][0]["text"].endswith(suffix))
                manifest = json.loads((output / "manifest.json").read_text())
                self.assertEqual(manifest["identity"]["model"]["thinking_level"], "minimal")
                self.assertEqual(report["model"]["api_key_env"], "GEMINI_API_KEY")
                self.model_call.reset_mock()
                self.assertTrue(self.run_eval(execute=True, resume=True, **options)["complete"])
                self.model_call.assert_not_called()
                for changes in ({"thinking_level": "high"}, {"thinking_level": None},
                                {"model": "gemini/gemma-4-31b-it"}):
                    with self.assertRaisesRegex(ValueError, "Resume mismatch"):
                        self.run_eval(execute=True, resume=True, **{**options, **changes})

    def test_deepseek_pilot_cli_builds_disabled_thinking_offline(self):
        from evals.__main__ import main

        with patch("builtins.print"):
            code = main(["--model", "deepseek/deepseek-flash", "--reasoning-effort", "none",
                         "--request-interval", "4", "--rate-limit-cooldown", "60",
                         "--image-detail", "high", "--max-output-tokens", "2048",
                         "--answer-format", "plain-or-bold", "--prompt-policy", "answer-only-v1",
                         "--data", "symbolic", "--repo-root", str(self.root),
                         "--output", str(self.output), "--limit", "1"])
        self.assertEqual(code, 0)
        self.model_call.assert_not_called()
        self.key_check.assert_not_called()
        report = json.loads((self.output / "dry_run.json").read_text())
        self.assertEqual(report["model"]["reasoning_effort"], "none")
        self.assertEqual(report["model"]["api_key_env"], "DEEPSEEK_API_KEY")
        preview = json.loads((self.output / "sample_requests.json").read_text())[0]
        self.assertEqual(preview["payload"]["thinking"], {"type": "disabled"})
        self.assertEqual(preview["payload"]["max_tokens"], 2048)

    def test_gemini_level_cli_builds_minimal_thinking_offline(self):
        from evals.__main__ import main

        with patch("builtins.print"):
            code = main(["--model", "gemini/gemini-3.5-flash-lite", "--thinking-level", "minimal",
                         "--repo-root", str(self.root), "--output", str(self.output), "--limit", "1"])
        self.assertEqual(code, 0)
        self.model_call.assert_not_called()
        self.key_check.assert_not_called()
        report = json.loads((self.output / "dry_run.json").read_text())
        self.assertEqual(report["model"]["thinking_level"], "minimal")
        preview = json.loads((self.output / "sample_requests.json").read_text())[0]
        self.assertEqual(preview["payload"]["generationConfig"]["thinkingConfig"], {"thinkingLevel": "minimal"})

    def test_gemini_disabled_thinking_is_recorded_and_used_for_all_formats(self):
        for fmt in ("symbolic", "natural-language", "identification-symbolic", "identification"):
            with self.subTest(data=fmt):
                self.model_call.reset_mock()
                output = self.output.parent / f"gemini-{fmt}"
                answers = [response("A"), response("AF")] if fmt.startswith("identification") else None
                result = self.execute(answers, model="gemini/gemini-2.5-flash", data=fmt,
                                      thinking_budget=0, output=output)
                self.assertTrue(result["complete"])
                self.assertEqual(result["model"]["thinking_budget"], 0)
                manifest = json.loads((output / "manifest.json").read_text())
                self.assertEqual(manifest["identity"]["model"]["thinking_budget"], 0)
                for call in self.model_call.call_args_list:
                    self.assertEqual(call.args[1]["generationConfig"]["thinkingConfig"], {"thinkingBudget": 0})
                    self.assertNotIn("target", json.dumps(call.args[1]))
                self.model_call.reset_mock()
                self.key_check.reset_mock()
                self.assert_preflight_failure(model="gemini/gemini-2.5-flash", data=fmt,
                                              output=output, resume=True)
                self.assert_preflight_failure(model="gemini/gemini-2.5-flash", data=fmt,
                                              output=output, resume=True, thinking_budget=1024)
                result = self.run_eval(model="gemini/gemini-2.5-flash", data=fmt, thinking_budget=0,
                                       output=output, execute=True, resume=True)
                self.assertTrue(result["complete"])
                self.model_call.assert_not_called()
                self.key_check.assert_not_called()

    def test_gemini_budget_cli_builds_disabled_thinking_offline(self):
        from evals.__main__ import main

        with patch("builtins.print"):
            code = main(["--model", "gemini/gemini-2.5-flash", "--thinking-budget", "0",
                         "--repo-root", str(self.root), "--output", str(self.output), "--limit", "1"])
        self.assertEqual(code, 0)
        self.model_call.assert_not_called()
        self.key_check.assert_not_called()
        report = json.loads((self.output / "dry_run.json").read_text())
        self.assertEqual(report["model"]["thinking_budget"], 0)
        preview = json.loads((self.output / "sample_requests.json").read_text())[0]
        self.assertEqual(preview["payload"]["generationConfig"]["thinkingConfig"], {"thinkingBudget": 0})

    def test_custom_relative_and_absolute_data_paths(self):
        for index, data in enumerate(["dataset/hob_vl_v1/model_inputs.jsonl", str(self.input_file)]):
            with self.subTest(data=data):
                result = self.run_eval(data=data, output=self.root / "runs" / str(index))
                self.assertTrue((Path(result["output_dir"]) / "dry_run.json").is_file())
        self.key_check.assert_not_called()
        self.model_call.assert_not_called()

    def test_identification_dry_run_validates_label_targets(self):
        result = self.run_eval(data="identification", output=self.root / "runs" / "ident-dry")
        self.assertEqual(result["selected_rows"], 2)
        self.assertEqual(result["counts"]["track"], {"photo": 2})
        self.key_check.assert_not_called()
        self.model_call.assert_not_called()

    def test_identification_execute_scores_labels_and_prose_is_invalid(self):
        self.model_call.side_effect = [response("AF"), response("The answer is A")]
        result = self.run_eval(data="identification-symbolic", execute=True,
                               output=self.root / "runs" / "ident-exec")
        predictions = read_jsonl(self.root / "runs" / "ident-exec" / "predictions.jsonl")
        self.assertEqual([row["prediction"] for row in predictions], ["AF", None])
        self.assertEqual([row["status"] for row in predictions], ["ok", "invalid"])
        self.assertEqual([row["correct"] for row in predictions], [False, False])
        self.assertEqual(result["overall"]["accuracy"], 0)
        self.assertEqual(result["parser"], "candidate-label-full-response-v1")

    def test_identification_execute_canonicalizes_label_case(self):
        self.model_call.side_effect = [response(" a. "), response("af!")]
        result = self.run_eval(data="identification", execute=True,
                               output=self.root / "runs" / "ident-case")
        predictions = read_jsonl(self.root / "runs" / "ident-case" / "predictions.jsonl")
        self.assertEqual([row["prediction"] for row in predictions], ["A", "AF"])
        self.assertEqual([row["correct"] for row in predictions], [True, True])
        self.assertEqual(result["overall"]["accuracy"], 1.0)

    def test_rejects_identification_target_outside_candidates_before_api(self):
        changed = [dict(row) for row in self.ident_gold]
        changed[0]["target"] = "Z"
        (self.dataset / "photo_identification_1000.json").write_text(
            json.dumps(changed), encoding="utf-8")
        self.assert_preflight_failure(data="identification",
                                      output=self.root / "runs" / "ident-bad-target")

    def test_scores_by_id_and_counts_invalid_and_errors_as_wrong(self):
        self.execute([response("Yes"), response("No"), response("Yes, since the condition holds."),
                      ProviderError("offline simulated provider failure")])
        self.key_check.assert_called_once()
        self.assertEqual(self.model_call.call_count, 4)
        predictions = read_jsonl(self.output / "predictions.jsonl")
        self.assertEqual([row["id"] for row in predictions], [row["id"] for row in self.inputs])
        self.assertEqual([row["target"] for row in predictions], ["Yes", "No", "Yes", "No"])
        self.assertEqual([row["status"] for row in predictions], ["ok", "ok", "invalid", "error"])
        self.assertEqual([row["correct"] for row in predictions], [True, True, False, False])
        summary = json.loads((self.output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["mode"], "execute")
        self.assertEqual(summary["overall"]["total"], 4)
        self.assertEqual(summary["overall"]["correct"], 2)
        self.assertEqual(summary["overall"]["accuracy"], 0.5)
        self.assertEqual(summary["overall"]["invalid"], 1)
        self.assertEqual(summary["overall"]["errors"], 1)

    def test_incomplete_response_is_not_credited_even_if_text_is_yes(self):
        self.execute([response("Yes", complete=False)], limit=1)
        row = read_jsonl(self.output / "predictions.jsonl")[0]
        self.assertFalse(row["correct"])
        self.assertNotEqual(row["status"], "ok")

    def test_authentication_failure_happens_before_any_model_call(self):
        self.key_check.side_effect = ValueError("missing test credential")
        with self.assertRaisesRegex(ValueError, "missing test credential"):
            self.run_eval(execute=True)
        self.model_call.assert_not_called()

    def test_successful_run_does_not_persist_the_api_key(self):
        self.execute()
        for artifact in self.output.iterdir():
            if artifact.is_file():
                with self.subTest(artifact=artifact.name):
                    self.assertNotIn("offline-test-key", artifact.read_text(encoding="utf-8"))

    def test_provider_receives_only_input_and_no_gold_metadata(self):
        self.execute()
        for call in self.model_call.call_args_list:
            payload = json.dumps(call.args[1])
            self.assertNotIn('"target"', payload)
            self.assertNotIn('"audit"', payload)
            self.assertNotIn("test-cohort", payload)
            self.assertNotIn("formula-", payload)

    def test_first_provider_error_stops_requests_and_resume_retries_the_error(self):
        result = self.execute([response("Yes"), ProviderError("offline rejection", uncertain=False), response("Yes")])
        self.assertEqual(self.model_call.call_count, 2)
        self.assertEqual(result["remaining_rows"], 2)
        self.assertEqual(result["pending_rows"], 1)
        self.assertTrue(result["stopped_on_error"])
        self.assertFalse(result["complete"])
        self.assertEqual(result["overall"]["accuracy"], 0.5)
        self.model_call.reset_mock()
        # The error-state row is re-attempted, then the two unattempted rows.
        resumed = self.execute([response("Yes"), response("Yes"), response("No")], resume=True)
        self.assertEqual(self.model_call.call_count, 3)
        self.assertTrue(resumed["complete"])
        self.assertEqual(resumed["overall"]["accuracy"], 0.75)
        self.assertEqual(resumed["overall"]["errors"], 0)
        self.assertEqual(resumed["superseded_error_rows"], 1)
        predictions = read_jsonl(self.output / "predictions.jsonl")
        self.assertEqual(len(predictions), 5)
        self.assertEqual(predictions[1]["status"], "error")
        self.assertEqual(predictions[1]["id"], predictions[2]["id"])

    def test_continue_on_timeout_persists_error_and_sends_later_questions(self):
        from evals.providers import call_model
        from evals.tests.test_providers import success

        self.model_call.side_effect = call_model
        with patch.dict("os.environ", {"HOB_VL_TEST_FAKE_KEY": "offline-fake-key"}), \
                patch("evals.providers._post_json", side_effect=[
                    TimeoutError("private timeout details"), success("openai"),
                    success("openai"), success("openai")]) as post:
            result = self.run_eval(execute=True, continue_on_error=True,
                                   api_key_env="HOB_VL_TEST_FAKE_KEY")
        self.assertEqual(post.call_count, 4)
        self.assertTrue(result["all_attempted"])
        self.assertFalse(result["complete"])
        self.assertFalse(result["stopped_on_error"])
        self.assertEqual(result["pending_ids"], ["example-0"])
        self.assertEqual(result["overall"]["total"], 4)
        self.assertEqual(result["overall"]["errors"], 1)
        self.assertEqual(result["continued_errors_this_invocation"], 1)
        self.assertFalse((self.output / "inflight.json").exists())
        rows = read_jsonl(self.output / "predictions.jsonl")
        self.assertEqual([row["id"] for row in rows], [row["id"] for row in self.inputs])
        self.assertTrue(rows[0]["uncertain"])
        self.assertTrue(rows[0]["continued_after_error"])
        self.assertNotIn("private timeout details", rows[0]["error"])

    def test_resume_continue_defers_saved_timeout_then_explicit_retry_only_calls_pending(self):
        self.execute([response("Yes"), ProviderError("offline timeout", transient=True)])
        manifest_before = (self.output / "manifest.json").read_bytes()
        saved_before = (self.output / "predictions.jsonl").read_bytes()
        self.model_call.reset_mock()
        report = self.execute([response("Yes"), response("No")], resume=True, continue_on_error=True)
        self.assertEqual(self.model_call.call_count, 2)
        self.assertEqual(report["deferred_pending_ids_this_invocation"], ["example-1"])
        self.assertEqual(report["pending_rows"], 1)
        self.assertEqual(report["remaining_rows"], 0)
        self.assertTrue(report["all_attempted"])
        self.assertFalse(report["complete"])
        self.assertEqual(report["overall"]["accuracy"], 0.75)
        self.assertEqual((self.output / "manifest.json").read_bytes(), manifest_before)
        self.assertTrue((self.output / "predictions.jsonl").read_bytes().startswith(saved_before))
        self.assertFalse((self.output / "inflight.json").exists())
        self.model_call.reset_mock()
        self.key_check.reset_mock()
        report = self.run_eval(execute=True, resume=True, continue_on_error=True)
        self.assertFalse(report["complete"])
        self.model_call.assert_not_called()
        self.key_check.assert_not_called()
        # Error evidence still requires explicit consent to re-submit.
        with self.assertRaisesRegex(ValueError, "retry-pending"):
            self.run_eval(execute=True, resume=True)
        report = self.execute([response("No")], resume=True, continue_on_error=True, retry_pending=True)
        self.assertEqual(self.model_call.call_count, 1)
        self.assertTrue(report["complete"])
        self.assertEqual(report["overall"]["accuracy"], 1.0)
        self.assertEqual(report["superseded_error_rows"], 1)
        self.assertEqual((self.output / "manifest.json").read_bytes(), manifest_before)

    def test_continue_retains_multiple_uncertain_errors_without_inflight_marker(self):
        failure = ProviderError("offline timeout", transient=True)
        report = self.execute([failure, response("No"), failure, response("No")],
                              continue_on_error=True, max_consecutive_errors=2)
        self.assertTrue(report["all_attempted"])
        self.assertFalse(report["stopped_on_error"])
        self.assertEqual(report["pending_ids"], ["example-0", "example-2"])
        self.assertFalse((self.output / "inflight.json").exists())
        self.model_call.reset_mock()
        with self.assertRaisesRegex(ValueError, "retry-pending"):
            self.run_eval(execute=True, resume=True)
        self.model_call.assert_not_called()
        rescored = rescore([self.output], output=self.output.parent / "rescore-pending",
                           repo_root=self.root)["runs"][0]
        self.assertFalse(rescored["complete"])
        self.assertEqual(rescored["pending_rows"], 2)
        self.assertEqual(rescored["rescored"]["overall"]["total"], 4)
        self.assertEqual(rescored["rescored"]["overall"]["accuracy"], 0.5)
        report = self.execute([response("Yes"), response("Yes")], resume=True,
                              continue_on_error=True, retry_pending=True)
        self.assertEqual(self.model_call.call_count, 2)
        self.assertTrue(report["complete"])

    def test_continue_stops_at_consecutive_error_limit(self):
        failure = ProviderError("offline timeout", transient=True)
        report = self.execute([failure, failure, failure, response("No")], continue_on_error=True)
        self.assertEqual(self.model_call.call_count, 3)
        self.assertTrue(report["stopped_on_error"])
        self.assertEqual(report["stop_reason"], "consecutive_error_limit")
        self.assertFalse(report["all_attempted"])
        self.assertEqual(report["pending_rows"], 3)
        self.assertEqual(report["remaining_rows"], 1)
        self.assertTrue((self.output / "inflight.json").exists())

    def test_continue_still_stops_on_auth_configuration_and_unexpected_errors(self):
        failures = [ProviderError("offline HTTP 401", uncertain=False, status_code=401),
                    ProviderError("offline HTTP 400", uncertain=False, status_code=400),
                    ValueError("private error body")]
        for index, failure in enumerate(failures):
            self.model_call.reset_mock()
            output = self.output.parent / f"fatal-{index}"
            report = self.execute([failure, response("No")], output=output, continue_on_error=True)
            self.assertEqual(self.model_call.call_count, 1)
            self.assertTrue(report["stopped_on_error"])
            self.assertEqual(report["stop_reason"], "non_transient_error")
            rows = read_jsonl(output / "predictions.jsonl")
            self.assertFalse(rows[0]["transient"])
            self.assertNotIn("private error body", rows[0]["error"])

    def test_continue_does_not_overwrite_an_unsaved_interrupted_request(self):
        failure = ProviderError("offline timeout", transient=True)
        with self.assertRaises(KeyboardInterrupt):
            self.execute([failure, KeyboardInterrupt()], continue_on_error=True)
        marker_before = (self.output / "inflight.json").read_bytes()
        saved_before = (self.output / "predictions.jsonl").read_bytes()
        self.model_call.reset_mock()
        with self.assertRaisesRegex(ValueError, "no saved uncertain error record"):
            self.run_eval(execute=True, resume=True, continue_on_error=True)
        self.model_call.assert_not_called()
        self.assertEqual((self.output / "inflight.json").read_bytes(), marker_before)
        self.assertEqual((self.output / "predictions.jsonl").read_bytes(), saved_before)
        result = self.execute(resume=True, continue_on_error=True, retry_pending=True)
        self.assertTrue(result["complete"])
        self.assertEqual(self.model_call.call_count, 4)

    def test_continue_policy_validation_before_credentials(self):
        for settings in ({"continue_on_error": "yes"}, {"max_consecutive_errors": 0},
                         {"max_consecutive_errors": True}, {"max_consecutive_errors": 1.5}):
            with self.subTest(settings=settings):
                self.assert_preflight_failure(**settings)
        for name in ("request_interval", "rate_limit_cooldown"):
            for value in (-1, float("inf"), float("nan"), True, "4"):
                with self.subTest(name=name, value=value):
                    self.assert_preflight_failure(**{name: value})

    def test_pacing_spaces_request_starts_without_inflating_request_latency(self):
        clock = [100.0]
        starts = []
        def answer(*args):
            starts.append(clock[0])
            clock[0] += 1
            return response("Yes")
        self.model_call.side_effect = answer
        with patch("evals.run.time.monotonic", side_effect=lambda: clock[0]), \
                patch("evals.run.time.sleep", side_effect=lambda delay: clock.__setitem__(0, clock[0] + delay)):
            report = self.run_eval(execute=True, request_interval=4)
        self.assertEqual(starts, [100, 104, 108, 112])
        self.assertTrue(report["complete"])
        self.assertEqual(report["elapsed_seconds_this_invocation"], 13)
        self.assertEqual([r["elapsed_seconds"] for r in read_jsonl(self.output / "predictions.jsonl")], [1]*4)

    def test_rate_limit_waits_honors_header_then_resumes_only_failed_ids(self):
        for header, expected in ((None, 60), (90, 90)):
            with self.subTest(retry_after=header):
                output = self.output.parent / f"cooldown-{header}"
                clock = [100.0]
                starts = []
                failure = ProviderError("offline HTTP 429", uncertain=False, transient=True,
                                        status_code=429, retry_after_seconds=header)
                answers = iter([failure, response("No"), response("Yes"), response("Explanation: No")])
                def answer(*args):
                    starts.append(clock[0])
                    item = next(answers)
                    if isinstance(item, Exception):
                        raise item
                    return item
                self.model_call.side_effect = answer
                with patch("evals.run.time.monotonic", side_effect=lambda: clock[0]), \
                        patch("evals.run.time.sleep", side_effect=lambda delay: clock.__setitem__(0, clock[0] + delay)):
                    report = self.run_eval(execute=True, output=output, continue_on_error=True, request_interval=4)
                self.assertEqual(starts, [100, 100+expected, 104+expected, 108+expected])
                self.assertEqual(report["overall"]["invalid"], 1)
                self.assertEqual(report["pending_rows"], 1)
                old_manifest = (output / "manifest.json").read_bytes()
                old_predictions = (output / "predictions.jsonl").read_bytes()
                self.model_call.reset_mock()
                report = self.execute([response("Yes")], output=output, resume=True, retry_pending=True,
                                      continue_on_error=True, request_interval=7, rate_limit_cooldown=120)
                self.assertTrue(report["complete"])
                self.assertEqual(self.model_call.call_count, 1)
                self.assertEqual(report["overall"]["invalid"], 1)
                self.assertEqual((output / "manifest.json").read_bytes(), old_manifest)
                self.assertTrue((output / "predictions.jsonl").read_bytes().startswith(old_predictions))
                self.assertEqual(report["execution_policy"]["request_interval"], 7)

    def test_interrupt_during_cooldown_preserves_saved_error_without_new_submission(self):
        failure = ProviderError("offline HTTP 429", uncertain=False, transient=True, status_code=429)
        with patch("evals.run.time.sleep", side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):
                self.execute([failure, response("No")], continue_on_error=True)
        self.assertEqual(self.model_call.call_count, 1)
        self.assertFalse((self.output / "inflight.json").exists())
        self.assertFalse((self.output / ".lock").exists())
        self.assertEqual(len(read_jsonl(self.output / "predictions.jsonl")), 1)

    def test_deepseek_nonobject_response_continues_and_remains_pending(self):
        from unittest.mock import MagicMock
        from evals.providers import call_model
        from evals.tests.test_providers import success
        opener = MagicMock()
        opener.open.return_value.__enter__.return_value.read.side_effect = [
            b"null", *[json.dumps(success("deepseek")).encode() for _ in range(3)]]
        self.model_call.side_effect = call_model
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "offline-fake-key"}), \
                patch("evals.providers.request.build_opener", return_value=opener):
            report = self.run_eval(execute=True, model="deepseek/deepseek-flash",
                                   reasoning_effort="none", continue_on_error=True)
        self.assertTrue(report["all_attempted"])
        self.assertFalse(report["stopped_on_error"])
        self.assertEqual(report["pending_ids"], ["example-0"])
        self.assertEqual(opener.open.call_count, 4)
        self.assertFalse((self.output / "inflight.json").exists())

    def test_cli_distinguishes_pending_pass_from_complete_and_aborted_runs(self):
        from evals.__main__ import main

        args = ["--model", "openai/test-model", "--repo-root", str(self.root),
                "--output", str(self.output), "--execute", "--continue-on-error"]
        self.model_call.side_effect = [ProviderError("offline timeout", transient=True),
                                      response("No"), response("Yes"), response("No")]
        with patch("builtins.print"):
            self.assertEqual(main(args), 3)
            self.model_call.side_effect = [response("Yes")]
            self.assertEqual(main(args + ["--resume", "--retry-pending"]), 0)
            args[args.index(str(self.output))] = str(self.output.parent / "aborted")
            self.model_call.side_effect = ProviderError("offline authentication failure", uncertain=False)
            self.assertEqual(main(args), 1)

    def test_timeout_requires_explicit_retry_and_retains_uncertainty(self):
        from evals.providers import call_model
        self.model_call.side_effect = call_model
        with patch.dict("os.environ", {"HOB_VL_TEST_FAKE_KEY": "offline-fake-key"}), \
                patch("evals.providers._post_json", side_effect=TimeoutError("offline timeout")):
            self.run_eval(execute=True, limit=1, api_key_env="HOB_VL_TEST_FAKE_KEY")
        row = read_jsonl(self.output / "predictions.jsonl")[0]
        self.assertTrue(row["uncertain"])
        self.assertTrue((self.output / "inflight.json").exists())
        self.model_call.reset_mock()
        with self.assertRaisesRegex(ValueError, "retry-pending"):
            self.run_eval(execute=True, limit=1, api_key_env="HOB_VL_TEST_FAKE_KEY", resume=True)
        self.model_call.assert_not_called()
        self.model_call.side_effect = [response("Yes")]
        report = self.run_eval(execute=True, limit=1, api_key_env="HOB_VL_TEST_FAKE_KEY",
                               resume=True, retry_pending=True)
        self.assertTrue(report["complete"])
        self.assertFalse((self.output / "inflight.json").exists())
        self.assertEqual(len(read_jsonl(self.output / "predictions.jsonl")), 2)

    def test_legacy_error_without_uncertainty_field_requires_explicit_retry(self):
        self.execute([ProviderError("offline timeout")], limit=1)
        path = self.output / "predictions.jsonl"
        rows = read_jsonl(path)
        rows[0].pop("uncertain")
        write_jsonl(path, rows)
        (self.output / "inflight.json").unlink()
        self.model_call.reset_mock()
        with self.assertRaisesRegex(ValueError, "retry-pending"):
            self.execute(limit=1, resume=True)
        self.model_call.assert_not_called()

    def test_interrupted_inflight_request_requires_explicit_retry(self):
        with self.assertRaises(KeyboardInterrupt):
            self.execute([response("Yes"), KeyboardInterrupt()])
        inflight = json.loads((self.output / "inflight.json").read_text(encoding="utf-8"))
        self.assertEqual(inflight["id"], "example-1")
        self.assertEqual(len(read_jsonl(self.output / "predictions.jsonl")), 1)
        self.model_call.reset_mock()
        with self.assertRaisesRegex(ValueError, "retry-pending"):
            self.execute(resume=True)
        self.model_call.assert_not_called()
        resumed = self.execute([response("No"), response("Yes"), response("No")],
                               resume=True, retry_pending=True)
        self.assertEqual(self.model_call.call_count, 3)
        self.assertTrue(resumed["complete"])
        self.assertEqual(resumed["overall"]["accuracy"], 1.0)
        self.assertFalse((self.output / "inflight.json").exists())

    def test_stale_inflight_marker_is_dropped_without_rebilling(self):
        self.execute()
        saved = read_jsonl(self.output / "predictions.jsonl")[0]
        keys = ("id", "target", "metadata", "image_path",
                "prepared_image_sha256", "prompt_sha256")
        (self.output / "inflight.json").write_text(
            json.dumps({key: saved[key] for key in keys}), encoding="utf-8")
        self.model_call.reset_mock()
        result = self.execute(resume=True)
        self.model_call.assert_not_called()
        self.assertTrue(result["complete"])
        self.assertFalse((self.output / "inflight.json").exists())

    def test_filter_offset_and_limit_select_expected_rows(self):
        self.execute([response("No")], track="photo", offset=1, limit=1)
        predictions = read_jsonl(self.output / "predictions.jsonl")
        self.assertEqual([row["id"] for row in predictions], ["example-3"])
        self.assertTrue(predictions[0]["correct"])

    def test_seed_reproduces_the_same_order_and_subset(self):
        orders = []
        for index in range(2):
            output = self.root / "runs" / f"seed-{index}"
            self.execute([response("Yes"), response("Yes")], output=output, seed=17, limit=2)
            orders.append([row["id"] for row in read_jsonl(output / "predictions.jsonl")])
        self.assertEqual(orders[0], orders[1])
        self.assertEqual(len(set(orders[0])), 2)

    def test_image_preparation_records_actual_resize(self):
        self.run_eval(max_image_edge=1)
        manifest = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
        for prepared in manifest["identity"]["images"].values():
            self.assertEqual(prepared["original_dimensions"], [2, 2])
            self.assertEqual(prepared["prepared_dimensions"], [1, 1])
            self.assertTrue(prepared["transformed"])
            self.assertNotEqual(prepared["original_sha256"], prepared["prepared_sha256"])
        self.model_call.assert_not_called()

    def test_existing_output_requires_explicit_resume(self):
        self.run_eval()
        self.assert_preflight_failure()

    def test_published_archive_rejects_resume_without_writing_or_authentication(self):
        self.execute()
        path = self.output / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["archive"] = {"read_only": True}
        path.write_text(json.dumps(manifest))
        before = {p.name: p.read_bytes() for p in self.output.iterdir() if p.is_file()}
        self.key_check.reset_mock()
        self.model_call.reset_mock()
        with self.assertRaisesRegex(ValueError, "Published result archives cannot be resumed"):
            self.run_eval(execute=True, resume=True, retry_pending=True)
        self.key_check.assert_not_called()
        self.model_call.assert_not_called()
        after = {p.name: p.read_bytes() for p in self.output.iterdir() if p.is_file()}
        self.assertEqual(before, after)

    def test_completed_resume_makes_no_additional_model_calls(self):
        self.execute()
        original = (self.output / "predictions.jsonl").read_bytes()
        self.model_call.reset_mock()
        self.model_call.side_effect = AssertionError("Already-scored IDs must be skipped")
        self.run_eval(execute=True, resume=True)
        self.model_call.assert_not_called()
        self.assertEqual((self.output / "predictions.jsonl").read_bytes(), original)

    def test_partial_resume_only_scores_missing_ids(self):
        self.execute()
        write_jsonl(self.output / "predictions.jsonl", read_jsonl(self.output / "predictions.jsonl")[:2])
        self.model_call.reset_mock()
        self.execute([response("Yes"), response("No")], resume=True)
        self.assertEqual(self.model_call.call_count, 2)
        rows = read_jsonl(self.output / "predictions.jsonl")
        self.assertEqual([row["id"] for row in rows], [row["id"] for row in self.inputs])
        summary = json.loads((self.output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["overall"]["total"], 4)
        self.assertEqual(summary["overall"]["accuracy"], 1.0)

    def test_resume_reads_saved_ids_after_acquiring_output_lock(self):
        self.execute()
        completed = read_jsonl(self.output / "predictions.jsonl")
        write_jsonl(self.output / "predictions.jsonl", completed[:2])
        self.model_call.reset_mock()
        self.model_call.side_effect = AssertionError("A completed concurrent request must not be repeated")
        actual_open = os.open

        def acquire_after_previous_writer_finishes(path, flags, *args, **kwargs):
            # Simulate the previous writer finishing immediately before we acquire its lock.
            if Path(path).name == ".lock":
                write_jsonl(self.output / "predictions.jsonl", completed)
            return actual_open(path, flags, *args, **kwargs)

        with patch("evals.run.os.open", side_effect=acquire_after_previous_writer_finishes):
            self.run_eval(execute=True, resume=True)
        self.model_call.assert_not_called()
        self.assertEqual(len(read_jsonl(self.output / "predictions.jsonl")), 4)

    def test_resume_rejects_duplicate_saved_prediction_ids(self):
        self.execute()
        rows = read_jsonl(self.output / "predictions.jsonl")
        write_jsonl(self.output / "predictions.jsonl", rows + [rows[0]])
        self.model_call.reset_mock()
        self.key_check.reset_mock()
        self.assert_preflight_failure(resume=True)

    def test_resume_rejects_saved_scores_that_disagree_with_response(self):
        self.execute()
        rows = read_jsonl(self.output / "predictions.jsonl")
        rows[0]["correct"] = False
        write_jsonl(self.output / "predictions.jsonl", rows)
        self.model_call.reset_mock()
        self.key_check.reset_mock()
        self.assert_preflight_failure(resume=True)

    def test_resume_rejects_saved_prompt_or_image_evidence_mismatch(self):
        self.execute()
        rows = read_jsonl(self.output / "predictions.jsonl")
        self.model_call.reset_mock()
        self.key_check.reset_mock()
        for field in ["prompt_sha256", "prepared_image_sha256", "image_path"]:
            with self.subTest(field=field):
                changed = [{**row} for row in rows]
                changed[0][field] = "different-evidence"
                write_jsonl(self.output / "predictions.jsonl", changed)
                self.assert_preflight_failure(resume=True)

    def test_resume_rejects_unfinished_checkpoint_line_before_api(self):
        self.execute()
        with (self.output / "predictions.jsonl").open("ab") as handle:
            handle.write(b'{"id":"unfinished')
        self.model_call.reset_mock()
        self.key_check.reset_mock()
        self.assert_preflight_failure(resume=True)

    def test_resume_rejects_changed_model_or_generation_config(self):
        self.execute()
        self.model_call.reset_mock()
        self.key_check.reset_mock()
        for config in [{"model": "openai/different-model"}, {"max_output_tokens": 20}, {"temperature": 0.1}]:
            with self.subTest(config=config):
                self.assert_preflight_failure(resume=True, **config)

    def test_resume_rejects_changed_input(self):
        self.execute()
        self.inputs[0]["prompt"] += " Changed."
        write_jsonl(self.input_file, self.inputs)
        self.model_call.reset_mock()
        self.key_check.reset_mock()
        self.assert_preflight_failure(resume=True)

    def test_resume_rejects_changed_image_bytes(self):
        self.execute()
        Image.new("RGB", (2, 2), color="white").save(self.root / self.inputs[0]["image_path"])
        self.model_call.reset_mock()
        self.key_check.reset_mock()
        self.assert_preflight_failure(resume=True)

    def test_resume_rejects_changed_gold_answer(self):
        self.execute()
        self.gold[0]["target"] = "No"
        write_jsonl(self.gold_file, self.gold)
        self.model_call.reset_mock()
        self.key_check.reset_mock()
        self.assert_preflight_failure(resume=True)

    def test_rejects_missing_gold_id_before_api(self):
        write_jsonl(self.gold_file, self.gold[1:])
        self.assert_preflight_failure()

    def test_rejects_duplicate_input_id_before_api(self):
        write_jsonl(self.input_file, self.inputs + [self.inputs[0]])
        self.assert_preflight_failure()

    def test_rejects_duplicate_gold_id_before_api(self):
        write_jsonl(self.gold_file, self.gold + [self.gold[0]])
        self.assert_preflight_failure()

    def test_rejects_missing_image_before_api(self):
        (self.root / self.inputs[-1]["image_path"]).unlink()
        self.assert_preflight_failure()

    def test_rejects_unreadable_image_before_api(self):
        (self.root / self.inputs[-1]["image_path"]).write_bytes(b"not a png")
        self.assert_preflight_failure()

    def test_rejects_image_path_outside_repo_before_api(self):
        outside = Path(self.temp.name) / "outside.png"
        Image.new("RGB", (2, 2)).save(outside)
        self.inputs[0]["image_path"] = "../outside.png"
        self.gold[0]["input"]["image_path"] = "../outside.png"
        write_jsonl(self.input_file, self.inputs)
        write_jsonl(self.gold_file, self.gold)
        self.assert_preflight_failure()

    def test_rejects_mismatched_input_and_gold_image_before_api(self):
        self.gold[0]["input"]["image_path"] = self.inputs[1]["image_path"]
        write_jsonl(self.gold_file, self.gold)
        self.assert_preflight_failure()

    def test_rejects_invalid_limits_offsets_and_empty_selection_before_api(self):
        for options in [{"limit": 0}, {"limit": -1}, {"offset": -1}, {"offset": 99}]:
            with self.subTest(options=options):
                self.assert_preflight_failure(**options)

    def test_rejects_invalid_target_before_api(self):
        self.gold[0]["target"] = "True"
        write_jsonl(self.gold_file, self.gold)
        self.assert_preflight_failure()

    def test_rejects_malformed_nested_gold_fields_before_api(self):
        for field in ["input", "audit"]:
            for invalid in [None, [], "wrong type"]:
                with self.subTest(field=field, invalid=invalid):
                    changed = [{**row} for row in self.gold]
                    changed[0][field] = invalid
                    write_jsonl(self.gold_file, changed)
                    self.assert_preflight_failure()


if __name__ == "__main__":
    unittest.main()
