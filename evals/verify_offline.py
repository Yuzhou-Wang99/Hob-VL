"""Reproduce the offline evidence: tests, all four full formats, four fake API runs.

Run ``python -m evals.verify_offline`` from the repository root. Socket connection
attempts are blocked for the entire verification. Model names and responses are
explicit fixtures, never real model results. No real API credentials are read.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import io
import json
import platform
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .data import DEFAULT_ROOT, file_sha256, load_dataset
from .run import evaluate
from .scoring import LABEL_PARSER_VERSION, parse_answer, parse_prediction, summarize


def verify(output: Path) -> dict:
    started = datetime.now(timezone.utc)
    evidence = {"kind": "offline_harness_verification", "started_at": started.isoformat(),
                "python": platform.python_version(), "real_model_calls": 0,
                "real_api_credentials_read": False, "socket_connections_blocked_by_test_guard": True}
    with ExitStack() as stack:
        guards = [stack.enter_context(patch(name, side_effect=AssertionError("Network forbidden in offline verification")))
                  for name in ("socket.create_connection", "socket.socket.connect", "socket.socket.connect_ex")]
        test_output = io.StringIO()
        suite = unittest.defaultTestLoader.discover(str(DEFAULT_ROOT / "evals/tests"), pattern="test_*.py")
        tests = unittest.TextTestRunner(stream=test_output, verbosity=2).run(suite)
        if not tests.wasSuccessful():
            raise AssertionError(test_output.getvalue())
        evidence["unit_tests"] = {"run": tests.testsRun, "failures": len(tests.failures), "errors": len(tests.errors)}
        print(f"Passed {tests.testsRun} offline tests", flush=True)
        runs = DEFAULT_ROOT / "evals/runs"
        runs.mkdir(exist_ok=True)
        # Keep bulky transient manifests/payload previews out of checked-in evidence.
        with tempfile.TemporaryDirectory(prefix="verification-", dir=runs) as work:
            directory = Path(work)
            evidence["full_dry_runs"] = []
            with patch("evals.run.check_api_key", side_effect=AssertionError("Credentials forbidden during dry run")), \
                 patch("evals.run.call_model", side_effect=AssertionError("Model calls forbidden during dry run")):
                for data in ("symbolic", "natural-language"):
                    report = evaluate(model="openai/offline-placeholder", data=data, output=directory / data)
                    assert report["selected_rows"] == report["payloads_built"] == report["validated_joins"] == 6000
                    assert report["unique_images_decoded"] == 1046
                    assert report["counts"]["track"] == {"photo": 2000, "synthetic": 4000}
                    assert report["counts"]["target"] == {"Yes": 3000, "No": 3000}
                    report.pop("output_dir")
                    evidence["full_dry_runs"].append(report)
                    print(f"Verified {data}: 6,000 payloads, 1,046 decoded images, zero API calls", flush=True)
                for data in ("identification", "identification-symbolic"):
                    report = evaluate(model="openai/offline-placeholder", data=data, output=directory / data)
                    assert report["selected_rows"] == report["payloads_built"] == report["validated_joins"] == 1000
                    assert report["unique_images_decoded"] == 46
                    assert report["counts"]["track"] == {"photo": 1000}
                    report.pop("output_dir")
                    evidence["full_dry_runs"].append(report)
                    print(f"Verified {data}: 1,000 payloads, 46 decoded images, zero API calls", flush=True)
            # Independent full scoring check using deliberately fabricated constant responses.
            evidence["scoring_fixtures"] = []
            for data in ("symbolic", "natural-language"):
                dataset = load_dataset(data)
                rows = [{"id": row.id, "target": row.target, "metadata": row.metadata,
                         "prediction": parse_answer("Yes"), "status": "ok", "correct": row.target == "Yes"}
                        for row in dataset.examples]
                result = summarize(rows)
                assert result["overall"]["total"] == 6000 and result["overall"]["accuracy"] == 0.5
                assert result["equivalent_pairs"]["complete_pairs"] == 3000
                assert result["equivalent_pairs"]["agreement"] == 1.0
                assert result["equivalent_pairs"]["both_correct_accuracy"] == 0.5
                evidence["scoring_fixtures"].append({"format": data, "fixture": "fabricated constant Yes, not a model", **result})
            identification = load_dataset("identification")
            expected_a = sum(row.target == "A" for row in identification.examples)
            rows = [{"id": row.id, "target": row.target, "metadata": row.metadata,
                     "prediction": parse_prediction("A", row), "status": "ok",
                     "correct": row.target == "A"}
                    for row in identification.examples]
            result = summarize(rows, task=identification.task)
            assert result["overall"]["total"] == 1000 and result["overall"]["correct"] == expected_a
            assert result["parser"] == LABEL_PARSER_VERSION
            assert result["equivalent_pairs"]["complete_pairs"] == 0
            evidence["scoring_fixtures"].append(
                {"format": "identification", "fixture": "fabricated constant A, not a model", **result})
            native_responses = {
                "openai": {"id": "offline", "status": "completed", "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": "Yes"}]}],
                    "usage": {"input_tokens": 7, "output_tokens": 1}},
                "anthropic": {"id": "offline", "content": [{"type": "text", "text": "Yes"}],
                              "stop_reason": "end_turn", "usage": {"input_tokens": 7, "output_tokens": 1}},
                "gemini": {"responseId": "offline", "candidates": [
                    {"content": {"parts": [{"text": "Yes"}]}, "finishReason": "STOP"}],
                    "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 1, "totalTokenCount": 8}},
                "openai-compatible": {"id": "offline", "choices": [
                    {"message": {"content": "Yes"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 1}},
            }
            selected = load_dataset("symbolic", limit=12, seed=7)
            expected_correct = sum(row.target == "Yes" for row in selected.examples)
            assert {row.metadata["track"] for row in selected.examples} == {"photo", "synthetic"}
            evidence["mocked_provider_runs"] = []
            for provider, fake_response in native_responses.items():
                kwargs = dict(model=f"{provider}/offline-placeholder", output=directory / provider,
                              execute=True, limit=12, seed=7, api_key_env="HOB_VL_OFFLINE_TEST_KEY")
                if provider == "openai-compatible":
                    kwargs["base_url"] = "https://offline.invalid/v1"
                with patch.dict("os.environ", {"HOB_VL_OFFLINE_TEST_KEY": "offline-fake-key"}), \
                     patch("evals.providers._post_json", return_value=fake_response) as fake_http:
                    report = evaluate(**kwargs)
                    assert fake_http.call_count == 12
                    assert report["overall"]["correct"] == expected_correct
                    assert report["overall"]["total"] == 12
                    assert report["overall"]["errors"] == report["overall"]["invalid"] == 0
                    fake_http.reset_mock()
                    resumed = evaluate(**kwargs, resume=True)
                    assert fake_http.call_count == 0
                    assert resumed["overall"] == report["overall"]
                evidence["mocked_provider_runs"].append({
                    "provider": provider, "fixture": "native API JSON fixture, not a model response",
                    "mock_transport_calls": 12, "real_transport_calls": 0,
                    "resume_additional_calls": 0, "overall": report["overall"], "usage": report["usage"],
                })
                print(f"Verified {provider}: 12 mocked requests, scoring, resume without repeated calls", flush=True)
        attempts = sum(guard.call_count for guard in guards)
        if attempts:
            raise AssertionError(f"Offline verification attempted {attempts} socket connections")
        evidence["socket_connection_attempts"] = attempts
    evidence["source_sha256"] = {
        str(path.relative_to(DEFAULT_ROOT)).replace("\\", "/"): file_sha256(path)
        for path in sorted((DEFAULT_ROOT / "evals").rglob("*.py"))
    }
    evidence["finished_at"] = datetime.now(timezone.utc).isoformat()
    evidence["elapsed_seconds"] = round((datetime.now(timezone.utc) - started).total_seconds(), 3)
    evidence["limitations"] = ["No live API/authentication/model-availability validation was performed.",
                               "Fabricated fixture scores prove harness behavior, not model performance.",
                               "Image resizing may affect label legibility; settings and exact byte hashes are recorded."]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".tests.txt").write_text(test_output.getvalue(), encoding="utf-8")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / "evals/evidence/offline_verification.json")
    args = parser.parse_args()
    report = verify(args.output)
    print(f"Evidence saved to {args.output} ({report['elapsed_seconds']} seconds)")


if __name__ == "__main__":
    main()
