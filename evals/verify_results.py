"""Verify the published result archive against the released data, offline.

Checks archive hashes, all saved attempt histories, all final answer parsings,
all prompt/target/image joins, recorded settings and completeness. Does not
regenerate model outputs, contact APIs, or modify the saved archive.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

from .checkpoints import read_predictions
from .data import DEFAULT_ROOT, file_sha256, load_dataset
from .scoring import parse_prediction, summarize


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def verify(root=DEFAULT_ROOT):
    root = Path(root)
    archive = json.loads((root / "results/archive_manifest.json").read_text())
    require(file_sha256(root / "results/configurations.json") == archive["configurations_sha256"], "Settings checksum mismatch")
    configurations = json.loads((root / "results/configurations.json").read_text())
    configurations = {c["configuration"]: c for c in configurations["configurations"]}
    datasets = {fmt: load_dataset(fmt, repo_root=root) for fmt in
                ("symbolic", "natural-language", "identification-symbolic", "identification")}
    source_hashes = {}
    final_count = attempt_count = error_count = 0
    require(len(archive["runs"]) == 36 and len({r["path"] for r in archive["runs"]}) == 36, "Wrong run coverage")
    for entry in archive["runs"]:
        folder = root / entry["path"]
        for filename, expected in entry["files"].items():
            require(file_sha256(folder / filename) == expected, f"Archive hash mismatch: {entry['path']}/{filename}")
        manifest = json.loads((folder / "manifest.json").read_text())
        summary = json.loads((folder / "summary.json").read_text())
        identity = manifest["identity"]
        require(manifest["archive"]["read_only"], "Missing archive marker")
        require(hashlib.sha256(canonical(identity)).hexdigest() == manifest["fingerprint"] == summary["fingerprint"], "Archive identity mismatch")
        require(identity["model"] == summary["model"], "Model mismatch")
        recorded = configurations[entry["configuration"]]
        require(identity["model"]["model"] == recorded["requested_model"] and identity["model"]["base_url"] == recorded["api_root"], "Settings index mismatch")
        require(manifest["created_at"] == recorded["run_manifest_creation_utc"][entry["format"]], "Creation date mismatch")
        dataset = datasets[entry["format"]]
        examples = {e.id: e for e in dataset.examples}
        require(identity["data"]["format"] == entry["format"], "Wrong format")
        for label, path in (("inputs", dataset.inputs_path), ("answers", dataset.answers_path)):
            require(identity["data"][f"{label}_sha256"] == file_sha256(path), f"Wrong {label} hash")
        require(identity["selection"]["ids_sha256"] == hashlib.sha256(canonical(list(examples))).hexdigest(), "Wrong selection")
        for relative, info in identity["images"].items():
            if relative not in source_hashes:
                source_hashes[relative] = file_sha256(root / relative)
            require(source_hashes[relative] == info["original_sha256"], "Released image changed")
        attempts, latest = read_predictions(folder / "predictions.jsonl")
        require(latest.keys() == examples.keys(), "Missing or unexpected IDs")
        require(len(attempts) == entry["saved_attempts"] and len(latest) == entry["final_responses"], "Archive count mismatch")
        policy = "plain-or-bold" if "bold" in identity["parser"] else "plain"
        suffix = identity.get("prompt_preparation", {}).get("suffix", "")
        for row in attempts:
            example = examples[row["id"]]
            require(row["target"] == example.target and row["image_path"] == example.image_path, "Dataset evidence mismatch")
            require(row["metadata"] == example.metadata, "Annotation metadata mismatch")
            prompt_hash = hashlib.sha256((example.prompt + suffix).encode()).hexdigest()
            require(row["prompt_sha256"] == prompt_hash, "Prompt changed during packaging")
            require(row["prepared_image_sha256"] == identity["images"][row["image_path"]]["prepared_sha256"], "Prepared-image evidence mismatch")
        for row in latest.values():
            require(row["status"] != "error", "Unresolved API error")
            parsed = parse_prediction(row["response"], examples[row["id"]], answer_format=policy) if row["complete"] else None
            require(parsed == row["prediction"] and row["correct"] == (parsed is not None and parsed == row["target"]), "Saved answer score mismatch")
        computed = summarize(list(latest.values()), dataset.task, answer_format=policy)
        for key in ("overall", "equivalent_pairs", "by_track", "by_cohort", "by_variant", "by_target", "by_scene_family"):
            require(computed[key] == summary[key], f"Stale summary: {key}")
        require(summary["complete"] and not summary["pending_rows"] and not summary["remaining_rows"], "Incomplete run")
        final_count += len(latest)
        attempt_count += len(attempts)
        errors = sum(r["status"] == "error" for r in attempts)
        require(errors == entry["historical_api_errors"], "Error-history mismatch")
        error_count += errors
        print(f"PASS: {entry['configuration']} / {entry['format']}", flush=True)
    require(final_count == archive["final_responses"] == 126000, "Wrong total responses")
    require(attempt_count == archive["saved_attempts"] and error_count == archive["historical_api_errors"], "Wrong history totals")
    result = dict(status="passed", completed_runs=36, final_responses=final_count,
                  saved_attempts=attempt_count, historical_api_errors=error_count, model_api_calls=0)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    verify()
