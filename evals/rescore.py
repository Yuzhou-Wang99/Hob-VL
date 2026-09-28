"""Rescore saved answers offline into a separate report; never modify a run."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from .data import DEFAULT_ROOT, file_sha256, load_dataset, sha256
from .prompts import prepare_prompts, recorded_prompt_policy
from .run import _json_bytes, _load_predictions, _write_json
from .scoring import (ANSWER_FORMATS, parser_for_task, parse_prediction,
                      reported_cost, summarize)


def _snapshot(source: Path) -> dict:
    if (source / ".lock").exists():
        raise ValueError(f"Run is locked; wait until it stops: {source}")
    return {name: file_sha256(source / name) for name in
            ("manifest.json", "predictions.jsonl")}


def _rescore_run(source: Path, answer_format: str, repo_root) -> dict:
    evidence = _snapshot(source)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    identity = manifest["identity"]
    if (identity.get("schema_version") != 1 or identity.get("mode") != "execute"
            or manifest.get("fingerprint") != sha256(_json_bytes(identity))):
        raise ValueError(f"Invalid execute manifest: {source}")
    task = identity["data"]["task"]
    policies = {parser_for_task(task, policy): policy for policy in ANSWER_FORMATS}
    original_policy = policies.get(identity["parser"])
    if original_policy is None:
        raise ValueError(f"Unsupported source parser: {identity['parser']}")
    selection = identity["selection"]
    dataset = load_dataset(
        manifest["inputs_path"], answers=manifest["answers_path"], repo_root=repo_root,
        **{key: selection[key] for key in ("limit", "offset", "track", "seed")})
    if (dataset.task != task
            or file_sha256(dataset.inputs_path) != identity["data"]["inputs_sha256"]
            or file_sha256(dataset.answers_path) != identity["data"]["answers_sha256"]
            or len(dataset.examples) != selection["selected_rows"]
            or sha256(_json_bytes([row.id for row in dataset.examples])) != selection["ids_sha256"]):
        raise ValueError(f"Source data/selection no longer matches the manifest: {source}")
    prompt_policy = recorded_prompt_policy(identity)
    dataset = prepare_prompts(dataset, prompt_policy)
    # Scoring needs no image decoding. Check saved response evidence against the
    # original image inventory, not a newly prepared image or a model request.
    assets = {name: SimpleNamespace(info=info) for name, info in identity["images"].items()}
    if set(assets) != {row.image_path for row in dataset.examples}:
        raise ValueError(f"Image inventory does not match selected examples: {source}")
    attempts, state = _load_predictions(source / "predictions.jsonl", dataset.examples,
                                       assets, answer_format=original_policy)
    original, rescored, changes = [], [], []
    for example in dataset.examples:
        if example.id not in state:
            continue
        row = state[example.id]
        prediction = (parse_prediction(row["response"], example, answer_format=answer_format)
                      if row["complete"] is True and row["status"] != "error" else None)
        updated = {**row, "prediction": prediction, "correct": prediction == example.target,
                   "status": "error" if row["status"] == "error" else
                             ("ok" if prediction else "invalid")}
        original.append(row)
        rescored.append(updated)
        fields = ("prediction", "correct", "status")
        before, after = ({key: record[key] for key in fields} for record in (row, updated))
        if before != after:
            changes.append({"id": example.id, "response": row["response"], "target": example.target,
                            "before": before, "after": after})
    settled = sum(row["status"] in ("ok", "invalid") for row in original)
    if _snapshot(source) != evidence:
        raise ValueError(f"Source changed during rescoring; try again after it stops: {source}")
    return {
        "source_run": str(source), "source_sha256": evidence,
        "model": identity["model"], "format": identity["data"]["format"],
        "prompt_policy": prompt_policy,
        "source_parser": identity["parser"], "parser": parser_for_task(task, answer_format),
        "planned_rows": len(dataset.examples), "attempted_rows": len(original),
        "settled_rows": settled, "pending_rows": len(original) - settled,
        "remaining_rows": len(dataset.examples) - len(original),
        "complete": settled == len(dataset.examples),
        "original": summarize(original, task=task, answer_format=original_policy),
        "rescored": summarize(rescored, task=task, answer_format=answer_format),
        "billing": reported_cost(attempts), "changed_rows": changes,
    }


def rescore(runs, *, answer_format="plain-or-bold", output=None, repo_root=None) -> dict:
    if answer_format not in ANSWER_FORMATS:
        raise ValueError("answer_format must be plain or plain-or-bold")
    sources = [Path(path).resolve() for path in runs]
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("Provide one or more distinct source run directories")
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = Path(repo_root or DEFAULT_ROOT) / "evals/comparisons" / f"rescore-{stamp}-{uuid.uuid4().hex[:8]}"
    output = Path(output).resolve()
    if output.exists() or any(output.is_relative_to(source) for source in sources):
        raise ValueError("Choose a new output directory outside the source runs")
    reports = [_rescore_run(source, answer_format, repo_root) for source in sources]
    for source, report in zip(sources, reports):
        if _snapshot(source) != report["source_sha256"]:
            raise ValueError(f"Source changed during rescoring: {source}")
    result = {
        "mode": "offline-rescore", "api_calls": 0, "answer_format": answer_format,
        "created_at": datetime.now(timezone.utc).isoformat(), "output_dir": str(output),
        "note": "Original runs are unchanged. Scores cover latest saved attempts, including pending "
                "errors as incorrect; unattempted rows are excluded. Partial runs are not full results. "
                "Image hashes are checked against saved manifests; images are not decoded again.",
        "runs": reports,
    }
    lines = ["# Offline rescoring", "", f"Answer format: `{answer_format}`. No API calls.", "",
             result["note"], "", "| Run / format | Complete | Attempted / planned | Correct before → after | Invalid before → after | Pending errors |",
             "|---|---|---:|---:|---:|---:|"]
    for report in reports:
        old, new = report["original"]["overall"], report["rescored"]["overall"]
        label = f"{Path(report['source_run']).parent.name} / {report['format']}"
        lines.append(f"| {label} | {report['complete']} | {report['attempted_rows']} / {report['planned_rows']} "
                     f"| {old['correct']} → {new['correct']} | {old['invalid']} → {new['invalid']} "
                     f"| {report['pending_rows']} |")
    lines.extend(["", "See report.json for changed answers, source hashes, group metrics, equivalent-pair "
                  "metrics, and provider-reported cost coverage. These reports cannot be resumed as inference runs.", ""])
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "report.json", result)
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="Source run directories containing manifests and predictions")
    parser.add_argument("--answer-format", choices=ANSWER_FORMATS, default="plain-or-bold")
    parser.add_argument("--output", help="New report directory; default: evals/comparisons/rescore-<unique-id>")
    parser.add_argument("--repo-root", help="Dataset repository root; manifest data paths must still exist")
    try:
        result = rescore(**vars(parser.parse_args(argv)))
    except (ValueError, OSError, KeyError) as exc:
        print(f"Rescoring failed: {exc}", file=sys.stderr)
        return 2
    print(f"Offline rescoring complete; no API calls. Report: {result['output_dir']}/report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
