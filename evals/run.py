"""A scriptable runner with an explicit boundary between preparation and API use."""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .checkpoints import read_predictions
from .data import (MAX_IMAGE_BYTES, file_sha256, load_dataset, prepare_images,
                   sha256)
from .providers import ProviderConfig, ProviderError, build_request, call_model, check_api_key
from .prompts import (DEFAULT_PROMPT_POLICY, prepare_prompts, prompt_preparation,
                      recorded_prompt_policy)
from .scoring import ANSWER_FORMATS, parser_for_task, parse_prediction, reported_cost, summarize


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _preview(value):
    """Retain request structure and text without persisting large image encodings."""
    if isinstance(value, dict):
        return {key: (f"<base64 image: {len(item)} characters>"
                      if key == "data" and isinstance(item, str) else _preview(item))
                for key, item in value.items()}
    if isinstance(value, list):
        return [_preview(item) for item in value]
    if isinstance(value, str) and value.startswith("data:image/"):
        header, encoded = value.split(",", 1)
        return f"{header},<base64 image: {len(encoded)} characters>"
    return value


def _check_prediction_evidence(row: dict, example, assets: dict) -> None:
    if row.get("target") != example.target or row.get("metadata") != example.metadata:
        raise ValueError(f"Prediction metadata/target mismatch: {row['id']}")
    if (row.get("image_path") != example.image_path
            or row.get("prompt_sha256") != sha256(example.prompt.encode("utf-8"))
            or row.get("prepared_image_sha256") != assets[example.image_path].info["prepared_sha256"]):
        raise ValueError(f"Prediction prompt/image mismatch: {row['id']}")


def _load_predictions(path: Path, examples: list, assets: dict, *,
                      answer_format: str = "plain") -> tuple[list[dict], dict[str, dict]]:
    """Return every saved terminal row plus the latest state per id.

    An id may appear more than once only when every earlier record is an
    ``error`` attempt; the last record per id is its current state. ``ok`` and
    ``invalid`` states are settled and are never re-attempted on resume.
    """
    rows, state = read_predictions(path)
    expected = {row.id: row for row in examples}
    for row in rows:
        identifier = row["id"]
        if identifier not in expected:
            raise ValueError(f"Unexpected prediction id: {identifier}")
        example = expected[identifier]
        _check_prediction_evidence(row, example, assets)
        if row.get("status") not in ("ok", "invalid", "error"):
            raise ValueError(f"Invalid saved prediction status: {identifier}")
        prediction = (parse_prediction(row.get("response"), example, answer_format=answer_format)
                      if row.get("complete") is True else None)
        status = "error" if row["status"] == "error" else ("ok" if prediction else "invalid")
        if status == "error":
            prediction = None
        correct = prediction == example.target
        if (row.get("prediction") != prediction or row.get("correct") is not correct
                or row["status"] != status):
            raise ValueError(f"Saved scoring does not match response: {identifier}")
    return rows, state


def _load_inflight(output_dir: Path, expected: dict, assets: dict) -> dict | None:
    """Evidence for a request interrupted after submission but before saving.

    Written before each request and removed after its record is persisted, so a
    leftover file means the previous run stopped mid-request with an uncertain
    billable outcome. A stale marker whose answer was saved is removed silently.
    """
    path = output_dir / "inflight.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    identifier = record.get("id")
    if identifier not in expected:
        raise ValueError(f"inflight.json names an unexpected id: {identifier}")
    _check_prediction_evidence(record, expected[identifier], assets)
    return record


def evaluate(
    *,
    model: str,
    data: str = "symbolic",
    output: str | Path | None = None,
    repo_root: str | Path | None = None,
    answers: str | Path | None = None,
    execute: bool = False,
    dry_run: bool = False,
    limit: int | None = None,
    offset: int = 0,
    track: str | None = None,
    seed: int | None = None,
    resume: bool = False,
    retry_pending: bool = False,
    continue_on_error: bool = False,
    max_consecutive_errors: int = 3,
    request_interval: float = 0,
    rate_limit_cooldown: float = 60,
    api_key_env: str | None = None,
    base_url: str | None = None,
    max_output_tokens: int = 4096,
    temperature: float | None = None,
    timeout: float = 120,
    retries: int = 0,
    max_image_edge: int = 2048,
    reasoning_effort: str | None = None,
    image_detail: str | None = None,
    thinking_budget: int | None = None,
    thinking_level: str | None = None,
    thinking_mode: str | None = None,
    answer_format: str = "plain",
    prompt_policy: str = DEFAULT_PROMPT_POLICY,
) -> dict:
    """Evaluate ``provider/model`` on a format alias or model-input JSONL path.

    Without ``execute=True``, fully validate selected images and construct every
    payload offline. No credentials are read and no provider is contacted.
    Real execution is sequential, saves each response immediately, and stops on
    the first request exception by default. ``continue_on_error`` moves past
    transient failures, up to ``max_consecutive_errors`` in a row, and defers
    saved errors on resume unless ``retry_pending`` is given. Settled IDs are
    always skipped. Resubmitting uncertain requests requires ``retry_pending``.
    Resume requires identical data/model configuration; continuation controls
    are local execution policy and may change without changing requests.
    """
    if execute and dry_run:
        raise ValueError("execute and dry_run are mutually exclusive")
    if answer_format not in ANSWER_FORMATS:
        raise ValueError("answer_format must be plain or plain-or-bold")
    if type(continue_on_error) is not bool:
        raise ValueError("continue_on_error must be a boolean")
    if type(max_consecutive_errors) is not int or max_consecutive_errors < 1:
        raise ValueError("max_consecutive_errors must be a positive integer")
    for name, value in (("request_interval", request_interval), ("rate_limit_cooldown", rate_limit_cooldown)):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be a finite nonnegative number of seconds")
    if resume and output is None:
        raise ValueError("resume requires an explicit output directory")
    if resume and not execute:
        raise ValueError("resume applies only to execute runs")
    if not isinstance(model, str) or "/" not in model:
        raise ValueError("model must be provider/model, e.g. openai/your-model-id")
    provider, model_name = model.split("/", 1)
    config = ProviderConfig(provider=provider, model=model_name, api_key_env=api_key_env,
                            base_url=base_url, max_output_tokens=max_output_tokens,
                            temperature=temperature, timeout=timeout, retries=retries,
                            reasoning_effort=reasoning_effort, image_detail=image_detail,
                            thinking_budget=thinking_budget, thinking_level=thinking_level,
                            thinking_mode=thinking_mode)
    config.validate()
    if type(max_image_edge) is not int or max_image_edge < 1:
        raise ValueError("max_image_edge must be a positive integer")
    # Input and image preflight is intentionally before authentication or any API call.
    dataset = load_dataset(data, repo_root=repo_root, answers=answers, limit=limit,
                           offset=offset, track=track, seed=seed)
    preparation = prompt_preparation(dataset.task, prompt_policy)
    dataset = prepare_prompts(dataset, prompt_policy)
    # Record the effective task policy. Unmodified prompts omit preparation,
    # preserving legacy fingerprints and default identification behavior.
    prompt_policy = preparation["policy"] if preparation else "original"
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        label = re.sub(r"[^a-zA-Z0-9_.-]+", "-", f"{model}-{dataset.format}")[:100]
        output_dir = dataset.root / "evals/runs" / f"{stamp}-{label}-{uuid.uuid4().hex[:8]}"
    else:
        # Output paths follow normal CLI conventions; data paths follow repo root.
        output_dir = Path(output).resolve()
    if output_dir.exists() and not resume:
        raise ValueError(f"Output already exists; choose a new directory or --resume: {output_dir}")
    if resume and not (output_dir / "manifest.json").is_file():
        raise ValueError("Cannot resume: manifest.json is missing")
    if resume:
        archived = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        if archived.get("archive", {}).get("read_only"):
            raise ValueError("Published result archives cannot be resumed; choose a new output directory for inference")
    assets = prepare_images(dataset, max_image_edge=max_image_edge)
    import PIL

    identity = {
        "schema_version": 1, "harness_version": __version__, "parser": parser_for_task(dataset.task, answer_format),
        "mode": "execute" if execute else "dry-run", "model": config.public_dict(),
        "data": {"format": dataset.format, "task": dataset.task,
                 "inputs_sha256": file_sha256(dataset.inputs_path),
                 "answers_sha256": file_sha256(dataset.answers_path),
                 "input_rows": dataset.input_count, "answer_rows": dataset.answer_count},
        "selection": {"track": track, "seed": seed, "offset": offset, "limit": limit,
                      "selected_rows": len(dataset.examples),
                      "ids_sha256": sha256(_json_bytes([row.id for row in dataset.examples]))},
        "image_preparation": {"max_image_edge": max_image_edge, "max_bytes": MAX_IMAGE_BYTES,
                              "pillow_version": PIL.__version__},
        "images": {name: asset.info for name, asset in assets.items()},
    }
    if preparation is not None:
        identity["prompt_preparation"] = preparation
    fingerprint = sha256(_json_bytes(identity))
    manifest = {"fingerprint": fingerprint, "identity": identity,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "inputs_path": str(dataset.inputs_path), "answers_path": str(dataset.answers_path)}
    if not resume:
        output_dir.mkdir(parents=True, exist_ok=False)
    lock = output_dir / ".lock"
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError("Output is locked by another run; inspect/remove a stale .lock only after it stops") from exc
    os.close(lock_fd)
    try:
        # Load saved state only while holding the lock; a second resumer must see
        # any responses written by the first one before deciding what to call.
        if resume:
            previous = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            previous_policy = recorded_prompt_policy(previous["identity"])
            if previous_policy != prompt_policy:
                raise ValueError(
                    f"Resume mismatch: prompt policy changed from {previous_policy} to {prompt_policy}. "
                    "Use a new output directory for changed prompts, or retain the saved prompts "
                    f"with --prompt-policy {previous_policy}.")
            if (previous.get("fingerprint") != fingerprint
                    or previous.get("identity") != identity):
                raise ValueError("Resume mismatch: model, data, selection, images, or settings changed")
        rows, state = (_load_predictions(output_dir / "predictions.jsonl", dataset.examples, assets,
                                         answer_format=answer_format)
                       if resume else ([], {}))
        settled = {identifier for identifier, row in state.items() if row["status"] in ("ok", "invalid")}
        deferred = set()
        if resume:
            # Older error records have no uncertainty field. Treat them
            # conservatively rather than silently repeating a possibly billed call.
            uncertain_ids = {identifier for identifier, row in state.items()
                             if row["status"] == "error" and row.get("uncertain", True)}
            inflight = _load_inflight(output_dir, {row.id: row for row in dataset.examples}, assets)
            if inflight is not None:
                if inflight["id"] in settled:
                    # The answer was persisted before the interrupt; the marker is stale.
                    (output_dir / "inflight.json").unlink()
                else:
                    uncertain_ids.add(inflight["id"])
            if continue_on_error and not retry_pending:
                # Every deferred request must already have durable error evidence.
                # A marker without such a record still needs explicit recovery;
                # never overwrite it with the next question and lose the request.
                if inflight is not None and inflight["id"] not in settled:
                    saved = state.get(inflight["id"])
                    if saved is None or not saved.get("uncertain", True):
                        raise ValueError(
                            f"Interrupted request {inflight['id']} has no saved uncertain error record. "
                            "Inspect provider logs, then use --resume --retry-pending to re-attempt.")
                deferred = {identifier for identifier, row in state.items() if row["status"] == "error"}
            if uncertain_ids - deferred and not retry_pending:
                raise ValueError(
                    f"Request outcome is unknown for {', '.join(sorted(uncertain_ids))}. "
                    "Inspect provider logs, then resume with --retry-pending to re-attempt.")
            if deferred:
                print(f"Deferring {len(deferred)} saved API errors; continuing with unattempted IDs. "
                      "Use --resume --retry-pending to retry them later.", file=sys.stderr)
                if inflight is not None and inflight["id"] in deferred:
                    (output_dir / "inflight.json").unlink(missing_ok=True)
        if not resume:
            _write_json(output_dir / "manifest.json", manifest)
        if not execute:
            return _dry_run(dataset, config, assets, output_dir, fingerprint, prompt_policy)
        if len(settled | deferred) < len(dataset.examples):
            check_api_key(config)
        return _execute(dataset, config, assets, output_dir, fingerprint, rows, state, settled,
                        answer_format=answer_format, prompt_policy=prompt_policy,
                        continue_on_error=continue_on_error, max_consecutive_errors=max_consecutive_errors,
                        deferred=deferred, retry_pending=retry_pending,
                        request_interval=request_interval, rate_limit_cooldown=rate_limit_cooldown)
    finally:
        lock.unlink(missing_ok=True)


def _dry_run(dataset, config, assets, output_dir, fingerprint, prompt_policy):
    request_bytes = []
    previews = []
    preview_tracks = set()
    for row in dataset.examples:
        asset = assets[row.image_path]
        payload = build_request(config, row.prompt, asset.data, asset.mime_type)
        request_bytes.append(len(_json_bytes(payload)))
        track = row.metadata.get("track")
        if track not in preview_tracks and len(previews) < 3:
            preview_tracks.add(track)
            previews.append({"id": row.id, "image_path": row.image_path,
                             "image": asset.info, "payload": _preview(payload)})
    counts = {field: dict(sorted(Counter(str(row.metadata.get(field) or "unknown")
                                        for row in dataset.examples).items()))
              for field in ("track", "variant", "cohort")}
    counts["target"] = dict(Counter(row.target for row in dataset.examples))
    report = {
        "mode": "dry-run", "model": config.public_dict(), "format": dataset.format,
        "prompt_policy": prompt_policy,
        "output_dir": str(output_dir), "fingerprint": fingerprint,
        "api_calls": 0, "credentials_read": False,
        "input_rows": dataset.input_count, "answer_rows": dataset.answer_count,
        "selected_rows": len(dataset.examples), "validated_joins": dataset.input_count,
        "unique_images_decoded": len(assets),
        "transformed_images": sum(asset.info["transformed"] for asset in assets.values()),
        "payloads_built": len(request_bytes),
        "request_bytes": {"min": min(request_bytes), "max": max(request_bytes), "sum": sum(request_bytes)},
        "counts": counts,
        "note": "Offline preparation only. No predictions or model accuracy. Byte sizes are not token/cost estimates.",
    }
    _write_json(output_dir / "sample_requests.json", previews)
    _write_json(output_dir / "dry_run.json", report)
    return report


def _execute(dataset, config, assets, output_dir, fingerprint, rows, state, settled, *,
             answer_format="plain", prompt_policy="original", continue_on_error=False,
             max_consecutive_errors=3, deferred=None, retry_pending=False,
             request_interval=0, rate_limit_cooldown=60):
    calls = 0
    stopped = False
    stop_reason = None
    consecutive_errors = 0
    continued_errors = 0
    deferred = deferred or set()
    started = time.monotonic()
    inflight_path = output_dir / "inflight.json"
    next_request_at = 0.0
    try:
        with (output_dir / "predictions.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
            for example in dataset.examples:
                if example.id in settled or example.id in deferred:
                    continue
                asset = assets[example.image_path]
                payload = build_request(config, example.prompt, asset.data, asset.mime_type)
                # Wait before writing a submission marker. An interruption while
                # pacing must not make an unsubmitted question look in flight.
                while (delay := next_request_at - time.monotonic()) > 0:
                    time.sleep(min(delay, 60))
                start = time.monotonic()
                next_request_at = start + request_interval
                record = {
                    "id": example.id, "target": example.target, "metadata": example.metadata,
                    "image_path": example.image_path,
                    "prepared_image_sha256": asset.info["prepared_sha256"],
                    "prompt_sha256": sha256(example.prompt.encode("utf-8")),
                    "response": "", "prediction": None, "correct": False,
                    "status": "error", "complete": False, "usage": {},
                }
                _write_json(inflight_path, {key: record[key] for key in (
                    "id", "target", "metadata", "image_path",
                    "prepared_image_sha256", "prompt_sha256")})
                try:
                    calls += 1
                    response = call_model(config, payload)
                    prediction = (parse_prediction(response["text"], example, answer_format=answer_format)
                                  if response["complete"] is True else None)
                    record.update(response=response["text"], prediction=prediction,
                                  correct=prediction == example.target,
                                  status="ok" if prediction else "invalid",
                                  complete=response["complete"], usage=response.get("usage", {}),
                                  response_id=response.get("response_id"),
                                  finish_reason=response.get("finish_reason"))
                    if isinstance(response.get("reasoning_content"), str):
                        record["reasoning_content"] = response["reasoning_content"]
                    consecutive_errors = 0
                except Exception as exc:
                    # ProviderError is explicitly sanitized; never persist arbitrary exception bodies.
                    record["error"] = str(exc) if isinstance(exc, ProviderError) else type(exc).__name__
                    record["uncertain"] = exc.uncertain if isinstance(exc, ProviderError) else True
                    record["transient"] = isinstance(exc, ProviderError) and exc.transient
                    if isinstance(exc, ProviderError) and exc.status_code is not None:
                        record["http_status"] = exc.status_code
                    if isinstance(exc, ProviderError) and exc.retry_after_seconds is not None:
                        record["retry_after_seconds"] = exc.retry_after_seconds
                    consecutive_errors += 1
                    stopped = (not continue_on_error or not record["transient"]
                               or consecutive_errors >= max_consecutive_errors)
                    record["continued_after_error"] = not stopped
                    if stopped:
                        stop_reason = ("stop_on_error" if not continue_on_error else
                                       "non_transient_error" if not record["transient"] else
                                       "consecutive_error_limit")
                    else:
                        continued_errors += 1
                record["elapsed_seconds"] = round(time.monotonic() - start, 4)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                # When moving on, the flushed error row now preserves uncertainty;
                # inflight.json must describe only a request still awaiting a record.
                if record["status"] != "error" or not record["uncertain"] or not stopped:
                    inflight_path.unlink(missing_ok=True)
                rows.append(record)
                state[example.id] = record
                if record["status"] == "error":
                    action = f"Stopping ({stop_reason})" if stopped else "Continuing to next ID"
                    print(f"API error for {example.id}: {record['error']}. {action}; "
                          "this ID remains pending.", file=sys.stderr)
                if stopped:
                    break
                if record.get("http_status") == 429:
                    cooldown = max(rate_limit_cooldown, record.get("retry_after_seconds", 0))
                    next_request_at = max(next_request_at, time.monotonic() + cooldown)
                    print(f"Rate limited: waiting at least {cooldown:g}s before another request.",
                          file=sys.stderr)
                if len(state) % 100 == 0:
                    print(f"Saved {len(state)}/{len(dataset.examples)} predictions", file=sys.stderr)
    finally:
        # A signal may arrive after fsync but before in-memory bookkeeping.
        rows, state = _load_predictions(output_dir / "predictions.jsonl", dataset.examples, assets,
                                        answer_format=answer_format)
        final = [state[row.id] for row in dataset.examples if row.id in state]
        settled_rows = sum(row["status"] in ("ok", "invalid") for row in final)
        summary = {
            "mode": "execute", "model": config.public_dict(), "format": dataset.format,
            "prompt_policy": prompt_policy,
            "output_dir": str(output_dir), "fingerprint": fingerprint,
            "planned_rows": len(dataset.examples), "saved_rows": len(rows),
            "attempted_rows": len(final), "settled_rows": settled_rows,
            "pending_rows": len(final) - settled_rows,
            "pending_ids": [row["id"] for row in final if row["status"] == "error"],
            "superseded_error_rows": len(rows) - len(final),
            "remaining_rows": len(dataset.examples) - len(final),
            "in_flight_interrupted": inflight_path.exists(),
            "complete": settled_rows == len(dataset.examples),
            "all_attempted": len(final) == len(dataset.examples),
            "stopped_on_error": stopped,
            "stop_reason": stop_reason,
            "execution_policy": {"continue_on_error": continue_on_error,
                                 "max_consecutive_errors": max_consecutive_errors,
                                 "retry_pending": retry_pending,
                                 "request_interval": request_interval,
                                 "rate_limit_cooldown": rate_limit_cooldown},
            "deferred_pending_ids_this_invocation": sorted(deferred),
            "continued_errors_this_invocation": continued_errors,
            "provider_calls_this_invocation": calls,
            "elapsed_seconds_this_invocation": round(time.monotonic() - started, 4),
            **summarize(final, task=dataset.task, answer_format=answer_format),
            "billing": reported_cost(rows),
            "note": "Accuracy reflects the latest attempt per row; invalid/error states count as "
                    "incorrect. Error-state and in-flight rows stay pending for resume. "
                    "Observations share scenes and equivalent pairs; they are not independent.",
        }
        _write_json(output_dir / "summary.json", summary)
    return summary
