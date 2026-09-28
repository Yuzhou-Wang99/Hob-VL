"""Shared validation of append-only prediction histories, without API access."""

from pathlib import Path

from .data import read_jsonl


EVIDENCE_FIELDS = ("target", "metadata", "image_path", "prompt_sha256", "prepared_image_sha256")


def read_predictions(path: Path) -> tuple[list[dict], dict[str, dict]]:
    """Keep every attempt, but use only the latest state per ID for scoring.

    Only errors may be superseded. A retry must describe the same question,
    target, and image; duplicate settled answers indicate a damaged history.
    """
    if not path.exists():
        return [], {}
    if path.stat().st_size:
        with path.open("rb") as handle:
            handle.seek(-1, 2)
            if handle.read(1) != b"\n":
                raise ValueError(f"{path.name} has an unfinished final line; inspect it before resuming")
    rows, state = [], {}
    for row in read_jsonl(path):
        identifier = row["id"]
        status = row.get("status")
        if (status not in ("ok", "invalid", "error")
                or type(row.get("correct")) is not bool
                or type(row.get("complete")) is not bool
                or not isinstance(row.get("response"), str)
                or not isinstance(row.get("metadata"), dict)
                or any(not isinstance(row.get(key), str) or not row[key]
                       for key in EVIDENCE_FIELDS if key != "metadata")):
            raise ValueError(f"Invalid saved prediction evidence: {identifier}")
        if "uncertain" in row and type(row["uncertain"]) is not bool:
            raise ValueError(f"Invalid saved uncertainty flag: {identifier}")
        if status == "ok":
            if (not row["complete"] or not isinstance(row.get("prediction"), str)
                    or not row["prediction"]
                    or row["correct"] != (row["prediction"] == row["target"])):
                raise ValueError(f"Invalid saved scoring: {identifier}")
        elif row.get("prediction") is not None or row["correct"] or (status == "error" and row["complete"]):
            raise ValueError(f"Invalid saved scoring: {identifier}")
        previous = state.get(identifier)
        if previous is not None:
            if previous["status"] != "error":
                raise ValueError(f"Duplicate prediction for settled id: {identifier}")
            if any(previous[key] != row[key] for key in EVIDENCE_FIELDS):
                raise ValueError(f"Retry evidence changed for id: {identifier}")
        rows.append(row)
        state[identifier] = row
    return rows, state
