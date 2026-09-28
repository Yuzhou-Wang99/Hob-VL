"""Strict ID joins and deterministic image preparation. No network operations."""

from __future__ import annotations

import hashlib
import io
import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Iterator

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path("dataset/hob_vl_v1")
TASKS = ("yes-no", "identification")
FORMATS = {
    "symbolic": {"inputs": DATA_DIR / "model_inputs.jsonl",
                 "answers": DATA_DIR / "dataset.jsonl", "task": "yes-no"},
    "natural-language": {"inputs": DATA_DIR / "model_inputs_natural_language.jsonl",
                         "answers": DATA_DIR / "dataset.jsonl", "task": "yes-no"},
    "identification": {"inputs": DATA_DIR / "model_inputs_photo_identification_1000.jsonl",
                       "answers": DATA_DIR / "photo_identification_1000.json",
                       "task": "identification"},
    "identification-symbolic": {"inputs": DATA_DIR / "model_inputs_photo_identification_1000_symbolic.jsonl",
                                "answers": DATA_DIR / "photo_identification_1000.json",
                                "task": "identification"},
}
MAX_IMAGE_BYTES = 4 * 1024 * 1024


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def resolve_file(path: str | Path, root: Path) -> Path:
    path = Path(path)
    return (path if path.is_absolute() else root / path).resolve()


def _check_row(path: Path, number: int, row: object) -> dict:
    if not isinstance(row, dict):
        raise ValueError(f"{path}:{number}: expected a JSON object")
    if not isinstance(row.get("id"), str) or not row["id"].strip():
        raise ValueError(f"{path}:{number}: id must be a nonempty string")
    return row


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON: {exc.msg}") from exc
            yield _check_row(path, number, row)


def read_answers(path: Path) -> Iterator[dict]:
    """Answer records as JSONL lines or a single JSON list (first byte decides)."""
    with path.open(encoding="utf-8-sig") as handle:
        first = ""
        while True:
            char = handle.read(1)
            if not char or not char.isspace():
                first = char
                break
    if first != "[":
        yield from read_jsonl(path)
        return
    try:
        rows = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc.msg}") from exc
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON list of answer objects")
    for number, row in enumerate(rows, 1):
        yield _check_row(path, number, row)


@dataclass(frozen=True)
class Example:
    id: str
    prompt: str
    image_path: str
    target: str
    metadata: dict
    candidates: tuple | None = None


@dataclass(frozen=True)
class Dataset:
    examples: list[Example]
    root: Path
    inputs_path: Path
    answers_path: Path
    format: str
    task: str
    input_count: int
    answer_count: int


def load_dataset(
    data: str = "symbolic",
    *,
    repo_root: str | Path | None = None,
    answers: str | Path | None = None,
    limit: int | None = None,
    offset: int = 0,
    track: str | None = None,
    seed: int | None = None,
) -> Dataset:
    """Validate the full join, then filter, optionally shuffle, offset, and limit.

    Custom input subsets are supported; extra gold rows are allowed. Paths inside
    each input are always repository-relative, independent of the current cwd.
    Only the prompt and prepared image are ever passed to a provider.
    """
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer")
    if type(offset) is not int or offset < 0:
        raise ValueError("offset must be a nonnegative integer")
    if seed is not None and type(seed) is not int:
        raise ValueError("seed must be an integer")
    if track not in (None, "synthetic", "photo"):
        raise ValueError("track must be synthetic or photo")
    root = Path(repo_root or DEFAULT_ROOT).resolve()
    spec = FORMATS.get(str(data))
    inputs_path = resolve_file(spec["inputs"] if spec else data, root)
    answers_path = resolve_file(answers or (spec["answers"] if spec else DATA_DIR / "dataset.jsonl"), root)
    gold = {}
    for row in read_answers(answers_path):
        identifier = row["id"]
        if identifier in gold:
            raise ValueError(f"Duplicate answer id: {identifier}")
        if not isinstance(row.get("input"), dict) or not isinstance(row.get("audit", {}), dict):
            raise ValueError(f"{identifier}: input and audit must be JSON objects")
        expected_image = row["input"].get("image_path")
        if not isinstance(expected_image, str) or not expected_image:
            raise ValueError(f"{identifier}: answer record needs input.image_path")
        task = row.get("task")
        if task == "single_object_identification":
            gold_candidates = _check_candidates(row["input"].get("candidate_labels"),
                                                f"{identifier}: input.candidate_labels")
            if row.get("target") not in gold_candidates:
                raise ValueError(f"{identifier}: target must be one of its candidate_labels")
            gold_task = "identification"
        elif task is None:
            if row.get("target") not in ("Yes", "No"):
                raise ValueError(f"{identifier}: target must be Yes or No")
            gold_task = "yes-no"
        else:
            raise ValueError(f"{identifier}: unknown answer task: {task}")
        metadata = {key: row.get(key) for key in (
            "track", "variant", "scene_id", "scene_family", "formula_family_id",
            "semantic_function_id", "split", "image_index", "scene_target_family_id",
        )}
        if metadata["track"] is None and gold_task == "identification":
            metadata["track"] = "photo"
        metadata["cohort"] = row.get("audit", {}).get("cohort")
        gold[identifier] = (row["target"], expected_image, metadata, gold_task)
    seen = set()
    examples = []
    formats = set()
    input_tasks = set()
    for row in read_jsonl(inputs_path):
        identifier = row["id"]
        if identifier in seen:
            raise ValueError(f"Duplicate input id: {identifier}")
        seen.add(identifier)
        if identifier not in gold:
            raise ValueError(f"Missing answer for input id: {identifier}")
        if any(key in row for key in ("target", "answer", "audit")):
            raise ValueError(f"{identifier}: use model inputs without answers or audit fields")
        prompt, image_path = row.get("prompt"), row.get("image_path")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{identifier}: prompt must be a nonempty string")
        if not isinstance(image_path, str) or not image_path:
            raise ValueError(f"{identifier}: image_path must be a nonempty string")
        candidates = (_check_candidates(row["candidate_labels"], f"{identifier}: candidate_labels")
                      if "candidate_labels" in row else None)
        target, expected_image, metadata, gold_task = gold[identifier]
        input_task = "identification" if candidates is not None else "yes-no"
        input_tasks.add(input_task)
        if input_task != gold_task:
            raise ValueError(f"{identifier}: input task ({input_task}) does not match answer task ({gold_task})")
        if candidates is not None and target not in candidates:
            raise ValueError(f"{identifier}: target is not among input candidate_labels")
        if image_path.replace("\\", "/") != expected_image.replace("\\", "/"):
            raise ValueError(f"{identifier}: image_path does not match the answer record")
        safe_image_path(root, image_path)
        formats.add(row.get("format", "symbolic" if "formula" in row else "custom"))
        examples.append(Example(identifier, prompt, image_path, target, metadata, candidates))
    if len(formats) > 1:
        raise ValueError("Mixed input formats: evaluate symbolic and natural language separately")
    if len(input_tasks) > 1:
        raise ValueError("Mixed input tasks: candidate_labels must be present on all rows or none")
    task = next(iter(input_tasks), spec["task"] if spec else "yes-no")
    if spec and task != spec["task"]:
        raise ValueError(f"{data}: expected {spec['task']} inputs, found {task}")
    if track is not None:
        examples = [row for row in examples if row.metadata["track"] == track]
    if seed is not None:
        random.Random(seed).shuffle(examples)
    examples = examples[offset:None if limit is None else offset + limit]
    if not examples:
        raise ValueError("No input rows selected")
    format_name = str(data) if spec else next(iter(formats))
    if format_name == "controlled_natural_language_v1":
        format_name = "natural-language"
    return Dataset(examples, root, inputs_path, answers_path, format_name, task, len(seen), len(gold))


def _check_candidates(value: object, where: str) -> tuple:
    if (not isinstance(value, list) or not value
            or any(not isinstance(label, str) or not label.strip() for label in value)
            or len({label.upper() for label in value}) != len(value)):
        raise ValueError(f"{where} must be a list of unique nonempty strings")
    return tuple(value)


def safe_image_path(root: Path, relative: str) -> Path:
    normalized = relative.replace("\\", "/")
    if Path(normalized).is_absolute() or PureWindowsPath(normalized).drive:
        raise ValueError(f"Image path must be relative to repository root: {relative}")
    path = (root / normalized).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Image path escapes repository root: {relative}")
    return path


@dataclass(frozen=True)
class ImageAsset:
    data: bytes
    mime_type: str
    info: dict


def prepare_image(path: Path, *, max_image_edge: int = 2048) -> ImageAsset:
    """Decode every image; fit to a configurable edge and a 4 MiB inline cap.

    Preserve the original bytes when within both limits. Otherwise apply EXIF
    orientation, fit without cropping, and encode lossless RGB PNG. Record both
    byte hashes, dimensions, and actual scaling for reproducible evaluation.
    """
    from PIL import Image, ImageOps, UnidentifiedImageError

    if type(max_image_edge) is not int or max_image_edge < 1:
        raise ValueError("max_image_edge must be a positive integer")
    original = path.read_bytes()
    mime_types = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(original)) as check:
                check.verify()
            with Image.open(io.BytesIO(original)) as opened:
                source_format = opened.format
                if source_format not in mime_types:
                    raise ValueError(f"Unsupported image type: {source_format}")
                if getattr(opened, "n_frames", 1) != 1:
                    raise ValueError("Animated/multi-frame images are not supported")
                opened.load()
                original_size = list(opened.size)
                im = ImageOps.exif_transpose(opened)
                needs_orientation = im.size != opened.size or opened.getexif().get(274, 1) != 1
                prepared = original
                mime_type = mime_types[source_format]
                if max(im.size) > max_image_edge or len(original) > MAX_IMAGE_BYTES or needs_orientation:
                    # Flatten transparency on white to make the transmitted scene explicit.
                    rgba = im.convert("RGBA")
                    im = Image.new("RGB", rgba.size, "white")
                    im.paste(rgba, mask=rgba.getchannel("A"))
                    im.thumbnail((max_image_edge, max_image_edge), Image.Resampling.LANCZOS)
                    while True:
                        buffer = io.BytesIO()
                        im.save(buffer, format="PNG")
                        prepared = buffer.getvalue()
                        if len(prepared) <= MAX_IMAGE_BYTES:
                            break
                        im = im.resize((max(1, int(im.width * .8)), max(1, int(im.height * .8))),
                                       Image.Resampling.LANCZOS)
                    mime_type = "image/png"
                prepared_size = list(im.size)
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError(f"Cannot decode image {path}: {exc}") from exc
    return ImageAsset(prepared, mime_type, {
        "original_sha256": sha256(original),
        "prepared_sha256": sha256(prepared),
        "original_bytes": len(original), "prepared_bytes": len(prepared),
        "original_dimensions": original_size, "prepared_dimensions": prepared_size,
        "transformed": prepared != original, "mime_type": mime_type,
    })


def prepare_images(dataset: Dataset, *, max_image_edge: int = 2048) -> dict[str, ImageAsset]:
    assets = {}
    for row in dataset.examples:
        if row.image_path not in assets:
            path = safe_image_path(dataset.root, row.image_path)
            assets[row.image_path] = prepare_image(path, max_image_edge=max_image_edge)
    return assets
