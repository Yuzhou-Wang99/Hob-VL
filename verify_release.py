"""Verify packaged bytes, all task/format joins, pairing, and image decoding offline."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from PIL import Image

from evals.data import load_dataset, safe_image_path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify() -> dict:
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "release_manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        path = safe_image_path(root, relative)
        require(path.is_file(), f"Missing packaged file: {relative}")
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        require(actual == expected, f"File checksum mismatch: {relative}")
    print(f"PASS: {len(manifest['files'])} packaged file hashes", flush=True)

    datasets = {}
    for name, expected in (("symbolic", 6000), ("natural-language", 6000),
                           ("identification-symbolic", 1000), ("identification", 1000)):
        data = load_dataset(name, repo_root=root)
        require(len(data.examples) == data.input_count == data.answer_count == expected,
                f"Wrong row count: {name}")
        require(all(e.id.startswith("hob_vl_") for e in data.examples), f"Unexpected ID: {name}")
        datasets[name] = data
        print(f"PASS: {name}, {expected} input/answer joins", flush=True)

    for left, right in (("symbolic", "natural-language"),
                        ("identification-symbolic", "identification")):
        a, b = datasets[left].examples, datasets[right].examples
        require([e.id for e in a] == [e.id for e in b], f"ID/order mismatch: {left}, {right}")
        require(all((x.image_path, x.target, x.candidates) == (y.image_path, y.target, y.candidates)
                    for x, y in zip(a, b)), f"Paired evidence/target mismatch: {left}, {right}")

    boolean = datasets["symbolic"].examples
    require(Counter(e.target for e in boolean) == {"Yes": 3000, "No": 3000}, "Unbalanced Boolean labels")
    require(Counter(e.metadata["track"] for e in boolean) == {"photo": 2000, "synthetic": 4000},
            "Unexpected Boolean track counts")
    groups = defaultdict(list)
    for row in boolean:
        groups[(row.metadata["scene_id"], row.metadata["formula_family_id"])].append(row)
    require(len(groups) == 3000, "Unexpected equivalent-pair count")
    for members in groups.values():
        require(len(members) == 2 and {e.metadata["variant"] for e in members} == {"base", "de_morgan"},
                "Invalid base/De Morgan pair")
        require(len({(e.image_path, e.target) for e in members}) == 1, "Pair image/target mismatch")

    paths = {row.image_path for data in datasets.values() for row in data.examples}
    photos = {p for p in paths if p.startswith("images/photos/")}
    require(len(paths) == 1046 and len(photos) == 46, "Unexpected image counts")
    for relative in sorted(paths):
        with Image.open(safe_image_path(root, relative)) as im:
            im.verify()
    print("PASS: paired formats, 3,000 equivalent-expression pairs, and 1,046 decoded images", flush=True)
    return {"status": "passed", "question_ids": 7000, "prompt_rows": 14000,
            "images": 1046, "equivalent_expression_pairs_per_boolean_format": 3000,
            "model_api_calls": 0,
            "scope": "Automated checks of packaged bytes, schema, joins, counts, pair alignment, and image decoding."}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
