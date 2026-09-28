"""Join a symbolic and a natural-language run by id; report per-condition and paired accuracy.

Usage:
  python -m evals.compare RUN_A RUN_B [--output DIR]
  python -m evals.compare --model provider/model [--runs-dir DIR] [--output DIR]

Each RUN is a run directory containing predictions.jsonl and manifest.json,
or a predictions JSONL path with its manifest alongside it. Settings and image
evidence must match across the pair. --model finds the newest run of each format under --runs-dir (default
evals/runs) and compares every complete format pair: symbolic/natural-language
and identification-symbolic/identification.

Every invocation creates a fresh output directory (default
evals/comparisons/<stamp>-<model>-<id>) containing comparison.json and
comparison.md; the markdown file holds the same tables plus a LaTeX block for
Overleaf. When --model finds both format pairs, each task's report lands in a
yes-no/ or identification/ subdirectory of the output directory. Invalid/error
responses count as incorrect, matching evals scoring.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .prompts import recorded_prompt_policy

from .checkpoints import read_predictions
from .data import DEFAULT_ROOT
from .scoring import metrics

CATEGORIES = ("both_correct", "symbolic_only", "natural_language_only", "neither")
# Each pair maps (formal/symbolic side, natural-language side); the first element
# fills the report's "symbolic" slot and the second the "natural_language" slot.
FORMAT_PAIRS = (("symbolic", "natural-language"), ("identification-symbolic", "identification"))
PAIR_TASKS = {pair: task for pair, task in zip(FORMAT_PAIRS, ("yes-no", "identification"))}


def _find_predictions(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "predictions.jsonl"
        if candidate.is_file():
            return candidate
        raise ValueError(f"No predictions.jsonl in run directory: {path}")
    if not path.is_file():
        raise ValueError(f"Not a run directory or JSONL file: {path}")
    return path


def _model_string(model: object) -> str | None:
    if isinstance(model, dict) and model.get("provider") and model.get("model"):
        return f"{model['provider']}/{model['model']}"
    return None


def _run_info(path: Path) -> dict:
    directory = path if path.is_dir() else path.parent
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        raise ValueError(f"Missing {manifest}; keep the original manifest alongside predictions "
                         "so comparison can verify experimental settings")
    identity = json.loads(manifest.read_text(encoding="utf-8")).get("identity")
    if (not isinstance(identity, dict) or identity.get("mode") != "execute"
            or not isinstance(identity.get("model"), dict)
            or not isinstance(identity.get("data"), dict)
            or not isinstance(identity.get("images"), dict)
            or not isinstance(identity.get("image_preparation"), dict)
            or not identity.get("parser")):
        raise ValueError(f"Incomplete execute-run manifest: {manifest}")
    model = identity["model"]
    if (any(key not in model for key in ("provider", "model", "base_url", "max_output_tokens"))
            or _model_string(model) is None or not identity["data"].get("answers_sha256")):
        raise ValueError(f"Missing model or answer-file evidence in {manifest}")
    info = {"format": identity["data"].get("format"), "model": _model_string(model),
            "complete": None, "identity": identity}
    summary = directory / "summary.json"
    if summary.is_file():
        data = json.loads(summary.read_text(encoding="utf-8"))
        if data.get("format") != info["format"] or data.get("model") != model:
            raise ValueError(f"Summary and manifest disagree in {directory}")
        info["complete"] = data.get("complete")
    return info


def _check_conditions(first: dict, second: dict) -> None:
    """Compare formats only when their output instructions and settings match."""
    a, b = first["identity"], second["identity"]
    # Credential variable names and transport controls do not change the prompt
    # or generation settings. Compare all other fields, including future knobs.
    ignored = {"api_key_env", "timeout", "retries"}
    for key in sorted((a["model"].keys() | b["model"].keys()) - ignored):
        if a["model"].get(key) != b["model"].get(key):
            raise ValueError(f"Cannot compare: model setting {key!r} differs across runs")
    for key in ("task", "answers_sha256"):
        if a["data"].get(key) != b["data"].get(key):
            raise ValueError(f"Cannot compare: dataset {key!r} differs across runs")
    for key in ("parser", "image_preparation"):
        if a[key] != b[key]:
            raise ValueError(f"Cannot compare: {key} differs across runs")
    if recorded_prompt_policy(a) != recorded_prompt_policy(b):
        raise ValueError("Cannot compare: prompt_preparation differs across runs")


def _check_images(rows: dict[str, dict], info: dict) -> None:
    images = info["identity"]["images"]
    for identifier, row in rows.items():
        image = images.get(row["image_path"])
        if (not isinstance(image, dict)
                or image.get("prepared_sha256") != row["prepared_image_sha256"]):
            raise ValueError(f"Prediction image does not match its manifest for id: {identifier}")


def _find_model_runs(model: str, runs_dir: Path) -> dict[str, tuple[Path, Path]]:
    """Newest run per format for ``model``; one entry per complete format pair."""
    if "/" not in model:
        raise ValueError("model must be provider/model, e.g. openai/your-model-id")
    if not runs_dir.is_dir():
        raise ValueError(f"Runs directory does not exist: {runs_dir}")
    found: dict[str, list[Path]] = defaultdict(list)
    for child in sorted(runs_dir.iterdir()):
        if not child.is_dir() or not (child / "predictions.jsonl").is_file():
            continue
        try:
            info = _run_info(child)
        except (OSError, ValueError):
            continue
        if info["format"] and info["model"] == model:
            found[info["format"]].append(child)
    complete = {PAIR_TASKS[pair]: (found[pair[0]][-1], found[pair[1]][-1])
                for pair in FORMAT_PAIRS if all(found.get(fmt) for fmt in pair)}
    if not complete:
        raise ValueError(f"No complete format pair for {model} under {runs_dir}; "
                         f"found formats: {sorted(found) or 'none'}")
    return complete


def _load_rows(path: Path) -> dict[str, dict]:
    _, rows = read_predictions(path)
    if not rows:
        raise ValueError(f"No prediction rows in {path}")
    return rows


def _joint_table(sym_rows: dict, nl_rows: dict, ids: list[str]) -> dict:
    counts = dict.fromkeys(CATEGORIES, 0)
    discordant = {"symbolic_only": [], "natural_language_only": []}
    for identifier in ids:
        s = sym_rows[identifier]["correct"]
        n = nl_rows[identifier]["correct"]
        key = ("both_correct" if s and n else "symbolic_only" if s
               else "natural_language_only" if n else "neither")
        counts[key] += 1
        if key in discordant:
            discordant[key].append(identifier)
    total = len(ids)
    return {
        "paired_rows": total,
        "counts": counts,
        "rates": {key: value / total if total else None for key, value in counts.items()},
        "paired_accuracy": {
            "symbolic": (counts["both_correct"] + counts["symbolic_only"]) / total if total else None,
            "natural_language": (counts["both_correct"] + counts["natural_language_only"]) / total if total else None,
        },
        "discordant_ids": discordant,
    }


def compare(first: str | Path, second: str | Path, *, include_ids: bool = True) -> dict:
    paths = [_find_predictions(Path(p)) for p in (first, second)]
    infos = [_run_info(Path(p)) for p in (first, second)]
    detected = [info["format"] for info in infos]
    matching = [pair for pair in FORMAT_PAIRS if set(detected) == set(pair)]
    if not matching:
        raise ValueError(f"Formats {detected[0]!r} and {detected[1]!r} are not a comparable pair")
    _check_conditions(*infos)
    sym_i = detected.index(matching[0][0])
    nl_i = 1 - sym_i
    model = infos[sym_i]["model"]

    sym_rows = _load_rows(paths[sym_i])
    nl_rows = _load_rows(paths[nl_i])
    _check_images(sym_rows, infos[sym_i])
    _check_images(nl_rows, infos[nl_i])
    common = sorted(sym_rows.keys() & nl_rows.keys())
    if not common:
        raise ValueError("No shared IDs to compare")
    for identifier in common:
        for key in ("target", "metadata", "image_path", "prepared_image_sha256"):
            if sym_rows[identifier][key] != nl_rows[identifier][key]:
                raise ValueError(f"Paired {key} mismatch across runs for id: {identifier}")

    paired = _joint_table(sym_rows, nl_rows, common)
    if not include_ids:
        del paired["discordant_ids"]

    by_track = defaultdict(list)
    for identifier in common:
        track = sym_rows[identifier].get("metadata", {}).get("track") or "unknown"
        by_track[str(track)].append(identifier)
    track_tables = {}
    for track, ids in sorted(by_track.items()):
        table = _joint_table(sym_rows, nl_rows, ids)
        if not include_ids:
            del table["discordant_ids"]
        track_tables[track] = table

    return {
        "model": model,
        "prompt_policy": recorded_prompt_policy(infos[sym_i]["identity"]),
        "symbolic": {"path": str(paths[sym_i]), "format": detected[sym_i] or "assumed",
                     "complete": infos[sym_i]["complete"],
                     **metrics(list(sym_rows.values()))},
        "natural_language": {"path": str(paths[nl_i]), "format": detected[nl_i] or "assumed",
                             "complete": infos[nl_i]["complete"],
                             **metrics(list(nl_rows.values()))},
        "paired": paired,
        "unpaired": {"symbolic_only_ids": len(sym_rows) - len(common),
                     "natural_language_only_ids": len(nl_rows) - len(common)},
        "by_track": track_tables,
        "note": "Latest state per ID; correct=False includes invalid/error responses. "
                "Model settings, prompt policy, answer-file hashes, image preparation, and paired image evidence verified.",
    }


def _pct(value) -> str:
    return "—" if value is None else f"{100 * value:.2f}%"


def _side_names(report: dict) -> tuple[str, str]:
    names = {"symbolic": "Symbolic", "identification-symbolic": "Symbolic",
             "natural-language": "Natural language", "identification": "Natural language"}
    sym = names.get(report["symbolic"]["format"], report["symbolic"]["format"].replace("-", " ").title())
    nl = names.get(report["natural_language"]["format"], report["natural_language"]["format"].replace("-", " ").title())
    return sym, nl


def _tex_escape(text: str) -> str:
    return re.sub(r"([_%&#$])", r"\\\1", text)


def _accuracy_md(report: dict) -> list[str]:
    sym_name, nl_name = _side_names(report)
    rows = [(sym_name, report["symbolic"]), (nl_name, report["natural_language"])]
    lines = ["| Condition | Accuracy | Correct | Total | Valid | Invalid | Errors |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, m in rows:
        lines.append(f"| {name} | {_pct(m['accuracy'])} | {m['correct']} | {m['total']} "
                     f"| {m['valid']} | {m['invalid']} | {m['errors']} |")
    return lines


def _paired_md(table: dict, sym_name: str = "Symbolic", nl_name: str = "NL") -> list[str]:
    c = table["counts"]
    sym_total = c["both_correct"] + c["symbolic_only"]
    nl_total = c["both_correct"] + c["natural_language_only"]
    return [f"| | {nl_name} correct | {nl_name} wrong | Total |",
            "|---|---:|---:|---:|",
            f"| **{sym_name} correct** | {c['both_correct']} | {c['symbolic_only']} | {sym_total} |",
            f"| **{sym_name} wrong** | {c['natural_language_only']} | {c['neither']} "
            f"| {c['natural_language_only'] + c['neither']} |",
            f"| **Total** | {nl_total} | {c['symbolic_only'] + c['neither']} | {table['paired_rows']} |"]


def _latex(report: dict) -> str:
    sym, nl = report["symbolic"], report["natural_language"]
    sym_name, nl_name = _side_names(report)
    c = report["paired"]["counts"]
    acc = ["\\begin{tabular}{lrrrrr}", "\\toprule",
           "Condition & Accuracy & Correct & Total & Invalid & Errors \\\\", "\\midrule",
           f"{sym_name} & {_pct(sym['accuracy']).replace('%', '\\%')} & {sym['correct']} "
           f"& {sym['total']} & {sym['invalid']} & {sym['errors']} \\\\",
           f"{nl_name} & {_pct(nl['accuracy']).replace('%', '\\%')} & {nl['correct']} "
           f"& {nl['total']} & {nl['invalid']} & {nl['errors']} \\\\",
           "\\bottomrule", "\\end{tabular}", ""]
    pair = ["\\begin{tabular}{lrr}", "\\toprule",
            f" & {nl_name} correct & {nl_name} wrong \\\\", "\\midrule",
            f"{sym_name} correct & {c['both_correct']} & {c['symbolic_only']} \\\\",
            f"{sym_name} wrong & {c['natural_language_only']} & {c['neither']} \\\\",
            "\\bottomrule", "\\end{tabular}"]
    return "\n".join(acc + pair)


def _markdown(report: dict, output_dir: Path) -> str:
    title = _tex_escape(report["model"]) if report["model"] else "Model comparison"
    sym_name, nl_name = _side_names(report)
    lines = [f"# {title} — {report['symbolic']['format']} vs {report['natural_language']['format']}", "",
             f"- {sym_name} run: `{report['symbolic']['path']}`",
             f"- {nl_name} run: `{report['natural_language']['path']}`",
             f"- Prompt policy: `{report['prompt_policy']}`",
             "", "## Accuracy", ""]
    lines += _accuracy_md(report)
    lines += ["", f"## Paired outcomes (N = {report['paired']['paired_rows']})", ""]
    lines += _paired_md(report["paired"], sym_name, nl_name)
    if report["unpaired"]["symbolic_only_ids"] or report["unpaired"]["natural_language_only_ids"]:
        lines += ["", f"Unpaired rows: {report['unpaired']['symbolic_only_ids']} "
                      f"{report['symbolic']['format']}-only, "
                      f"{report['unpaired']['natural_language_only_ids']} "
                      f"{report['natural_language']['format']}-only."]
    for track, table in report["by_track"].items():
        lines += ["", f"### Track: {track} (N = {table['paired_rows']})", ""]
        lines += _paired_md(table, sym_name, nl_name)
    lines += ["", "## LaTeX (Overleaf)", "", "```latex", _latex(report), "```", "",
              "_Invalid/error responses count as incorrect, matching evals scoring policy._", ""]
    return "\n".join(lines)


def _new_output_dir(output: str | None, label: str) -> Path:
    if output is not None:
        directory = Path(output).resolve()
        if directory.exists():
            raise ValueError(f"Output already exists; choose a new directory: {directory}")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", label)[:100]
        directory = DEFAULT_ROOT / "evals/comparisons" / f"{stamp}-{safe}-{uuid.uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", metavar="RUN",
                        help="Two run directories or predictions.jsonl files; "
                             "each must have its original manifest alongside predictions")
    parser.add_argument("--model", help="provider/model; finds the newest run of each format under --runs-dir")
    parser.add_argument("--runs-dir", default=str(DEFAULT_ROOT / "evals/runs"),
                        help="Directory of eval runs scanned by --model (default: evals/runs)")
    parser.add_argument("--output", help="New output directory; default: evals/comparisons/<unique>")
    parser.add_argument("--no-ids", action="store_true",
                        help="Omit discordant id lists from the report")
    args = parser.parse_args(argv)
    try:
        if args.model:
            if args.runs:
                raise ValueError("Pass either two RUN arguments or --model, not both")
            pairs = _find_model_runs(args.model, Path(args.runs_dir))
        else:
            if len(args.runs) != 2:
                raise ValueError("Pass exactly two RUN arguments, or use --model")
            pairs = {"comparison": (Path(args.runs[0]), Path(args.runs[1]))}
        reports = {name: compare(first, second, include_ids=not args.no_ids)
                   for name, (first, second) in pairs.items()}
        label = args.model or reports[next(iter(reports))]["model"] or "comparison"
        output_dir = _new_output_dir(args.output, label)
        for name, report in reports.items():
            target = output_dir if len(reports) == 1 else output_dir / name
            target.mkdir(parents=True, exist_ok=True)
            report["output_dir"] = str(target)
            markdown = _markdown(report, target)
            (target / "comparison.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            (target / "comparison.md").write_text(markdown, encoding="utf-8")
        result = (reports[next(iter(reports))] if len(reports) == 1
                  else {"comparisons": reports, "output_dir": str(output_dir)})
    except (ValueError, OSError) as exc:
        print(f"Comparison failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
