"""Build paper tables from validated saved runs, without model API access.

Run ``python -m evals.paper_results`` from the repository root. Original run
files are read only; only generated reports and CSV tables are written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from .checkpoints import read_predictions
from .compare import compare
from .data import DEFAULT_ROOT, DATA_DIR, read_jsonl
from .scoring import parse_answer, parse_label, summarize


RUNS = (
    ("Luna", "luna-none-full"),
    ("Haiku 4.5", "haiku-4.5-none"),
    ("Grok 4.3", "grok-4.3-none-one-word-full"),
    ("Gemini Flash-Lite", "gemini-3.5-flash-lite-minimal-full"),
    ("DeepSeek Flash", "deepseek-v4.1-flash-none-full"),
    ("GLM FlashX (off)", "glm-4.6v-flashx-none-full"),
    ("Llama 4 Scout", "llama-4-scout-none-full"),
    ("Qwen3-VL 8B", "qwen3-vl-8b-instruct-none-full"),
    ("GLM FlashX (on)", "glm-4.6v-flashx-thinking-full"),
)
FORMATS = ("symbolic", "natural-language", "identification-symbolic", "identification")
COUNTS = (6000, 6000, 1000, 1000)
COHORTS = (
    ("quadratic_parity", "XOR of conjunctions"),
    ("majority_parity", "XOR of majorities"),
    ("nested_mux", "Nested multiplexers"),
    ("dense_cnf", "Dense CNF"),
    ("sparse_cnf_control", "Sparse CNF control"),
    ("polarity_trap_composition", "Polarity trap + parity"),
    ("parity_control", "Parity control"),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pct(value: float) -> str:
    return str((Decimal(str(value)) * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def build(runs_dir: Path, output: Path) -> dict:
    candidates = {r["id"]: r["candidate_labels"] for r in read_jsonl(
        DEFAULT_ROOT / DATA_DIR / "model_inputs_photo_identification_1000.jsonl")}
    report = {
        "protocol": "Latest saved state per ID; original scoring; invalid outputs incorrect; no API calls.",
        "boolean_questions_per_format": 6000, "identification_questions_per_format": 1000,
        "equivalent_pairs_per_boolean_format": 3000,
        "identification_uniform_guess_expected_accuracy": sum(1 / len(c) for c in candidates.values()) / len(candidates),
        "identification_candidate_count_range": [min(map(len, candidates.values())), max(map(len, candidates.values()))],
        "runs": [],
    }
    model_states = {}
    reference = {}
    for name, directory in RUNS:
        run = {"name": name, "directory": directory, "formats": {}, "paired_formats": {}}
        states = {}
        for fmt, expected in zip(FORMATS, COUNTS):
            path = runs_dir / directory / fmt
            manifest = json.loads((path / "manifest.json").read_text())
            identity = manifest["identity"]
            saved = json.loads((path / "summary.json").read_text())
            attempts, latest = read_predictions(path / "predictions.jsonl")
            require(saved["complete"] and not saved["pending_rows"] and not saved["remaining_rows"],
                    f"Incomplete run: {directory}/{fmt}")
            require(len(latest) == expected and identity["data"]["format"] == fmt,
                    f"Unexpected selection: {directory}/{fmt}")
            answer_format = "plain-or-bold" if "bold" in identity["parser"] else "plain"
            task = "yes-no" if expected == 6000 else "identification"
            rows = list(latest.values())
            computed = summarize(rows, task, answer_format=answer_format)
            for key in ("overall", "equivalent_pairs", "by_track", "by_cohort", "by_variant"):
                require(computed[key] == saved[key], f"Stale {key}: {directory}/{fmt}")
            require(computed["overall"]["errors"] == 0, f"Pending API error: {directory}/{fmt}")
            if fmt in reference:
                require(latest.keys() == reference[fmt].keys(), "Unmatched IDs across models")
                for identifier, row in latest.items():
                    require(all(row[k] == reference[fmt][identifier][k] for k in ("target", "metadata", "image_path")),
                            f"Different benchmark evidence: {identifier}")
            else:
                reference[fmt] = latest
            invalid = [r for r in rows if r["status"] == "invalid"]
            finishes = Counter(str(r.get("finish_reason")) for r in invalid)
            uniform_correct = uniform_valid = 0
            for row in rows:
                prediction = None
                if row["complete"] and row["status"] != "error":
                    prediction = (parse_answer(row["response"], answer_format="plain-or-bold")
                                  if task == "yes-no" else parse_label(
                                      row["response"], candidates[row["id"]], answer_format="plain-or-bold"))
                uniform_valid += prediction is not None
                uniform_correct += prediction is not None and prediction == row["target"]
            pairs = computed["equivalent_pairs"]
            if task == "yes-no":
                require(pairs["complete_pairs"] == 3000, "Missing equivalent pairs")
                # Validate equivalent-pair targets independently of the summary.
                families = {}
                for row in rows:
                    family = row["metadata"]["formula_family_id"]
                    families.setdefault(family, []).append(row)
                require(all(len(g) == 2 and g[0]["target"] == g[1]["target"]
                            and g[0]["image_path"] == g[1]["image_path"]
                            and g[0]["prepared_image_sha256"] == g[1]["prepared_image_sha256"]
                            for g in families.values()), "Invalid equivalent-pair evidence")
                pairs["valid_pair_coverage"] = pairs["both_valid"] / 3000
                pairs["valid_pair_conflicts"] = pairs["both_valid"] - pairs["agree"]
                pairs["valid_pair_conflict_rate"] = pairs["valid_pair_conflicts"] / pairs["both_valid"]
            run["formats"][fmt] = {
                "model": identity["model"], "prompt_policy": identity.get("prompt_preparation", {}).get("policy", "original"),
                "parser": identity["parser"], "image_preparation": identity["image_preparation"],
                "metrics": computed, "saved_attempts": len(attempts),
                "historical_api_errors": sum(r["status"] == "error" for r in attempts),
                "invalid_finish_reasons": dict(finishes),
                "token_limit_invalid": sum(finishes[k] for k in ("length", "MAX_TOKENS", "max_output_tokens")),
                "yes_fraction_all_items": sum(r["status"] == "ok" and r["prediction"] == "Yes" for r in rows) / expected,
                "uniform_bold_parser_sensitivity": {"correct": uniform_correct, "valid": uniform_valid},
                "source_sha256": {f: sha256(path / f) for f in ("manifest.json", "summary.json", "predictions.jsonl")},
            }
            states[fmt] = latest
        for sym, nl in (("symbolic", "natural-language"), ("identification-symbolic", "identification")):
            paired = compare(runs_dir / directory / sym, runs_dir / directory / nl, include_ids=False)
            require(paired["paired"]["paired_rows"] == len(states[sym]), "Unmatched formats")
            run["paired_formats"][sym] = paired["paired"]
        report["runs"].append(run)
        model_states[directory] = states

    off, on = (model_states[f"glm-4.6v-flashx-{mode}-full"] for mode in ("none", "thinking"))
    for fmt in FORMATS:
        for identifier in off[fmt]:
            require(all(off[fmt][identifier][k] == on[fmt][identifier][k]
                        for k in ("target", "prompt_sha256", "prepared_image_sha256")),
                    "GLM on/off contrast has different prompts or prepared images")
    report["glm_contrast_same_prompts_and_prepared_images"] = True
    label_counts = Counter(row["target"] for row in reference["identification"].values())
    most_common_label, correct_count = label_counts.most_common(1)[0]
    report["identification_constant_label_baseline"] = {
        "label": most_common_label, "correct": correct_count, "total": len(candidates),
        "accuracy": correct_count / len(candidates),
        "label_is_candidate_for_every_question": all(most_common_label in c for c in candidates.values()),
        "note": "Most frequent reference label in this evaluation set; descriptive baseline, not a held-out fitted predictor.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    def table(filename, header, rows):
        with (output / f"{filename}.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

    entries = report["runs"]
    def metrics(run, fmt):
        return run["formats"][fmt]["metrics"]

    table("accuracy", ["configuration", *FORMATS],
          [[r["name"], *[pct(metrics(r, fmt)["overall"]["accuracy"]) for fmt in FORMATS]] for r in entries])
    rows = []
    for r in entries:
        row = [r["name"]]
        for fmt in FORMATS[:2]:
            p = metrics(r, fmt)["equivalent_pairs"]
            row += [pct(p["both_correct_accuracy"]), pct(p["valid_pair_conflict_rate"]), p["both_valid"]]
        rows.append(row)
    table("equivalence", ["configuration", "symbolic_both_correct_percent", "symbolic_conflict_percent",
                          "symbolic_valid_pairs", "nl_both_correct_percent", "nl_conflict_percent", "nl_valid_pairs"], rows)
    table("invalid", ["configuration", *FORMATS, "token_limit_total"],
          [[r["name"], *[metrics(r, fmt)["overall"]["invalid"] for fmt in FORMATS],
            sum(r["formats"][fmt]["token_limit_invalid"] for fmt in FORMATS)] for r in entries])
    rows = []
    for cohort, label in COHORTS:
        rows.append([label, metrics(entries[5], "symbolic")["by_cohort"][cohort]["total"],
                     *[pct(metrics(entries[i], fmt)["by_cohort"][cohort]["accuracy"])
                       for fmt in FORMATS[:2] for i in (5, 8)]])
    table("glm_cohorts", ["family", "questions_per_format", "symbolic_thinking_off", "symbolic_thinking_on",
                          "nl_thinking_off", "nl_thinking_on"], rows)
    table("tracks", ["configuration", "symbolic_synthetic", "symbolic_photo", "nl_synthetic", "nl_photo"],
          [[r["name"], *[pct(metrics(r, fmt)["by_track"][track]["accuracy"])
                        for fmt in FORMATS[:2] for track in ("synthetic", "photo")]] for r in entries])
    table("label_bias", ["configuration", "symbolic_yes_percent_all_items", "nl_yes_percent_all_items"],
          [[r["name"], *[pct(r["formats"][fmt]["yes_fraction_all_items"]) for fmt in FORMATS[:2]]] for r in entries])
    rows = []
    for r in entries:
        row = [r["name"]]
        for fmt in ("symbolic", "identification-symbolic"):
            p = r["paired_formats"][fmt]
            row += [pct(p["rates"]["both_correct"]), p["counts"]["symbolic_only"], p["counts"]["natural_language_only"]]
        rows.append(row)
    table("format_pairs", ["configuration", "boolean_both_correct_percent", "boolean_symbolic_only_count",
                           "boolean_nl_only_count", "identification_both_correct_percent",
                           "identification_symbolic_only_count", "identification_nl_only_count"], rows)
    settings, invalid_rows = [], []
    for r in entries:
        for fmt in FORMATS:
            record = r["formats"][fmt]
            model = record["model"]
            manifest = json.loads((runs_dir / r["directory"] / fmt / "manifest.json").read_text())
            settings.append([r["name"], fmt, model["provider"], model["model"], model["base_url"],
                             model.get("reasoning_effort"), model.get("thinking_mode"), model.get("thinking_level"),
                             model.get("thinking_budget"), model["max_output_tokens"],
                             model.get("temperature") if model.get("temperature") is not None else "omitted (provider default)",
                             model.get("image_detail"), record["prompt_policy"], record["parser"], manifest["created_at"]])
            for row in model_states[r["directory"]][fmt].values():
                if row["status"] != "invalid":
                    continue
                if not row["complete"]:
                    reason = "incomplete_generation"
                elif not row["response"].strip():
                    reason = "empty_final_response"
                else:
                    reason = "not_a_single_allowed_answer_under_saved_parser"
                invalid_rows.append([r["name"], fmt, row["id"], row["target"], reason,
                                     row.get("finish_reason"), row["response"]])
    table("settings", ["configuration", "format", "provider", "requested_model", "api_root",
                       "reasoning_effort", "thinking_mode", "thinking_level", "thinking_budget",
                       "output_token_cap", "temperature", "image_detail", "prompt_policy", "parser",
                       "manifest_created_at_not_completion_time"], settings)
    table("invalid_outputs", ["configuration", "format", "id", "reference_answer", "invalid_reason",
                              "finish_reason", "full_final_response"], invalid_rows)
    md = ["# Saved benchmark results", "", "Regenerate offline with `python -m evals.paper_results`.", "",
          "Eight model identities, nine inference configurations, four formats, 126,000 final responses.",
          "Percentages use all questions; invalid answers count as incorrect. Historical API failures are superseded by retries.", "",
          "| Configuration | Boolean symbolic | Boolean NL | Identification symbolic | Identification NL |",
          "|---|---:|---:|---:|---:|"]
    for r in entries:
        md.append("| " + " | ".join([r["name"]] + [pct(metrics(r, f)["overall"]["accuracy"]) + "%" for f in FORMATS]) + " |")
    md += ["", "NL means structured natural language. GLM off/on mean thinking disabled/enabled; their caps are 2,048/4,096 tokens.",
           "", f"Identification baselines: uniform candidate guessing {pct(report['identification_uniform_guess_expected_accuracy'])}%; "
           f"always predict {most_common_label}: {pct(correct_count / len(candidates))}% ({correct_count}/{len(candidates)}).",
           "", "`results.json` contains exact counts, all subgroup metrics, paired results, parser sensitivity, and source hashes.",
           "`settings.csv` records requested identifiers and settings; `../configurations.json` additionally describes API request fields.",
           "`invalid_outputs.csv` preserves the full final response and a parser/completion-based reason for every invalid answer.",
           "", "Equivalent-expression both-correct accuracy uses all 3,000 pairs; conflict uses only pairs with two valid answers.",
           "A constant-label predictor has 50% both-correct accuracy and zero conflict. Read these with accuracy and label frequencies.",
           "Independent generations also include sampling variability; no repeated-identical-prompt control was run.",
           "Thinking settings, output caps, and some prompt policies differ. These are descriptive configuration comparisons."]
    (output / "README.md").write_text("\n".join(md) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_ROOT / "results/runs")
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / "results/generated")
    args = parser.parse_args()
    build(args.runs_dir.resolve(), args.output.resolve())
    print("Validated 36 completed runs and 126,000 final responses; generated nine CSV files, JSON, and Markdown.")


if __name__ == "__main__":
    main()
