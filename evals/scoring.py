"""Strict answer scoring; errors and invalid completions count as incorrect."""

from __future__ import annotations

import re
from collections import Counter, defaultdict

PARSER_VERSION = "yes-no-full-response-v1"
LABEL_PARSER_VERSION = "candidate-label-full-response-v1"
PARSER_FOR_TASK = {"yes-no": PARSER_VERSION, "identification": LABEL_PARSER_VERSION}
ANSWER_FORMATS = ("plain", "plain-or-bold")
BOLD_PARSER_FOR_TASK = {"yes-no": "yes-no-full-response-bold-v1",
                        "identification": "candidate-label-full-response-bold-v1"}


def parser_for_task(task: str, answer_format: str = "plain") -> str:
    if answer_format not in ANSWER_FORMATS:
        raise ValueError("answer_format must be plain or plain-or-bold")
    return (PARSER_FOR_TASK if answer_format == "plain" else BOLD_PARSER_FOR_TASK)[task]


def _parse_token(text, alternatives: str, answer_format: str) -> str | None:
    parser_for_task("yes-no", answer_format)
    if not isinstance(text, str):
        return None
    match = re.fullmatch(rf"\s*({alternatives})[.!]?\s*", text, re.IGNORECASE)
    if match:
        return match[1]
    if answer_format == "plain-or-bold":
        match = re.fullmatch(
            rf"\s*(?P<wrap>\*\*|__)(?P<answer>{alternatives})(?P<inside>[.!]?)"
            rf"(?P=wrap)(?P<outside>[.!]?)\s*", text, re.IGNORECASE)
        if match and not (match["inside"] and match["outside"]):
            return match["answer"]
    return None


def parse_answer(text: str, *, answer_format: str = "plain") -> str | None:
    """Accept only Yes/No, ignoring case, surrounding space, and one . or !.

    Deliberately reject prose, both labels, true/false, JSON, and numeric answers.
    A response mentioning the correct label is not necessarily a prediction.
    """
    answer = _parse_token(text, "yes|no", answer_format)
    return answer.capitalize() if answer else None


def parse_label(text: str, candidates, *, answer_format: str = "plain") -> str | None:
    """Accept only one complete candidate label, ignoring case and one . or !.

    Multi-letter labels (e.g. AF) are matched before their prefixes; prose,
    multiple labels, and labels outside the candidate set are invalid.
    """
    if not isinstance(text, str) or not candidates:
        return None
    lookup = {str(label).upper(): str(label) for label in candidates}
    alternatives = "|".join(sorted(map(re.escape, lookup), key=len, reverse=True))
    answer = _parse_token(text, alternatives, answer_format)
    return lookup[answer.upper()] if answer else None


def parse_prediction(text: str, example, *, answer_format: str = "plain") -> str | None:
    """Dispatch on the example's task: candidate label vs Yes/No."""
    if example.candidates:
        return parse_label(text, example.candidates, answer_format=answer_format)
    return parse_answer(text, answer_format=answer_format)


def reported_cost(rows: list[dict]) -> dict:
    """Subtotal of provider-reported charges, with explicit missing coverage."""
    amounts = [row.get("usage", {}).get("cost_in_usd_ticks") for row in rows]
    amounts = [value for value in amounts if type(value) is int and value >= 0]
    return {
        "reported_cost_usd": sum(amounts) / 10_000_000_000 if amounts else None,
        "saved_attempts_with_cost": len(amounts),
        "saved_attempts_without_cost": len(rows) - len(amounts),
        "note": "Provider-reported subtotal for saved attempts only. Missing amounts and "
                "requests without a saved response are not included; missing does not mean free.",
    }


def metrics(rows: list[dict]) -> dict:
    total = len(rows)
    correct = sum(row["correct"] is True for row in rows)
    invalid = sum(row["status"] == "invalid" for row in rows)
    errors = sum(row["status"] == "error" for row in rows)
    return {
        "total": total, "correct": correct,
        "accuracy": correct / total if total else None,
        "invalid": invalid, "errors": errors,
        "valid": total - invalid - errors,
    }


def summarize(rows: list[dict], task: str = "yes-no", *, answer_format: str = "plain") -> dict:
    result = {"overall": metrics(rows)}
    for field in ("track", "variant", "cohort", "target", "scene_family"):
        groups = defaultdict(list)
        for row in rows:
            value = row["target"] if field == "target" else row["metadata"].get(field)
            groups[str(value) if value is not None else "unknown"].append(row)
        result[f"by_{field}"] = {name: metrics(group) for name, group in sorted(groups.items())}
    families = defaultdict(list)
    for row in rows:
        family = row["metadata"].get("formula_family_id")
        if family:
            families[family].append(row)
    pairs = [group for group in families.values() if len(group) == 2
             and {row["metadata"].get("variant") for row in group} == {"base", "de_morgan"}]
    both_valid = sum(all(row["status"] == "ok" for row in pair) for pair in pairs)
    agreement = sum(all(row["status"] == "ok" for row in pair)
                    and pair[0]["prediction"] == pair[1]["prediction"] for pair in pairs)
    both_correct = sum(all(row["correct"] for row in pair) for pair in pairs)
    result["equivalent_pairs"] = {
        "complete_pairs": len(pairs), "both_valid": both_valid,
        "agree": agreement, "agreement": agreement / len(pairs) if pairs else None,
        "both_correct": both_correct,
        "both_correct_accuracy": both_correct / len(pairs) if pairs else None,
        "excluded_incomplete_or_nonpair_families": len(families) - len(pairs),
        "denominator": "complete base/de_morgan pairs; invalid/error answers never agree",
    }
    usage = Counter()
    for row in rows:
        for key, value in row.get("usage", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                usage[key] += value
    result["usage"] = dict(usage)
    result["parser"] = parser_for_task(task, answer_format)
    return result
