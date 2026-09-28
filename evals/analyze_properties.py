"""Descriptive, offline analysis of Boolean annotations and saved predictions.

Run: python -m evals.analyze_properties
Requires numpy, pandas and matplotlib. No API requests or source-run edits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .checkpoints import read_predictions
from .paper_results import RUNS, FORMATS, COUNTS
from .scoring import parse_answer, summarize


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset/hob_vl_v1"
FEATURES = ("certificate_size", "two_fact_predictability", "total_influence")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, obj):
    # pandas can introduce NaN into empty-denominator cells; serialize as null.
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)) and not np.isfinite(x):
            return None
        if isinstance(x, np.generic):
            return x.item()
        return x
    path.write_text(json.dumps(clean(obj), indent=2, allow_nan=False) + "\n")


def display_path(path):
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def annotations():
    source = read_jsonl(DATA / "dataset.jsonl")
    require(len(source) == 6000, "Expected 6,000 Boolean questions")
    rows = []
    for r in source:
        audit, s = r["audit"], r["audit"]["semantics"]
        rows.append(dict(
            id=r["id"], target=r["target"], track=r["track"],
            scene_id=r["scene_id"], scene_family=r["scene_family"],
            pair_id=r["formula_family_id"], variant=r["variant"],
            family=audit["cohort"], image_path=r["input"]["image_path"],
            certificate_size=s["instance_certificate_size"],
            two_fact_predictability=s["best_2_atom_predictor"]["accuracy"],
            total_influence=sum(s["influences"].values()),
            min_influence=s["min_influence"],
            uniform_true_rate=s["uniform_true_rate"],
            broad_dependence=s["broad_dependence_pass"],
            weak_two_fact_clues=s["weak_two_atom_clues_pass"],
            ast_nodes=audit["syntax"]["node_count"],
            depth=audit["syntax"]["depth_edges"],
            alternations=audit["syntax"]["max_and_or_alternations"],
        ))
    frame = pd.DataFrame(rows).set_index("id", verify_integrity=True)
    for _, pair in frame.groupby("pair_id"):
        require(len(pair) == 2 and set(pair.variant) == {"base", "de_morgan"}, "Broken pair")
        require(all(pair[c].nunique() == 1 for c in (*FEATURES, "target", "scene_id", "family")),
                "Equivalent-pair invariant mismatch")
    return frame


def load_results(frame, runs_dir, snapshot_path):
    snapshot = json.loads(snapshot_path.read_text())
    reference = {r["directory"]: r for r in snapshot["runs"]}
    input_paths = {"symbolic": DATA / "model_inputs.jsonl",
                   "natural-language": DATA / "model_inputs_natural_language.jsonl"}
    prompts = {fmt: {r["id"]: r for r in read_jsonl(path)} for fmt, path in input_paths.items()}
    source_hashes = {display_path(p): sha(p)
                     for p in (DATA / "dataset.jsonl", snapshot_path, *input_paths.values())}
    answers_sha = source_hashes[display_path(DATA / "dataset.jsonl")]
    frames, verified = [], []
    for model, directory in RUNS:
        for fmt, expected in zip(FORMATS, COUNTS):
            folder = runs_dir / directory / fmt
            saved = json.loads((folder / "summary.json").read_text())
            manifest = json.loads((folder / "manifest.json").read_text())["identity"]
            for filename, digest in reference[directory]["formats"][fmt]["source_sha256"].items():
                path = folder / filename
                actual = sha(path)
                require(actual == digest, f"Saved paper snapshot differs from {path}; regenerate/audit first")
                source_hashes[display_path(path)] = actual
            attempts, latest = read_predictions(folder / "predictions.jsonl")
            require(len(latest) == expected and saved["complete"] and not saved["pending_rows"]
                    and not saved["remaining_rows"], f"Incomplete run: {folder}")
            require(not any(r["status"] == "error" for r in latest.values()), f"Pending errors: {folder}")
            require(sum(r["correct"] for r in latest.values()) == saved["overall"]["correct"],
                    f"Incorrect saved accuracy: {folder}")
            verified.append(dict(model=model, format=fmt, questions=len(latest),
                                 correct=saved["overall"]["correct"], invalid=saved["overall"]["invalid"],
                                 historical_attempts=len(attempts), complete=True))
            if fmt not in input_paths:
                continue
            require(set(latest) == set(frame.index), f"ID mismatch: {folder}")
            require(manifest["data"]["answers_sha256"] == answers_sha, "Answer dataset changed")
            require(manifest["data"]["inputs_sha256"] == source_hashes[display_path(input_paths[fmt])],
                    "Prompt dataset changed")
            answer_format = "plain-or-bold" if "bold" in manifest["parser"] else "plain"
            recomputed = summarize(list(latest.values()), answer_format=answer_format)
            require(recomputed["overall"] == saved["overall"], "Summary mismatch")
            require(recomputed["equivalent_pairs"] == saved["equivalent_pairs"], "Pair summary mismatch")
            suffix = manifest.get("prompt_preparation", {}).get("suffix", "")
            joined = frame.copy()
            extras = []
            for identifier in frame.index:
                r, original = latest[identifier], prompts[fmt][identifier]
                require(r["target"] == frame.loc[identifier, "target"], "Target mismatch")
                require(r["image_path"] == original["image_path"] == frame.loc[identifier, "image_path"],
                        "Image path mismatch")
                require(r["metadata"]["cohort"] == frame.loc[identifier, "family"] and
                        r["metadata"]["formula_family_id"] == frame.loc[identifier, "pair_id"], "Metadata mismatch")
                prompt = original["prompt"] + suffix
                require(hashlib.sha256(prompt.encode()).hexdigest() == r["prompt_sha256"], "Prompt mismatch")
                require(manifest["images"][r["image_path"]]["prepared_sha256"] == r["prepared_image_sha256"],
                        "Prepared-image evidence mismatch")
                parsed = parse_answer(r["response"], answer_format=answer_format) if r["complete"] else None
                require(parsed == r["prediction"] and (parsed == r["target"]) == r["correct"], "Parser mismatch")
                extras.append(dict(id=identifier, model=model, format=fmt,
                                   correct=int(r["correct"]), valid=int(r["status"] == "ok"),
                                   prediction=r["prediction"],
                                   token_limit=int(r["status"] == "invalid" and r.get("finish_reason")
                                                   in ("length", "MAX_TOKENS", "max_output_tokens")),
                                   prompt_words=len(prompt.split())))
            joined = joined.join(pd.DataFrame(extras).set_index("id"))
            frames.append(joined.reset_index())
            print(f"Verified {model}: {fmt}", flush=True)
    return pd.concat(frames, ignore_index=True), verified, source_hashes


def metrics(g):
    n, correct, valid = len(g), int(g.correct.sum()), int(g.valid.sum())
    yes, no = g[g.target == "Yes"], g[g.target == "No"]
    p = g.groupby("pair_id").agg(valid=("valid", "sum"), correct=("correct", "sum"),
                                 distinct=("prediction", "nunique"), n=("valid", "size"))
    p = p[p.n == 2]
    nv = int((p.valid == 2).sum())
    conflicts = int(((p.valid == 2) & (p.distinct == 2)).sum())
    return dict(n=n, scenes=int(g.scene_id.nunique()), reference_yes=len(yes), reference_no=len(no),
                correct=correct, accuracy=correct/n, failure_rate=1-correct/n,
                valid=valid, wrong_valid=valid-correct, invalid=n-valid,
                token_limit=int(g.token_limit.sum()), accuracy_on_valid=correct/valid if valid else None,
                balanced_accuracy=(float(yes.correct.mean()+no.correct.mean())/2 if len(yes) and len(no) else None),
                majority_label_baseline=max(len(yes), len(no))/n,
                prediction_yes=int((g.prediction == "Yes").sum()),
                pairs=len(p), valid_pairs=nv, pair_conflicts=conflicts,
                pair_conflict_rate=conflicts/nv if nv else None,
                pairs_both_correct=int((p.correct == 2).sum()),
                pairs_both_correct_accuracy=float((p.correct == 2).mean()) if len(p) else None)


def add_groups(df):
    df["certificate_band"] = pd.cut(df.certificate_size, [2,4,6,10], labels=["3-4", "5-6", "7-10"]).astype(str)
    df["predictability_band"] = pd.cut(df.two_fact_predictability, [0,.55,.60,.70,1],
                                       labels=["<=0.55", "(0.55,0.60]", "(0.60,0.70]", ">0.70"]).astype(str)
    df["influence_band"] = pd.cut(df.total_influence, [0,3,5,10],
                                   labels=["<=3", "(3,5]", ">5"]).astype(str)
    df["word_quartile"] = df.groupby(["model","format"]).prompt_words.transform(
        lambda x: pd.qcut(x, 4, labels=["Q1", "Q2", "Q3", "Q4"]).astype(str))
    return df


def grouped_metrics(df):
    rows = []
    fields = ("certificate_size", "certificate_band", "predictability_band", "influence_band",
              "broad_dependence", "weak_two_fact_clues", "family", "word_quartile", "variant", "track")
    for (model, fmt), group in df.groupby(["model", "format"], sort=False):
        for field in fields:
            for value, g in group.groupby(field, sort=True):
                rows.append(dict(model=model, format=fmt, property=field, group=str(value), **metrics(g)))
    return rows


def within_family(df):
    rows = []
    for (model, fmt, family), group in df.groupby(["model", "format", "family"], sort=False):
        for field in ("certificate_size", "predictability_band", "influence_band", "weak_two_fact_clues"):
            for value, g in group.groupby(field, sort=True):
                rows.append(dict(model=model, format=fmt, family=family,
                                 property=field, group=str(value), **metrics(g)))
    return rows


def slopes(df):
    """Descriptive FWL linear-probability slopes, without iid p-values/CIs.

    One logical feature at a time; controls: family x track x gold label,
    log request word count and rewrite indicator. Standardization is global
    within the given run, before residualization. Not causal estimates.
    """
    rows = []
    for (model, fmt), g in df.groupby(["model", "format"], sort=False):
        strata = g.family + "/" + g.track + "/" + g.target
        controls = np.column_stack([pd.get_dummies(strata).to_numpy(dtype=float),
                                    np.log(g.prompt_words), (g.variant == "de_morgan").astype(float)])
        outcomes = np.column_stack([g.correct, 1-g.valid, g.token_limit]).astype(float)
        y_res = outcomes - controls @ np.linalg.lstsq(controls, outcomes, rcond=None)[0]
        for feature in FEATURES:
            x = g[feature].to_numpy(dtype=float)
            sd = x.std()
            x = (x-x.mean())/sd
            xr = x-controls @ np.linalg.lstsq(controls, x, rcond=None)[0]
            share = float(xr @ xr / (x @ x))
            beta = xr @ y_res / (xr @ xr) if share > 1e-10 else [np.nan]*3
            raw = x @ outcomes / (x @ x)
            rows.append(dict(model=model, format=fmt, property=feature, sd=sd,
                             residual_variance_fraction=share,
                             residual_sd_in_original_units=sd*np.sqrt(share),
                             raw_accuracy_pp_per_sd=float(100*raw[0]),
                             adjusted_accuracy_pp_per_sd=float(100*beta[0]),
                             adjusted_accuracy_pp_per_residual_sd=float(100*beta[0]*np.sqrt(share)),
                             adjusted_invalid_pp_per_sd=float(100*beta[1]),
                             adjusted_token_limit_pp_per_sd=float(100*beta[2])))
    return rows


def length_and_pairs(df):
    """Matched format outcomes for extra length/output-validity diagnostics."""
    rows = []
    for model, g in df.groupby("model", sort=False):
        sym = g[g.format == "symbolic"].set_index("id")
        nl = g[g.format == "natural-language"].set_index("id").loc[sym.index]
        require((sym.target == nl.target).all(), "Cross-format labels differ")
        for label, mask in (("all", pd.Series(True, index=sym.index)),
                            ("valid_both_formats", (sym.valid == 1) & (nl.valid == 1))):
            s, n = sym.loc[mask], nl.loc[mask]
            rows.append(dict(model=model, subset=label, n=len(s), symbolic_correct=int(s.correct.sum()),
                             nl_correct=int(n.correct.sum()), symbolic_accuracy=float(s.correct.mean()),
                             nl_accuracy=float(n.correct.mean()),
                             nl_minus_symbolic_pp=float(100*(n.correct.mean()-s.correct.mean())),
                             symbolic_only_correct=int(((s.correct == 1) & (n.correct == 0)).sum()),
                             nl_only_correct=int(((s.correct == 0) & (n.correct == 1)).sum())))
    return rows


def property_support(frame):
    rows=[]
    for family,g in frame.groupby("family"):
        row=dict(family=family, questions=len(g), pairs=g.pair_id.nunique())
        for f in FEATURES:
            row[f]=dict(min=float(g[f].min()), max=float(g[f].max()), unique=int(g[f].nunique()))
        rows.append(row)
    return rows


def glm_transitions(df):
    rows=[]
    for fmt in FORMATS[:2]:
        off=df[(df.model == "GLM FlashX (off)") & (df.format == fmt)].set_index("id")
        on=df[(df.model == "GLM FlashX (on)") & (df.format == fmt)].set_index("id").loc[off.index]
        for band in ["all", "3-4", "5-6", "7-10"]:
            mask=off.certificate_band == band if band != "all" else pd.Series(True,index=off.index)
            a,b=off.loc[mask],on.loc[mask]
            rows.append(dict(format=fmt, certificate_band=band, n=len(a),
                             off_correct=int(a.correct.sum()), on_correct=int(b.correct.sum()),
                             recovered=int(((a.correct == 0)&(b.correct == 1)).sum()),
                             lost=int(((a.correct == 1)&(b.correct == 0)).sum()),
                             both_correct=int(((a.correct == 1)&(b.correct == 1)).sum()),
                             neither_correct=int(((a.correct == 0)&(b.correct == 0)).sum()),
                             change_pp=float(100*(b.correct.mean()-a.correct.mean()))))
    return rows


def report(grouped, within, adjusted, matched, support, verified, transitions, output):
    """Generate a descriptive report of the fixed archived results."""
    def get(model,fmt,prop,band):
        return next(r for r in grouped if (r['model'],r['format'],r['property'],r['group'])
                    == (model,fmt,prop,str(band)))
    def percent(x):
        return f"{100*x:.2f}"
    lines=["# Boolean-property analysis", "",
           "Reproduce offline with `python -m evals.analyze_properties` (numpy, pandas, matplotlib).",
           "Archived runs, annotations, and scores are read only; no API calls are made.", "",
           "## Evidence and coverage", "",
           f"Validated all {len(verified)} full task-format runs against the saved paper snapshot: "
           f"{sum(r['questions'] for r in verified):,} final responses, no pending API failures. "
           "The property analysis joins 108,000 Boolean responses from nine configurations and two formats "
           "to 6,000 question records by ID. Infrastructure retries are not counted as new observations. "
           "All Boolean final answers were reparsed under their saved policies; targets, input hashes, "
           "prepared-image evidence, prompt hashes, summary totals and equivalent-pair invariants were checked. "
           "`audit.json` records source and script hashes.", "",
           "## How to read the results", "",
           "- Accuracy includes invalid outputs as incorrect; failure rate is one minus this accuracy.",
           "- Balanced accuracy averages Yes-label and No-label recall, still treating invalid outputs as incorrect. "
           "Its constant-label baseline is 50% when both labels are present. It is undefined for a single-label subgroup.",
           "- The majority-label baseline in the JSON is calculated from each subgroup's observed target proportions; "
           "it is descriptive and not a learned held-out baseline.",
           "- A2 (two-fact predictability) is the optimal fixed-two-input predictor over the uniform truth-table domain. "
           "It is not measured model fact accuracy, and its value is not a baseline accuracy on these selected images.",
           "- Total influence is the sum of all ten input influences over the uniform truth-table domain. "
           "It is not a measurement of visual errors or of model attention.",
           "- Certificate size is instance-specific; all abstract completions are considered, as in the dataset annotation.",
           "- Bins were chosen before inspecting model scores: certificates 3–4/5–6/7–10; A2 <=.55/(.55,.60]/(.60,.70]/>.70; "
           "total influence <=3/(3,5]/>5. Exact certificate sizes are also reported, including sparse cells.",
           "- Counts include both expression variants. Observations share images and logical functions; no iid confidence "
           "intervals or p-values are claimed. All analyses are exploratory and descriptive, not causal.", "",
           "## Main finding: certificate size separates thinking-enabled GLM performance", "",
           "| Certificate size | Questions per format | GLM off symbolic | GLM on symbolic | GLM off NL | GLM on NL |",
           "|---|---:|---:|---:|---:|---:|"]
    for band in ["3-4","5-6","7-10"]:
        cells=[get(model,fmt,"certificate_band",band) for fmt in FORMATS[:2]
               for model in ("GLM FlashX (off)","GLM FlashX (on)")]
        vals=[percent(r['accuracy']) for r in cells]
        lines.append(f"| {band} | {cells[0]['n']:,} | " + " | ".join(v+"%" for v in vals)+" |")
    lines += ["", "For thinking-enabled GLM symbolic evaluation, balanced accuracies are " +
              ", ".join(percent(get("GLM FlashX (on)","symbolic","certificate_band",b)['balanced_accuracy'])+"%"
                        for b in ["3-4","5-6","7-10"])+
              ". Thus the pooled gradient is not explained by the small target-label imbalances in these groups. "
              "It remains an association across different family mixtures, not proof of difficulty caused by needing more facts.",
              "", "## Predictability and influence", "",
              "| Property | Group | N | GLM on symbolic accuracy | GLM on NL accuracy |",
              "|---|---|---:|---:|---:|"]
    for prop,bands in [("predictability_band",["<=0.55","(0.55,0.60]","(0.60,0.70]",">0.70"]),
                       ("influence_band",["<=3","(3,5]",">5"])]:
        for b in bands:
            s,n=(get("GLM FlashX (on)",f,prop,b) for f in FORMATS[:2])
            lines.append(f"| {prop} | {b} | {s['n']:,} | {percent(s['accuracy'])}% | {percent(n['accuracy'])}% |")
    lines += ["", "GLM's higher symbolic scores occur in groups with more informative two-fact predictors and lower total "
              "influence. The other eight configurations do not show a comparable, consistent advantage on these groups; "
              "the full nine-configuration balanced-accuracy heatmap is in `property_accuracy.png`/`.svg`. "
              "These results do not show that GLM actually uses two facts or that visual misreads explain its errors.",
              "", "## Essential qualification: properties overlap with construction family", "",
              "| Family | Questions | Certificate range | A2 range | Total influence range |",
              "|---|---:|---|---|---|"]
    for r in support:
        cells=[]
        for f in FEATURES:
            x=r[f]
            cells.append(f"{x['min']:.4g}" if x['unique']==1 else f"{x['min']:.4g}–{x['max']:.4g}")
        lines.append(f"| {r['family']} | {r['questions']} | "+" | ".join(cells)+" |")
    lines += ["", "A2 and total influence vary within only the two CNF families; the other five families each have fixed "
              "values. Certificate size varies within five families. Therefore most of the pooled A2/influence variation "
              "is a family comparison. `within_family.json` gives all models, both formats, exact certificate cells and "
              "within-family property bands; missing cells are not fabricated.", "",
              "### Within-family certificate results for thinking-enabled GLM", "",
              "| Family | Certificate | N | Symbolic accuracy | NL accuracy |",
              "|---|---:|---:|---:|---:|"]
    for family in [r['family'] for r in support if r['certificate_size']['unique']>1]:
        subset=[r for r in within if r['model']=="GLM FlashX (on)" and r['format']=="symbolic"
                and r['family']==family and r['property']=="certificate_size"]
        for s in subset:
            n=next(r for r in within if r['model']==s['model'] and r['format']=="natural-language"
                   and r['family']==family and r['property']==s['property'] and r['group']==s['group'])
            lines.append(f"| {family} | {s['group']} | {s['n']} | {percent(s['accuracy'])}% | {percent(n['accuracy'])}% |")
    lines += ["", "The relationship is not uniformly monotonic. For example, sparse-CNF symbolic accuracy decreases "
              "from certificate 3 through 5 but rises at size 6 (only 22 questions/11 expression pairs). "
              "Parity of conjunctions also has reversals. Certificate size should not be presented as a universal difficulty ordering.",
              "", "### Adjustment sensitivity check", "",
              "`adjusted_associations.json` reports one-feature-at-a-time descriptive linear-probability fits, "
              "controlling for family × image-source × reference-label strata, log actual prompt word count, and rewrite status. "
              "Logical properties are not mutually adjusted. Both original-SD and residual-SD slopes are recorded; "
              "there are no significance claims. Fits do not isolate perception, reasoning or other causes.", "",
              "| Property | Variance remaining after controls (symbolic) | GLM on accuracy change per residual SD (pp) |",
              "|---|---:|---:|"]
    for f in FEATURES:
        r=next(r for r in adjusted if r['model']=="GLM FlashX (on)" and r['format']=="symbolic" and r['property']==f)
        lines.append(f"| {f} | {100*r['residual_variance_fraction']:.3f}% | {r['adjusted_accuracy_pp_per_residual_sd']:+.2f} |")
    lines += ["", "Less than 0.1% of total-influence variance remains after these controls; an independent influence effect "
              "is poorly separated from family in this dataset. The pooled A2 association also does not persist as a positive "
              "adjusted slope. These checks constrain the interpretation rather than establish a model mechanism.",
              "", "## Additional analysis: invalid outputs and matched format comparison", "",
              "| Model | IDs valid in both formats | Symbolic accuracy on those IDs | NL accuracy on those IDs | NL − symbolic (pp) |",
              "|---|---:|---:|---:|---:|"]
    for r in matched:
        if r['subset']=="valid_both_formats":
            lines.append(f"| {r['model']} | {r['n']:,} | {percent(r['symbolic_accuracy'])}% | "
                         f"{percent(r['nl_accuracy'])}% | {r['nl_minus_symbolic_pp']:+.2f} |")
    lines += ["", "For thinking-enabled GLM, the full-sample symbolic advantage is 12.30 pp. It remains 8.25 pp on "
              "the same 5,421 IDs with valid answers in both formats. Invalid outputs therefore do not fully account for "
              "the observed format difference. This selected subset is a supplementary analysis, not a replacement score "
              "or an estimate of what truncated responses would have answered.",
              "", "### Thinking-enabled GLM and prompt length", "",
              "| Format | Word-count quartile | N | Accuracy | Invalid | Token-limit failures |",
              "|---|---|---:|---:|---:|---:|"]
    for fmt in FORMATS[:2]:
        for band in ["Q1","Q2","Q3","Q4"]:
            r=get("GLM FlashX (on)",fmt,"word_quartile",band)
            lines.append(f"| {fmt} | {band} | {r['n']} | {percent(r['accuracy'])}% | {r['invalid']} | {r['token_limit']} |")
    lines += ["", "Q1 contains shorter prompts and Q4 longer prompts, separately within each format. "
              "Ties can make quartile counts unequal. Longer prompts and family/structure co-vary; this is not an isolated "
              "causal effect of prompt length, and the invalid-output trend is not strictly monotonic.",
              "", "## Matched GLM off/on transitions", "",
              "| Format | Certificate | N | Off wrong → on correct | Off correct → on wrong | Net accuracy change (pp) |",
              "|---|---|---:|---:|---:|---:|"]
    for r in transitions:
        lines.append(f"| {r['format']} | {r['certificate_band']} | {r['n']} | {r['recovered']} | {r['lost']} | {r['change_pp']:+.2f} |")
    lines += ["", "Invalid outputs are incorrect in these transitions. Thinking mode and output cap differ together "
              "(2,048 off, 4,096 on), so these are configuration comparisons.",
              "", "## Interpretation and limitations", "",
              "The annotations reveal heterogeneous performance, most clearly for thinking-enabled GLM; "
              "they do not isolate the causes of its failures or define a universal difficulty scale.", "",
              "Further experiments still needed for stronger diagnostic attribution: atomic-fact queries, evaluation "
              "with correct truth values supplied, exact evaluation of predicted facts, repeated identical/rewrite prompts, "
              "and matched-budget/additional thinking configurations. None of these new model experiments was run here.", "",
              "## Files", "",
              "- `audit.json`: run completeness and source hashes.",
              "- `grouped_results.json`: all nine configurations, subgroup accuracies, label balance, invalid/length counts and pair outcomes.",
              "- `within_family.json`: corresponding within-family groups.",
              "- `adjusted_associations.json`: exploratory adjustment and residual-variance diagnostics.",
              "- `property_support.json`: which properties actually vary within each family.",
              "- `matched_formats.json`, `glm_transitions.json`: additional matched comparisons.",
              "- `question_results.jsonl.gz`: compressed joined per-question evidence (no raw response prose).",
              "- `property_accuracy.png` / `.svg`: figure with all configurations."]
    (output/"README.md").write_text("\n".join(lines)+"\n")


def plots(grouped, output):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "hob-vl-properties-v1"
    import matplotlib.pyplot as plt
    g=pd.DataFrame(grouped)
    # Match the ICLR text width so fonts remain readable at \linewidth.
    # Properties occupy rows; formats occupy columns, sharing model labels.
    fig, axes=plt.subplots(3,2,figsize=(5.5,7.4),layout="constrained")
    models=[r[0] for r in RUNS]
    specs=[("certificate_band",["3-4","5-6","7-10"],"Certificate size",
            ["3–4","5–6","7–10"]),
           ("predictability_band",["<=0.55","(0.55,0.60]","(0.60,0.70]",">0.70"],
            "Best two-fact predictability",
            [r"$\leq 0.55$", "$(0.55,$\n$0.60]$", "$(0.60,$\n$0.70]$", r"$>0.70$"]),
           ("influence_band",["<=3","(3,5]",">5"],"Total influence",
            [r"$\leq 3$", r"$(3,5]$", r"$>5$"])]
    format_labels={"symbolic":"Symbolic", "natural-language":"Structured natural language"}
    for i,(prop,order,title,tick_labels) in enumerate(specs):
        for j,fmt in enumerate(FORMATS[:2]):
            ax=axes[i,j]
            sub=g[(g.format == fmt)&(g.property == prop)]
            mat=sub.pivot(index="model",columns="group",values="balanced_accuracy").reindex(index=models,columns=order)*100
            im=ax.imshow(mat,vmin=30,vmax=90,cmap="viridis",aspect="auto")
            ax.set_xticks(range(len(order)),tick_labels,fontsize=8)
            ax.set_yticks(range(len(models)),models if j == 0 else [],fontsize=8)
            ax.tick_params(axis="both",length=0,pad=4)
            ax.set_title(title,fontsize=9,pad=7)
            if i == 0:
                ax.text(.5,1.22,format_labels[fmt],transform=ax.transAxes,
                        ha="center",va="bottom",fontsize=9,fontweight="bold")
            for spine in ax.spines.values():
                spine.set_linewidth(.5)
            for y in range(len(models)):
                for x in range(len(order)):
                    v=mat.iloc[y,x]
                    ax.text(x,y,f"{v:.1f}",ha="center",va="center",fontsize=8.5,
                            color="black" if v>65 else "white")
    colorbar=fig.colorbar(im,ax=axes,orientation="horizontal",fraction=.035,pad=.035,
                         shrink=.8,aspect=35,ticks=[30,40,50,60,70,80,90])
    colorbar.set_label("Balanced accuracy (%)",fontsize=9)
    colorbar.ax.tick_params(labelsize=8,length=3)
    fig.savefig(output/"property_accuracy.png",dpi=330)
    fig.savefig(output/"property_accuracy.svg", metadata={"Date": None})
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"results/property_analysis")
    parser.add_argument("--runs-dir", type=Path, default=ROOT/"results/runs")
    parser.add_argument("--snapshot", type=Path, default=ROOT/"results/generated/results.json")
    args=parser.parse_args()
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    frame=annotations()
    df,verified,hashes=load_results(frame, args.runs_dir.resolve(), args.snapshot.resolve())
    df=add_groups(df)
    grouped=grouped_metrics(df)
    within=within_family(df)
    adjusted=slopes(df)
    matched=length_and_pairs(df)
    support=property_support(frame)
    transitions=glm_transitions(df)
    write_json(output/"audit.json",dict(runs=verified,final_responses=sum(r["questions"] for r in verified),
                                       analyzed_boolean_responses=len(df),source_sha256=hashes,
                                       script_sha256=sha(Path(__file__))))
    write_json(output/"grouped_results.json",grouped)
    write_json(output/"within_family.json",within)
    write_json(output/"adjusted_associations.json",adjusted)
    write_json(output/"matched_formats.json",matched)
    write_json(output/"property_support.json",support)
    write_json(output/"glm_transitions.json",transitions)
    # Compact joined evidence, excluding raw prose, for independent recalculation.
    df.to_json(output/"question_results.jsonl.gz",orient="records",lines=True,double_precision=15,
               compression={"method":"gzip", "mtime":0})
    plots(grouped,output)
    report(grouped,within,adjusted,matched,support,verified,transitions,output)
    print(f"Saved offline analysis to {output}",flush=True)


if __name__ == "__main__":
    main()
