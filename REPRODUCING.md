# Reproducing the saved Hob-VL results

The release contains eight model identities evaluated in nine configurations:
Luna, Haiku 4.5, Grok 4.3, Gemini Flash-Lite, DeepSeek Flash, GLM FlashX with
thinking disabled, Llama 4 Scout, Qwen3-VL 8B, and GLM FlashX with thinking enabled.
Each configuration has 6,000 symbolic Boolean, 6,000 natural-language Boolean,
1,000 symbolic identification, and 1,000 natural-language identification responses.

## Environment and commands

Use Python 3.11 or 3.12. The saved reports were regenerated with Python 3.12,
NumPy 1.26.4, pandas 2.2.2, matplotlib 3.9.2, and Pillow 10.4.0. A virtual
environment keeps these optional analysis dependencies separate:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r evals/requirements-analysis.txt
python verify_release.py
python -m evals.verify_results
python -m evals.paper_results
python -m evals.analyze_properties
```

Run from the repository root. These commands are entirely offline after
dependency installation and do not read API keys. The two analysis commands
rewrite their derived outputs; the archived responses remain unchanged.

For an independent output directory without changing the shipped reports:

```sh
python -m evals.paper_results --output /tmp/hob-vl-tables
python -m evals.analyze_properties \
  --snapshot /tmp/hob-vl-tables/results.json \
  --output /tmp/hob-vl-properties
```

## Mapping results to files

| Reported result | Generated file |
| --- | --- |
| Accuracy across all configurations and formats | `results/generated/accuracy.csv` |
| Equivalent-expression both-correct accuracy and conflict | `results/generated/equivalence.csv` |
| Invalid counts and token-limit failures | `results/generated/invalid.csv` |
| Every invalid final response and its scoring reason | `results/generated/invalid_outputs.csv` |
| GLM thinking-off/on results by construction family | `results/generated/glm_cohorts.csv` |
| Synthetic versus photographic Boolean accuracy | `results/generated/tracks.csv` |
| Predicted Yes frequencies | `results/generated/label_bias.csv` |
| Matched symbolic/NL outcomes | `results/generated/format_pairs.csv` |
| Requested settings and run initialization dates | `results/generated/settings.csv` |
| Exact counts, parser sensitivity, and source checksums | `results/generated/results.json` |
| Certificate/predictability/influence groups | `results/property_analysis/grouped_results.json` |
| Within-family comparisons and property support | `results/property_analysis/within_family.json`, `property_support.json` |
| Descriptive adjusted associations | `results/property_analysis/adjusted_associations.json` |
| Matched format and GLM configuration transitions | `results/property_analysis/matched_formats.json`, `glm_transitions.json` |
| Boolean-property heatmap | `results/property_analysis/property_accuracy.png` and `.svg` |
| Joined Boolean per-question analysis records | `results/property_analysis/question_results.jsonl.gz` |

The numerical tables are CSV, and the explanations are Markdown. No manuscript
source or Overleaf project is required. The identification uniform-candidate
guessing baseline and candidate-count range are in `results.json`; the baseline
averages `1 / number_of_candidates` over identification questions. The same
JSON records the constant-label baseline, its count, and candidate coverage.

## What is preserved

Each `results/runs/<configuration>/<format>/` has `manifest.json`,
`predictions.jsonl`, and `summary.json`. Predictions retain the complete saved
final-response text, any separately stored thinking text, parsed answer,
correctness, invalid/error status, token usage, finish reason, and latency.
All saved attempts remain in order, including earlier infrastructure failures.
Only the last response for each ID contributes to reported accuracy.

The archive uses the release's IDs and image paths. Original absolute machine
paths have been replaced with relative paths. Qwen's account-specific endpoint
prefix is replaced with `WORKSPACE`; the provider and region are retained.
Dataset-file hashes, selected-ID hashes, and manifest fingerprints were updated
to refer to this release. The manifest's `archive` block retains original file
hashes, data hashes, and fingerprint so these changes are explicit.
Prompts, reference answers, model response text, thinking text, scoring, token
usage, and image bytes were not changed by result packaging.

The archive is for inspection and analysis, not resuming paid inference. Use a
new output directory under `evals/runs/` to run models again. See `evals/README.md`.

## Settings and interpretive limits

`results/configurations.json` gives provider/API routes, requested model IDs,
generation fields, prompt policies, parsers, image preparation, and the original
manifest-creation timestamp for each format. These timestamps are **not** the
completion time or a complete request-by-request time log. Unspecified sampling
parameters use provider defaults; their numeric defaults were not retained.
Requested aliases do not independently identify immutable backend versions.
In particular, the DeepSeek request used `deepseek-flash`; its archive directory
name is historical and does not verify a versioned backend.

- Primary accuracy includes invalid outputs as incorrect. A completed response
  containing explanation or multiple labels is not converted to a correct
  answer merely because it mentions the reference label. `invalid_outputs.csv`
  distinguishes incomplete generations, empty final responses, and saved-parser
  format violations; it does not infer the model's intended answer.
- Equivalent-expression both-correct accuracy uses all 3,000 pairs per Boolean
  format. Conflict uses only pairs with two valid outputs. A constant-label
  predictor attains 50% both-correct accuracy and zero conflict. Read these
  metrics with item accuracy and predicted-label frequencies.
- Generations are independent. Without repeated identical-prompt controls,
  conflict combines sampling variability and possible presentation sensitivity.
- GLM thinking off/on uses caps of 2,048/4,096 output tokens. The contrast does
  not isolate thinking from the changed cap. Other prompt policies also differ
  across configurations; see the recorded settings.
- Logical properties co-vary with construction family. Pooled associations
  describe the measured conditions, not isolated causal effects or a universal
  difficulty scale. `within_family.json` and `property_support.json` expose
  this confounding. Repeated scenes and equivalent expressions are dependent
  observations; these scripts do not claim independent-trial significance.
- Recorded usage is not a complete billing statement. Failed or interrupted
  requests can lack usage information; absent cost fields do not imply free use.

## Software checks

```sh
python -m unittest discover -s evals/tests
```

The tests use simulated providers. `verify_release.py` checks release checksums,
all 14,000 prompt/answer joins, pair alignment, and image decoding.
`evals.verify_results` additionally validates all 126,000 final scores and all
saved attempt histories against the released prompts, targets, and image hashes.
These are software/integrity checks, not substitutes for visual annotation review.
