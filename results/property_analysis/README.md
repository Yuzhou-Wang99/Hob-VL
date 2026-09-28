# Boolean-property analysis

Reproduce offline with `python -m evals.analyze_properties` (numpy, pandas, matplotlib).
Archived runs, annotations, and scores are read only; no API calls are made.

## Evidence and coverage

Validated all 36 full task-format runs against the saved paper snapshot: 126,000 final responses, no pending API failures. The property analysis joins 108,000 Boolean responses from nine configurations and two formats to 6,000 question records by ID. Infrastructure retries are not counted as new observations. All Boolean final answers were reparsed under their saved policies; targets, input hashes, prepared-image evidence, prompt hashes, summary totals and equivalent-pair invariants were checked. `audit.json` records source and script hashes.

## How to read the results

- Accuracy includes invalid outputs as incorrect; failure rate is one minus this accuracy.
- Balanced accuracy averages Yes-label and No-label recall, still treating invalid outputs as incorrect. Its constant-label baseline is 50% when both labels are present. It is undefined for a single-label subgroup.
- The majority-label baseline in the JSON is calculated from each subgroup's observed target proportions; it is descriptive and not a learned held-out baseline.
- A2 (two-fact predictability) is the optimal fixed-two-input predictor over the uniform truth-table domain. It is not measured model fact accuracy, and its value is not a baseline accuracy on these selected images.
- Total influence is the sum of all ten input influences over the uniform truth-table domain. It is not a measurement of visual errors or of model attention.
- Certificate size is instance-specific; all abstract completions are considered, as in the dataset annotation.
- Bins were chosen before inspecting model scores: certificates 3–4/5–6/7–10; A2 <=.55/(.55,.60]/(.60,.70]/>.70; total influence <=3/(3,5]/>5. Exact certificate sizes are also reported, including sparse cells.
- Counts include both expression variants. Observations share images and logical functions; no iid confidence intervals or p-values are claimed. All analyses are exploratory and descriptive, not causal.

## Main finding: certificate size separates thinking-enabled GLM performance

| Certificate size | Questions per format | GLM off symbolic | GLM on symbolic | GLM off NL | GLM on NL |
|---|---:|---:|---:|---:|---:|
| 3-4 | 1,772 | 45.71% | 82.62% | 52.88% | 62.19% |
| 5-6 | 2,856 | 49.02% | 57.35% | 48.98% | 47.41% |
| 7-10 | 1,372 | 51.38% | 51.60% | 50.00% | 44.90% |

For thinking-enabled GLM symbolic evaluation, balanced accuracies are 82.62%, 57.44%, 51.60%. Thus the pooled gradient is not explained by the small target-label imbalances in these groups. It remains an association across different family mixtures, not proof of difficulty caused by needing more facts.

## Predictability and influence

| Property | Group | N | GLM on symbolic accuracy | GLM on NL accuracy |
|---|---|---:|---:|---:|
| predictability_band | <=0.55 | 2,000 | 53.35% | 46.65% |
| predictability_band | (0.55,0.60] | 1,030 | 52.52% | 45.05% |
| predictability_band | (0.60,0.70] | 2,488 | 71.95% | 54.34% |
| predictability_band | >0.70 | 482 | 85.48% | 67.01% |
| influence_band | <=3 | 2,436 | 76.97% | 57.68% |
| influence_band | (3,5] | 2,564 | 56.01% | 47.07% |
| influence_band | >5 | 1,000 | 49.90% | 46.00% |

GLM's higher symbolic scores occur in groups with more informative two-fact predictors and lower total influence. The other eight configurations do not show a comparable, consistent advantage on these groups; the full nine-configuration balanced-accuracy heatmap is in `property_accuracy.png`/`.svg`. These results do not show that GLM actually uses two facts or that visual misreads explain its errors.

## Essential qualification: properties overlap with construction family

| Family | Questions | Certificate range | A2 range | Total influence range |
|---|---:|---|---|---|
| dense_cnf | 1000 | 4–8 | 0.5889–0.7158 | 2.551–3.305 |
| majority_parity | 1000 | 6 | 0.5703 | 3.75 |
| nested_mux | 1500 | 4–5 | 0.625 | 2.859 |
| parity_control | 500 | 10 | 0.5 | 10 |
| polarity_trap_composition | 500 | 7–10 | 0.5 | 6.5 |
| quadratic_parity | 1000 | 5–10 | 0.5312 | 5 |
| sparse_cnf_control | 500 | 3–6 | 0.6816–0.835 | 1.766–2.477 |

A2 and total influence vary within only the two CNF families; the other five families each have fixed values. Certificate size varies within five families. Therefore most of the pooled A2/influence variation is a family comparison. `within_family.json` gives all models, both formats, exact certificate cells and within-family property bands; missing cells are not fabricated.

### Within-family certificate results for thinking-enabled GLM

| Family | Certificate | N | Symbolic accuracy | NL accuracy |
|---|---:|---:|---:|---:|
| dense_cnf | 4 | 38 | 65.79% | 52.63% |
| dense_cnf | 5 | 718 | 57.80% | 49.30% |
| dense_cnf | 6 | 204 | 60.29% | 44.61% |
| dense_cnf | 7 | 38 | 52.63% | 34.21% |
| dense_cnf | 8 | 2 | 50.00% | 0.00% |
| nested_mux | 4 | 1328 | 81.25% | 58.96% |
| nested_mux | 5 | 172 | 75.00% | 55.23% |
| polarity_trap_composition | 7 | 486 | 47.74% | 50.41% |
| polarity_trap_composition | 10 | 14 | 35.71% | 21.43% |
| quadratic_parity | 5 | 232 | 74.14% | 51.72% |
| quadratic_parity | 6 | 436 | 47.71% | 48.17% |
| quadratic_parity | 7 | 238 | 62.18% | 39.92% |
| quadratic_parity | 8 | 74 | 43.24% | 56.76% |
| quadratic_parity | 9 | 14 | 50.00% | 21.43% |
| quadratic_parity | 10 | 6 | 16.67% | 50.00% |
| sparse_cnf_control | 3 | 330 | 90.61% | 77.27% |
| sparse_cnf_control | 4 | 76 | 80.26% | 57.89% |
| sparse_cnf_control | 5 | 72 | 69.44% | 43.06% |
| sparse_cnf_control | 6 | 22 | 95.45% | 36.36% |

The relationship is not uniformly monotonic. For example, sparse-CNF symbolic accuracy decreases from certificate 3 through 5 but rises at size 6 (only 22 questions/11 expression pairs). Parity of conjunctions also has reversals. Certificate size should not be presented as a universal difficulty ordering.

### Adjustment sensitivity check

`adjusted_associations.json` reports one-feature-at-a-time descriptive linear-probability fits, controlling for family × image-source × reference-label strata, log actual prompt word count, and rewrite status. Logical properties are not mutually adjusted. Both original-SD and residual-SD slopes are recorded; there are no significance claims. Fits do not isolate perception, reasoning or other causes.

| Property | Variance remaining after controls (symbolic) | GLM on accuracy change per residual SD (pp) |
|---|---:|---:|
| certificate_size | 10.422% | -2.80 |
| two_fact_predictability | 2.954% | -0.64 |
| total_influence | 0.075% | +0.85 |

Less than 0.1% of total-influence variance remains after these controls; an independent influence effect is poorly separated from family in this dataset. The pooled A2 association also does not persist as a positive adjusted slope. These checks constrain the interpretation rather than establish a model mechanism.

## Additional analysis: invalid outputs and matched format comparison

| Model | IDs valid in both formats | Symbolic accuracy on those IDs | NL accuracy on those IDs | NL − symbolic (pp) |
|---|---:|---:|---:|---:|
| Luna | 5,998 | 49.07% | 48.68% | -0.38 |
| Haiku 4.5 | 5,930 | 49.83% | 50.39% | +0.56 |
| Grok 4.3 | 6,000 | 48.75% | 49.57% | +0.82 |
| Gemini Flash-Lite | 6,000 | 48.60% | 48.52% | -0.08 |
| DeepSeek Flash | 5,927 | 50.23% | 49.77% | -0.46 |
| GLM FlashX (off) | 6,000 | 48.58% | 50.37% | +1.78 |
| Llama 4 Scout | 6,000 | 49.98% | 48.93% | -1.05 |
| Qwen3-VL 8B | 6,000 | 50.57% | 50.37% | -0.20 |
| GLM FlashX (on) | 5,421 | 64.38% | 56.13% | -8.25 |

For thinking-enabled GLM, the full-sample symbolic advantage is 12.30 pp. It remains 8.25 pp on the same 5,421 IDs with valid answers in both formats. Invalid outputs therefore do not fully account for the observed format difference. This selected subset is a supplementary analysis, not a replacement score or an estimate of what truncated responses would have answered.

### Thinking-enabled GLM and prompt length

| Format | Word-count quartile | N | Accuracy | Invalid | Token-limit failures |
|---|---|---:|---:|---:|---:|
| symbolic | Q1 | 1516 | 84.56% | 5 | 5 |
| symbolic | Q2 | 1484 | 62.33% | 8 | 8 |
| symbolic | Q3 | 1523 | 53.05% | 17 | 17 |
| symbolic | Q4 | 1477 | 53.83% | 17 | 16 |
| natural-language | Q1 | 1502 | 62.78% | 78 | 78 |
| natural-language | Q2 | 1498 | 49.80% | 127 | 127 |
| natural-language | Q3 | 1500 | 45.47% | 182 | 182 |
| natural-language | Q4 | 1500 | 46.73% | 149 | 149 |

Q1 contains shorter prompts and Q4 longer prompts, separately within each format. Ties can make quartile counts unequal. Longer prompts and family/structure co-vary; this is not an isolated causal effect of prompt length, and the invalid-output trend is not strictly monotonic.

## Matched GLM off/on transitions

| Format | Certificate | N | Off wrong → on correct | Off correct → on wrong | Net accuracy change (pp) |
|---|---|---:|---:|---:|---:|
| symbolic | all | 6000 | 1804 | 909 | +14.92 |
| symbolic | 3-4 | 1772 | 806 | 152 | +36.91 |
| symbolic | 5-6 | 2856 | 734 | 496 | +8.33 |
| symbolic | 7-10 | 1372 | 264 | 261 | +0.22 |
| natural-language | all | 6000 | 998 | 948 | +0.83 |
| natural-language | 3-4 | 1772 | 469 | 304 | +9.31 |
| natural-language | 5-6 | 2856 | 394 | 439 | -1.58 |
| natural-language | 7-10 | 1372 | 135 | 205 | -5.10 |

Invalid outputs are incorrect in these transitions. Thinking mode and output cap differ together (2,048 off, 4,096 on), so these are configuration comparisons.

## Interpretation and limitations

The annotations reveal heterogeneous performance, most clearly for thinking-enabled GLM; they do not isolate the causes of its failures or define a universal difficulty scale.

Further experiments still needed for stronger diagnostic attribution: atomic-fact queries, evaluation with correct truth values supplied, exact evaluation of predicted facts, repeated identical/rewrite prompts, and matched-budget/additional thinking configurations. None of these new model experiments was run here.

## Files

- `audit.json`: run completeness and source hashes.
- `grouped_results.json`: all nine configurations, subgroup accuracies, label balance, invalid/length counts and pair outcomes.
- `within_family.json`: corresponding within-family groups.
- `adjusted_associations.json`: exploratory adjustment and residual-variance diagnostics.
- `property_support.json`: which properties actually vary within each family.
- `matched_formats.json`, `glm_transitions.json`: additional matched comparisons.
- `question_results.jsonl.gz`: compressed joined per-question evidence (no raw response prose).
- `property_accuracy.png` / `.svg`: figure with all configurations.
