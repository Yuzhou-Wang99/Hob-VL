# Saved benchmark results

Regenerate offline with `python -m evals.paper_results`.

Eight model identities, nine inference configurations, four formats, 126,000 final responses.
Percentages use all questions; invalid answers count as incorrect. Historical API failures are superseded by retries.

| Configuration | Boolean symbolic | Boolean NL | Identification symbolic | Identification NL |
|---|---:|---:|---:|---:|
| Luna | 49.05% | 48.67% | 35.70% | 40.00% |
| Haiku 4.5 | 49.37% | 50.22% | 32.20% | 43.00% |
| Grok 4.3 | 48.75% | 49.57% | 21.60% | 25.70% |
| Gemini Flash-Lite | 48.60% | 48.52% | 27.10% | 41.10% |
| DeepSeek Flash | 49.62% | 49.67% | 16.50% | 24.30% |
| GLM FlashX (off) | 48.58% | 50.37% | 31.30% | 35.00% |
| Llama 4 Scout | 49.98% | 48.93% | 27.80% | 33.30% |
| Qwen3-VL 8B | 50.57% | 50.37% | 23.00% | 28.60% |
| GLM FlashX (on) | 63.50% | 51.20% | 73.70% | 72.10% |

NL means structured natural language. GLM off/on mean thinking disabled/enabled; their caps are 2,048/4,096 tokens.

Identification baselines: uniform candidate guessing 16.90%; always predict B: 19.20% (192/1000).

`results.json` contains exact counts, all subgroup metrics, paired results, parser sensitivity, and source hashes.
`settings.csv` records requested identifiers and settings; `../configurations.json` additionally describes API request fields.
`invalid_outputs.csv` preserves the full final response and a parser/completion-based reason for every invalid answer.

Equivalent-expression both-correct accuracy uses all 3,000 pairs; conflict uses only pairs with two valid answers.
A constant-label predictor has 50% both-correct accuracy and zero conflict. Read these with accuracy and label frequencies.
Independent generations also include sampling variability; no repeated-identical-prompt control was run.
Thinking settings, output caps, and some prompt policies differ. These are descriptive configuration comparisons.
