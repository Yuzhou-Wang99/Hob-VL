# Hob-VL evaluation

This runner loads the supplied prompts and images, joins answers by ID, sends
only the prepared prompt and image, saves full responses, and scores answers.
The completed paper runs are bundled under `results/runs/`; each new inference
run creates separate records under `evals/runs/` by default.

## Installation and offline checks

Use Python 3.11+ and Pillow. From the package root:

```sh
python -m pip install -r evals/requirements.txt
python verify_release.py
python -m evals --model openai/offline-placeholder --data symbolic --limit 5 --dry-run
python -m evals.verify_offline
```

The placeholder model is for offline checks only. Dry-run mode is the default;
it validates joins, prepares selected images, and builds requests without
reading API keys or contacting providers. A dry run does not measure model
accuracy or cost. The full offline verifier also exercises simulated requests
and resume behavior, with network connections blocked.

| `--data` | Task | Inputs under `dataset/hob_vl_v1/` |
| --- | --- | --- |
| `symbolic` | Yes/No | `model_inputs.jsonl` |
| `natural-language` | Yes/No | `model_inputs_natural_language.jsonl` |
| `identification-symbolic` | Candidate label | `model_inputs_photo_identification_1000_symbolic.jsonl` |
| `identification` | Candidate label | `model_inputs_photo_identification_1000.jsonl` |

Boolean formats each contain 6,000 rows; identification formats each contain
1,000. Omitting `--limit` selects all rows for the chosen format. The four
conditions together require 14,000 model responses. `--seed` shuffles
deterministically; otherwise file order is preserved. `--track`, `--offset`,
and `--limit` can select subsets, which may omit equivalent-expression pairs.

## Prompts and scoring

`--prompt-policy original` uses dataset prompts without a suffix. The default
`boolean-one-word-v1` appends the following instruction to Boolean prompts:

> Respond with exactly one word: Yes or No. Do not include explanations, Markdown, punctuation, or any other text.

`--prompt-policy answer-only-v1` also appends this to identification prompts:

> Respond with exactly one candidate label. Do not include explanations, Markdown, punctuation, or any other text.

Suffixes are applied in memory and recorded in manifests and prompt hashes;
the dataset files remain unchanged. Use matching policies across formats and
models for matched comparisons.

The default `--answer-format plain` accepts only a whole Yes/No answer or one
candidate label, allowing case differences, surrounding whitespace, and one
terminal period or exclamation mark. `plain-or-bold` additionally accepts one
answer enclosed in Markdown bold. Explanatory prose, multiple answers, empty
outputs, and incomplete generations are invalid and count as incorrect. The
full response is retained even when parsing fails. No answer is recovered by
searching prose for the reference label.

## Live inference

Select a vision-capable model and endpoint available to your provider account.
The runner implements these provider prefixes and key-variable defaults:

| Prefix | Protocol | Default key environment variable |
| --- | --- | --- |
| `openai/` | Responses | `OPENAI_API_KEY` |
| `anthropic/` | Messages | `ANTHROPIC_API_KEY` |
| `gemini/` | generateContent | `GEMINI_API_KEY` |
| `openai-compatible/` | Chat Completions; explicit API root required | `OPENAI_API_KEY` |
| `qwen/` | Chat Completions; explicit regional/workspace API root required | `DASHSCOPE_API_KEY` |
| `zai/` | Chat Completions | `ZAI_API_KEY` |
| `deepseek/` | Chat Completions | `DEEPSEEK_API_KEY` |
| `deepinfra/` | Chat Completions | `DEEPINFRA_API_KEY` |

Configure the key in the environment; never put credentials in dataset files,
Git, or the model-ID argument. `--api-key-env NAME` selects a different key
variable. The manifest records its name, not its value. `.env` files are not
automatically loaded.

After configuring your provider and replacing the model placeholder, a small
live run can be started with:

```sh
python -m evals --model openai/YOUR_VISION_MODEL_ID \
  --data symbolic --limit 5 \
  --prompt-policy answer-only-v1 --answer-format plain-or-bold \
  --max-output-tokens 2048 --execute --output evals/runs/example-symbolic
```

`--execute` sends real requests and may incur charges. The output-token limit
is not a dollar spending cap. Temperature is omitted unless supplied. Image
preparation defaults to a maximum edge of 2,048 pixels and a 4 MiB inline limit.
Thinking controls are provider-specific; `python -m evals --help` lists the
supported flags, and manifests record requested settings. A folder name does
not establish a model's thinking mode or immutable backend version.

## Saved output and resume

Each run writes a manifest and, for live runs, full responses in
`predictions.jsonl` and scores in `summary.json`. The summary includes validity,
track/family breakdowns, usage when reported, and equivalent-expression results.
`inflight.json` records an interrupted request whose billable outcome may be
uncertain. The checkpoint history is append-only.

Resume with the same command, settings, and output directory, adding `--resume`.
Saved valid or invalid model outputs are retained. API/transport failures stay
pending; `--retry-pending` explicitly permits retrying pending requests, including
requests that may already have been billed. Changed data, prompts, settings, or
images require a new output directory. Runs created against a differently named
dataset snapshot cannot be resumed against this package.

`--continue-on-error` moves past transient failures; repeated failures can still
stop the run, and authentication/configuration failures stop immediately. Read
the final summary rather than assuming every started question has settled.

## Comparisons and offline rescoring

Use `python -m evals.compare --help` for paired-format comparisons and
`python -m evals.rescore --help` for explicit offline rescoring of saved outputs.
Neither operation performs model inference. Equivalent-expression pairing
(base versus De Morgan) is distinct from symbolic/NL pairing by question ID.
Both-correct accuracy and agreement have different interpretations: a
constant-label Boolean predictor has 50% item and both-correct accuracy and
perfect agreement on this balanced dataset.

## Saved paper results

The completed experimental archive lives under `results/runs/`, separate from
new inference outputs in `evals/runs/`. To inspect or regenerate its tables and
logical-property analyses without API calls, see [REPRODUCING.md](../REPRODUCING.md).
Published archives cannot be resumed with the inference runner.
