# Hob-VL

Hob-VL evaluates visually grounded Boolean composition in vision-language models.
It contains two tasks, each in symbolic and structured natural-language formats.

| Task | Question IDs | Symbolic prompts | Natural-language prompts |
| --- | ---: | ---: | ---: |
| Boolean Yes/No evaluation | 6,000 | 6,000 | 6,000 |
| Object identification | 1,000 | 1,000 | 1,000 |
| Total | 7,000 | 7,000 | 7,000 |

The images are 1,000 generated scenes and 46 labeled photographs. The Boolean
task contains 4,000 generated-scene questions and 2,000 photographic questions,
balanced between Yes and No within each track. Its 3,000 base/De Morgan pairs
have equivalent Boolean expressions and shared answers. Identification uses
the same 46 photographs. Two prompt formats are alternative presentations of
the same questions, not additional independent questions.

## Contents

- `dataset/hob_vl_v1/`: final prompts, answers, scene records, photo facts, and
  dataset metadata. See the [dataset card](dataset/hob_vl_v1/DATASET_CARD.md).
- `images/synthetic/` and `images/photos/`: the exact images referenced by the
  dataset, with neutral scene-based filenames.
- `evals/`: inference, scoring, resume, comparison, and offline verification.
  See the [evaluation guide](evals/README.md).
- `results/`: the 36 completed task/format runs for eight model identities and
  nine inference configurations, with full responses, retry histories, saved
  settings, scores, CSV tables, and Boolean-property analyses.
  See the [results guide](results/README.md).
- `verify_release.py` and `release_manifest.json`: file-integrity and dataset
  join checks for this snapshot.

This package contains final data, evaluation code, and the completed experimental
results. It supports both new evaluations and offline regeneration of reported
statistics from the saved responses. Dataset-generation and annotation-generation
scripts, pilot or abandoned evaluations, manuscript sources, and Overleaf
integration are not included. It does not regenerate the dataset from scratch
or guarantee identical responses from a provider on a new run.

## Reproduce the reported results

No API key or paid model request is needed for these commands:

```sh
python -m pip install -r evals/requirements-analysis.txt
python verify_release.py
python -m evals.verify_results
python -m evals.paper_results
python -m evals.analyze_properties
```

See [REPRODUCING.md](REPRODUCING.md) for the output files, settings, scoring
definitions, and limits of the comparisons. The saved archive contains 126,000
final responses and 92 earlier API-error attempts; invalid final answers remain
counted as incorrect. Inference cannot resume into the published archives.

## Start here

Use Python 3.11 or later. Run these commands from the folder containing this
README:

```sh
python -m pip install -r evals/requirements.txt
python verify_release.py
python -m evals --model openai/offline-placeholder --data symbolic --limit 5 --dry-run
```

The last command is an offline payload check; `offline-placeholder` is not a
real model identifier. It makes no API calls and reads no API credentials.
Choose a new output directory for each check or run. For the complete offline
test suite, all four full datasets, and simulated provider responses:

```sh
python -m evals.verify_offline
```

Simulated responses verify the software, not model performance. Real model
requests require an explicit `--execute` flag and the selected provider's key.
Only the image and prepared prompt are sent to a model. Answers and annotation
evidence are used locally for scoring.

## Data conventions

Question IDs use `hob_vl_` and `hob_vl_ident_` prefixes. Join prompts and answers
by ID. All image paths are relative to the root of this repository.

## Licensing

The code uses the [MIT License](LICENSE). The dataset, annotations, and images
use [CC BY 4.0](LICENSE-DATA). See [LICENSING.md](LICENSING.md) for the scope of
these permissions and attribution guidance.
