# Hob-VL dataset card

## Tasks and scope

The Boolean task contains 6,000 questions over 1,000 generated scenes and 46
labeled photographs. Each question combines ten visual statements with Boolean
operators. The photo identification task contains 1,000 questions over the same
46 photographs, with two to five spatial predicates and one candidate-label
answer. Both tasks have symbolic and structured natural-language presentations.

This is an evaluation-only dataset. Repeated images, related formulas, and
paired presentations are dependent observations, not independent train/test
splits. Keep related questions and scene families together in any future split.

## Files

| File | Contents |
| --- | --- |
| `model_inputs.jsonl` | 6,000 symbolic Boolean prompts |
| `model_inputs_natural_language.jsonl` | 6,000 corresponding structured NL Boolean prompts |
| `dataset.jsonl` | 6,000 Boolean answers and logical/visual annotations |
| `model_inputs_photo_identification_1000_symbolic.jsonl` | 1,000 symbolic identification prompts |
| `model_inputs_photo_identification_1000.jsonl` | 1,000 corresponding NL identification prompts |
| `photo_identification_1000.json` | 1,000 identification answers and candidate-level annotations |
| `scenes.json` | 1,046 scene records, objects, statements, and provenance fields |
| `photo_facts.json` | Visual statement inventories for the 46 photographs |
| `manifest.json` | Boolean dataset statistics and current exported-file checksums |

Image paths are relative to the package root, not this directory. All model
input files contain `id`, `image_path`, and `prompt`. Boolean inputs also carry
statement information; symbolic inputs carry a formula. Identification inputs
include `candidate_labels`. Preserve NL indentation and logical scope when
sending prompts to a model.

Join inputs to answers by `id`. Boolean answers are `target: "Yes"` or
`target: "No"` in `dataset.jsonl`. Identification answers are candidate labels
in `photo_identification_1000.json`; labels may have multiple letters. Never
send the answer files, `audit` objects, scene annotations, or reference truth
values to a model being evaluated.

## Pairing and construction evidence

Each task's symbolic and NL files have the same IDs, image paths, and row order.
The 6,000 Boolean questions additionally form 3,000 base/De Morgan pairs,
identified by `formula_family_id` and `variant`. Each such pair shares a scene
and answer. Equivalent-expression pairing is distinct from matching the two
prompt formats for the same ID.

Boolean records retain their expression trees, atom values and evidence,
construction-family labels, rewrite information, and function/structure
measurements. Identification records retain predicates, candidate-level truth
information, uncertainty, and the unique reference answer. These annotations
support inspection and analysis; they are not model inputs.

The construction families are parity of conjunctions, parity of majorities,
nested multiplexers, dense CNF, sparse chain CNF, polarity-trap composition,
and ten-input parity. Family counts and Boolean balance are in `manifest.json`.
The final records preserve construction seeds and annotations, but this package
does not include the generation programs.

## Grounding and limitations

Generated scenes are tied to deterministic renderer specifications. Their
statements concern independently controllable colors or shapes of labeled
objects.

Identification uses approximate object-reference points rather than bounding
boxes or segmentation labels. Close spatial comparisons can be unknown; the
stored logical checks require a unique answer across their possible completions.

For photographs, influence and certificate annotations concern abstract Boolean
assignments, not a verified space of physically realizable image edits. A
mathematical dependence property is not a guarantee of model difficulty.
Candidate counts vary from 3 to 32; identification targets are not exactly
balanced. Repeated scenes and targets should be accounted for in analysis.

## Anonymized file identity

Question IDs have neutral `hob_vl_`/`hob_vl_ident_` prefixes and images have
scene-based paths under `images/`. Prompt text, targets, logical annotations,
and the released image bytes are unchanged by this packaging step. Original
photograph filenames have been replaced in scene records by `source_image_id`.
For photographs, `source_sha256` identifies the unannotated original image;
those originals are not included. It is not the checksum of the released
labeled PNG. Current released-file hashes are in the root release manifest.

Historical validation reports and their obsolete file checksums are not
included. `python verify_release.py` checks the current files and joins;
`python -m evals.verify_offline` checks the runner, all four prompt collections,
image decoding/preparation, and scoring against simulated responses. These
commands test file integrity and software behavior using offline checks.

## License

The dataset, annotations, dataset documentation, and images are licensed under
[CC BY 4.0](../../LICENSE-DATA), copyright 2026 Hob-VL contributors. The evaluation
code uses the [MIT License](../../LICENSE). See [LICENSING.md](../../LICENSING.md)
for the scope of these permissions and attribution guidance.
