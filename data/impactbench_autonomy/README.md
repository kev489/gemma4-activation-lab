# ImpactBench Autonomy Subsets

This directory contains literal full-record subsets extracted from the local ImpactBench transcript archive.

Regenerate with:

```bash
python3 scripts/extract_impactbench_autonomy_subsets.py
```

## Files

- `autonomy_preservation_records.jsonl`
  - Full ImpactBench transcript records where `locations[].subarea_name == "Autonomy Preservation"`.
- `self_determination_records.jsonl`
  - Full ImpactBench transcript records where `locations[].subarea_name == "Self-Determination"`.
- `manifest.json`
  - Counts by verdict, benchmark, metric, and source model.

## Important

These files preserve the original ImpactBench record shape, including:

- `scenario`
- `samples`
- `locations`
- `verdict`
- transcript metadata

They are not converted into the repo's single-turn probe format.

## `judge_eval/`

Blinded sub-samples used to calibrate an LLM judge (`src/gemma4_activation_lab/judge.py`) against ImpactBench `verdict.result` labels. Each row carries `rubric`, `conversation`, `ground_truth`, and `metadata` — the judge code only sees `rubric` and `conversation`. Generated with `scripts/build_judge_eval_set.py`; consumed by `scripts/run_judge_eval.py`. See the "LLM judge" section in `AGENTS.md` for the full workflow.

Files:

- `autonomy_preservation_eval_50.jsonl`, `self_determination_eval_50.jsonl` — 50 rows (25 yes / 25 no) per subarea, seed=0.
- `autonomy_preservation_eval_20.jsonl`, `self_determination_eval_20.jsonl` — 20-row variants for quota-constrained runs.

## `activation_examples/v1/`

Canonical tracked activation examples exported from validated turn-localization
annotations:

- `examples.jsonl` contains one deduplicated approved row per source assistant
  turn and exact pooling span, including the source criterion's `harmful`
  polarity flag.
- `excluded_or_review.jsonl` preserves confounded rows and duplicate annotation
  disagreements without including them in default training.
- `manifest.json` records counts, hashes, source runs, and export policy.

Current v1 has `2,616` examples and `148` excluded/review groups. The labels are
criterion-direction labels. For the initial Autonomy Preservation and
Self-Determination vectors, use Gemma-source `harmful=false` rows, exclude rows
tagged with both target subareas, split by source record or scenario, and
balance metrics within each label. See `activation_examples/v1/README.md` for
the exact target-slice counts.

Regenerate from the current local annotation roots with:

```bash
python3 scripts/export_turn_localization_dataset.py
```

Load and resolve rows with
`gemma4_activation_lab.activation_datasets`. The loader reconstructs full
conversation context from the compressed transcript archives rather than
duplicating every conversation in the export.
