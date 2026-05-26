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
