Each JSONL row is a single matched conversational example.

Required fields:

- `scenario_id`
- `user_message`
- `response_style_label`
- `assistant_target_text`
- `notes`

The probe scaffold expects multiple rows with the same `scenario_id` and `user_message`, differing mainly in `response_style_label` and `assistant_target_text`.

## ImpactBench autonomy/self-determination subsets

Dedicated ImpactBench-derived files live under `data/impactbench_autonomy/`.

Regenerate them with:

```bash
python3 scripts/extract_impactbench_autonomy_subsets.py
```

These files preserve full ImpactBench transcript records whose `locations[].subarea_name` is either:

- `Autonomy Preservation`
- `Self-Determination`

Primary files:

- `data/impactbench_autonomy/autonomy_preservation_records.jsonl`
- `data/impactbench_autonomy/self_determination_records.jsonl`
- `data/impactbench_autonomy/manifest.json`

These files are not converted into the repo's single-turn probe schema.

Blinded judge-eval sub-samples (recommended production judge is `claude-opus-4-7`; Codex `gpt-5.4-mini` is the current code default but is being deprecated for production grading — see the judge section in `AGENTS.md` and section `7` of `results/run_history.md`):

- `data/impactbench_autonomy/judge_eval/autonomy_preservation_eval_{20,50}.jsonl`
- `data/impactbench_autonomy/judge_eval/self_determination_eval_{20,50}.jsonl`

Canonical localized activation examples:

- `data/impactbench_autonomy/activation_examples/v1/examples.jsonl`
- `data/impactbench_autonomy/activation_examples/v1/excluded_or_review.jsonl`
- `data/impactbench_autonomy/activation_examples/v1/manifest.json`

These rows preserve multi-turn source references and exact localized pooling
spans. They are intentionally separate from the single-turn matched
conversation schema described above. Their `activation_label` values indicate
criterion satisfaction or violation, not universal beneficial or harmful
behavior. For first-pass subarea vectors, use `harmful=false`, split by source
record or scenario, and follow the target-slice policy in
`activation_examples/v1/README.md`.
