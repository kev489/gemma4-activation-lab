# ImpactBench Opus Prompt Templates

These prompt templates support building cleaner activation-vector training data
from ImpactBench autonomy transcripts.

Use them in this order:

1. `01_turn_localization_opus.md`
   - Input one full ImpactBench record.
   - Output local evidence labels for each assistant turn.
   - Purpose: identify which assistant turn actually satisfies, violates, or
     fails to address the metric criterion.

2. `02_activation_contrast_filter_opus.md`
   - Input one candidate assistant turn plus the record metadata and the
     localization output.
   - Output whether that assistant turn is suitable for activation-vector data.
   - Purpose: discard muddy turns before computing directions.

3. `03_pairing_review_opus.md`
   - Input two filtered candidate turns.
   - Output whether they make a strong positive/negative contrast pair.
   - Purpose: prevent directions from being dominated by source-model style,
     length, scenario difficulty, or unrelated content.

4. `04_gemma_response_grading_opus.md`
   - Input Gemma's generated response in an ImpactBench scenario.
   - Output yes/no criterion grading plus turn-level notes.
   - Purpose: evaluate existing and future steering runs before using
     ImpactBench data for new vectors.

Important:

- Do not compute activation vectors from whole-chat pooled activations.
- Use the full chat only as context.
- For vector extraction, pool the localized assistant answer span only.
- Prefer high-confidence examples matched by `metric_id`, `scenario_id`, user
  pressure turn, and approximate response length.
- Treat `verdict.result` as weak conversation-level supervision, not as a
  direct assistant-turn label.

