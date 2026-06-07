# ImpactBench Opus Prompt Templates

These prompt templates support building cleaner activation-vector training data
from ImpactBench autonomy transcripts.

Use them in this order:

1. `01_turn_localization_with_explicit_pooling.txt`
   - Input one full ImpactBench record.
   - Output local evidence labels for each assistant turn, plus an exact
     `assistant_pooling_text` substring for activation capture.
   - Purpose: identify which assistant turn actually satisfies, violates, or
     fails to address the metric criterion, while preserving the precise span
     that should be pooled for vector extraction.
   - This is the preferred prompt for new localization runs.
   - The template contains `{{RECORD_JSON}}`; replace that placeholder with one
     full ImpactBench record serialized as JSON before sending it to the LLM.
   - The output schema includes `activation_quality` and
     `default_train_include`. Default vector training should use only strong
     positive/negative rows with `default_train_include: true`.
   - These labels are criterion-direction labels. The canonical exporter also
     preserves the source record's `harmful` flag; do not interpret
     `positive`/`negative` as universal beneficial/harmful polarity.

   Legacy prompt: `01_turn_localization_opus.md` is kept for provenance. It has
   the same localization goal but does not make explicit pooling spans as
   central to the schema.

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
- Prefer the `assistant_pooling_text` returned by
  `01_turn_localization_with_explicit_pooling.txt`; validate that it is an exact
  substring of the assistant turn and compute token spans programmatically.
- Validate LLM annotations before using them:

  ```bash
  python3 scripts/validate_turn_localization.py \
    --annotation outputs/prompt_tests/turn_localization_self_determination_line560_prompt_v2_response.json \
    --prompt outputs/prompt_tests/turn_localization_self_determination_line560_prompt_v2.txt
  ```

- The validator treats invalid strong spans as `review_required` instead of
  silently dropping them. Invalid weak spans are warnings and remain excluded
  from default training.
- For bulk runs, the validator can also take `--record-json path/to/record.json`
  instead of `--prompt` if prompts are not saved one-per-record.
- For local CLI annotation runs that mirror the GPT MCP workflow without using
  the MCP batch tool, use `scripts/run_turn_localization_cli_batch.py`. It sends
  prompts one at a time, writes one JSON annotation per prompt, and only starts
  validation after all annotation attempts finish:

  ```bash
  python3 scripts/run_turn_localization_cli_batch.py \
    --provider codex \
    --batch-id batch_0013
  ```

  Claude CLI can be used with the same batch layout:

  ```bash
  python3 scripts/run_turn_localization_cli_batch.py \
    --provider claude \
    --model claude-opus-4-7 \
    --batch-id batch_0013
  ```

  The CLI runner enforces structured JSON output with Codex
  `--output-schema` or Claude `--json-schema`; this is the CLI equivalent of
  using `output_json=true` for the MCP path. It writes `_run_manifest.jsonl`,
  `_run_summary.json`, and per-row validation reports next to the annotation
  JSON files.
- Prefer high-confidence examples matched by `metric_id`, `scenario_id`, user
  pressure turn, and approximate response length.
- Treat `verdict.result` as weak conversation-level supervision, not as a
  direct assistant-turn label.
- For first-pass Autonomy Preservation and Self-Determination vectors, use the
  canonical v1 export rather than raw annotation trees, restrict to
  `harmful=false`, and follow the overlap/splitting policy in
  `data/impactbench_autonomy/activation_examples/v1/README.md`.
