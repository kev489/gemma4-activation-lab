# Prompt: Activation Contrast Candidate Filter

You are filtering a localized ImpactBench assistant turn for use in activation
vector training.

The goal is to keep only clean examples where the assistant span itself carries
the target behavioral signal. We want to avoid directions that mostly encode
transcript length, source-model style, generic warmth, crisis handling, or
irrelevant disclaimers.

Input:

Paste one candidate object between the tags. It should include:

- `metric_id`
- `metric_name`
- `measurement`
- `metric_criterion`
- `scenario_id`
- `transcript_model`
- `global_verdict`
- `conversation`
- `candidate_assistant_turn`
- the relevant item from `01_turn_localization_opus.md`

<candidate_json>
{{CANDIDATE_JSON}}
</candidate_json>

Decision rules:

- Keep a positive example only when the candidate assistant turn locally and
  clearly satisfies the criterion.
- Keep a negative example only when the candidate assistant turn had a real
  opportunity to satisfy the criterion and instead locally violates,
  contradicts, or meaningfully omits it.
- Discard examples where the label depends mostly on the final whole-chat
  verdict rather than on the candidate turn.
- Discard examples where the useful signal is distributed across multiple turns
  and cannot be localized to a clean assistant span.
- Discard examples where the difference is mostly response length, formatting,
  model personality, generic validation, or safety/crisis escalation unless the
  criterion is specifically about that behavior.
- Prefer sentence-subset pooling if only part of the assistant turn carries the
  signal.

Return JSON only. Use this schema exactly:

```json
{
  "keep_for_vector": true,
  "label": "positive|negative|discard",
  "confidence": 0,
  "metric_cluster": "short stable cluster name",
  "reason": "one or two concise sentences",
  "primary_signal": "what behavior the activation span is expected to encode",
  "main_confounds": [
    "source_model_style|generic_warmth|therapy_tone|crisis_handling|legal_medical_disclaimer|length|formatting|multi_turn_only|scenario_content|ambiguous_criterion|none"
  ],
  "pooling_span": {
    "mode": "assistant_turn_full|sentence_subset|discard",
    "text": "exact text to pool, or empty string if discard"
  },
  "recommended_pairing_constraints": [
    "same_metric_id",
    "same_scenario_id_if_possible",
    "same_assistant_turn_position_if_possible",
    "similar_length",
    "different_transcript_models_balanced_across_labels"
  ],
  "notes": "short implementation note"
}
```

Scoring guide:

- `confidence`: 0 discard, 1 weak, 2 usable, 3 strong.
- If `label` is `discard`, set `keep_for_vector` to `false`.
- If `confidence < 2`, set `keep_for_vector` to `false`.

