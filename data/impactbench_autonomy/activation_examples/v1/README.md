# ImpactBench Activation Examples v1

This directory is the canonical tracked export of validated turn-localization
work. It contains compact activation examples, not the full annotation run
trees under `outputs/`.

## Files

- `examples.jsonl`
  - One canonical row per approved `(record_id, assistant_turn_index)`.
  - Contains the full source assistant turn, exact pooling substring and
    character offsets, criterion-direction activation label, ImpactBench
    `harmful` polarity flag, metric metadata, hashes, and annotation provenance.
- `excluded_or_review.jsonl`
  - Strong directional rows withheld from v1 because of confounds, conflicting
    include decisions, label conflicts, or duplicate annotations that selected
    different pooling spans.
- `manifest.json`
  - Counts, content hashes, source runs, and the export policy.

Full multi-turn context is reconstructed from the tracked compressed archives
under `impactBench/impactbench_transcripts/all_models/`. The export does not
duplicate the full conversation for every localized span.

## Regenerate

The default command uses the current local Gemma and Claude-Haiku annotation
roots:

```bash
python3 scripts/export_turn_localization_dataset.py
```

The exporter requires those ignored `outputs/` annotation roots to exist. It
fails on missing source records, invalid validation reports, non-exact spans,
or source hash mismatches.

## v1 Inclusion Policy

A canonical example must be:

- from a validator-clean annotation file
- a strong positive or negative turn
- exact-source-span valid
- marked `default_train_include=true` in every annotation copy
- label- and span-consistent across duplicate annotations

Duplicate rows use `(record_id, assistant_turn_index)` as the key. Exact
duplicates collapse to one example with all provenance retained. Any label,
span, or include disagreement is withheld in `excluded_or_review.jsonl`.

Current v1 contains:

- `2,616` examples
- Gemma 4 31B: `1,879` (`993` positive, `886` negative)
- Claude Haiku 4.5: `737` (`621` positive, `116` negative)
- `148` excluded/review groups

## Label Semantics

`activation_label` preserves the annotation prompt's criterion-direction
semantics:

- `positive`: the localized turn was annotated as satisfying the criterion
- `negative`: the localized turn was annotated as violating the criterion

This is not always the same as beneficial versus harmful autonomy behavior.
ImpactBench criteria with `harmful=true` describe harmful behavior, so their
criterion-direction labels require a separate polarity audit before they are
mixed into an autonomy-preservation or self-determination direction.

For the first-pass subarea vectors, use `harmful=false` rows only. Keep
`harmful=true` rows for a later metric-level relabeling pass rather than
mechanically mixing them into the default mean difference.

Recommended first-pass Gemma-only target slices also exclude rows tagged with
both target subareas:

- Autonomy Preservation: `1,055` rows (`581` positive, `474` negative)
- Self-Determination: `503` rows (`262` positive, `241` negative)

Split by source record or scenario before activation capture, and balance
metrics within each label so records with more localized turns do not dominate.

Use Gemma as the primary first-pass construction set. Claude-Haiku is useful as
a source-model sensitivity check, but its target slices are too
negative-sparse to treat as co-equal primary directions. Do not test a combined
direction until source-specific results are understood, and do not weight
duplicate records or metrics merely because they produced more localized spans.
