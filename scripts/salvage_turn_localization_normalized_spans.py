#!/usr/bin/env python3
"""Repair turn-localization annotations with exact and normalized span matches.

This is the second salvage pass after exact-span remapping. It still avoids
semantic guessing: a non-exact span is recovered only when its normalized text
has exactly one contiguous match in the normalized source assistant turns. The
output pooling text is replaced with the exact source substring so the standard
validator remains the final gate.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import salvage_turn_localization_exact_spans as exact


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "outputs"
    / "turn_localization_annotations_salvaged"
    / "normalized_span_remap_from_gpt-mcp_20260528"
)

STRIP_CHARS = set("*_`#>")
QUOTE_MAP = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u201f": '"',
}
DASH_CHARS = {"\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"}


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    source_indices: list[int] = []
    pending_space_index: int | None = None

    for source_index, raw_char in enumerate(text):
        char = unicodedata.normalize("NFKC", raw_char)
        char = QUOTE_MAP.get(char, char)
        if char in DASH_CHARS:
            char = "-"

        if char in STRIP_CHARS:
            continue

        if char.isspace():
            if chars and chars[-1] != " ":
                pending_space_index = source_index
            continue

        if pending_space_index is not None:
            chars.append(" ")
            source_indices.append(pending_space_index)
            pending_space_index = None

        chars.append(char.lower())
        source_indices.append(source_index)

    while chars and chars[-1] == " ":
        chars.pop()
        source_indices.pop()
    return "".join(chars), source_indices


def normalized_span_matches(
    span: str,
    source_turns: list[dict[str, Any]],
    *,
    min_normalized_chars: int,
) -> list[tuple[int, int, int, str]]:
    normalized_span, _ = normalize_with_map(span)
    normalized_span = normalized_span.strip()
    if len(normalized_span) < min_normalized_chars:
        return []

    matches: list[tuple[int, int, int, str]] = []
    for source_turn in source_turns:
        normalized_source, source_map = normalize_with_map(source_turn["text"])
        start = normalized_source.find(normalized_span)
        while start != -1:
            end = start + len(normalized_span)
            source_start = source_map[start]
            source_end = source_map[end - 1] + 1
            source_start, source_end = expand_markdown_edges(source_turn["text"], source_start, source_end)
            exact_source_span = source_turn["text"][source_start:source_end]
            matches.append(
                (
                    source_turn["assistant_turn_index"],
                    source_start,
                    source_end,
                    exact_source_span,
                )
            )
            start = normalized_source.find(normalized_span, start + 1)
    return matches


def expand_markdown_edges(text: str, start: int, end: int) -> tuple[int, int]:
    while start > 0 and text[start - 1] in STRIP_CHARS:
        start -= 1
    while end < len(text) and text[end] in STRIP_CHARS:
        end += 1
    return start, end


def repair_turn(
    item: dict[str, Any],
    source_turns: list[dict[str, Any]],
    stats: Counter[str],
    *,
    min_normalized_chars: int,
) -> dict[str, Any] | None:
    repaired = exact.repair_turn(item, source_turns, stats)
    if repaired is not None:
        return repaired

    candidate = copy.deepcopy(item)
    label = candidate.get("activation_label")
    quality = candidate.get("activation_quality")
    if label not in {"positive", "negative"} or quality not in {"strong", "weak"}:
        return None
    pooling = candidate.get("pooling")
    if not isinstance(pooling, dict):
        return None
    span = pooling.get("assistant_pooling_text")
    if not isinstance(span, str) or not span:
        return None

    matches = normalized_span_matches(
        span,
        source_turns,
        min_normalized_chars=min_normalized_chars,
    )
    if len(matches) != 1:
        stats["normalized_span_unmatched" if not matches else "normalized_span_ambiguous"] += 1
        return None

    source_idx, source_start, source_end, exact_source_span = matches[0]
    source_turn = source_turns[source_idx - 1]
    if candidate.get("assistant_turn_index") != source_idx:
        stats["normalized_turn_remapped"] += 1
    else:
        stats["normalized_same_turn_repaired"] += 1

    candidate["assistant_turn_index"] = source_turn["assistant_turn_index"]
    candidate["conversation_turn_index"] = source_turn["conversation_turn_index"]
    candidate["preceding_user_turn_index"] = source_turn["preceding_user_turn_index"]
    candidate["pooling"]["assistant_pooling_text"] = exact_source_span
    candidate["pooling"]["pooling_span_char_start"] = source_start
    candidate["pooling"]["pooling_span_char_end"] = source_end
    if candidate["pooling"].get("mode") not in {"assistant_turn_full", "sentence_subset"}:
        candidate["pooling"]["mode"] = "sentence_subset"
    return candidate


def repair_annotation(
    annotation: dict[str, Any],
    record: dict[str, Any],
    stats: Counter[str],
    *,
    min_normalized_chars: int,
) -> dict[str, Any]:
    source_turns = exact.source_assistant_turns(record)
    repaired = copy.deepcopy(annotation)

    repaired["record_id"] = exact.source_record_id(record)
    for field in exact.METADATA_FIELDS:
        if field in record:
            repaired[field] = record[field]
    repaired["global_verdict_used_only_as_context"] = True

    per_source: dict[int, dict[str, Any]] = {}
    collisions = 0
    for item in annotation.get("assistant_turns", []):
        if not isinstance(item, dict):
            continue
        repaired_item = repair_turn(
            item,
            source_turns,
            stats,
            min_normalized_chars=min_normalized_chars,
        )
        if repaired_item is None:
            continue
        source_idx = repaired_item["assistant_turn_index"]
        existing = per_source.get(source_idx)
        if existing is None or exact.priority(repaired_item) > exact.priority(existing):
            per_source[source_idx] = repaired_item
        if existing is not None:
            collisions += 1

    if collisions:
        stats["source_turn_collisions"] += collisions

    ordered_turns: list[dict[str, Any]] = []
    for source_turn in source_turns:
        item = per_source.get(source_turn["assistant_turn_index"])
        if item is None:
            stats["discard_inserted"] += 1
            item = exact.discard_turn(
                source_turn,
                "Inserted by normalized-span salvage because the original annotation omitted this assistant turn or had no unique source span.",
            )
        ordered_turns.append(item)

    repaired["assistant_turns"] = ordered_turns
    repaired["best_positive_turn_indices"] = [
        item["assistant_turn_index"]
        for item in ordered_turns
        if item.get("activation_label") == "positive" and item.get("activation_quality") == "strong"
    ]
    repaired["best_negative_turn_indices"] = [
        item["assistant_turn_index"]
        for item in ordered_turns
        if item.get("activation_label") == "negative" and item.get("activation_quality") == "strong"
    ]
    if not repaired.get("record_usefulness"):
        repaired["record_usefulness"] = "none"
    if not isinstance(repaired.get("notes"), str):
        repaired["notes"] = ""
    repaired["notes"] = (repaired.get("notes") or "") + "\n[salvage] exact-plus-normalized deterministic span repair applied."
    return repaired


def batch_ids(start: int, end: int) -> list[str]:
    return [f"batch_{i:04d}" for i in range(start, end + 1)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", action="append", dest="batch_ids", help="Batch id to process. Repeatable.")
    parser.add_argument("--batch-start", type=int, default=21)
    parser.add_argument("--batch-end", type=int, default=41)
    parser.add_argument("--prompt-batch-root", type=Path, default=exact.DEFAULT_PROMPT_BATCH_ROOT)
    parser.add_argument("--input-root", type=Path, default=exact.DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--exact-summary", type=Path, default=exact.DEFAULT_OUTPUT_ROOT / "_salvage_summary.json")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--min-normalized-chars", type=int, default=20)
    args = parser.parse_args()

    selected_batches = args.batch_ids or batch_ids(args.batch_start, args.batch_end)
    rows: list[dict[str, Any]] = []
    repair_stats: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    before_status_counts: Counter[str] = Counter()
    before_clean_turns = 0
    after_clean_turns = 0

    exact_clean_turns = None
    if args.exact_summary.exists():
        exact_summary = exact.read_json(args.exact_summary)
        exact_clean_turns = exact_summary.get("after_default_train_eligible")

    for batch_id in selected_batches:
        input_dir = args.input_root / batch_id
        prompt_dir = args.prompt_batch_root / batch_id / "prompts"
        output_dir = args.output_root / batch_id
        validation_dir = output_dir / "validation"
        validation_dir.mkdir(parents=True, exist_ok=True)

        for annotation_path in sorted(input_dir.glob("[0-9]*.json")):
            stem = annotation_path.stem
            prompt_path = prompt_dir / f"{stem}.txt"
            old_validation_path = input_dir / "validation" / f"{stem}.validation.json"
            if not prompt_path.exists():
                continue

            annotation = exact.read_json(annotation_path)
            record = exact.load_record_from_prompt(prompt_path)
            repaired = repair_annotation(
                annotation,
                record,
                repair_stats,
                min_normalized_chars=args.min_normalized_chars,
            )

            repaired_path = output_dir / annotation_path.name
            validation_path = validation_dir / f"{stem}.validation.json"
            exact.write_json(repaired_path, repaired)
            validation = exact.validate(prompt_path, repaired_path, validation_path, args.python)
            status_counts[validation["status"]] += 1

            before_status = "unknown"
            before_summary: dict[str, Any] = {}
            if old_validation_path.exists():
                old_report = exact.read_json(old_validation_path)
                before_summary = old_report.get("summary", {})
                if before_summary.get("has_errors"):
                    before_status = "validation_error"
                elif before_summary.get("review_required"):
                    before_status = "review_required"
                else:
                    before_status = "ok"
                before_clean_turns += int(before_summary.get("default_train_eligible", 0))
            before_status_counts[before_status] += 1

            after_summary = validation.get("summary", {})
            after_clean_turns += int(after_summary.get("default_train_eligible", 0) or 0)
            rows.append(
                {
                    "batch_id": batch_id,
                    "stem": stem,
                    "input_annotation_path": str(annotation_path),
                    "prompt_path": str(prompt_path),
                    "repaired_annotation_path": str(repaired_path),
                    "validation_path": str(validation_path),
                    "before_status": before_status,
                    "after_status": validation["status"],
                    "before_default_train_eligible": before_summary.get("default_train_eligible"),
                    "after_default_train_eligible": after_summary.get("default_train_eligible"),
                    "after_has_warnings": after_summary.get("has_warnings"),
                    "after_turn_warnings": after_summary.get("turn_warnings"),
                    "after_top_level_warnings": after_summary.get("top_level_warnings"),
                }
            )

    manifest_path = args.output_root / "_salvage_manifest.jsonl"
    summary_path = args.output_root / "_salvage_summary.json"
    args.output_root.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")

    summary = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "prompt_batch_root": str(args.prompt_batch_root),
        "batch_ids": selected_batches,
        "n_annotations": len(rows),
        "before_status_counts": dict(sorted(before_status_counts.items())),
        "after_status_counts": dict(sorted(status_counts.items())),
        "before_default_train_eligible": before_clean_turns,
        "after_default_train_eligible": after_clean_turns,
        "exact_pass_after_default_train_eligible": exact_clean_turns,
        "delta_vs_original": after_clean_turns - before_clean_turns,
        "delta_vs_exact_pass": None if exact_clean_turns is None else after_clean_turns - int(exact_clean_turns),
        "min_normalized_chars": args.min_normalized_chars,
        "repair_stats": dict(sorted(repair_stats.items())),
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
    }
    exact.write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 1 if status_counts.get("validation_crash") else 0


if __name__ == "__main__":
    raise SystemExit(main())
