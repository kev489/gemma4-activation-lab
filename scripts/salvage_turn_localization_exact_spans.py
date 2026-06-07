#!/usr/bin/env python3
"""Deterministically repair turn-localization annotations using exact spans.

This handles only high-confidence mechanical repairs:

- copy source metadata fields back onto the annotation
- repair offsets when pooling text is an exact substring
- remap a turn when its pooling text appears exactly in one source assistant turn
- fill omitted source assistant turns as discard rows

It deliberately does not do fuzzy matching or semantic relabeling.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PROMPT_BATCH_ROOT = (
    ROOT
    / "outputs"
    / "turn_localization_prompt_batches"
    / "claude-haiku-4-5_20260528__20_per_batch"
)
DEFAULT_INPUT_ROOT = (
    ROOT
    / "outputs"
    / "turn_localization_annotations"
    / "gpt-mcp_from_claude-haiku-4-5_20260528"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "outputs"
    / "turn_localization_annotations_salvaged"
    / "exact_span_remap_from_gpt-mcp_20260528"
)

METADATA_FIELDS = (
    "scenario_id",
    "scenario_title",
    "transcript_model",
    "benchmark",
    "metric_id",
    "metric_name",
    "behavior_type",
    "measurement",
    "metric_criterion",
    "subarea_name",
    "global_verdict_result",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_record_from_prompt(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"<impactbench_record_json>\s*(.*?)\s*</impactbench_record_json>", text, re.S)
    if not match:
        raise ValueError(f"No <impactbench_record_json> block found in {path}")
    record = json.loads(match.group(1))
    if not isinstance(record, dict):
        raise ValueError(f"Record block in {path} did not parse to an object")
    return record


def source_record_id(record: dict[str, Any]) -> str:
    if record.get("record_id"):
        return str(record["record_id"])
    return f"{record.get('scenario_id')}::{record.get('metric_id')}::{record.get('transcript_model')}"


def source_assistant_turns(record: dict[str, Any]) -> list[dict[str, Any]]:
    samples = record.get("samples")
    if not isinstance(samples, list) or not samples or not isinstance(samples[0], list):
        raise ValueError("Record does not contain samples[0] conversation")

    turns: list[dict[str, Any]] = []
    preceding_user_turn_index: int | None = None
    for conversation_turn_index, message in enumerate(samples[0], start=1):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "user":
            preceding_user_turn_index = conversation_turn_index
        elif role == "assistant":
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError(f"Assistant message at conversation turn {conversation_turn_index} has non-string content")
            turns.append(
                {
                    "assistant_turn_index": len(turns) + 1,
                    "conversation_turn_index": conversation_turn_index,
                    "preceding_user_turn_index": preceding_user_turn_index,
                    "text": content,
                }
            )
    return turns


def find_all(text: str, needle: str) -> list[int]:
    if not needle:
        return []
    starts: list[int] = []
    pos = text.find(needle)
    while pos != -1:
        starts.append(pos)
        pos = text.find(needle, pos + 1)
    return starts


def discard_turn(source_turn: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "assistant_turn_index": source_turn["assistant_turn_index"],
        "conversation_turn_index": source_turn["conversation_turn_index"],
        "preceding_user_turn_index": source_turn["preceding_user_turn_index"],
        "criterion_effect": "irrelevant",
        "activation_label": "discard",
        "activation_quality": "discard",
        "default_train_include": False,
        "confidence": 0,
        "evidence_strength": 0,
        "opportunity_strength": 0,
        "evidence_quotes": [],
        "reason": reason,
        "confounds": [],
        "pooling": {
            "mode": "discard",
            "assistant_pooling_text": "",
            "pooling_span_char_start": None,
            "pooling_span_char_end": None,
            "pooling_span_reason": "No source-aligned annotation was available for this assistant turn.",
        },
    }


def priority(item: dict[str, Any]) -> tuple[int, int, int, int]:
    label = item.get("activation_label")
    quality = item.get("activation_quality")
    include = item.get("default_train_include") is True
    directional = label in {"positive", "negative"}
    pool = item.get("pooling") if isinstance(item.get("pooling"), dict) else {}
    span = pool.get("assistant_pooling_text") if isinstance(pool, dict) else ""
    return (
        1 if include else 0,
        1 if quality == "strong" and directional else 0,
        int(item.get("confidence") or 0) + int(item.get("evidence_strength") or 0),
        len(span) if isinstance(span, str) else 0,
    )


def repair_turn(
    item: dict[str, Any],
    source_turns: list[dict[str, Any]],
    stats: Counter[str],
) -> dict[str, Any] | None:
    repaired = copy.deepcopy(item)
    label = repaired.get("activation_label")
    quality = repaired.get("activation_quality")
    pooling = repaired.get("pooling")

    if label == "discard" or quality == "discard":
        source_idx = repaired.get("assistant_turn_index")
        if not isinstance(source_idx, int) or not (1 <= source_idx <= len(source_turns)):
            return None
        source_turn = source_turns[source_idx - 1]
        repaired["assistant_turn_index"] = source_turn["assistant_turn_index"]
        repaired["conversation_turn_index"] = source_turn["conversation_turn_index"]
        repaired["preceding_user_turn_index"] = source_turn["preceding_user_turn_index"]
        repaired["activation_label"] = "discard"
        repaired["activation_quality"] = "discard"
        repaired["default_train_include"] = False
        repaired["pooling"] = {
            "mode": "discard",
            "assistant_pooling_text": "",
            "pooling_span_char_start": None,
            "pooling_span_char_end": None,
            "pooling_span_reason": "Discard turn normalized during deterministic salvage.",
        }
        stats["discard_normalized"] += 1
        return repaired

    if not isinstance(pooling, dict):
        return None
    span = pooling.get("assistant_pooling_text")
    if not isinstance(span, str) or not span:
        return None

    matches: list[tuple[int, int]] = []
    for source_turn in source_turns:
        starts = find_all(source_turn["text"], span)
        for start in starts:
            matches.append((source_turn["assistant_turn_index"], start))
    if len(matches) != 1:
        stats["exact_span_unmatched" if not matches else "exact_span_ambiguous"] += 1
        return None

    source_idx, start = matches[0]
    source_turn = source_turns[source_idx - 1]
    old_idx = repaired.get("assistant_turn_index")
    if old_idx != source_idx:
        stats["turn_remapped"] += 1
    else:
        stats["same_turn_offset_repaired"] += 1
    repaired["assistant_turn_index"] = source_turn["assistant_turn_index"]
    repaired["conversation_turn_index"] = source_turn["conversation_turn_index"]
    repaired["preceding_user_turn_index"] = source_turn["preceding_user_turn_index"]
    repaired["pooling"]["pooling_span_char_start"] = start
    repaired["pooling"]["pooling_span_char_end"] = start + len(span)
    if repaired["pooling"].get("mode") not in {"assistant_turn_full", "sentence_subset"}:
        repaired["pooling"]["mode"] = "sentence_subset"
    return repaired


def repair_annotation(
    annotation: dict[str, Any],
    record: dict[str, Any],
    stats: Counter[str],
) -> dict[str, Any]:
    source_turns = source_assistant_turns(record)
    repaired = copy.deepcopy(annotation)

    repaired["record_id"] = source_record_id(record)
    for field in METADATA_FIELDS:
        if field in record:
            repaired[field] = record[field]
    repaired["global_verdict_used_only_as_context"] = True

    per_source: dict[int, dict[str, Any]] = {}
    collisions = 0
    for item in annotation.get("assistant_turns", []):
        if not isinstance(item, dict):
            continue
        repaired_item = repair_turn(item, source_turns, stats)
        if repaired_item is None:
            continue
        source_idx = repaired_item["assistant_turn_index"]
        existing = per_source.get(source_idx)
        if existing is None or priority(repaired_item) > priority(existing):
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
            item = discard_turn(
                source_turn,
                "Inserted by deterministic salvage because the original annotation omitted this assistant turn or had no exact source span.",
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
    return repaired


def validate(prompt_path: Path, annotation_path: Path, validation_path: Path, python_exe: str) -> dict[str, Any]:
    cmd = [
        python_exe,
        str(ROOT / "scripts" / "validate_turn_localization.py"),
        "--annotation",
        str(annotation_path),
        "--prompt",
        str(prompt_path),
        "--output",
        str(validation_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return {
            "status": "validation_crash",
            "error": proc.stderr.strip() or proc.stdout.strip(),
        }
    report = read_json(validation_path)
    summary = report.get("summary", {})
    if summary.get("has_errors"):
        status = "validation_error"
    elif summary.get("review_required"):
        status = "review_required"
    else:
        status = "ok"
    return {
        "status": status,
        "summary": summary,
    }


def batch_ids(start: int, end: int) -> list[str]:
    return [f"batch_{i:04d}" for i in range(start, end + 1)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", action="append", dest="batch_ids", help="Batch id to process. Repeatable.")
    parser.add_argument("--batch-start", type=int, default=21)
    parser.add_argument("--batch-end", type=int, default=41)
    parser.add_argument("--prompt-batch-root", type=Path, default=DEFAULT_PROMPT_BATCH_ROOT)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    selected_batches = args.batch_ids or batch_ids(args.batch_start, args.batch_end)
    rows: list[dict[str, Any]] = []
    repair_stats: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    before_status_counts: Counter[str] = Counter()
    before_clean_turns = 0
    after_clean_turns = 0

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

            annotation = read_json(annotation_path)
            record = load_record_from_prompt(prompt_path)
            repaired = repair_annotation(annotation, record, repair_stats)

            repaired_path = output_dir / annotation_path.name
            validation_path = validation_dir / f"{stem}.validation.json"
            write_json(repaired_path, repaired)
            validation = validate(prompt_path, repaired_path, validation_path, args.python)
            status_counts[validation["status"]] += 1

            before_status = "unknown"
            before_summary: dict[str, Any] = {}
            if old_validation_path.exists():
                old_report = read_json(old_validation_path)
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
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
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
        "repair_stats": dict(sorted(repair_stats.items())),
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 1 if status_counts.get("validation_crash") else 0


if __name__ == "__main__":
    raise SystemExit(main())
