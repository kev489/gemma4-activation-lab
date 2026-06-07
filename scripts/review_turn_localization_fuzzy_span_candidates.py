#!/usr/bin/env python3
"""Generate a review-only report for fuzzy turn-localization span candidates.

This script does not write repaired annotations. It looks only at directional
annotation turns that the exact and normalized salvage passes cannot repair, then
proposes likely source spans for human review.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import salvage_turn_localization_exact_spans as exact
import salvage_turn_localization_normalized_spans as normalized


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "outputs"
    / "turn_localization_salvage_reviews"
    / "fuzzy_span_candidates_from_gpt-mcp_20260528"
)

TOKEN_RE = re.compile(r"\S+")


def batch_ids(start: int, end: int) -> list[str]:
    return [f"batch_{i:04d}" for i in range(start, end + 1)]


def token_spans(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(0), match.start(), match.end()) for match in TOKEN_RE.finditer(text)]


def source_slice_from_normalized_window(
    source_text: str,
    normalized_map: list[int],
    normalized_start: int,
    normalized_end: int,
) -> tuple[int, int, str]:
    source_start = normalized_map[normalized_start]
    source_end = normalized_map[normalized_end - 1] + 1
    source_start, source_end = normalized.expand_markdown_edges(source_text, source_start, source_end)
    return source_start, source_end, source_text[source_start:source_end]


def similarity(span_norm: str, candidate_norm: str) -> float:
    return difflib.SequenceMatcher(None, span_norm, candidate_norm).ratio()


def candidate_windows_for_turn(
    *,
    span_norm: str,
    source_turn: dict[str, Any],
    span_token_count: int,
    max_windows_per_turn: int,
) -> list[dict[str, Any]]:
    source_norm, source_map = normalized.normalize_with_map(source_turn["text"])
    tokens = token_spans(source_norm)
    if not tokens:
        return []

    window_sizes = {
        max(4, int(round(span_token_count * scale)))
        for scale in (0.55, 0.75, 1.0, 1.25, 1.5)
    }
    candidates: dict[tuple[int, int], dict[str, Any]] = {}

    for window_size in sorted(window_sizes):
        if window_size > len(tokens):
            continue
        step = max(1, window_size // 8)
        starts = list(range(0, len(tokens) - window_size + 1, step))
        last_start = len(tokens) - window_size
        if starts[-1] != last_start:
            starts.append(last_start)

        for token_start in starts:
            token_end = token_start + window_size
            normalized_start = tokens[token_start][1]
            normalized_end = tokens[token_end - 1][2]
            candidate_norm = source_norm[normalized_start:normalized_end]
            score = similarity(span_norm, candidate_norm)
            source_start, source_end, exact_source_span = source_slice_from_normalized_window(
                source_turn["text"],
                source_map,
                normalized_start,
                normalized_end,
            )
            key = (source_start, source_end)
            existing = candidates.get(key)
            row = {
                "assistant_turn_index": source_turn["assistant_turn_index"],
                "conversation_turn_index": source_turn["conversation_turn_index"],
                "preceding_user_turn_index": source_turn["preceding_user_turn_index"],
                "source_start": source_start,
                "source_end": source_end,
                "score": score,
                "candidate_text": exact_source_span,
                "candidate_normalized": candidate_norm,
            }
            if existing is None or score > existing["score"]:
                candidates[key] = row

    ranked = sorted(candidates.values(), key=lambda row: row["score"], reverse=True)
    return ranked[:max_windows_per_turn]


def fuzzy_candidates(
    *,
    span: str,
    source_turns: list[dict[str, Any]],
    min_normalized_chars: int,
    min_score: float,
    max_candidates: int,
    max_windows_per_turn: int,
) -> list[dict[str, Any]]:
    span_norm, _ = normalized.normalize_with_map(span)
    span_norm = span_norm.strip()
    span_tokens = token_spans(span_norm)
    if len(span_norm) < min_normalized_chars or len(span_tokens) < 4:
        return []

    rows: list[dict[str, Any]] = []
    for source_turn in source_turns:
        rows.extend(
            candidate_windows_for_turn(
                span_norm=span_norm,
                source_turn=source_turn,
                span_token_count=len(span_tokens),
                max_windows_per_turn=max_windows_per_turn,
            )
        )

    deduped: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in rows:
        key = (row["assistant_turn_index"], row["source_start"], row["source_end"])
        existing = deduped.get(key)
        if existing is None or row["score"] > existing["score"]:
            deduped[key] = row

    ranked = [
        row
        for row in sorted(deduped.values(), key=lambda item: item["score"], reverse=True)
        if row["score"] >= min_score
    ]
    return ranked[:max_candidates]


def already_mechanically_repairable(
    item: dict[str, Any],
    source_turns: list[dict[str, Any]],
    min_normalized_chars: int,
) -> bool:
    probe_stats: Counter[str] = Counter()
    repaired = normalized.repair_turn(
        item,
        source_turns,
        probe_stats,
        min_normalized_chars=min_normalized_chars,
    )
    return repaired is not None


def review_bucket(score: float, best_minus_second: float | None) -> str:
    if score >= 0.92 and (best_minus_second is None or best_minus_second >= 0.04):
        return "high_confidence_review"
    if score >= 0.84:
        return "medium_confidence_review"
    return "low_confidence_review"


def preview(text: str, limit: int = 500) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def write_markdown_review(path: Path, rows: list[dict[str, Any]], limit: int) -> None:
    lines = [
        "# Fuzzy Span Candidate Review",
        "",
        "Review-only output. No annotations were repaired by this script.",
        "",
    ]
    for row in rows[:limit]:
        candidates = row["candidates"]
        best = candidates[0]
        lines.extend(
            [
                f"## {row['batch_id']} / {row['stem']} / item {row['annotation_item_index']}",
                "",
                f"- label: `{row['activation_label']}`",
                f"- quality: `{row['activation_quality']}`",
                f"- default_train_include: `{row['default_train_include']}`",
                f"- original_assistant_turn_index: `{row['original_assistant_turn_index']}`",
                f"- review_bucket: `{row['review_bucket']}`",
                f"- best_score: `{best['score']:.3f}`",
                f"- best_source_turn: `{best['assistant_turn_index']}`",
                "",
                "Original span:",
                "",
                "```text",
                row["original_span"],
                "```",
                "",
                "Best candidate source span:",
                "",
                "```text",
                best["candidate_text"],
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", action="append", dest="batch_ids", help="Batch id to process. Repeatable.")
    parser.add_argument("--batch-start", type=int, default=21)
    parser.add_argument("--batch-end", type=int, default=41)
    parser.add_argument("--prompt-batch-root", type=Path, default=exact.DEFAULT_PROMPT_BATCH_ROOT)
    parser.add_argument("--input-root", type=Path, default=exact.DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--min-normalized-chars", type=int, default=20)
    parser.add_argument("--min-score", type=float, default=0.72)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--max-windows-per-turn", type=int, default=10)
    parser.add_argument("--markdown-limit", type=int, default=40)
    args = parser.parse_args()

    selected_batches = args.batch_ids or batch_ids(args.batch_start, args.batch_end)
    args.output_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_root / "fuzzy_candidates.jsonl"
    csv_path = args.output_root / "fuzzy_candidates.csv"
    markdown_path = args.output_root / "fuzzy_candidates_review.md"
    summary_path = args.output_root / "summary.json"

    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()

    for batch_id in selected_batches:
        input_dir = args.input_root / batch_id
        prompt_dir = args.prompt_batch_root / batch_id / "prompts"
        for annotation_path in sorted(input_dir.glob("[0-9]*.json")):
            stem = annotation_path.stem
            prompt_path = prompt_dir / f"{stem}.txt"
            if not prompt_path.exists():
                continue

            annotation = exact.read_json(annotation_path)
            record = exact.load_record_from_prompt(prompt_path)
            source_turns = exact.source_assistant_turns(record)

            for item_index, item in enumerate(annotation.get("assistant_turns", []), start=1):
                if not isinstance(item, dict):
                    continue
                label = item.get("activation_label")
                quality = item.get("activation_quality")
                if label not in {"positive", "negative"} or quality not in {"strong", "weak"}:
                    continue
                pooling = item.get("pooling")
                if not isinstance(pooling, dict):
                    continue
                span = pooling.get("assistant_pooling_text")
                if not isinstance(span, str) or not span:
                    continue

                counters["directional_spans_seen"] += 1
                if already_mechanically_repairable(item, source_turns, args.min_normalized_chars):
                    counters["already_exact_or_normalized"] += 1
                    continue
                counters["needs_fuzzy_review"] += 1

                candidates = fuzzy_candidates(
                    span=span,
                    source_turns=source_turns,
                    min_normalized_chars=args.min_normalized_chars,
                    min_score=args.min_score,
                    max_candidates=args.max_candidates,
                    max_windows_per_turn=args.max_windows_per_turn,
                )
                if not candidates:
                    counters["no_candidate_above_threshold"] += 1
                    continue

                best = candidates[0]
                second_score = candidates[1]["score"] if len(candidates) > 1 else None
                margin = None if second_score is None else best["score"] - second_score
                bucket = review_bucket(best["score"], margin)
                counters[bucket] += 1
                rows.append(
                    {
                        "batch_id": batch_id,
                        "stem": stem,
                        "annotation_path": str(annotation_path),
                        "prompt_path": str(prompt_path),
                        "annotation_item_index": item_index,
                        "original_assistant_turn_index": item.get("assistant_turn_index"),
                        "activation_label": label,
                        "activation_quality": quality,
                        "default_train_include": item.get("default_train_include"),
                        "original_span": span,
                        "original_span_preview": preview(span),
                        "candidate_count": len(candidates),
                        "best_score": best["score"],
                        "second_score": second_score,
                        "best_minus_second": margin,
                        "review_bucket": bucket,
                        "candidates": candidates,
                    }
                )

    rows.sort(key=lambda row: (row["best_score"], row.get("best_minus_second") or 0), reverse=True)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "batch_id",
            "stem",
            "annotation_item_index",
            "original_assistant_turn_index",
            "activation_label",
            "activation_quality",
            "default_train_include",
            "review_bucket",
            "best_score",
            "second_score",
            "best_minus_second",
            "candidate_count",
            "best_assistant_turn_index",
            "best_source_start",
            "best_source_end",
            "original_span_preview",
            "best_candidate_preview",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            best = row["candidates"][0]
            writer.writerow(
                {
                    "batch_id": row["batch_id"],
                    "stem": row["stem"],
                    "annotation_item_index": row["annotation_item_index"],
                    "original_assistant_turn_index": row["original_assistant_turn_index"],
                    "activation_label": row["activation_label"],
                    "activation_quality": row["activation_quality"],
                    "default_train_include": row["default_train_include"],
                    "review_bucket": row["review_bucket"],
                    "best_score": f"{row['best_score']:.6f}",
                    "second_score": "" if row["second_score"] is None else f"{row['second_score']:.6f}",
                    "best_minus_second": "" if row["best_minus_second"] is None else f"{row['best_minus_second']:.6f}",
                    "candidate_count": row["candidate_count"],
                    "best_assistant_turn_index": best["assistant_turn_index"],
                    "best_source_start": best["source_start"],
                    "best_source_end": best["source_end"],
                    "original_span_preview": row["original_span_preview"],
                    "best_candidate_preview": preview(best["candidate_text"]),
                }
            )

    write_markdown_review(markdown_path, rows, args.markdown_limit)

    summary = {
        "input_root": str(args.input_root),
        "prompt_batch_root": str(args.prompt_batch_root),
        "output_root": str(args.output_root),
        "batch_ids": selected_batches,
        "min_normalized_chars": args.min_normalized_chars,
        "min_score": args.min_score,
        "max_candidates": args.max_candidates,
        "max_windows_per_turn": args.max_windows_per_turn,
        "n_review_rows": len(rows),
        "counters": dict(sorted(counters.items())),
        "jsonl_path": str(jsonl_path),
        "csv_path": str(csv_path),
        "markdown_path": str(markdown_path),
        "summary_path": str(summary_path),
    }
    exact.write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
