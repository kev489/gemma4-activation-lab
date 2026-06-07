#!/usr/bin/env python3
"""Export validated turn-localization annotations into a tracked activation dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gemma4_activation_lab.activation_datasets import (  # noqa: E402
    ACTIVATION_EXAMPLE_SCHEMA_VERSION,
    canonical_record_json,
    impactbench_record_id,
    load_impactbench_record_index,
    sha256_text,
    source_assistant_turn,
    validate_activation_examples,
)
from gemma4_activation_lab.io_utils import write_jsonl  # noqa: E402


DEFAULT_TRANSCRIPT_DIR = ROOT / "impactBench" / "impactbench_transcripts" / "all_models"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "impactbench_autonomy" / "activation_examples" / "v1"
DEFAULT_GEMMA_ROOT = (
    ROOT
    / "outputs"
    / "turn_localization_annotations"
    / "codex_from_gemma-4-31b_20260531_codex10_parallel"
)
DEFAULT_CLAUDE_ROOT = (
    ROOT
    / "outputs"
    / "turn_localization_annotations_salvaged"
    / "normalized_span_remap_from_gpt-mcp_20260528"
)


@dataclass(frozen=True)
class AnnotationSource:
    name: str
    transcript_model: str
    annotation_root: Path
    salvage_method: str


@dataclass(frozen=True)
class Candidate:
    source_name: str
    salvage_method: str
    annotation_path: str
    record_id: str
    annotation: dict[str, Any]
    turn: dict[str, Any]
    validation_turn: dict[str, Any]
    record: dict[str, Any]
    assistant_text: str
    source_archive: str

    @property
    def key(self) -> tuple[str, int]:
        return self.record_id, self.turn["assistant_turn_index"]

    @property
    def span_signature(self) -> tuple[str, int, int]:
        pooling = self.turn["pooling"]
        return (
            pooling["assistant_pooling_text"],
            pooling["pooling_span_char_start"],
            pooling["pooling_span_char_end"],
        )


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def annotation_paths(root: Path) -> list[Path]:
    return [
        path
        for path in sorted(root.glob("batch_*/*.json"))
        if not path.name.startswith("_") and "validation" not in path.parts
    ]


def source_subarea_names(record: dict[str, Any]) -> list[str]:
    return sorted(
        {
            location["subarea_name"]
            for location in record.get("locations") or []
            if isinstance(location, dict) and isinstance(location.get("subarea_name"), str)
        }
    )


def candidate_payload(candidate: Candidate) -> dict[str, Any]:
    turn = candidate.turn
    pooling = turn["pooling"]
    return {
        "annotation_path": candidate.annotation_path,
        "source_run": candidate.source_name,
        "salvage_method": candidate.salvage_method,
        "activation_label": turn["activation_label"],
        "activation_quality": turn["activation_quality"],
        "default_train_include": turn["default_train_include"],
        "criterion_effect": turn["criterion_effect"],
        "confidence": turn["confidence"],
        "evidence_strength": turn["evidence_strength"],
        "opportunity_strength": turn["opportunity_strength"],
        "confounds": turn.get("confounds") or [],
        "reason": turn.get("reason", ""),
        "pooling": {
            "mode": pooling["mode"],
            "assistant_pooling_text": pooling["assistant_pooling_text"],
            "pooling_span_char_start": pooling["pooling_span_char_start"],
            "pooling_span_char_end": pooling["pooling_span_char_end"],
            "pooling_span_reason": pooling.get("pooling_span_reason", ""),
        },
    }


def load_candidates(
    source: AnnotationSource,
    record_index: dict[str, dict[str, Any]],
    transcript_dir: Path,
) -> tuple[list[Candidate], dict[str, Any]]:
    candidates: list[Candidate] = []
    annotation_count = 0
    warning_counts: Counter[str] = Counter()
    for annotation_path in annotation_paths(source.annotation_root):
        annotation_count += 1
        annotation = load_json(annotation_path)
        record_id = annotation["record_id"]
        if record_id not in record_index:
            raise KeyError(f"Missing source record {record_id} for {annotation_path}")
        record = record_index[record_id]
        if impactbench_record_id(record) != record_id:
            raise ValueError(f"Source record id mismatch for {annotation_path}")
        if record.get("transcript_model") != source.transcript_model:
            raise ValueError(f"Transcript model mismatch for {annotation_path}")

        validation_path = annotation_path.parent / "validation" / f"{annotation_path.stem}.validation.json"
        validation = load_json(validation_path)
        summary = validation["summary"]
        if summary["has_errors"] or summary["review_required"]:
            raise ValueError(f"Annotation is not validator-clean: {annotation_path}")
        validation_turns = {
            turn["assistant_turn_index"]: turn
            for turn in validation["turns"]
        }
        for validation_turn in validation["turns"]:
            for issue in validation_turn.get("issues") or []:
                warning_counts[issue["code"]] += 1

        for turn in annotation["assistant_turns"]:
            if turn.get("activation_quality") != "strong":
                continue
            if turn.get("activation_label") not in {"positive", "negative"}:
                continue
            turn_index = turn["assistant_turn_index"]
            validation_turn = validation_turns[turn_index]
            if validation_turn["review_required"]:
                raise ValueError(f"Strong directional turn still requires review: {annotation_path}:{turn_index}")
            if not validation_turn["span_valid_for_training"]:
                continue
            if any(issue["severity"] == "error" for issue in validation_turn.get("issues") or []):
                continue
            if turn["default_train_include"] and not validation_turn["default_train_eligible"]:
                raise ValueError(f"Included turn is not validator-eligible: {annotation_path}:{turn_index}")

            assistant_turn = source_assistant_turn(record, turn_index)
            pooling = turn["pooling"]
            start = pooling["pooling_span_char_start"]
            end = pooling["pooling_span_char_end"]
            pooling_text = pooling["assistant_pooling_text"]
            if assistant_turn.text[start:end] != pooling_text:
                raise ValueError(f"Non-exact pooling span: {annotation_path}:{turn_index}")
            if assistant_turn.conversation_turn_index != turn["conversation_turn_index"]:
                raise ValueError(f"Conversation turn mismatch: {annotation_path}:{turn_index}")
            if assistant_turn.preceding_user_turn_index != turn["preceding_user_turn_index"]:
                raise ValueError(f"Preceding user turn mismatch: {annotation_path}:{turn_index}")

            candidates.append(
                Candidate(
                    source_name=source.name,
                    salvage_method=source.salvage_method,
                    annotation_path=relative(annotation_path),
                    record_id=record_id,
                    annotation=annotation,
                    turn=turn,
                    validation_turn=validation_turn,
                    record=record,
                    assistant_text=assistant_turn.text,
                    source_archive=relative(
                        transcript_dir / f"{source.transcript_model}_transcripts.jsonl.gz"
                    ),
                )
            )

    return candidates, {
        "annotation_files": annotation_count,
        "strong_directional_valid_candidates": len(candidates),
        "validation_warning_counts": dict(sorted(warning_counts.items())),
    }


def merged_confounds(candidates: list[Candidate]) -> list[str]:
    values = {
        confound
        for candidate in candidates
        for confound in candidate.turn.get("confounds") or []
        if confound != "none"
    }
    return sorted(values) if values else ["none"]


def unique_strings(values: list[str]) -> list[str]:
    return sorted({value for value in values if value})


def build_example(candidates: list[Candidate]) -> dict[str, Any]:
    first = candidates[0]
    turn = first.turn
    record = first.record
    pooling_text, pooling_start, pooling_end = first.span_signature
    annotation_provenance = [
        {
            "annotation_path": candidate.annotation_path,
            "source_run": candidate.source_name,
            "salvage_method": candidate.salvage_method,
        }
        for candidate in sorted(candidates, key=lambda item: item.annotation_path)
    ]
    example_id = f"{first.record_id}::assistant_turn_{turn['assistant_turn_index']:02d}"
    return {
        "schema_version": ACTIVATION_EXAMPLE_SCHEMA_VERSION,
        "example_id": example_id,
        "record_id": first.record_id,
        "scenario_id": record["scenario_id"],
        "transcript_model": record["transcript_model"],
        "benchmark": record["benchmark"],
        "metric_id": record["metric_id"],
        "metric_name": record["metric_name"],
        "metric_criterion": record["metric_criterion"],
        "behavior_type": record.get("behavior_type", ""),
        "measurement": record.get("measurement", ""),
        "harmful": record["harmful"],
        "subarea_names": source_subarea_names(record),
        "global_verdict_result": (record.get("verdict") or {}).get("result"),
        "assistant_turn_index": turn["assistant_turn_index"],
        "conversation_turn_index": turn["conversation_turn_index"],
        "preceding_user_turn_index": turn["preceding_user_turn_index"],
        "activation_label": turn["activation_label"],
        "activation_quality": "strong",
        "criterion_effects": unique_strings([candidate.turn["criterion_effect"] for candidate in candidates]),
        "confidence_min": min(candidate.turn["confidence"] for candidate in candidates),
        "evidence_strength_min": min(candidate.turn["evidence_strength"] for candidate in candidates),
        "opportunity_strength_min": min(candidate.turn["opportunity_strength"] for candidate in candidates),
        "confounds": merged_confounds(candidates),
        "assistant_turn_text": first.assistant_text,
        "assistant_turn_sha256": sha256_text(first.assistant_text),
        "source_record_sha256": sha256_text(canonical_record_json(record)),
        "source_archive": first.source_archive,
        "pooling": {
            "mode": turn["pooling"]["mode"],
            "assistant_pooling_text": pooling_text,
            "pooling_span_char_start": pooling_start,
            "pooling_span_char_end": pooling_end,
            "pooling_text_sha256": sha256_text(pooling_text),
            "pooling_span_reasons": unique_strings(
                [candidate.turn["pooling"].get("pooling_span_reason", "") for candidate in candidates]
            ),
        },
        "annotation_reasons": unique_strings([candidate.turn.get("reason", "") for candidate in candidates]),
        "annotation_provenance": annotation_provenance,
    }


def build_review_row(
    key: tuple[str, int],
    candidates: list[Candidate],
    *,
    review_reason: str,
) -> dict[str, Any]:
    first = candidates[0]
    record = first.record
    return {
        "schema_version": "impactbench_activation_review_v1",
        "review_id": f"{key[0]}::assistant_turn_{key[1]:02d}",
        "review_reason": review_reason,
        "record_id": key[0],
        "scenario_id": record["scenario_id"],
        "transcript_model": record["transcript_model"],
        "benchmark": record["benchmark"],
        "metric_id": record["metric_id"],
        "metric_name": record["metric_name"],
        "metric_criterion": record["metric_criterion"],
        "subarea_names": source_subarea_names(record),
        "assistant_turn_index": key[1],
        "assistant_turn_text": first.assistant_text,
        "assistant_turn_sha256": sha256_text(first.assistant_text),
        "source_record_sha256": sha256_text(canonical_record_json(record)),
        "source_archive": first.source_archive,
        "candidates": [
            candidate_payload(candidate)
            for candidate in sorted(candidates, key=lambda item: item.annotation_path)
        ],
    }


def classify_groups(
    candidates: list[Candidate],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    grouped: dict[tuple[str, int], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.key].append(candidate)

    examples: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    for key in sorted(grouped):
        group = grouped[key]
        included = [candidate for candidate in group if candidate.turn["default_train_include"] is True]
        excluded = [candidate for candidate in group if candidate.turn["default_train_include"] is False]
        if not included:
            reason = "confounded_strong_not_included"
            review_rows.append(build_review_row(key, group, review_reason=reason))
            stats[reason] += 1
            continue
        if excluded:
            reason = "mixed_include_decision"
            review_rows.append(build_review_row(key, group, review_reason=reason))
            stats[reason] += 1
            continue

        labels = {candidate.turn["activation_label"] for candidate in included}
        spans = {candidate.span_signature for candidate in included}
        if len(labels) > 1:
            reason = "label_conflict"
            review_rows.append(build_review_row(key, included, review_reason=reason))
            stats[reason] += 1
            continue
        if len(spans) > 1:
            reason = "same_label_different_span"
            review_rows.append(build_review_row(key, included, review_reason=reason))
            stats[reason] += 1
            continue

        examples.append(build_example(included))
        if len(included) > 1:
            stats["exact_duplicate_collapsed"] += len(included) - 1
        stats["exported_examples"] += 1

    return examples, review_rows, stats


def file_metadata(path: Path, rows: int) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": relative(path),
        "rows": rows,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def nested_counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def model_label_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts[row["transcript_model"]][row["activation_label"]] += 1
    return {
        model: dict(sorted(label_counts.items()))
        for model, label_counts in sorted(counts.items())
    }


def harmful_label_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts[str(row["harmful"]).lower()][row["activation_label"]] += 1
    return {
        harmful: dict(sorted(label_counts.items()))
        for harmful, label_counts in sorted(counts.items())
    }


def harmful_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(
        sorted(Counter(str(row["harmful"]).lower() for row in rows).items())
    )


def subarea_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(row["subarea_names"])
    return dict(sorted(counts.items()))


def build_manifest(
    *,
    examples_path: Path,
    review_path: Path,
    examples: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    source_summaries: dict[str, dict[str, Any]],
    classification_stats: Counter[str],
    sources: list[AnnotationSource],
    transcript_dir: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "impactbench_activation_dataset_manifest_v1",
        "dataset_version": "v1",
        "generation_command": "python3 scripts/export_turn_localization_dataset.py",
        "files": {
            "examples": file_metadata(examples_path, len(examples)),
            "excluded_or_review": file_metadata(review_path, len(review_rows)),
        },
        "source_runs": [
            {
                "name": source.name,
                "transcript_model": source.transcript_model,
                "annotation_root": relative(source.annotation_root),
                "source_archive": relative(
                    transcript_dir / f"{source.transcript_model}_transcripts.jsonl.gz"
                ),
                "salvage_method": source.salvage_method,
                **source_summaries[source.name],
            }
            for source in sources
        ],
        "export_policy": {
            "include": [
                "validator-clean annotation file",
                "strong positive or negative turn",
                "valid exact pooling span",
                "default_train_include=true in every annotation copy",
                "all annotation copies agree on label and exact span",
            ],
            "dedupe_key": ["record_id", "assistant_turn_index"],
            "exact_duplicate_action": "collapse to one example with all annotation provenance",
            "span_disagreement_action": "exclude to review queue",
            "label_conflict_action": "exclude to review queue",
            "mixed_include_decision_action": "exclude to review queue",
            "confounded_strong_not_included_action": "preserve in review queue, exclude from v1",
        },
        "examples": {
            "rows": len(examples),
            "label_counts": nested_counts(examples, "activation_label"),
            "transcript_model_counts": nested_counts(examples, "transcript_model"),
            "transcript_model_label_counts": model_label_counts(examples),
            "harmful_counts": harmful_counts(examples),
            "harmful_label_counts": harmful_label_counts(examples),
            "subarea_counts": subarea_counts(examples),
            "benchmark_counts": nested_counts(examples, "benchmark"),
            "unique_records": len({row["record_id"] for row in examples}),
            "unique_metrics": len({row["metric_id"] for row in examples}),
        },
        "excluded_or_review": {
            "rows": len(review_rows),
            "reason_counts": nested_counts(review_rows, "review_reason"),
        },
        "classification_stats": dict(sorted(classification_stats.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript-dir", type=Path, default=DEFAULT_TRANSCRIPT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gemma-root", type=Path, default=DEFAULT_GEMMA_ROOT)
    parser.add_argument("--claude-root", type=Path, default=DEFAULT_CLAUDE_ROOT)
    args = parser.parse_args()

    sources = [
        AnnotationSource(
            name="gemma_4_31b_codex10_promoted",
            transcript_model="gemma-4-31b",
            annotation_root=args.gemma_root,
            salvage_method=(
                "exact_plus_normalized_span_repair; evidence_quote_repair; "
                "clean_include_cleanup_20260604"
            ),
        ),
        AnnotationSource(
            name="claude_haiku_4_5_normalized_salvage",
            transcript_model="claude-haiku-4-5",
            annotation_root=args.claude_root,
            salvage_method="exact_plus_normalized_span_repair; clean_include_cleanup_20260604",
        ),
    ]
    record_index = load_impactbench_record_index(
        args.transcript_dir,
        transcript_models=[source.transcript_model for source in sources],
    )

    all_candidates: list[Candidate] = []
    source_summaries: dict[str, dict[str, Any]] = {}
    for source in sources:
        candidates, summary = load_candidates(source, record_index, args.transcript_dir)
        all_candidates.extend(candidates)
        source_summaries[source.name] = summary

    examples, review_rows, classification_stats = classify_groups(all_candidates)
    examples.sort(key=lambda row: row["example_id"])
    review_rows.sort(key=lambda row: row["review_id"])
    validate_activation_examples(examples, source=args.output_dir / "examples.jsonl")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    examples_path = args.output_dir / "examples.jsonl"
    review_path = args.output_dir / "excluded_or_review.jsonl"
    write_jsonl(examples_path, examples)
    write_jsonl(review_path, review_rows)
    manifest = build_manifest(
        examples_path=examples_path,
        review_path=review_path,
        examples=examples,
        review_rows=review_rows,
        source_summaries=source_summaries,
        classification_stats=classification_stats,
        sources=sources,
        transcript_dir=args.transcript_dir,
    )
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
