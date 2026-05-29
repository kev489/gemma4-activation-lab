from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

CRITERION_EFFECTS = {"satisfies", "violates", "mixed", "irrelevant"}
ACTIVATION_LABELS = {"positive", "negative", "discard"}
ACTIVATION_QUALITIES = {"strong", "weak", "discard"}
POOLING_MODES = {"assistant_turn_full", "sentence_subset", "discard"}
RECORD_USEFULNESS = {"high", "medium", "low", "none"}

TOP_LEVEL_REQUIRED = {
    "record_id",
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
    "global_verdict_used_only_as_context",
    "assistant_turns",
    "best_positive_turn_indices",
    "best_negative_turn_indices",
    "record_usefulness",
    "notes",
}

TURN_REQUIRED = {
    "assistant_turn_index",
    "conversation_turn_index",
    "preceding_user_turn_index",
    "criterion_effect",
    "activation_label",
    "activation_quality",
    "default_train_include",
    "confidence",
    "evidence_strength",
    "opportunity_strength",
    "evidence_quotes",
    "reason",
    "confounds",
    "pooling",
}

POOLING_REQUIRED = {
    "mode",
    "assistant_pooling_text",
    "pooling_span_char_start",
    "pooling_span_char_end",
    "pooling_span_reason",
}


@dataclass(frozen=True)
class SourceAssistantTurn:
    assistant_turn_index: int
    conversation_turn_index: int
    preceding_user_turn_index: int | None
    text: str


def read_text_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def load_record_from_prompt(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"<impactbench_record_json>\s*(.*?)\s*</impactbench_record_json>", text, re.S)
    if not match:
        raise ValueError(f"No <impactbench_record_json> block found in {path}")
    record = json.loads(match.group(1))
    if not isinstance(record, dict):
        raise ValueError(f"Record block in {path} did not parse to an object")
    return record


def load_record_from_json(path: Path) -> dict[str, Any]:
    data = read_text_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a single JSON object in {path}")
    return data


def infer_prompt_path(annotation_path: Path) -> Path | None:
    name = annotation_path.name
    for suffix in ("_response.json", "_response.txt"):
        if name.endswith(suffix):
            candidate = annotation_path.with_name(name.removesuffix(suffix) + ".txt")
            if candidate.exists():
                return candidate
    return None


def source_record_id(record: dict[str, Any]) -> str:
    if record.get("record_id"):
        return str(record["record_id"])
    return f"{record.get('scenario_id')}::{record.get('metric_id')}::{record.get('transcript_model')}"


def source_assistant_turns(record: dict[str, Any]) -> list[SourceAssistantTurn]:
    samples = record.get("samples")
    if not isinstance(samples, list) or not samples or not isinstance(samples[0], list):
        raise ValueError("Record does not contain samples[0] conversation")

    turns: list[SourceAssistantTurn] = []
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
                SourceAssistantTurn(
                    assistant_turn_index=len(turns) + 1,
                    conversation_turn_index=conversation_turn_index,
                    preceding_user_turn_index=preceding_user_turn_index,
                    text=content,
                )
            )
    return turns


def add_issue(issues: list[dict[str, str]], severity: str, code: str, message: str) -> None:
    issues.append({"severity": severity, "code": code, "message": message})


def find_all(text: str, needle: str) -> list[int]:
    if not needle:
        return []
    starts: list[int] = []
    pos = text.find(needle)
    while pos != -1:
        starts.append(pos)
        pos = text.find(needle, pos + 1)
    return starts


def validate_int_score(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 3


def validate_span(
    item: dict[str, Any],
    source_turn: SourceAssistantTurn | None,
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    pooling = item.get("pooling")
    if not isinstance(pooling, dict):
        add_issue(issues, "error", "missing_pooling", "Turn pooling must be an object")
        return {
            "span_status": "invalid",
            "span_valid_for_training": False,
            "review_required": item.get("activation_quality") == "strong",
        }

    missing = sorted(POOLING_REQUIRED - set(pooling))
    for key in missing:
        add_issue(issues, "error", "missing_pooling_field", f"Missing pooling field: {key}")

    mode = pooling.get("mode")
    span = pooling.get("assistant_pooling_text")
    start = pooling.get("pooling_span_char_start")
    end = pooling.get("pooling_span_char_end")

    if mode not in POOLING_MODES:
        add_issue(issues, "error", "bad_pooling_mode", f"Unsupported pooling mode: {mode!r}")
    if not isinstance(span, str):
        add_issue(issues, "error", "bad_pooling_text", "assistant_pooling_text must be a string")
        span = ""

    label = item.get("activation_label")
    quality = item.get("activation_quality")
    strong = quality == "strong" and label in {"positive", "negative"}
    span_issue_severity = "error" if strong else "warning"

    if mode == "discard" or label == "discard" or quality == "discard":
        if span != "" or start is not None or end is not None:
            add_issue(issues, "error", "discard_span_not_empty", "Discard turns must use empty span and null offsets")
        return {
            "span_status": "discard",
            "span_valid_for_training": False,
            "review_required": False,
            "repaired_start": None,
            "repaired_end": None,
        }

    if source_turn is None:
        add_issue(issues, "error", "missing_source_turn", "Cannot validate span without source assistant turn")
        return {
            "span_status": "invalid",
            "span_valid_for_training": False,
            "review_required": strong,
        }

    if span == "":
        add_issue(issues, span_issue_severity, "empty_non_discard_span", "Non-discard turns must provide assistant_pooling_text")
        return {
            "span_status": "invalid",
            "span_valid_for_training": False,
            "review_required": strong,
        }

    offsets_are_ints = isinstance(start, int) and not isinstance(start, bool) and isinstance(end, int) and not isinstance(end, bool)
    offset_slice_matches = False
    if offsets_are_ints:
        if start < 0 or end > len(source_turn.text) or start >= end:
            add_issue(issues, span_issue_severity, "bad_offsets", f"Invalid offsets ({start}, {end}) for source length {len(source_turn.text)}")
        else:
            offset_slice_matches = source_turn.text[start:end] == span
            if not offset_slice_matches:
                add_issue(issues, "warning", "offset_slice_mismatch", "Provided offsets do not slice to assistant_pooling_text")
    elif start is not None or end is not None:
        add_issue(issues, span_issue_severity, "bad_offset_type", "Offsets must both be integers or both be null")

    occurrences = find_all(source_turn.text, span)
    if not occurrences:
        add_issue(issues, span_issue_severity, "span_not_found", "assistant_pooling_text is not an exact substring of the source assistant turn")
        return {
            "span_status": "invalid",
            "span_valid_for_training": False,
            "review_required": strong,
            "occurrences": 0,
        }

    if offset_slice_matches:
        return {
            "span_status": "valid",
            "span_valid_for_training": True,
            "review_required": False,
            "repaired_start": start,
            "repaired_end": end,
            "occurrences": len(occurrences),
        }

    if len(occurrences) == 1:
        repaired_start = occurrences[0]
        repaired_end = repaired_start + len(span)
        add_issue(
            issues,
            "warning",
            "offsets_repaired",
            f"Span text is unique; use repaired offsets ({repaired_start}, {repaired_end})",
        )
        return {
            "span_status": "offsets_repaired",
            "span_valid_for_training": True,
            "review_required": False,
            "repaired_start": repaired_start,
            "repaired_end": repaired_end,
            "occurrences": 1,
        }

    add_issue(
        issues,
        span_issue_severity,
        "ambiguous_span",
        f"assistant_pooling_text occurs {len(occurrences)} times and offsets did not disambiguate it",
    )
    return {
        "span_status": "ambiguous",
        "span_valid_for_training": False,
        "review_required": strong,
        "occurrences": len(occurrences),
    }


def validate_turn(
    item: Any,
    source_by_assistant_index: dict[int, SourceAssistantTurn],
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    if not isinstance(item, dict):
        return {
            "assistant_turn_index": None,
            "issues": [{"severity": "error", "code": "bad_turn", "message": "assistant_turns item must be an object"}],
            "training_decision": "exclude",
            "default_train_eligible": False,
            "review_required": False,
        }

    missing = sorted(TURN_REQUIRED - set(item))
    for key in missing:
        add_issue(issues, "error", "missing_turn_field", f"Missing turn field: {key}")

    assistant_turn_index = item.get("assistant_turn_index")
    source_turn = source_by_assistant_index.get(assistant_turn_index) if isinstance(assistant_turn_index, int) else None

    if not isinstance(assistant_turn_index, int) or isinstance(assistant_turn_index, bool):
        add_issue(issues, "error", "bad_assistant_turn_index", "assistant_turn_index must be an integer")
    elif source_turn is None:
        add_issue(issues, "error", "unknown_assistant_turn_index", f"No source assistant turn {assistant_turn_index}")

    if source_turn is not None:
        if item.get("conversation_turn_index") != source_turn.conversation_turn_index:
            add_issue(
                issues,
                "error",
                "conversation_turn_index_mismatch",
                f"Expected conversation_turn_index {source_turn.conversation_turn_index}",
            )
        if item.get("preceding_user_turn_index") != source_turn.preceding_user_turn_index:
            add_issue(
                issues,
                "error",
                "preceding_user_turn_index_mismatch",
                f"Expected preceding_user_turn_index {source_turn.preceding_user_turn_index}",
            )

    if item.get("criterion_effect") not in CRITERION_EFFECTS:
        add_issue(issues, "error", "bad_criterion_effect", f"Unsupported criterion_effect: {item.get('criterion_effect')!r}")
    if item.get("activation_label") not in ACTIVATION_LABELS:
        add_issue(issues, "error", "bad_activation_label", f"Unsupported activation_label: {item.get('activation_label')!r}")
    if item.get("activation_quality") not in ACTIVATION_QUALITIES:
        add_issue(issues, "error", "bad_activation_quality", f"Unsupported activation_quality: {item.get('activation_quality')!r}")
    if not isinstance(item.get("default_train_include"), bool):
        add_issue(issues, "error", "bad_default_train_include", "default_train_include must be boolean")

    for score_field in ("confidence", "evidence_strength", "opportunity_strength"):
        if not validate_int_score(item.get(score_field)):
            add_issue(issues, "error", "bad_score", f"{score_field} must be an integer from 0 to 3")

    label = item.get("activation_label")
    quality = item.get("activation_quality")
    default_train_include = item.get("default_train_include")

    if label == "discard" and quality != "discard":
        add_issue(issues, "error", "discard_quality_mismatch", "activation_label discard must use activation_quality discard")
    if quality == "discard" and default_train_include:
        add_issue(issues, "error", "discard_included", "Discard quality cannot be included in default training")
    if default_train_include and (quality != "strong" or label not in {"positive", "negative"}):
        add_issue(issues, "error", "bad_default_train_include_combo", "default_train_include requires strong positive/negative label")
    if quality == "strong" and label in {"positive", "negative"} and default_train_include is not True:
        add_issue(issues, "warning", "strong_not_included", "Strong positive/negative examples are expected to set default_train_include true")
    if label in {"positive", "negative"} and isinstance(item.get("confidence"), int) and item["confidence"] < 2:
        add_issue(issues, "error", "low_confidence_directional_label", "Directional labels require confidence >= 2")

    evidence_quotes = item.get("evidence_quotes")
    if not isinstance(evidence_quotes, list) or not all(isinstance(q, str) for q in evidence_quotes):
        add_issue(issues, "error", "bad_evidence_quotes", "evidence_quotes must be a list of strings")
    elif source_turn is not None:
        for quote in evidence_quotes:
            if quote and quote not in source_turn.text:
                add_issue(issues, "warning", "evidence_quote_not_found", f"Evidence quote not found in source assistant turn: {quote!r}")

    span_result = validate_span(item, source_turn, issues)
    has_error = any(issue["severity"] == "error" for issue in issues)
    wants_default_train = bool(default_train_include)
    span_valid = bool(span_result.get("span_valid_for_training"))

    default_train_eligible = wants_default_train and not has_error and span_valid
    review_required = bool(span_result.get("review_required"))
    if review_required:
        training_decision = "review_required"
    elif default_train_eligible:
        training_decision = "include"
    else:
        training_decision = "exclude"

    weak_invalid_ignored = quality == "weak" and not span_valid

    return {
        "assistant_turn_index": assistant_turn_index,
        "conversation_turn_index": item.get("conversation_turn_index"),
        "activation_label": label,
        "activation_quality": quality,
        "default_train_include": default_train_include,
        "span_status": span_result.get("span_status"),
        "span_valid_for_training": span_valid,
        "repaired_start": span_result.get("repaired_start"),
        "repaired_end": span_result.get("repaired_end"),
        "default_train_eligible": default_train_eligible,
        "training_decision": training_decision,
        "review_required": review_required,
        "weak_invalid_ignored": weak_invalid_ignored,
        "issues": issues,
    }


def validate_annotation(annotation: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    missing = sorted(TOP_LEVEL_REQUIRED - set(annotation))
    for key in missing:
        add_issue(issues, "error", "missing_top_level_field", f"Missing top-level field: {key}")

    expected_record_id = source_record_id(record)
    if annotation.get("record_id") != expected_record_id:
        add_issue(issues, "error", "record_id_mismatch", f"Expected record_id {expected_record_id!r}")

    for field in ("scenario_id", "scenario_title", "transcript_model", "benchmark", "metric_id", "metric_name", "behavior_type", "measurement", "metric_criterion"):
        if field in annotation and field in record and annotation.get(field) != record.get(field):
            add_issue(issues, "warning", "record_metadata_mismatch", f"Annotation {field} does not match source record")

    if annotation.get("record_usefulness") not in RECORD_USEFULNESS:
        add_issue(issues, "error", "bad_record_usefulness", f"Unsupported record_usefulness: {annotation.get('record_usefulness')!r}")

    source_turns = source_assistant_turns(record)
    source_by_assistant_index = {turn.assistant_turn_index: turn for turn in source_turns}
    annotated_turns = annotation.get("assistant_turns")
    if not isinstance(annotated_turns, list):
        add_issue(issues, "error", "bad_assistant_turns", "assistant_turns must be a list")
        annotated_turns = []

    if len(annotated_turns) != len(source_turns):
        add_issue(
            issues,
            "error",
            "assistant_turn_count_mismatch",
            f"Expected {len(source_turns)} assistant turns, got {len(annotated_turns)}",
        )

    turn_reports = [validate_turn(item, source_by_assistant_index) for item in annotated_turns]

    strong_positive_indices = [
        r["assistant_turn_index"]
        for r in turn_reports
        if r["activation_label"] == "positive" and r["activation_quality"] == "strong"
    ]
    strong_negative_indices = [
        r["assistant_turn_index"]
        for r in turn_reports
        if r["activation_label"] == "negative" and r["activation_quality"] == "strong"
    ]

    if annotation.get("best_positive_turn_indices") != strong_positive_indices:
        add_issue(
            issues,
            "warning",
            "best_positive_mismatch",
            f"Expected best_positive_turn_indices {strong_positive_indices}",
        )
    if annotation.get("best_negative_turn_indices") != strong_negative_indices:
        add_issue(
            issues,
            "warning",
            "best_negative_mismatch",
            f"Expected best_negative_turn_indices {strong_negative_indices}",
        )

    all_issues = issues + [issue for report in turn_reports for issue in report["issues"]]
    summary = {
        "source_assistant_turns": len(source_turns),
        "annotated_assistant_turns": len(annotated_turns),
        "top_level_errors": sum(1 for issue in issues if issue["severity"] == "error"),
        "top_level_warnings": sum(1 for issue in issues if issue["severity"] == "warning"),
        "turn_errors": sum(1 for report in turn_reports for issue in report["issues"] if issue["severity"] == "error"),
        "turn_warnings": sum(1 for report in turn_reports for issue in report["issues"] if issue["severity"] == "warning"),
        "default_train_requested": sum(1 for report in turn_reports if report["default_train_include"] is True),
        "default_train_eligible": sum(1 for report in turn_reports if report["default_train_eligible"]),
        "review_required": sum(1 for report in turn_reports if report["review_required"]),
        "weak_invalid_ignored": sum(1 for report in turn_reports if report["weak_invalid_ignored"]),
    }
    summary["has_errors"] = bool(summary["top_level_errors"] or summary["turn_errors"])
    summary["has_warnings"] = bool(summary["top_level_warnings"] or summary["turn_warnings"])

    return {
        "record_id": annotation.get("record_id"),
        "expected_record_id": expected_record_id,
        "summary": summary,
        "top_level_issues": issues,
        "turns": turn_reports,
        "clean_default_training_turn_indices": [
            r["assistant_turn_index"] for r in turn_reports if r["default_train_eligible"]
        ],
        "review_turn_indices": [
            r["assistant_turn_index"] for r in turn_reports if r["review_required"]
        ],
        "all_issue_count": len(all_issues),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate LLM turn-localization annotations against a source ImpactBench record.")
    parser.add_argument("--annotation", type=Path, required=True, help="LLM annotation JSON file.")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--prompt", type=Path, help="Prompt file containing <impactbench_record_json>.")
    source_group.add_argument("--record-json", type=Path, help="Single full ImpactBench record JSON file.")
    parser.add_argument("--output", type=Path, help="Optional path for the detailed validation report JSON.")
    parser.add_argument("--fail-on-issues", action="store_true", help="Exit non-zero if validation finds errors or review-required turns.")
    args = parser.parse_args()

    prompt_path = args.prompt
    if prompt_path is None and args.record_json is None:
        prompt_path = infer_prompt_path(args.annotation)
        if prompt_path is None:
            parser.error("provide --prompt or --record-json; could not infer prompt path from annotation name")

    try:
        annotation = read_text_json(args.annotation)
        if not isinstance(annotation, dict):
            raise ValueError("Annotation must be a JSON object")
        record = load_record_from_json(args.record_json) if args.record_json else load_record_from_prompt(prompt_path)  # type: ignore[arg-type]
        report = validate_annotation(annotation, record)
    except Exception as exc:
        print(f"validation failed before report generation: {exc}", file=sys.stderr)
        return 2

    report_json = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_json + "\n", encoding="utf-8")

    compact = {
        "annotation": str(args.annotation),
        "source": str(args.record_json or prompt_path),
        **report["summary"],
        "clean_default_training_turn_indices": report["clean_default_training_turn_indices"],
        "review_turn_indices": report["review_turn_indices"],
    }
    print(json.dumps(compact, ensure_ascii=True, indent=2, sort_keys=True))

    if args.fail_on_issues and (report["summary"]["has_errors"] or report["summary"]["review_required"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
