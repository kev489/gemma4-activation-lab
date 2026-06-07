from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .io_utils import load_jsonl


ACTIVATION_EXAMPLE_SCHEMA_VERSION = "impactbench_activation_example_v1"
ACTIVATION_LABELS = {"positive", "negative"}

ACTIVATION_EXAMPLE_REQUIRED_FIELDS = {
    "schema_version",
    "example_id",
    "record_id",
    "scenario_id",
    "transcript_model",
    "benchmark",
    "metric_id",
    "metric_name",
    "metric_criterion",
    "behavior_type",
    "measurement",
    "harmful",
    "subarea_names",
    "global_verdict_result",
    "assistant_turn_index",
    "conversation_turn_index",
    "preceding_user_turn_index",
    "activation_label",
    "activation_quality",
    "assistant_turn_text",
    "assistant_turn_sha256",
    "source_record_sha256",
    "source_archive",
    "pooling",
    "annotation_provenance",
}

POOLING_REQUIRED_FIELDS = {
    "mode",
    "assistant_pooling_text",
    "pooling_span_char_start",
    "pooling_span_char_end",
    "pooling_text_sha256",
}


@dataclass(frozen=True)
class SourceAssistantTurn:
    assistant_turn_index: int
    conversation_turn_index: int
    preceding_user_turn_index: int | None
    text: str


@dataclass(frozen=True)
class ResolvedActivationExample:
    example: dict[str, Any]
    record: dict[str, Any]
    messages: list[dict[str, str]]
    assistant_turn: SourceAssistantTurn


@dataclass(frozen=True)
class RenderedActivationExample:
    text: str
    token_span_start: int
    token_span_end: int
    rendered_char_start: int
    rendered_char_end: int


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_record_json(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def impactbench_record_id(record: dict[str, Any]) -> str:
    return f"{record.get('scenario_id')}::{record.get('metric_id')}::{record.get('transcript_model')}"


def source_assistant_turns(record: dict[str, Any]) -> list[SourceAssistantTurn]:
    samples = record.get("samples")
    if not isinstance(samples, list) or not samples or not isinstance(samples[0], list):
        raise ValueError("ImpactBench record does not contain a samples[0] conversation")

    turns: list[SourceAssistantTurn] = []
    preceding_user_turn_index: int | None = None
    for conversation_turn_index, message in enumerate(samples[0], start=1):
        if not isinstance(message, dict):
            raise ValueError(f"Non-object message at conversation turn {conversation_turn_index}")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"Bad message at conversation turn {conversation_turn_index}")
        if role == "user":
            preceding_user_turn_index = conversation_turn_index
        elif role == "assistant":
            turns.append(
                SourceAssistantTurn(
                    assistant_turn_index=len(turns) + 1,
                    conversation_turn_index=conversation_turn_index,
                    preceding_user_turn_index=preceding_user_turn_index,
                    text=content,
                )
            )
    return turns


def source_assistant_turn(record: dict[str, Any], assistant_turn_index: int) -> SourceAssistantTurn:
    turns = source_assistant_turns(record)
    if assistant_turn_index < 1 or assistant_turn_index > len(turns):
        raise IndexError(
            f"assistant_turn_index={assistant_turn_index} outside 1..{len(turns)} "
            f"for {impactbench_record_id(record)}"
        )
    return turns[assistant_turn_index - 1]


def validate_activation_examples(
    rows: list[dict[str, Any]],
    *,
    source: Path,
) -> None:
    if not rows:
        raise ValueError(f"Activation example dataset is empty: {source}")

    example_ids: set[str] = set()
    turn_keys: set[tuple[str, int]] = set()
    for row_index, row in enumerate(rows, start=1):
        missing = sorted(ACTIVATION_EXAMPLE_REQUIRED_FIELDS - set(row))
        if missing:
            raise ValueError(f"Activation example row {row_index} missing fields: {', '.join(missing)}")
        if row["schema_version"] != ACTIVATION_EXAMPLE_SCHEMA_VERSION:
            raise ValueError(
                f"Activation example row {row_index} has unsupported schema_version="
                f"{row['schema_version']!r}"
            )
        if row["activation_label"] not in ACTIVATION_LABELS:
            raise ValueError(f"Activation example row {row_index} has bad activation_label")
        if row["activation_quality"] != "strong":
            raise ValueError(f"Activation example row {row_index} is not strong quality")
        if not isinstance(row["harmful"], bool):
            raise ValueError(f"Activation example row {row_index} has non-boolean harmful flag")

        example_id = row["example_id"]
        if not isinstance(example_id, str) or not example_id:
            raise ValueError(f"Activation example row {row_index} has bad example_id")
        if example_id in example_ids:
            raise ValueError(f"Duplicate example_id at row {row_index}: {example_id}")
        example_ids.add(example_id)

        assistant_turn_index = row["assistant_turn_index"]
        if not isinstance(assistant_turn_index, int) or assistant_turn_index < 1:
            raise ValueError(f"Activation example row {row_index} has bad assistant_turn_index")
        turn_key = (row["record_id"], assistant_turn_index)
        if turn_key in turn_keys:
            raise ValueError(f"Duplicate record/turn key at row {row_index}: {turn_key}")
        turn_keys.add(turn_key)

        assistant_text = row["assistant_turn_text"]
        if not isinstance(assistant_text, str) or not assistant_text:
            raise ValueError(f"Activation example row {row_index} has empty assistant_turn_text")
        if sha256_text(assistant_text) != row["assistant_turn_sha256"]:
            raise ValueError(f"Activation example row {row_index} has assistant text hash mismatch")

        pooling = row["pooling"]
        if not isinstance(pooling, dict):
            raise ValueError(f"Activation example row {row_index} has non-object pooling")
        pooling_missing = sorted(POOLING_REQUIRED_FIELDS - set(pooling))
        if pooling_missing:
            raise ValueError(
                f"Activation example row {row_index} pooling missing fields: "
                f"{', '.join(pooling_missing)}"
            )
        pooling_text = pooling["assistant_pooling_text"]
        start = pooling["pooling_span_char_start"]
        end = pooling["pooling_span_char_end"]
        if not isinstance(pooling_text, str) or not pooling_text:
            raise ValueError(f"Activation example row {row_index} has empty pooling text")
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or start >= end:
            raise ValueError(f"Activation example row {row_index} has bad pooling offsets")
        if end > len(assistant_text) or assistant_text[start:end] != pooling_text:
            raise ValueError(f"Activation example row {row_index} pooling span is not exact")
        if sha256_text(pooling_text) != pooling["pooling_text_sha256"]:
            raise ValueError(f"Activation example row {row_index} has pooling text hash mismatch")


def load_activation_examples(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    validate_activation_examples(rows, source=path)
    return rows


def load_impactbench_record_index(
    transcript_dir: Path,
    *,
    transcript_models: Iterable[str] | None = None,
) -> dict[str, dict[str, Any]]:
    models = sorted(set(transcript_models or []))
    if models:
        paths = [transcript_dir / f"{model}_transcripts.jsonl.gz" for model in models]
    else:
        paths = sorted(transcript_dir.glob("*_transcripts.jsonl.gz"))
    if not paths:
        raise FileNotFoundError(f"No transcript archives found under {transcript_dir}")

    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                record = json.loads(line)
                record_id = impactbench_record_id(record)
                if record_id in records:
                    raise ValueError(f"Duplicate ImpactBench record id in archives: {record_id}")
                records[record_id] = record
    return records


def conversation_through_assistant_turn(
    record: dict[str, Any],
    assistant_turn_index: int,
) -> list[dict[str, str]]:
    target = source_assistant_turn(record, assistant_turn_index)
    messages = record["samples"][0][: target.conversation_turn_index]
    normalized: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"Bad source message in {impactbench_record_id(record)}")
        normalized.append({"role": role, "content": content})
    if not normalized or normalized[-1]["role"] != "assistant":
        raise RuntimeError(f"Target conversation does not end at an assistant turn: {impactbench_record_id(record)}")
    return normalized


def resolve_activation_example(
    example: dict[str, Any],
    record_index: dict[str, dict[str, Any]],
) -> ResolvedActivationExample:
    record_id = example["record_id"]
    if record_id not in record_index:
        raise KeyError(f"Missing source record for activation example: {record_id}")
    record = record_index[record_id]
    if sha256_text(canonical_record_json(record)) != example["source_record_sha256"]:
        raise ValueError(f"Source record hash mismatch for {example['example_id']}")

    turn = source_assistant_turn(record, example["assistant_turn_index"])
    if turn.text != example["assistant_turn_text"]:
        raise ValueError(f"Source assistant text mismatch for {example['example_id']}")
    if turn.conversation_turn_index != example["conversation_turn_index"]:
        raise ValueError(f"Conversation turn index mismatch for {example['example_id']}")
    if turn.preceding_user_turn_index != example["preceding_user_turn_index"]:
        raise ValueError(f"Preceding user turn index mismatch for {example['example_id']}")

    return ResolvedActivationExample(
        example=example,
        record=record,
        messages=conversation_through_assistant_turn(record, example["assistant_turn_index"]),
        assistant_turn=turn,
    )


def _find_last_occurrence(text: str, substring: str) -> int:
    start = text.rfind(substring)
    if start < 0:
        raise RuntimeError("Assistant turn text was not preserved by chat templating")
    return start


def render_activation_example(
    processor: Any,
    resolved: ResolvedActivationExample,
) -> RenderedActivationExample:
    import torch

    from .modeling import apply_chat_template_text

    full_text = apply_chat_template_text(
        processor,
        resolved.messages,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    assistant_rendered_start = _find_last_occurrence(full_text, resolved.assistant_turn.text)
    pooling = resolved.example["pooling"]
    rendered_char_start = assistant_rendered_start + pooling["pooling_span_char_start"]
    rendered_char_end = assistant_rendered_start + pooling["pooling_span_char_end"]
    if full_text[rendered_char_start:rendered_char_end] != pooling["assistant_pooling_text"]:
        raise RuntimeError(f"Rendered pooling span mismatch for {resolved.example['example_id']}")

    tokenizer = getattr(processor, "tokenizer", processor)
    encoded = tokenizer(text=full_text, return_offsets_mapping=True, return_tensors="pt")
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise RuntimeError("Tokenizer does not provide offset_mapping")
    model_input_ids = processor(text=full_text, return_tensors="pt")["input_ids"]
    if not torch.equal(encoded["input_ids"], model_input_ids):
        raise RuntimeError("Tokenizer input ids differ from processor input ids")

    token_indices = [
        token_index
        for token_index, (start, end) in enumerate(offsets[0].tolist())
        if end > rendered_char_start and start < rendered_char_end
    ]
    if not token_indices:
        raise RuntimeError(f"Pooling span tokenized to no tokens for {resolved.example['example_id']}")
    if token_indices != list(range(token_indices[0], token_indices[-1] + 1)):
        raise RuntimeError(f"Pooling token span is not contiguous for {resolved.example['example_id']}")

    return RenderedActivationExample(
        text=full_text,
        token_span_start=token_indices[0],
        token_span_end=token_indices[-1] + 1,
        rendered_char_start=rendered_char_start,
        rendered_char_end=rendered_char_end,
    )
