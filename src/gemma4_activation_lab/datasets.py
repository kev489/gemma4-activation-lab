from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .io_utils import load_jsonl

MATCHED_REQUIRED_FIELDS = {
    "scenario_id",
    "user_message",
    "response_style_label",
    "assistant_target_text",
}

HELDOUT_REQUIRED_FIELDS = {
    "scenario_id",
    "user_message",
}


def validate_rows(
    rows: list[dict[str, Any]],
    *,
    required_fields: set[str],
    source: Path,
    row_kind: str,
) -> None:
    if not rows:
        raise ValueError(f"{row_kind} dataset is empty: {source}")

    for row_index, row in enumerate(rows, start=1):
        missing = sorted(required_fields - set(row))
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"{row_kind} row {row_index} in {source} is missing required field(s): {missing_text}")


def load_validated_jsonl(
    path: Path,
    *,
    required_fields: set[str],
    row_kind: str,
) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    validate_rows(rows, required_fields=required_fields, source=path, row_kind=row_kind)
    return rows


def load_matched_rows(path: Path) -> list[dict[str, Any]]:
    return load_validated_jsonl(
        path,
        required_fields=MATCHED_REQUIRED_FIELDS,
        row_kind="matched conversation",
    )


def load_heldout_rows(path: Path) -> list[dict[str, Any]]:
    return load_validated_jsonl(
        path,
        required_fields=HELDOUT_REQUIRED_FIELDS,
        row_kind="held-out prompt",
    )


def require_labels(rows: Iterable[dict[str, Any]], labels: Iterable[str]) -> None:
    available = {row["response_style_label"] for row in rows}
    missing = [label for label in labels if label not in available]
    if missing:
        available_text = ", ".join(sorted(available))
        missing_text = ", ".join(missing)
        raise KeyError(f"Missing label(s): {missing_text}. Available labels: {available_text}")
