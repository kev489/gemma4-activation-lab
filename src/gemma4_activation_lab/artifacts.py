from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from .io_utils import save_json, slugify


BASELINE_FIELDNAMES = [
    "scenario_id",
    "user_message",
    "baseline_output",
]

STEERING_FIELDNAMES = [
    "scenario_id",
    "module_name",
    "alpha",
    "position_mode",
    "user_message",
    "baseline_output",
    "intervened_output",
]

PROBE_STEERING_FIELDNAMES = [
    "scenario_id",
    "module_name",
    "alpha",
    "user_message",
    "baseline_output",
    "intervened_output",
]


def append_jsonl(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def write_rows_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_direction_vectors(
    run_dir: Path,
    directions: dict[str, torch.Tensor],
    *,
    positive_label: str,
    negative_label: str,
) -> list[dict[str, Any]]:
    vector_dir = run_dir / "vectors"
    vector_dir.mkdir(parents=True, exist_ok=True)
    direction_metadata: list[dict[str, Any]] = []

    for module_name, direction in directions.items():
        save_path = vector_dir / f"{slugify(module_name)}__{slugify(positive_label)}_minus_{slugify(negative_label)}.pt"
        torch.save(
            {
                "module_name": module_name,
                "positive_label": positive_label,
                "negative_label": negative_label,
                "vector": direction.cpu(),
            },
            save_path,
        )
        direction_metadata.append(
            {
                "module_name": module_name,
                "path": str(save_path),
                "vector_norm": float(torch.linalg.vector_norm(direction).item()),
            }
        )

    return direction_metadata


def save_probe_artifacts(
    run_dir: Path,
    *,
    summary: dict[str, Any],
    capture_records: list[dict[str, Any]],
    steering_results: list[dict[str, Any]],
) -> None:
    save_json(run_dir / "summary.json", summary)
    save_json(run_dir / "capture_records.json", {"records": capture_records})
    save_json(run_dir / "steering_results.json", {"rows": steering_results})
    write_rows_csv(run_dir / "steering_summary.csv", steering_results, PROBE_STEERING_FIELDNAMES)
