#!/usr/bin/env python3
"""Build ImpactBench turn-localization prompt folders and batch manifests.

The builder mirrors the completed Haiku run layout:

- one filled prompt per matching ImpactBench record
- subset prompt folders under `outputs/turn_localization_prompts/<run_id>/`
- 20-prompt batch folders under
  `outputs/turn_localization_prompt_batches/<run_id>__20_per_batch/`

It only prepares prompts and manifests. Annotation runs are handled by
`scripts/run_turn_localization_cli_batch.py` or an external batch runner.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TEMPLATE = (
    ROOT
    / "data"
    / "impactbench_autonomy"
    / "prompts"
    / "01_turn_localization_with_explicit_pooling.txt"
)

DEFAULT_SOURCES = {
    "autonomy_preservation": ROOT
    / "data"
    / "impactbench_autonomy"
    / "autonomy_preservation_records.jsonl",
    "self_determination": ROOT
    / "data"
    / "impactbench_autonomy"
    / "self_determination_records.jsonl",
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def slug_part(value: Any, *, max_len: int = 96) -> str:
    text = str(value or "missing").lower()
    text = re.sub(r"[^a-z0-9_.-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:max_len].strip("-") or "missing"


def record_id(record: dict[str, Any]) -> str:
    return f"{record.get('scenario_id')}::{record.get('metric_id')}::{record.get('transcript_model')}"


def subarea_name(record: dict[str, Any]) -> str | None:
    locations = record.get("locations")
    if isinstance(locations, list):
        for location in locations:
            if isinstance(location, dict) and location.get("subarea_name"):
                return str(location["subarea_name"])
    return None


def verdict_result(record: dict[str, Any]) -> str:
    verdict = record.get("verdict")
    if isinstance(verdict, dict) and verdict.get("result") is not None:
        return str(verdict["result"])
    return "missing"


def load_records(path: Path, transcript_model: str) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            record = json.loads(line)
            if record.get("transcript_model") == transcript_model:
                rows.append((line_no, record))
    return rows


def prompt_filename(index: int, record: dict[str, Any]) -> str:
    metric_id = slug_part(record.get("metric_id"))
    scenario_id = slug_part(record.get("scenario_id"), max_len=64)
    verdict = slug_part(verdict_result(record), max_len=24)
    return f"{index:04d}__{metric_id}__{scenario_id}__{verdict}.txt"


def manifest_record(
    *,
    subset: str,
    source_path: Path,
    source_line_no: int,
    prompt_path: Path,
    record: dict[str, Any],
) -> dict[str, Any]:
    return {
        "behavior_type": record.get("behavior_type"),
        "measurement": record.get("measurement"),
        "metric_id": record.get("metric_id"),
        "metric_name": record.get("metric_name"),
        "prompt_path": rel(prompt_path),
        "record_id": record_id(record),
        "scenario_id": record.get("scenario_id"),
        "scenario_title": record.get("scenario_title"),
        "source_line_no": source_line_no,
        "source_path": rel(source_path),
        "subset": subset,
        "transcript_model": record.get("transcript_model"),
        "verdict_result": verdict_result(record),
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def build_prompts(
    *,
    transcript_model: str,
    run_id: str,
    template_path: Path,
    prompt_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    template = template_path.read_text(encoding="utf-8")
    if "{{RECORD_JSON}}" not in template:
        raise ValueError(f"template does not contain {{RECORD_JSON}}: {template_path}")

    all_rows: list[dict[str, Any]] = []
    subset_summaries: dict[str, Any] = {}
    for subset, source_path in DEFAULT_SOURCES.items():
        rows = load_records(source_path, transcript_model)
        if not rows:
            raise ValueError(f"no records for transcript_model={transcript_model!r} in {source_path}")

        prompt_dir = prompt_root / subset
        prompt_dir.mkdir(parents=True, exist_ok=True)
        metric_counts: Counter[str] = Counter()
        verdict_counts: Counter[str] = Counter()

        for index, (source_line_no, record) in enumerate(rows, start=1):
            record_json = json.dumps(record, ensure_ascii=True, separators=(",", ":"))
            prompt_text = template.replace("{{RECORD_JSON}}", record_json)
            prompt_path = prompt_dir / prompt_filename(index, record)
            prompt_path.write_text(prompt_text, encoding="utf-8")

            metric_counts[str(record.get("metric_id"))] += 1
            verdict_counts[verdict_result(record)] += 1
            all_rows.append(
                manifest_record(
                    subset=subset,
                    source_path=source_path,
                    source_line_no=source_line_no,
                    prompt_path=prompt_path,
                    record=record,
                )
            )

        subset_summaries[subset] = {
            "metric_prompt_counts": dict(sorted(metric_counts.items())),
            "prompt_dir": rel(prompt_dir),
            "prompts": len(rows),
            "source_path": rel(source_path),
            "unique_metric_ids": len(metric_counts),
            "verdict_counts": dict(sorted(verdict_counts.items())),
        }

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest_path": rel(prompt_root / "manifest.json"),
        "manifest_records_path": rel(prompt_root / "manifest_records.jsonl"),
        "output_dir": rel(prompt_root),
        "subsets": subset_summaries,
        "template_path": rel(template_path),
        "total_prompts": len(all_rows),
        "transcript_model": transcript_model,
    }
    write_json(prompt_root / "manifest.json", manifest)
    write_jsonl(prompt_root / "manifest_records.jsonl", all_rows)
    return manifest, all_rows


def build_batches(
    *,
    run_id: str,
    batch_size: int,
    prompt_rows: list[dict[str, Any]],
    batch_root: Path,
    annotation_root: Path,
) -> dict[str, Any]:
    batch_rows: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    n_batches = math.ceil(len(prompt_rows) / batch_size)

    for batch_index in range(1, n_batches + 1):
        batch_id = f"batch_{batch_index:04d}"
        batch_dir = batch_root / batch_id
        prompts_dir = batch_dir / "prompts"
        annotation_output_dir = annotation_root / batch_id
        prompts_dir.mkdir(parents=True, exist_ok=True)

        chunk = prompt_rows[(batch_index - 1) * batch_size : batch_index * batch_size]
        prompt_manifest_rows: list[dict[str, Any]] = []
        for row in chunk:
            source_prompt_path = ROOT / row["prompt_path"]
            target_prompt_path = prompts_dir / source_prompt_path.name
            shutil.copy2(source_prompt_path, target_prompt_path)

            batch_row = dict(row)
            batch_row.update(
                {
                    "annotation_output_path": rel(annotation_output_dir / (target_prompt_path.stem + ".json")),
                    "batch_id": batch_id,
                    "batch_index": batch_index,
                    "batch_prompt_path": rel(target_prompt_path),
                }
            )
            prompt_manifest_rows.append(batch_row)

        batch_summary = {
            "annotation_output_dir": rel(annotation_output_dir),
            "batch_id": batch_id,
            "batch_index": batch_index,
            "batch_manifest_jsonl": rel(batch_dir / "batch_manifest.jsonl"),
            "prompt_count": len(chunk),
            "prompts_dir": rel(prompts_dir),
        }
        write_jsonl(batch_dir / "batch_manifest.jsonl", prompt_manifest_rows)
        write_json(batch_dir / "batch_manifest.json", batch_summary)
        batches.append(batch_summary)
        batch_rows.append(batch_summary)

    top_manifest = {
        "batch_root": rel(batch_root),
        "batch_size": batch_size,
        "batches": batches,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest_jsonl": rel(batch_root / "manifest.jsonl"),
        "prompt_run": rel(ROOT / "outputs" / "turn_localization_prompts" / run_id),
        "total_batches": len(batches),
        "total_prompts": len(prompt_rows),
    }
    write_json(batch_root / "manifest.json", top_manifest)
    write_jsonl(batch_root / "manifest.jsonl", batch_rows)

    readme = f"""# Turn Localization Prompt Batches

Source prompt run: `outputs/turn_localization_prompts/{run_id}`

This directory splits the {len(prompt_rows)} filled turn-localization prompts into {len(batches)} folders of up to {batch_size} prompts each.

Use each `batch_XXXX/prompts/` folder as one GPT MCP batch input.

Write JSON annotations to the matching folder under:

`{rel(annotation_root)}`

Each batch has:

- `prompts/`: copied prompt `.txt` files
- `batch_manifest.jsonl`: one row per prompt with source metadata and recommended output path
- `batch_manifest.json`: compact batch summary

Top-level files:

- `manifest.json`: complete batch summary
- `manifest.jsonl`: one row per batch
"""
    (batch_root / "README.md").write_text(readme, encoding="utf-8")
    return top_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript-model", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    today = datetime.now().strftime("%Y%m%d")
    run_id = args.run_id or f"{args.transcript_model}_{today}"
    prompt_root = ROOT / "outputs" / "turn_localization_prompts" / run_id
    batch_root = (
        ROOT
        / "outputs"
        / "turn_localization_prompt_batches"
        / f"{run_id}__{args.batch_size}_per_batch"
    )
    annotation_root = ROOT / "outputs" / "turn_localization_annotations" / f"gpt-mcp_from_{run_id}"

    for path in (prompt_root, batch_root):
        if path.exists():
            if not args.force:
                raise SystemExit(f"{path} already exists; pass --force to replace it")
            shutil.rmtree(path)

    prompt_manifest, prompt_rows = build_prompts(
        transcript_model=args.transcript_model,
        run_id=run_id,
        template_path=args.template,
        prompt_root=prompt_root,
    )
    batch_manifest = build_batches(
        run_id=run_id,
        batch_size=args.batch_size,
        prompt_rows=prompt_rows,
        batch_root=batch_root,
        annotation_root=annotation_root,
    )

    summary = {
        "annotation_root": rel(annotation_root),
        "batch_root": rel(batch_root),
        "batch_size": args.batch_size,
        "prompt_root": rel(prompt_root),
        "total_batches": batch_manifest["total_batches"],
        "total_prompts": prompt_manifest["total_prompts"],
        "transcript_model": args.transcript_model,
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
