#!/usr/bin/env python3
"""Run turn-localization prompt batches through Codex CLI or Claude CLI.

This is the CLI analogue of the GPT MCP batch workflow:

- reads each prompt file from a `batch_XXXX/prompts/` directory
- sends prompts with configurable concurrency
- writes one parsed annotation JSON file per prompt
- after all annotation attempts finish, validates the completed annotations
- writes `_run_manifest.jsonl` and `_run_summary.json`

The script does not rewrite, move, or edit source prompts.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PROMPT_BATCH_ROOT = (
    ROOT
    / "outputs"
    / "turn_localization_prompt_batches"
    / "claude-haiku-4-5_20260528__20_per_batch"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "outputs"
    / "turn_localization_annotations"
    / "gpt-mcp_from_claude-haiku-4-5_20260528"
)

DEFAULT_CODEX_MODEL = "gpt-5.4-mini"
DEFAULT_CLAUDE_MODEL = "claude-opus-4-7"


ANNOTATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "record_id": {"type": "string"},
        "scenario_id": {"type": "string"},
        "scenario_title": {"type": "string"},
        "transcript_model": {"type": "string"},
        "benchmark": {"type": "string"},
        "metric_id": {"type": "string"},
        "metric_name": {"type": "string"},
        "behavior_type": {"type": "string"},
        "measurement": {"type": "string"},
        "metric_criterion": {"type": "string"},
        "subarea_name": {"type": "string"},
        "global_verdict_result": {"type": "string"},
        "global_verdict_used_only_as_context": {"type": "boolean"},
        "assistant_turns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "assistant_turn_index": {"type": "integer"},
                    "conversation_turn_index": {"type": "integer"},
                    "preceding_user_turn_index": {
                        "anyOf": [{"type": "integer"}, {"type": "null"}]
                    },
                    "criterion_effect": {
                        "type": "string",
                        "enum": ["satisfies", "violates", "mixed", "irrelevant"],
                    },
                    "activation_label": {
                        "type": "string",
                        "enum": ["positive", "negative", "discard"],
                    },
                    "activation_quality": {
                        "type": "string",
                        "enum": ["strong", "weak", "discard"],
                    },
                    "default_train_include": {"type": "boolean"},
                    "confidence": {"type": "integer", "minimum": 0, "maximum": 3},
                    "evidence_strength": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 3,
                    },
                    "opportunity_strength": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 3,
                    },
                    "evidence_quotes": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "reason": {"type": "string"},
                    "confounds": {"type": "array", "items": {"type": "string"}},
                    "pooling": {
                        "type": "object",
                        "properties": {
                            "mode": {
                                "type": "string",
                                "enum": [
                                    "assistant_turn_full",
                                    "sentence_subset",
                                    "discard",
                                ],
                            },
                            "assistant_pooling_text": {"type": "string"},
                            "pooling_span_char_start": {
                                "anyOf": [{"type": "integer"}, {"type": "null"}]
                            },
                            "pooling_span_char_end": {
                                "anyOf": [{"type": "integer"}, {"type": "null"}]
                            },
                            "pooling_span_reason": {"type": "string"},
                        },
                        "required": [
                            "mode",
                            "assistant_pooling_text",
                            "pooling_span_char_start",
                            "pooling_span_char_end",
                            "pooling_span_reason",
                        ],
                        "additionalProperties": False,
                    },
                },
                "required": [
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
                ],
                "additionalProperties": False,
            },
        },
        "best_positive_turn_indices": {"type": "array", "items": {"type": "integer"}},
        "best_negative_turn_indices": {"type": "array", "items": {"type": "integer"}},
        "record_usefulness": {
            "type": "string",
            "enum": ["high", "medium", "low", "none"],
        },
        "notes": {"type": "string"},
    },
    "required": [
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
    ],
    "additionalProperties": False,
}


def parse_jsonish(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("model output was not a JSON object")
    return parsed


def parse_claude_stdout(stdout: str) -> dict[str, Any]:
    envelope = json.loads(stdout)
    if not isinstance(envelope, dict):
        raise ValueError("Claude stdout was not a JSON object")
    if envelope.get("is_error"):
        raise RuntimeError(
            "Claude returned an error envelope: "
            + json.dumps(envelope, ensure_ascii=True)[:1000]
        )
    structured = envelope.get("structured_output")
    if structured is not None:
        if not isinstance(structured, dict):
            raise ValueError("Claude structured_output was not a JSON object")
        return structured
    result = envelope.get("result")
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        return parse_jsonish(result)
    raise ValueError("Claude stdout had neither structured_output nor JSON result")


def run_codex(
    *,
    codex_cli: str,
    model: str,
    reasoning_effort: str | None,
    prompt_text: str,
    output_path: Path,
    timeout_s: int,
    schema_path: Path,
    sandbox: str,
    cwd: Path,
    codex_home: Path | None = None,
) -> dict[str, Any]:
    tmp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    cmd = [
        codex_cli,
        "exec",
        "--model",
        model,
        "--ephemeral",
        "--sandbox",
        sandbox,
        "--cd",
        str(cwd),
        "--skip-git-repo-check",
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(tmp_output_path),
        "--color",
        "never",
        "-",
    ]
    if reasoning_effort is not None:
        cmd[4:4] = ["-c", f'model_reasoning_effort="{reasoning_effort}"']
    started = time.monotonic()
    proc = subprocess.run(
        cmd,
        input=prompt_text,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
        env={**os.environ, "CODEX_HOME": str(codex_home)} if codex_home else None,
    )
    duration_ms = int((time.monotonic() - started) * 1000)
    output_text = tmp_output_path.read_text(encoding="utf-8") if tmp_output_path.exists() else proc.stdout
    if proc.returncode != 0:
        raise RuntimeError(
            f"codex exited {proc.returncode}: stdout={proc.stdout[:1000]!r} "
            f"stderr={proc.stderr[:1000]!r}"
        )
    annotation = parse_jsonish(output_text)
    annotation["_cli_metadata"] = {
        "provider": "codex",
        "model": model,
        "duration_ms": duration_ms,
    }
    return annotation


def prepare_isolated_codex_home(path: Path, source_home: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for filename in ("auth.json", "config.toml", "installation_id", "models_cache.json"):
        source = source_home / filename
        if source.exists():
            shutil.copy2(source, path / filename)


def run_claude(
    *,
    claude_cli: str,
    model: str,
    prompt_text: str,
    timeout_s: int,
    schema: dict[str, Any],
) -> dict[str, Any]:
    cmd = [
        claude_cli,
        "-p",
        "--model",
        model,
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(schema, separators=(",", ":")),
        "--no-session-persistence",
        "--disable-slash-commands",
        "--permission-mode",
        "dontAsk",
    ]
    started = time.monotonic()
    proc = subprocess.run(
        cmd,
        input=prompt_text,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    duration_ms = int((time.monotonic() - started) * 1000)
    if proc.returncode != 0:
        raise RuntimeError(
            f"claude exited {proc.returncode}: stdout={proc.stdout[:1000]!r} "
            f"stderr={proc.stderr[:1000]!r}"
        )
    annotation = parse_claude_stdout(proc.stdout)
    annotation["_cli_metadata"] = {
        "provider": "claude",
        "model": model,
        "duration_ms": duration_ms,
    }
    return annotation


def write_annotation(path: Path, annotation: dict[str, Any]) -> None:
    # Keep validation-facing files exactly in the prompt schema; metadata stays in
    # the manifest instead of adding an unexpected top-level annotation key.
    annotation = dict(annotation)
    annotation.pop("_cli_metadata", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(annotation, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_validation(
    *,
    prompt_path: Path,
    annotation_path: Path,
    validation_path: Path,
    python_exe: str,
) -> tuple[str, dict[str, Any] | None, str | None]:
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
        return "validation_error", None, proc.stderr.strip() or proc.stdout.strip()
    try:
        report = json.loads(validation_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return "validation_error", None, f"could not read validation report: {exc}"
    summary = report.get("summary", {})
    if summary.get("has_errors"):
        return "validation_error", report, None
    if summary.get("review_required"):
        return "review_required", report, None
    return "ok", report, None


def prompt_paths_for_batch(prompt_dir: Path) -> list[Path]:
    paths = sorted(prompt_dir.glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"no .txt prompts found in {prompt_dir}")
    return paths


def make_pending_row(
    *,
    batch_id: str,
    provider: str,
    model: str,
    prompt_path: Path,
    annotation_path: Path,
    validation_path: Path,
) -> dict[str, Any]:
    return {
        "batch_id": batch_id,
        "provider": provider,
        "model": model,
        "prompt_path": str(prompt_path),
        "annotation_path": str(annotation_path),
        "validation_path": str(validation_path),
        "annotation_status": "pending",
        "validation_status": "pending",
        "status": "pending",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default="batch_0013")
    parser.add_argument("--provider", choices=["codex", "claude"], required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt-batch-root", type=Path, default=DEFAULT_PROMPT_BATCH_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--codex-cli", default="codex")
    parser.add_argument("--claude-cli", default="claude")
    parser.add_argument("--codex-cwd", type=Path, default=Path("/tmp"))
    parser.add_argument("--codex-sandbox", default="read-only")
    parser.add_argument(
        "--codex-reasoning-effort",
        choices=["low", "medium", "high", "xhigh"],
        default=None,
        help="Override Codex reasoning effort for codex exec calls.",
    )
    parser.add_argument(
        "--codex-home-mode",
        choices=["shared", "isolated"],
        default="shared",
        help="Use a unique temporary CODEX_HOME per Codex worker to avoid shared-state races.",
    )
    parser.add_argument(
        "--codex-home-source",
        type=Path,
        default=Path.home() / ".codex",
        help="Source CODEX_HOME to copy auth/config files from when --codex-home-mode=isolated.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of prompts to annotate concurrently.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    cli = args.codex_cli if args.provider == "codex" else args.claude_cli
    if shutil.which(cli) is None:
        raise SystemExit(f"`{cli}` not found on PATH")

    model = args.model
    if model is None:
        model = DEFAULT_CODEX_MODEL if args.provider == "codex" else DEFAULT_CLAUDE_MODEL

    prompt_dir = args.prompt_batch_root / args.batch_id / "prompts"
    output_dir = args.output_root / args.batch_id
    validation_dir = output_dir / "validation"
    manifest_path = output_dir / "_run_manifest.jsonl"
    summary_path = output_dir / "_run_summary.json"
    prompt_paths = prompt_paths_for_batch(prompt_dir)
    if args.max_prompts is not None:
        prompt_paths = prompt_paths[: args.max_prompts]

    if args.dry_run:
        print(f"provider={args.provider} model={model}")
        print(f"prompt_dir={prompt_dir}")
        print(f"output_dir={output_dir}")
        print(f"validation_dir={validation_dir}")
        for prompt_path in prompt_paths:
            print(f"{prompt_path} -> {output_dir / (prompt_path.stem + '.json')}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    schema_path = output_dir / "_annotation_schema.json"
    schema_path.write_text(
        json.dumps(ANNOTATION_SCHEMA, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    print(
        f"[annotations] {args.provider} {model}"
        + (
            f" effort={args.codex_reasoning_effort}"
            if args.provider == "codex" and args.codex_reasoning_effort is not None
            else ""
        )
        + f": {len(prompt_paths)} prompts "
        f"from {prompt_dir} with workers={args.workers}",
        flush=True,
    )

    def annotate_one(index: int, prompt_path: Path) -> tuple[int, dict[str, Any]]:
        annotation_path = output_dir / (prompt_path.stem + ".json")
        validation_path = validation_dir / (prompt_path.stem + ".validation.json")
        row = make_pending_row(
            batch_id=args.batch_id,
            provider=args.provider,
            model=model,
            prompt_path=prompt_path,
            annotation_path=annotation_path,
            validation_path=validation_path,
        )
        if args.skip_existing and annotation_path.exists():
            row["annotation_status"] = "skipped_existing"
            return index, row

        try:
            prompt_text = prompt_path.read_text(encoding="utf-8")
            if args.provider == "codex":
                annotation = run_codex(
                    codex_cli=args.codex_cli,
                    model=model,
                    reasoning_effort=args.codex_reasoning_effort,
                    prompt_text=prompt_text,
                    output_path=annotation_path,
                    timeout_s=args.timeout_s,
                    schema_path=schema_path,
                    sandbox=args.codex_sandbox,
                    cwd=args.codex_cwd,
                    codex_home=codex_home_for(index),
                )
            else:
                annotation = run_claude(
                    claude_cli=args.claude_cli,
                    model=model,
                    prompt_text=prompt_text,
                    timeout_s=args.timeout_s,
                    schema=ANNOTATION_SCHEMA,
                )
            row["duration_ms"] = annotation.get("_cli_metadata", {}).get("duration_ms")
            write_annotation(annotation_path, annotation)
            row["annotation_status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            row["annotation_status"] = "error"
            row["status"] = "error"
            row["error"] = str(exc)
        return index, row

    rows_by_index: dict[int, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="turn-localization-cli-") as tmp_dir:
        tmp_root = Path(tmp_dir)

        def codex_home_for(index: int) -> Path | None:
            if args.provider != "codex" or args.codex_home_mode != "isolated":
                return None
            codex_home = tmp_root / f"codex-home-{index:04d}"
            prepare_isolated_codex_home(codex_home, args.codex_home_source)
            return codex_home

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(annotate_one, index, prompt_path): (index, prompt_path)
                for index, prompt_path in enumerate(prompt_paths, start=1)
            }
            for future in as_completed(futures):
                index, prompt_path = futures[future]
                try:
                    completed_index, row = future.result()
                except Exception as exc:  # noqa: BLE001
                    annotation_path = output_dir / (prompt_path.stem + ".json")
                    validation_path = validation_dir / (prompt_path.stem + ".validation.json")
                    completed_index = index
                    row = make_pending_row(
                        batch_id=args.batch_id,
                        provider=args.provider,
                        model=model,
                        prompt_path=prompt_path,
                        annotation_path=annotation_path,
                        validation_path=validation_path,
                    )
                    row["annotation_status"] = "error"
                    row["status"] = "error"
                    row["error"] = str(exc)
                rows_by_index[completed_index] = row
                status = row["annotation_status"]
                if status == "ok":
                    print(f"[annotations] {completed_index}/{len(prompt_paths)} ok {prompt_path.name}", flush=True)
                elif status == "skipped_existing":
                    print(
                        f"[annotations] {completed_index}/{len(prompt_paths)} skipped {prompt_path.name}",
                        flush=True,
                    )
                else:
                    print(
                        f"[annotations] {completed_index}/{len(prompt_paths)} error "
                        f"{prompt_path.name}: {row.get('error')}",
                        flush=True,
                    )
    rows = [rows_by_index[index] for index in sorted(rows_by_index)]

    print("[validation] starting after annotation phase completed", flush=True)
    for index, row in enumerate(rows, start=1):
        prompt_path = Path(row["prompt_path"])
        annotation_path = Path(row["annotation_path"])
        validation_path = Path(row["validation_path"])
        if row["annotation_status"] not in {"ok", "skipped_existing"}:
            row["validation_status"] = "not_run"
            row["status"] = "error"
            continue
        if not annotation_path.exists():
            row["validation_status"] = "not_run"
            row["status"] = "error"
            row["error"] = "annotation file missing"
            continue

        status, report, error = run_validation(
            prompt_path=prompt_path,
            annotation_path=annotation_path,
            validation_path=validation_path,
            python_exe=args.python,
        )
        row["validation_status"] = status
        row["status"] = status
        if error:
            row["validation_error"] = error
        if report is not None:
            summary = report.get("summary", {})
            row["validation_has_errors"] = bool(summary.get("has_errors"))
            row["validation_has_warnings"] = bool(summary.get("has_warnings"))
            row["validation_warning_count"] = int(summary.get("top_level_warnings", 0)) + int(
                summary.get("turn_warnings", 0)
            )
            row["review_required_count"] = int(summary.get("review_required", 0))
            row["default_train_eligible"] = int(summary.get("default_train_eligible", 0))
        print(f"[validation] {index}/{len(rows)} {status} {prompt_path.name}", flush=True)

    counts = Counter(row["status"] for row in rows)
    annotation_counts = Counter(row["annotation_status"] for row in rows)
    validation_counts = Counter(row["validation_status"] for row in rows)
    warning_total = sum(int(row.get("validation_warning_count", 0)) for row in rows)
    review_required_total = sum(int(row.get("review_required_count", 0)) for row in rows)

    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")

    summary = {
        "batch_id": args.batch_id,
        "provider": args.provider,
        "model": model,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "prompt_dir": str(prompt_dir),
        "output_dir": str(output_dir),
        "validation_dir": str(validation_dir),
        "n_prompts": len(prompt_paths),
        "status_counts": dict(sorted(counts.items())),
        "annotation_status_counts": dict(sorted(annotation_counts.items())),
        "validation_status_counts": dict(sorted(validation_counts.items())),
        "ok": counts.get("ok", 0),
        "error": counts.get("error", 0) + counts.get("validation_error", 0),
        "review_required": counts.get("review_required", 0),
        "review_required_turns": review_required_total,
        "validation_warnings": warning_total,
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 1 if summary["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
