"""Run the LLM judge against a blinded eval set.

By default this uses `codex exec` with GPT-5.4 mini, which relies on the local
Codex / ChatGPT login rather than an OpenAI API key. The older Claude CLI judge
path remains available with `--provider claude`.

Outputs:
  outputs/judge_eval/<eval-stem>_<timestamp>/results.jsonl
  outputs/judge_eval/<eval-stem>_<timestamp>/summary.json

This script does NOT run automatically; invoke it explicitly.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gemma4_activation_lab.judge import (  # noqa: E402
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_CODEX_MODEL,
    DEFAULT_PROVIDER,
    Judge,
    JudgeError,
    Rubric,
)


CODEX_MODEL = DEFAULT_CODEX_MODEL
SONNET_MODEL = DEFAULT_CLAUDE_MODEL
OPUS_MODEL = "claude-opus-4-7"


def load_eval_set(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"bad JSONL at {path}:{line_no}") from exc
    return rows


def grade_one(judge: Judge, row: dict[str, Any]) -> dict[str, Any]:
    rubric = Rubric(**row["rubric"])
    conversation = row["conversation"]
    try:
        judgment = judge.grade(rubric, conversation)
        return {"ok": True, **judgment.to_dict()}
    except JudgeError as exc:
        return {
            "ok": False,
            "error": str(exc),
            "model": judge.model,
            "provider": judge.provider,
        }


def run_pass(
    rows: list[dict[str, Any]],
    judge: Judge,
    workers: int,
    label: str,
) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    started = time.time()
    print(f"[{label}] grading {len(rows)} rows with {judge.model} ({workers} workers)", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(grade_one, judge, r): r["row_id"] for r in rows}
        done = 0
        for fut in as_completed(futures):
            row_id = futures[fut]
            results[row_id] = fut.result()
            done += 1
            if done % 5 == 0 or done == len(rows):
                elapsed = time.time() - started
                print(f"[{label}] {done}/{len(rows)} ({elapsed:.1f}s)", flush=True)
    return results


def summarize(rows: list[dict[str, Any]], merged: dict[int, dict[str, Any]]) -> dict[str, Any]:
    overall = Counter()
    by_behavior: dict[str, Counter] = defaultdict(Counter)
    by_measurement: dict[str, Counter] = defaultdict(Counter)
    by_model: dict[str, Counter] = defaultdict(Counter)
    confusion = Counter()
    errors = 0
    total_cost = 0.0
    fallback_used = 0
    primary_provider = None
    primary_model = None
    fallback_provider = None
    fallback_model = None

    for row in rows:
        rid = row["row_id"]
        entry = merged.get(rid, {})
        gt = row["ground_truth"]["result"]
        primary = entry.get("primary", {})
        final = entry.get("final", primary)
        primary_provider = primary_provider or primary.get("provider")
        primary_model = primary_model or primary.get("model")
        fallback = entry.get("fallback")
        if fallback is not None:
            fallback_provider = fallback_provider or fallback.get("provider")
            fallback_model = fallback_model or fallback.get("model")
        if not final.get("ok"):
            errors += 1
            continue
        pred = final["result"]
        match = pred == gt
        overall["total"] += 1
        overall["match"] += int(match)
        by_behavior[row["rubric"]["behavior_type"]]["total"] += 1
        by_behavior[row["rubric"]["behavior_type"]]["match"] += int(match)
        by_measurement[row["rubric"]["measurement"]]["total"] += 1
        by_measurement[row["rubric"]["measurement"]]["match"] += int(match)
        tm = row["metadata"].get("transcript_model") or "unknown"
        by_model[tm]["total"] += 1
        by_model[tm]["match"] += int(match)
        confusion[(gt, pred)] += 1
        total_cost += float(primary.get("cost_usd") or 0.0)
        if fallback is not None:
            fallback_used += 1
            total_cost += float(fallback.get("cost_usd") or 0.0)

    def rate(c: Counter) -> dict[str, Any]:
        return {
            "total": c["total"],
            "match": c["match"],
            "rate": (c["match"] / c["total"]) if c["total"] else None,
        }

    return {
        "n_rows": len(rows),
        "errors": errors,
        "primary_provider": primary_provider,
        "primary_model": primary_model,
        "fallback_provider": fallback_provider,
        "fallback_model": fallback_model,
        "primary_only": len(rows) - fallback_used - errors,
        "fallback_on_disagreement": fallback_used,
        "overall": rate(overall),
        "by_behavior_type": {k: rate(v) for k, v in by_behavior.items()},
        "by_measurement": {k: rate(v) for k, v in by_measurement.items()},
        "by_transcript_model": {k: rate(v) for k, v in by_model.items()},
        "confusion_matrix": {f"gt={gt}|pred={pred}": n for (gt, pred), n in confusion.items()},
        "reported_cost_usd_informational": round(total_cost, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "judge_eval")
    parser.add_argument("--provider", choices=["codex", "claude"], default=DEFAULT_PROVIDER)
    parser.add_argument("--model", default=None,
                        help="Primary judge model. Defaults to gpt-5.4-mini for codex, "
                             "claude-sonnet-4-6 for claude.")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Optional smoke-test limit applied after loading the eval set.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Default: 1 for codex, 4 for claude.")
    parser.add_argument("--codex-cli", default="codex")
    parser.add_argument("--claude-cli", default="claude")
    parser.add_argument("--codex-cwd", type=Path, default=Path("/tmp"))
    parser.add_argument("--codex-sandbox", default="read-only")
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--claude-opus-on-disagree", action="store_true",
                        help="Re-grade primary disagreements with Claude Opus.")
    parser.add_argument("--sonnet-model", default=SONNET_MODEL,
                        help=argparse.SUPPRESS)
    parser.add_argument("--opus-model", default=OPUS_MODEL)
    parser.add_argument("--skip-opus", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    rows = load_eval_set(args.eval_set)
    if args.max_rows is not None:
        rows = rows[:args.max_rows]
    if not rows:
        raise SystemExit("empty eval set")
    workers = args.workers if args.workers is not None else (1 if args.provider == "codex" else 4)
    primary_model = args.model
    if primary_model is None:
        primary_model = args.sonnet_model if args.provider == "claude" else CODEX_MODEL

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / f"{args.eval_set.stem}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    primary_judge = Judge(
        provider=args.provider,
        model=primary_model,
        claude_cli=args.claude_cli,
        codex_cli=args.codex_cli,
        timeout_s=args.timeout_s,
        codex_cwd=args.codex_cwd,
        codex_sandbox=args.codex_sandbox,
    )
    primary_results = run_pass(rows, primary_judge, workers, label=f"{args.provider}-primary")

    disagreements: list[dict[str, Any]] = []
    for row in rows:
        primary = primary_results.get(row["row_id"], {})
        if primary.get("ok") and primary["result"] != row["ground_truth"]["result"]:
            disagreements.append(row)

    fallback_results: dict[int, dict[str, Any]] = {}
    use_opus_fallback = args.claude_opus_on_disagree and not args.skip_opus
    if disagreements and use_opus_fallback:
        fallback_judge = Judge(
            provider="claude",
            model=args.opus_model,
            claude_cli=args.claude_cli,
            timeout_s=args.timeout_s,
        )
        fallback_results = run_pass(
            disagreements,
            fallback_judge,
            workers,
            label="claude-opus-on-disagree",
        )

    merged: dict[int, dict[str, Any]] = {}
    for row in rows:
        rid = row["row_id"]
        primary = primary_results.get(rid, {"ok": False, "error": "missing"})
        fallback = fallback_results.get(rid)
        final = fallback if fallback and fallback.get("ok") else primary
        merged[rid] = {"primary": primary, "fallback": fallback, "final": final}

    results_path = run_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        for row in rows:
            rid = row["row_id"]
            entry = merged[rid]
            primary = entry["primary"]
            fallback = entry["fallback"]
            final = entry["final"]
            f.write(json.dumps({
                "row_id": rid,
                "metric_id": row["rubric"]["metric_id"],
                "transcript_model": row["metadata"].get("transcript_model"),
                "gt_result": row["ground_truth"]["result"],
                "gt_justification": row["ground_truth"]["justification"],
                "primary": primary,
                "fallback": fallback,
                "final_result": final.get("result") if final.get("ok") else None,
                "final_justification": final.get("justification") if final.get("ok") else None,
                "match_primary": (primary.get("ok") and primary["result"] == row["ground_truth"]["result"]),
                "match_final": (final.get("ok") and final["result"] == row["ground_truth"]["result"]),
            }, ensure_ascii=True, sort_keys=True) + "\n")

    summary = summarize(rows, merged)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nresults: {results_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
