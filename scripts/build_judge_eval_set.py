"""Build a stratified blinded judge-evaluation set from an ImpactBench JSONL.

Output rows have shape:
  {
    "row_id": int,
    "rubric": {metric_id, metric_name, behavior_type, measurement, metric_criterion},
    "conversation": [{role, content}, ...],
    "ground_truth": {"result": "yes"|"no", "justification": "..."},
    "metadata": {"scenario_id": ..., "transcript_model": ..., "benchmark": ...}
  }

The judge-facing code only ever reads `rubric` and `conversation`, so ground
truth lives in a separate top-level key and is not seen by the model.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def stratified_sample(
    records: list[dict[str, Any]],
    n_per_class: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        verdict = ((r.get("verdict") or {}).get("result") or "").strip()
        if verdict not in {"yes", "no"}:
            continue
        if not (r.get("samples") and isinstance(r["samples"][0], list) and r["samples"][0]):
            continue
        buckets[(verdict, r["metric_id"])].append(r)

    for bucket in buckets.values():
        rng.shuffle(bucket)

    selected: list[dict[str, Any]] = []
    for cls in ("yes", "no"):
        metric_ids = sorted({mid for (v, mid) in buckets if v == cls})
        rng.shuffle(metric_ids)
        positions = {mid: 0 for mid in metric_ids}

        class_picks: list[dict[str, Any]] = []
        while len(class_picks) < n_per_class:
            made_progress = False
            for mid in metric_ids:
                if len(class_picks) >= n_per_class:
                    break
                bucket = buckets[(cls, mid)]
                p = positions[mid]
                if p < len(bucket):
                    class_picks.append(bucket[p])
                    positions[mid] = p + 1
                    made_progress = True
            if not made_progress:
                raise RuntimeError(
                    f"only {len(class_picks)} records available for class {cls!r}, "
                    f"needed {n_per_class}"
                )
        selected.extend(class_picks)
    return selected


def blinded_row(record: dict[str, Any], row_id: int) -> dict[str, Any]:
    verdict = record.get("verdict") or {}
    return {
        "row_id": row_id,
        "rubric": {
            "metric_id": record["metric_id"],
            "metric_name": record["metric_name"],
            "behavior_type": record.get("behavior_type", ""),
            "measurement": record.get("measurement", ""),
            "metric_criterion": record["metric_criterion"],
        },
        "conversation": record["samples"][0],
        "ground_truth": {
            "result": verdict.get("result"),
            "justification": verdict.get("justification", ""),
        },
        "metadata": {
            "scenario_id": record.get("scenario_id"),
            "transcript_model": record.get("transcript_model"),
            "benchmark": record.get("benchmark"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-per-class", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    picks = stratified_sample(records, args.n_per_class, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(picks):
            f.write(json.dumps(blinded_row(rec, i), ensure_ascii=True, sort_keys=True) + "\n")

    summary = {
        "input": str(args.input.relative_to(ROOT)) if args.input.is_absolute() else str(args.input),
        "output": str(args.output.relative_to(ROOT)) if args.output.is_absolute() else str(args.output),
        "n_per_class": args.n_per_class,
        "n_total": len(picks),
        "seed": args.seed,
        "unique_metric_ids": len({p["metric_id"] for p in picks}),
        "unique_transcript_models": len({p.get("transcript_model") for p in picks}),
        "class_counts": {
            "yes": sum(1 for p in picks if p["verdict"]["result"] == "yes"),
            "no": sum(1 for p in picks if p["verdict"]["result"] == "no"),
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
