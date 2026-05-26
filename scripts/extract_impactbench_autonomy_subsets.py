from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPT_DIR = ROOT / "impactBench" / "impactbench_transcripts" / "all_models"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "impactbench_autonomy"

SUBAREAS = {
    "Autonomy Preservation": "autonomy_preservation_records.jsonl",
    "Self-Determination": "self_determination_records.jsonl",
}


def transcript_paths(transcript_dir: Path) -> list[Path]:
    paths = sorted(transcript_dir.glob("*_transcripts.jsonl.gz"))
    if not paths:
        raise FileNotFoundError(f"No transcript archives found under {transcript_dir}")
    return paths


def matching_subareas(record: dict[str, Any]) -> list[str]:
    names = []
    for location in record.get("locations") or []:
        subarea_name = location.get("subarea_name")
        if subarea_name in SUBAREAS and subarea_name not in names:
            names.append(subarea_name)
    return names


def extract_subsets(transcript_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    output_handles = {
        subarea: (output_dir / filename).open("w", encoding="utf-8")
        for subarea, filename in SUBAREAS.items()
    }
    counts: dict[str, Counter[str]] = {subarea: Counter() for subarea in SUBAREAS}
    benchmarks: dict[str, Counter[str]] = {subarea: Counter() for subarea in SUBAREAS}
    metrics: dict[str, Counter[str]] = {subarea: Counter() for subarea in SUBAREAS}
    models: dict[str, Counter[str]] = {subarea: Counter() for subarea in SUBAREAS}
    scenario_ids: dict[str, set[str]] = defaultdict(set)

    try:
        for path in transcript_paths(transcript_dir):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    subareas = matching_subareas(record)
                    if not subareas:
                        continue
                    verdict = (record.get("verdict") or {}).get("result", "missing")
                    transcript_model = record.get("transcript_model") or path.name.removesuffix("_transcripts.jsonl.gz")
                    for subarea in subareas:
                        output_handles[subarea].write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
                        counts[subarea][verdict] += 1
                        benchmarks[subarea][record.get("benchmark", "missing")] += 1
                        metrics[subarea][record.get("metric_name", "missing")] += 1
                        models[subarea][transcript_model] += 1
                        scenario_ids[subarea].add(record.get("scenario_id", "missing"))
    finally:
        for handle in output_handles.values():
            handle.close()

    manifest = {
        "source_transcript_dir": str(transcript_dir.relative_to(ROOT)),
        "output_dir": str(output_dir.relative_to(ROOT)),
        "record_shape": "Original full ImpactBench transcript records. No first-turn extraction or verdict relabeling.",
        "subsets": {},
    }
    for subarea, filename in SUBAREAS.items():
        output_path = output_dir / filename
        manifest["subsets"][subarea] = {
            "path": str(output_path.relative_to(ROOT)),
            "rows": sum(counts[subarea].values()),
            "verdict_counts": dict(sorted(counts[subarea].items())),
            "unique_scenario_ids": len(scenario_ids[subarea]),
            "benchmark_counts": dict(sorted(benchmarks[subarea].items())),
            "metric_count": len(metrics[subarea]),
            "model_counts": dict(sorted(models[subarea].items())),
        }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract literal ImpactBench autonomy-related full transcript subsets.")
    parser.add_argument("--transcript-dir", type=Path, default=TRANSCRIPT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    manifest = extract_subsets(args.transcript_dir, args.output_dir)
    print(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
