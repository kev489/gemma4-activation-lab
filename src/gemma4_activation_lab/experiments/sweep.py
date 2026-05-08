from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from gemma4_activation_lab.artifacts import (
    BASELINE_FIELDNAMES,
    STEERING_FIELDNAMES,
    append_jsonl,
)
from gemma4_activation_lab.config import DATA_DIR, MODEL_ID, OUTPUT_DIR, SEED
from gemma4_activation_lab.datasets import load_heldout_rows
from gemma4_activation_lab.generation import generate_for_user
from gemma4_activation_lab.intervene import AddVectorIntervention, PositionMode, register_additive_intervention
from gemma4_activation_lab.io_utils import make_run_dir, save_json
from gemma4_activation_lab.modeling import load_processor_and_model, set_reproducible_seed


@dataclass(frozen=True)
class SteeringSweepConfig:
    source_run: Path
    model_id: str = MODEL_ID
    seed: int = SEED
    heldout_dataset: Path = DATA_DIR / "activation_steering_warm_boundary_heldout_40.jsonl"
    alphas: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)
    position_modes: tuple[PositionMode, ...] = ("last", "all")
    selected_modules: tuple[str, ...] | None = None
    max_new_tokens: int = 64
    output_dir: Path = OUTPUT_DIR
    run_name: str = "saved_steering_sweep"
    progress: bool = True


@dataclass(frozen=True)
class SteeringSweepResult:
    run_dir: Path
    summary: dict[str, Any]


def load_directions(source_run: Path) -> dict[str, torch.Tensor]:
    summary_path = source_run / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    directions: dict[str, torch.Tensor] = {}
    for item in summary["direction_metadata"]:
        path = Path(item["path"])
        if not path.is_absolute():
            path = source_run / path
        payload = torch.load(path, map_location="cpu")
        directions[payload["module_name"]] = payload["vector"].float()
    return directions


def filter_directions(
    directions: dict[str, torch.Tensor],
    selected_modules: tuple[str, ...] | None,
) -> dict[str, torch.Tensor]:
    if selected_modules is None:
        return directions

    missing = [module_name for module_name in selected_modules if module_name not in directions]
    if missing:
        available = "\n".join(f"  - {module_name}" for module_name in directions)
        missing_text = ", ".join(missing)
        raise ValueError(f"Requested module(s) not found: {missing_text}\nAvailable modules:\n{available}")

    return {module_name: directions[module_name] for module_name in selected_modules}


def run_saved_steering_sweep(config: SteeringSweepConfig) -> SteeringSweepResult:
    set_reproducible_seed(config.seed)

    heldout_rows = load_heldout_rows(config.heldout_dataset)
    directions = filter_directions(load_directions(config.source_run), config.selected_modules)
    run_dir = make_run_dir(config.output_dir, config.run_name)

    planned_summary = {
        "model_id": config.model_id,
        "seed": config.seed,
        "source_run": str(config.source_run),
        "heldout_dataset": str(config.heldout_dataset),
        "modules": list(directions),
        "alphas": list(config.alphas),
        "position_modes": list(config.position_modes),
        "baseline_count": len(heldout_rows),
        "planned_steering_results_count": len(heldout_rows)
        * len(directions)
        * len(config.position_modes)
        * len(config.alphas),
        "streaming_files": {
            "baselines_jsonl": "baselines.jsonl",
            "baselines_csv": "baselines.csv",
            "steering_results_jsonl": "steering_results.jsonl",
            "steering_summary_csv": "steering_summary.csv",
        },
    }
    save_json(run_dir / "planned_summary.json", planned_summary)

    processor, model = load_processor_and_model(model_id=config.model_id)

    baselines: dict[str, str] = {}
    results: list[dict[str, Any]] = []

    with (
        (run_dir / "baselines.jsonl").open("w", encoding="utf-8") as baseline_jsonl,
        (run_dir / "baselines.csv").open("w", newline="", encoding="utf-8") as baseline_csv,
        (run_dir / "steering_results.jsonl").open("w", encoding="utf-8") as results_jsonl,
        (run_dir / "steering_summary.csv").open("w", newline="", encoding="utf-8") as results_csv,
    ):
        baseline_writer = csv.DictWriter(baseline_csv, fieldnames=BASELINE_FIELDNAMES)
        baseline_writer.writeheader()
        baseline_csv.flush()

        results_writer = csv.DictWriter(results_csv, fieldnames=STEERING_FIELDNAMES)
        results_writer.writeheader()
        results_csv.flush()

        for baseline_index, row in enumerate(heldout_rows, start=1):
            baseline_row = {
                "scenario_id": row["scenario_id"],
                "user_message": row["user_message"],
                "baseline_output": generate_for_user(
                    processor,
                    model,
                    row["user_message"],
                    max_new_tokens=config.max_new_tokens,
                ).decoded_output,
            }
            baselines[row["scenario_id"]] = baseline_row["baseline_output"]
            append_jsonl(baseline_jsonl, baseline_row)
            baseline_writer.writerow(baseline_row)
            baseline_csv.flush()
            if config.progress:
                print(f"[baseline {baseline_index}/{len(heldout_rows)}] {row['scenario_id']}", flush=True)

        result_index = 0
        planned_result_count = planned_summary["planned_steering_results_count"]
        for module_name, vector in directions.items():
            for position_mode in config.position_modes:
                for alpha in config.alphas:
                    controller = register_additive_intervention(
                        model,
                        AddVectorIntervention(
                            module_name=module_name,
                            vector=vector,
                            alpha=alpha,
                            position_mode=position_mode,
                            enabled=True,
                        ),
                        hook_mode="forward",
                    )
                    try:
                        for row in heldout_rows:
                            result_row = {
                                "scenario_id": row["scenario_id"],
                                "module_name": module_name,
                                "alpha": alpha,
                                "position_mode": position_mode,
                                "user_message": row["user_message"],
                                "baseline_output": baselines[row["scenario_id"]],
                                "intervened_output": generate_for_user(
                                    processor,
                                    model,
                                    row["user_message"],
                                    max_new_tokens=config.max_new_tokens,
                                ).decoded_output,
                            }
                            results.append(result_row)
                            append_jsonl(results_jsonl, result_row)
                            results_writer.writerow(result_row)
                            results_csv.flush()
                            result_index += 1
                            if config.progress:
                                print(
                                    f"[result {result_index}/{planned_result_count}] "
                                    f"{module_name} alpha={alpha} mode={position_mode} {row['scenario_id']}",
                                    flush=True,
                                )
                    finally:
                        controller.remove()

    summary = {
        "model_id": config.model_id,
        "seed": config.seed,
        "source_run": str(config.source_run),
        "heldout_dataset": str(config.heldout_dataset),
        "modules": list(directions),
        "alphas": list(config.alphas),
        "position_modes": list(config.position_modes),
        "baseline_count": len(baselines),
        "steering_results_count": len(results),
        "streaming_files": planned_summary["streaming_files"],
    }
    save_json(run_dir / "summary.json", summary)
    save_json(run_dir / "steering_results.json", {"rows": results})

    return SteeringSweepResult(run_dir=run_dir, summary=summary)
