#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gemma4_activation_lab.config import DATA_DIR, MODEL_ID, OUTPUT_DIR, SEED  # noqa: E402

PositionMode = Literal["last", "all"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep saved steering vectors over held-out prompts.")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--source-run", type=Path, required=True, help="Run directory containing summary.json and vectors/.")
    parser.add_argument(
        "--heldout-dataset",
        type=Path,
        default=DATA_DIR / "activation_steering_warm_boundary_heldout_40.jsonl",
    )
    parser.add_argument("--alphas", default="0.5,1,2,4,8", help="Comma-separated alpha values.")
    parser.add_argument("--position-modes", default="last,all", help="Comma-separated modes: last,all.")
    parser.add_argument(
        "--modules",
        default=None,
        help="Optional comma-separated exact module names to sweep, e.g. model.language_model.layers.21.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--run-name", default="saved_steering_sweep")
    return parser.parse_args()


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def parse_str_list(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_position_modes(value: str) -> tuple[PositionMode, ...]:
    modes: list[PositionMode] = []
    for item in value.split(","):
        mode = item.strip()
        if not mode:
            continue
        if mode not in {"last", "all"}:
            raise ValueError(f"Unsupported position mode {mode!r}; use 'last' or 'all'.")
        modes.append(mode)  # type: ignore[arg-type]
    return tuple(modes)


def main() -> None:
    args = parse_args()

    from gemma4_activation_lab.experiments.sweep import SteeringSweepConfig, run_saved_steering_sweep

    result = run_saved_steering_sweep(
        SteeringSweepConfig(
            model_id=args.model_id,
            seed=args.seed,
            source_run=args.source_run,
            heldout_dataset=args.heldout_dataset,
            alphas=parse_float_list(args.alphas),
            position_modes=parse_position_modes(args.position_modes),
            selected_modules=parse_str_list(args.modules),
            max_new_tokens=args.max_new_tokens,
            output_dir=args.output_dir,
            run_name=args.run_name,
        )
    )
    print(f"Saved steering sweep to: {result.run_dir}")


if __name__ == "__main__":
    main()
