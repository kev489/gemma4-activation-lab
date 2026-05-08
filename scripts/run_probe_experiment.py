#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gemma4_activation_lab.config import (  # noqa: E402
    DATA_DIR,
    DEFAULT_NEGATIVE_LABEL,
    DEFAULT_POSITIVE_LABEL,
    DEFAULT_STEERING_ALPHA,
    MODEL_ID,
    OUTPUT_DIR,
    SEED,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a matched-prompt probe and steering experiment.")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dataset", type=Path, default=DATA_DIR / "matched_conversations.jsonl")
    parser.add_argument("--heldout-dataset", type=Path, default=DATA_DIR / "heldout_prompts.jsonl")
    parser.add_argument("--layers", default="0,-1", help="Comma-separated layer indices, e.g. 0,10,-1")
    parser.add_argument("--positive-label", default=DEFAULT_POSITIVE_LABEL)
    parser.add_argument("--negative-label", default=DEFAULT_NEGATIVE_LABEL)
    parser.add_argument("--alpha", type=float, default=DEFAULT_STEERING_ALPHA)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--run-name", default="probe_experiment")
    return parser.parse_args()


def parse_layers(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    args = parse_args()

    from gemma4_activation_lab.experiments.probe import ProbeExperimentConfig, run_probe_experiment

    result = run_probe_experiment(
        ProbeExperimentConfig(
            model_id=args.model_id,
            seed=args.seed,
            dataset=args.dataset,
            heldout_dataset=args.heldout_dataset,
            layer_indices=parse_layers(args.layers),
            positive_label=args.positive_label,
            negative_label=args.negative_label,
            alpha=args.alpha,
            max_new_tokens=args.max_new_tokens,
            output_dir=args.output_dir,
            run_name=args.run_name,
        )
    )
    print(f"Probe experiment saved to: {result.run_dir}")
    print(f"Steering module: {result.steering_module}")


if __name__ == "__main__":
    main()
