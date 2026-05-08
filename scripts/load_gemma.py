#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gemma4_activation_lab.config import DEFAULT_MESSAGES, GENERATION_SETTINGS, MODEL_ID, SEED  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load google/gemma-4-E4B-it and run a smoke test.")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-new-tokens", type=int, default=GENERATION_SETTINGS["max_new_tokens"])
    parser.add_argument("--messages-json", help="Optional JSON list of chat messages.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from gemma4_activation_lab.generation import generate_from_messages
    from gemma4_activation_lab.hooks import enumerate_modules
    from gemma4_activation_lab.modeling import (
        first_parameter_dtype_device,
        load_processor_and_model,
        parameter_count,
        print_package_versions,
        set_reproducible_seed,
    )

    set_reproducible_seed(args.seed)
    print_package_versions()

    messages = json.loads(args.messages_json) if args.messages_json else DEFAULT_MESSAGES
    processor, model = load_processor_and_model(model_id=args.model_id)
    result = generate_from_messages(
        processor,
        model,
        messages,
        max_new_tokens=args.max_new_tokens,
    )
    dtype, device = first_parameter_dtype_device(model)
    top_level_module_names = [name for name, _ in model.named_children()]

    print("\nModel summary")
    print(f"model_id: {args.model_id}")
    print(f"model_class: {model.__class__.__name__}")
    print(f"dtype: {dtype}")
    print(f"device: {device}")
    print(f"parameter_count_estimate: {parameter_count(model):,}")
    print(f"top_level_module_names: {top_level_module_names[:12]}")
    print(f"named_module_count: {len(enumerate_modules(model))}")
    print("\nPrompt")
    print(result.prompt_text)
    print("\nSmoke test generation")
    print(result.decoded_output)


if __name__ == "__main__":
    main()
