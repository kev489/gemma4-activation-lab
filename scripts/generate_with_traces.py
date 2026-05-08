#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gemma4_activation_lab.config import (  # noqa: E402
    DEFAULT_MESSAGES,
    GENERATION_SETTINGS,
    MODEL_ID,
    OUTPUT_DIR,
    SEED,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate with Gemma 4 and save prompt, ids, text, and traces.")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--messages-file", type=Path, help="JSON file containing a list of chat messages.")
    parser.add_argument("--messages-json", help="Inline JSON list of chat messages.")
    parser.add_argument("--max-new-tokens", type=int, default=GENERATION_SETTINGS["max_new_tokens"])
    parser.add_argument("--output-attentions", action="store_true", help="Request generation attentions.")
    parser.add_argument("--run-name", default="traced_generation")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--skip-hidden-state-save", action="store_true")
    return parser.parse_args()


def load_messages(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.messages_file:
        return json.loads(args.messages_file.read_text(encoding="utf-8"))
    if args.messages_json:
        return json.loads(args.messages_json)
    return DEFAULT_MESSAGES


def main() -> None:
    args = parse_args()

    from gemma4_activation_lab.generation import generate_from_messages, save_generation_hidden_states
    from gemma4_activation_lab.io_utils import make_run_dir, save_json, save_tensor, utc_timestamp
    from gemma4_activation_lab.modeling import load_processor_and_model, print_package_versions, set_reproducible_seed

    set_reproducible_seed(args.seed)
    print_package_versions()

    messages = load_messages(args)
    processor, model = load_processor_and_model(model_id=args.model_id)
    run_dir = make_run_dir(args.output_dir, args.run_name)

    result = generate_from_messages(
        processor,
        model,
        messages,
        max_new_tokens=args.max_new_tokens,
        output_hidden_states=True,
        output_attentions=args.output_attentions,
    )

    hidden_state_files = []
    if not args.skip_hidden_state_save:
        hidden_state_files = save_generation_hidden_states(run_dir, getattr(result.raw_outputs, "hidden_states", None))

    (run_dir / "prompt.txt").write_text(result.prompt_text, encoding="utf-8")
    (run_dir / "decoded_output.txt").write_text(result.decoded_output, encoding="utf-8")
    save_tensor(run_dir / "input_ids.pt", result.input_ids)
    save_tensor(run_dir / "generated_sequences.pt", result.sequences)
    save_tensor(run_dir / "new_token_ids.pt", result.new_token_ids)

    metadata = {
        "timestamp_utc": utc_timestamp(),
        "model_id": args.model_id,
        "seed": args.seed,
        "messages": messages,
        "prompt_text": result.prompt_text,
        "generation_settings": {
            "max_new_tokens": args.max_new_tokens,
            "return_dict_in_generate": True,
            "output_hidden_states": True,
            "output_attentions": args.output_attentions,
            "do_sample": False,
        },
        "input_length": int(result.input_ids.shape[-1]),
        "generated_sequence_length": int(result.sequences.shape[-1]),
        "new_tokens_length": int(result.new_token_ids.shape[-1]),
        "hidden_state_files": hidden_state_files,
        "output_attentions": args.output_attentions,
    }
    save_json(run_dir / "metadata.json", metadata)

    print(f"Saved traced generation to: {run_dir}")
    print(result.decoded_output)


if __name__ == "__main__":
    main()
