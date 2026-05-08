#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List Gemma module names for hook placement.")
    parser.add_argument("--contains", help="Optional substring filter.")
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--show-default-hook-points", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from gemma4_activation_lab.hooks import collect_default_hook_points, module_names
    from gemma4_activation_lab.modeling import load_processor_and_model, print_package_versions, set_reproducible_seed

    set_reproducible_seed()
    print_package_versions()
    _, model = load_processor_and_model()

    names = module_names(model, contains=args.contains)
    for name in names[: args.limit]:
        print(name)

    if args.show_default_hook_points:
        print("\nDefault hook points")
        hook_points = collect_default_hook_points(model, layer_indices=[0, -1])
        for group, group_names in hook_points.items():
            print(f"[{group}]")
            for name in group_names:
                print(f"  {name}")


if __name__ == "__main__":
    main()
