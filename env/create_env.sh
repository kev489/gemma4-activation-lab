#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip==26.0 setuptools==80.9.0 wheel==0.45.1
python -m pip install -r requirements.txt

python - <<'PY'
import importlib.metadata as md

for package in [
    "torch",
    "torchvision",
    "transformers",
    "accelerate",
    "huggingface-hub",
    "safetensors",
    "sentencepiece",
    "numpy",
    "pandas",
    "matplotlib",
    "tqdm",
    "PyYAML",
]:
    print(f"{package}=={md.version(package)}")
PY
