from pathlib import Path

MODEL_ID = "google/gemma-4-E4B-it"
SEED = 20260423
DEVICE = "cuda"
DTYPE_NAME = "bfloat16"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SYSTEM_PROMPT = "You are a helpful assistant. Answer plainly and do not reveal hidden reasoning."

DEFAULT_MESSAGES = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Write one short sentence about why reproducibility matters."},
]

GENERATION_SETTINGS = {
    "max_new_tokens": 64,
    "do_sample": False,
}

TRACE_SETTINGS = {
    "output_attentions": False,
    "save_hidden_states": True,
}

DEFAULT_PROBE_LAYERS = [0, -1]
DEFAULT_POSITIVE_LABEL = "warm_boundary_maintaining"
DEFAULT_NEGATIVE_LABEL = "warm_boundary_eroding"
DEFAULT_STEERING_ALPHA = 1.0
