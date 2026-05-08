# Gemma 4 Activation Lab

This repo uses `AGENTS.md` as the canonical project guide.

See:

- [AGENTS.md](./AGENTS.md)
- [HANDOFF.md](./HANDOFF.md)

`AGENTS.md` contains:

- environment setup
- pinned dependencies
- Hugging Face auth and cache details
- verified model-loading notes
- smoke-test commands
- hook and intervention guidance
- Gemma 4 specific caveats

`HANDOFF.md` summarizes the current repo state, completed experiment outputs, qualitative findings, and recommended next steps.

## Working Layout

- `src/gemma4_activation_lab/modeling.py`: direct Hugging Face processor/model loading, dtype/device checks, tokenization helpers.
- `src/gemma4_activation_lab/prompting.py`: text-only chat prompt rendering with `enable_thinking=False` and assistant-span calculation.
- `src/gemma4_activation_lab/generation.py`: deterministic generation helpers and trace hidden-state persistence.
- `src/gemma4_activation_lab/hooks.py`: module discovery and activation capture hooks.
- `src/gemma4_activation_lab/intervene.py`: additive vector intervention implementation.
- `src/gemma4_activation_lab/datasets.py`: JSONL loaders and schema checks for matched and held-out examples.
- `src/gemma4_activation_lab/artifacts.py`: run artifact writers for vectors, summaries, JSONL, and CSV.
- `src/gemma4_activation_lab/experiments/`: reusable probe and saved-steering sweep workflows.
- `scripts/`: thin CLI wrappers around the reusable library workflows.
- `outputs/`: ignored scratch runs from probes, sweeps, and traces.
- `results/`: curated summaries/tables/notes worth committing after reviewing an output run.

The intended workflow is:

1. Add or refine matched examples under `data/`.
2. Run `scripts/run_probe_experiment.py` to capture activations and compute direction vectors.
3. Run `scripts/sweep_saved_steering.py` to compare modules, alphas, and token-position modes.
4. Promote only useful lightweight artifacts from `outputs/` into `results/`.
5. Add evaluation code under `src/gemma4_activation_lab/experiments/` once the steering workflow is stable.
