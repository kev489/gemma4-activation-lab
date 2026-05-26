# Gemma 4 Activation Lab

This repo uses `AGENTS.md` as the canonical project guide.

See:

- [AGENTS.md](./AGENTS.md) — environment setup, pinned dependencies, Hugging Face auth and cache details, verified model-loading notes, smoke-test commands, hook and intervention policy, judge harness, and Gemma 4 specific caveats.
- [results/run_history.md](./results/run_history.md) — per-experiment record: completed run directories, vector norms, qualitative reads, findings, recommended next steps, and long-form regenerate-commands.

## Intended workflow

1. Add or refine matched examples under `data/`.
2. Run `scripts/run_probe_experiment.py` to capture activations and compute direction vectors.
3. Run `scripts/sweep_saved_steering.py` to compare modules, alphas, and token-position modes.
4. Promote only useful lightweight artifacts from `outputs/` into `results/`.
5. Append a new entry to `results/run_history.md` summarizing the run.
6. Add evaluation code under `src/gemma4_activation_lab/experiments/` once the steering workflow is stable.
