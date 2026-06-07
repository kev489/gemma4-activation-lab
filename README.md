# Gemma 4 Activation Lab

This repo uses `AGENTS.md` as the canonical project guide.

See:

- [AGENTS.md](./AGENTS.md) — environment setup, pinned dependencies, Hugging Face auth and cache details, verified model-loading notes, smoke-test commands, hook and intervention policy, judge harness, and Gemma 4 specific caveats.
- [results/run_history.md](./results/run_history.md) — per-experiment record: completed run directories, vector norms, qualitative reads, findings, recommended next steps, and long-form regenerate-commands.

## Intended workflow

For the canonical ImpactBench autonomy work:

1. Load `data/impactbench_autonomy/activation_examples/v1/examples.jsonl`
   through `gemma4_activation_lab.activation_datasets`.
2. For the first two vectors, use Gemma-source, `harmful=false` rows and
   exclude examples tagged with both Autonomy Preservation and
   Self-Determination.
3. Split by source record or scenario, balance metrics within each label, and
   capture only the localized pooling-token span.
4. Compare layers and vector-construction choices on held-out records before
   steering evaluation.
5. Promote only useful lightweight artifacts from `outputs/` into `results/`
   and append the run to `results/run_history.md`.

The older `scripts/run_probe_experiment.py` path remains the scaffold for the
single-turn matched datasets. It is not yet wired to the canonical ImpactBench
v1 export.
