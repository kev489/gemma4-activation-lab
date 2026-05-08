# Gemma 4 Activation Lab Handoff

Updated: 2026-05-05

This handoff summarizes the current state of `/home/kchafloq/gemma4-activation-lab`, what has been built, what has been run, and what should be done next. `AGENTS.md` remains the canonical operating guide for environment setup, model-loading rules, and caveats.

## Current Status

- The repo is a text-only activation-access research scaffold for `google/gemma-4-E4B-it`.
- The main path uses direct `AutoProcessor.from_pretrained(...)` and `AutoModelForCausalLM.from_pretrained(...)`.
- The baseline path keeps Gemma thinking disabled with `enable_thinking=False`.
- The intended runtime is GPU `bfloat16`; the known-good environment used an NVIDIA L40S.
- The local Hugging Face cache for the model is at `/home/kchafloq/.cache/huggingface/hub/models--google--gemma-4-E4B-it`.
- At handoff time, this directory was not a Git repository, so there is no local commit history to audit with `git status`.

## Environment

Project root:

```bash
cd /home/kchafloq/gemma4-activation-lab
```

Virtualenv:

```bash
/home/kchafloq/gemma4-activation-lab/.venv
```

Create or refresh the environment:

```bash
bash env/create_env.sh
```

Manual equivalent:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip==26.0 setuptools==80.9.0 wheel==0.45.1
python -m pip install -r requirements.txt
```

Important package constraints:

- `torch==2.11.0`
- `torchvision==0.26.0`
- `transformers==5.6.1`
- `accelerate==1.13.0`
- `huggingface-hub==1.11.0`
- `safetensors==0.7.0`
- `sentencepiece==0.2.1`
- `numpy==2.4.4`
- `pandas==3.0.2`
- `matplotlib==3.10.8`
- `tqdm==4.67.3`
- `PyYAML==6.0.3`

Do not casually change the Torch/Torchvision pairing. `torchvision` is required even for text-only work because the Gemma 4 processor stack imports multimodal processor classes.

## Verified Model State

The model has been loaded successfully end to end in this environment.

Known-good observations:

- `torch.cuda.is_available() == True`
- `torch.cuda.is_bf16_supported() == True`
- visible device: `NVIDIA L40S`
- model class: `Gemma4ForConditionalGeneration`
- model dtype: `torch.bfloat16`
- model device: `cuda:0`
- estimated parameter count: about `7.94B`

Fast sanity check:

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/load_gemma.py --max-new-tokens 16
```

## Source Layout

Core library code:

- `src/gemma4_activation_lab/config.py`
- `src/gemma4_activation_lab/modeling.py`
- `src/gemma4_activation_lab/prompting.py`
- `src/gemma4_activation_lab/generation.py`
- `src/gemma4_activation_lab/hooks.py`
- `src/gemma4_activation_lab/intervene.py`
- `src/gemma4_activation_lab/interventions.py`
- `src/gemma4_activation_lab/datasets.py`
- `src/gemma4_activation_lab/artifacts.py`
- `src/gemma4_activation_lab/io_utils.py`
- `src/gemma4_activation_lab/experiments/probe.py`
- `src/gemma4_activation_lab/experiments/sweep.py`

Runnable scripts:

- `scripts/load_gemma.py`
- `scripts/list_modules.py`
- `scripts/generate_with_traces.py`
- `scripts/run_probe_experiment.py`
- `scripts/sweep_saved_steering.py`

Data:

- `data/matched_conversations.jsonl`: small toy matched set, 6 rows
- `data/heldout_prompts.jsonl`: small toy held-out set, 2 rows
- `data/activation_steering_warm_boundary_training_100_matched.jsonl`: 200 matched rows
- `data/activation_steering_warm_boundary_heldout_40.jsonl`: 40 held-out prompts
- `data/warm_boundary_self_agency_contrast_raw.jsonl`: 80 raw rows kept for provenance
- `data/warm_boundary_self_agency_contrast_pairs.jsonl`: 80 repaired pair rows
- `data/warm_boundary_self_agency_contrast_matched.jsonl`: 160 probe-compatible matched rows
- `activation_lab_dataset/`: local duplicate/import dump of the warm-boundary data, ignored by Git; canonical tracked copies live under `data/`

Artifacts:

- `outputs/`: ignored scratch run directories for traces, probes, sweeps, vectors, and generated outputs
- `results/`: curated summaries, tables, and notes promoted from `outputs/` when they are worth tracking

## Implemented Behavior

### Loading

`src/gemma4_activation_lab/modeling.py` handles:

- reproducible seeds across Python, NumPy, Torch, and CUDA
- CUDA and `bfloat16` checks
- package version printing
- direct processor/model loading
- eager attention for more inspectable hooks
- deterministic chat templating with `enable_thinking=False`
- generation decoding for only newly generated tokens

### Hooking

`src/gemma4_activation_lab/hooks.py` handles:

- module enumeration and substring filtering
- loud failure when a requested module is missing
- first-tensor extraction from hook outputs
- forward hook registration
- CPU-detached activation capture
- default hook discovery for Gemma 4 language-model layers

Preferred hook points:

- transformer blocks: `model.language_model.layers.N`
- attention output projection: `model.language_model.layers.N.self_attn.o_proj`
- MLP output: `model.language_model.layers.N.mlp`
- final norm: `model.language_model.norm`

Recommended first-pass layers:

- `0`
- `21`
- `41`

### Interventions

`src/gemma4_activation_lab/intervene.py` handles:

- additive vector interventions
- `forward_pre` and `forward` hook modes
- scalar `alpha`
- token targeting modes: `all`, `last`, and `span`
- explicit enable, disable, and remove controls
- width and shape checks before modifying activations

### Probe Experiments

`src/gemma4_activation_lab/experiments/probe.py` implements the reusable workflow. `scripts/run_probe_experiment.py` is a thin CLI wrapper.

The probe workflow:

- loads matched JSONL rows
- teacher-forces target assistant text
- captures activations over the assistant span
- mean-pools selected activations
- averages activations by response-style label
- computes `positive_label - negative_label` directions
- saves vector `.pt` files under `outputs/<run>/vectors/`
- runs baseline and intervened generation on held-out prompts
- writes `summary.json`, `capture_records.json`, `steering_results.json`, and `steering_summary.csv`

### Saved Steering Sweeps

`src/gemma4_activation_lab/experiments/sweep.py` implements the reusable workflow. `scripts/sweep_saved_steering.py` is a thin CLI wrapper.

The saved steering sweep workflow:

- loads vectors from an existing probe run's `summary.json`
- filters to selected modules if requested
- caches baseline outputs once
- sweeps alphas and position modes
- streams `baselines.jsonl`, `baselines.csv`, `steering_results.jsonl`, and `steering_summary.csv`
- writes final `summary.json` and `steering_results.json`

This script is safer for long GPU runs because it writes incremental outputs before the full sweep finishes.

## Completed Runs

### 1. Warm Boundary Full Probe

Run directory:

```text
outputs/probe_warm_boundary_full_20260424T024108Z
```

Purpose:

- Compute warm-boundary directions from the 200-row matched training set.

Labels:

- positive: `warm_boundary_maintaining`
- negative: `warm_boundary_eroding`

Selected modules:

- `model.language_model.layers.0`
- `model.language_model.layers.21`
- `model.language_model.layers.41`

Results:

- capture records: `600`
- held-out steering results: `40`
- layer 0 vector norm: `3.3818`
- layer 21 vector norm: `12.6603`
- layer 41 vector norm: `4.5969`

Key artifacts:

- `outputs/probe_warm_boundary_full_20260424T024108Z/summary.json`
- `outputs/probe_warm_boundary_full_20260424T024108Z/capture_records.json`
- `outputs/probe_warm_boundary_full_20260424T024108Z/steering_summary.csv`
- `outputs/probe_warm_boundary_full_20260424T024108Z/vectors/`

### 2. Broad Saved Steering Sweep

Run directory:

```text
outputs/saved_steering_sweep_20260424T034836Z
```

Source run:

```text
outputs/probe_warm_boundary_full_20260424T024108Z
```

Sweep:

- modules: layers `0`, `21`, `41`
- alphas: `0.5`, `1.0`, `2.0`, `4.0`, `8.0`
- position modes: `last`, `all`
- baselines: `40`
- steering results: `1200`

Qualitative read:

- Layer 21 is the most useful intervention point from this pass.
- `layer 21`, `alpha=1.0`, `position_mode=last` was the best first candidate.
- `alpha=2.0` and `position_mode=all` caused broader rewrites and lower baseline similarity.
- The main failure mode was generic supportive or therapy-like rewriting rather than clean warm-boundary behavior.

Key artifacts:

- `outputs/saved_steering_sweep_20260424T034836Z/summary.json`
- `outputs/saved_steering_sweep_20260424T034836Z/steering_summary.csv`
- `outputs/saved_steering_sweep_20260424T034836Z/qualitative_layer21_inspection.md`

### 3. Layer 21 Last-Token Alpha Calibration

Run directory:

```text
outputs/layer21_last_alpha_calibration_20260424T054527Z
```

Source run:

```text
outputs/probe_warm_boundary_full_20260424T024108Z
```

Sweep:

- module: `model.language_model.layers.21`
- position mode: `last`
- alphas: `0.75`, `1.0`, `1.25`, `1.5`, `1.75`, `2.0`
- baselines: `40`
- steering results: `240`

Qualitative read:

- Best defended setting: `layer 21`, `alpha=1.0`, `position_mode=last`.
- Conservative setting: `layer 21`, `alpha=0.75`, `position_mode=last`.
- `alpha=1.25` is possible but not clearly better.
- `alpha=1.5` through `2.0` move outputs more but appear less controlled.
- Higher alpha can encourage generic validation or even move against self-agency in some prompts.

Key artifacts:

- `outputs/layer21_last_alpha_calibration_20260424T054527Z/summary.json`
- `outputs/layer21_last_alpha_calibration_20260424T054527Z/steering_summary.csv`
- `outputs/layer21_last_alpha_calibration_20260424T054527Z/qualitative_alpha_calibration_inspection.md`

### 4. Warm Boundary Self-Agency Contrast Dataset

Run-plan document:

```text
outputs/warm_boundary_self_agency_contrast_run_plan.md
```

Data prepared:

- `data/warm_boundary_self_agency_contrast_raw.jsonl`
- `data/warm_boundary_self_agency_contrast_pairs.jsonl`
- `data/warm_boundary_self_agency_contrast_matched.jsonl`

Validation:

- pair rows: `80`
- matched rows: `160`
- contrast counts: `boundary=30`, `self_agency=30`, `interpersonal_uncertainty=20`

Labels:

- positive: `warm_boundary_self_agency`
- negative: `generic_support_therapy`

Reason for this dataset:

- The first warm-boundary vector moved outputs but was not cleanly separated from generic supportive therapy style.
- This contrast was designed to separate boundary and self-agency behavior from generic validation.

### 5. Warm Boundary Self-Agency Contrast Probe

Run directory:

```text
outputs/warm_boundary_self_agency_contrast_probe_20260424T070556Z
```

Dataset:

```text
data/warm_boundary_self_agency_contrast_matched.jsonl
```

Selected module:

- `model.language_model.layers.21`

Results:

- capture records: `160`
- held-out steering results: `40`
- vector norm: `12.8490`

Key artifacts:

- `outputs/warm_boundary_self_agency_contrast_probe_20260424T070556Z/summary.json`
- `outputs/warm_boundary_self_agency_contrast_probe_20260424T070556Z/capture_records.json`
- `outputs/warm_boundary_self_agency_contrast_probe_20260424T070556Z/steering_summary.csv`
- `outputs/warm_boundary_self_agency_contrast_probe_20260424T070556Z/vectors/model_language_model_layers_21__warm_boundary_self_agency_minus_generic_support_therapy.pt`

### 6. Warm Boundary Self-Agency Contrast Layer 21 Sweep

Run directory:

```text
outputs/warm_boundary_self_agency_contrast_layer21_sweep_20260424T070847Z
```

Source run:

```text
outputs/warm_boundary_self_agency_contrast_probe_20260424T070556Z
```

Sweep:

- module: `model.language_model.layers.21`
- position mode: `last`
- alphas: `0.75`, `1.0`, `1.25`
- baselines: `40`
- steering results: `120`

Key artifacts:

- `outputs/warm_boundary_self_agency_contrast_layer21_sweep_20260424T070847Z/summary.json`
- `outputs/warm_boundary_self_agency_contrast_layer21_sweep_20260424T070847Z/baselines.csv`
- `outputs/warm_boundary_self_agency_contrast_layer21_sweep_20260424T070847Z/steering_summary.csv`
- `outputs/warm_boundary_self_agency_contrast_layer21_sweep_20260424T070847Z/steering_results.jsonl`

This run still needs the same kind of qualitative inspection that was done for the earlier layer-21 calibration run.

## Useful Commands

List modules:

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/list_modules.py --limit 300
PYTHONPATH=src python scripts/list_modules.py --contains self_attn --limit 200
PYTHONPATH=src python scripts/list_modules.py --contains mlp --limit 200
PYTHONPATH=src python scripts/list_modules.py --show-default-hook-points
```

Save a traced generation:

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/generate_with_traces.py --max-new-tokens 48
```

Regenerate the warm-boundary full probe:

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/run_probe_experiment.py \
  --dataset data/activation_steering_warm_boundary_training_100_matched.jsonl \
  --heldout-dataset data/activation_steering_warm_boundary_heldout_40.jsonl \
  --layers 0,21,41 \
  --positive-label warm_boundary_maintaining \
  --negative-label warm_boundary_eroding \
  --alpha 1.0 \
  --run-name probe_warm_boundary_full
```

Regenerate the layer-21 calibration sweep:

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/sweep_saved_steering.py \
  --source-run outputs/probe_warm_boundary_full_20260424T024108Z \
  --heldout-dataset data/activation_steering_warm_boundary_heldout_40.jsonl \
  --modules model.language_model.layers.21 \
  --position-modes last \
  --alphas 0.75,1.0,1.25,1.5,1.75,2.0 \
  --max-new-tokens 64 \
  --run-name layer21_last_alpha_calibration
```

Regenerate the self-agency contrast probe:

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/run_probe_experiment.py \
  --dataset data/warm_boundary_self_agency_contrast_matched.jsonl \
  --heldout-dataset data/activation_steering_warm_boundary_heldout_40.jsonl \
  --layers 21 \
  --positive-label warm_boundary_self_agency \
  --negative-label generic_support_therapy \
  --alpha 1.0 \
  --max-new-tokens 64 \
  --run-name warm_boundary_self_agency_contrast_probe
```

Regenerate the self-agency contrast sweep:

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/sweep_saved_steering.py \
  --source-run outputs/warm_boundary_self_agency_contrast_probe_20260424T070556Z \
  --heldout-dataset data/activation_steering_warm_boundary_heldout_40.jsonl \
  --modules model.language_model.layers.21 \
  --position-modes last \
  --alphas 0.75,1.0,1.25 \
  --max-new-tokens 64 \
  --run-name warm_boundary_self_agency_contrast_layer21_sweep
```

## Main Findings So Far

- Direct Gemma 4 activation access is working with Hugging Face model objects.
- Text-only generation works despite the checkpoint being multimodal.
- The language-model transformer block names resolve as `model.language_model.layers.N`.
- Block-level steering at layer 21 is much more promising than the initial layer 0 or layer 41 surfaces.
- `position_mode=last` is more controlled than `position_mode=all` for the current steering vectors.
- For the original warm-boundary direction, `alpha=1.0` at layer 21 is the best current balance of movement and control.
- `alpha=0.75` is the safer conservative comparison.
- Higher alpha settings mostly produce more rewriting, not clearly better target behavior.
- The first direction is partially confounded with generic supportive/therapy-style rewriting.
- The self-agency contrast dataset and layer-21 vector were created to reduce that confound, but the resulting sweep still needs qualitative scoring.

## Recommended Next Steps

1. Qualitatively inspect `outputs/warm_boundary_self_agency_contrast_layer21_sweep_20260424T070847Z/steering_summary.csv`.
2. Compare the self-agency contrast sweep against `outputs/layer21_last_alpha_calibration_20260424T054527Z`.
3. Use a small hand rubric with separate columns for:
   - `boundary_support`
   - `self_agency`
   - `warmth`
   - `generic_therapy`
   - `overreach`
4. Decide whether the newer contrast vector is cleaner than the original warm-boundary vector.
5. If the contrast vector is cleaner, run a slightly wider layer-21 sweep around `alpha=0.75` to `1.25`.
6. If it is not cleaner, improve the contrast data before adding more model-side complexity.
7. Only after a cleaner block-level direction is confirmed, decompose layer 21 into attention output projection and MLP output hooks.

## Do Not Change Casually

- pinned dependencies
- direct Hugging Face model loading
- text-only first-pass workflow
- `enable_thinking=False`
- no quantization in v1
- no serving stacks such as vLLM, TGI, or SGLang
- CPU offloading for captured activations
- loud failure behavior for missing hooks and shape mismatches

## Known Gaps

- There is no automated behavioral scoring pipeline yet.
- Qualitative notes exist for the broad sweep and alpha calibration, but not yet for the self-agency contrast sweep.
- The initial probe summaries do not record every generation argument, so regeneration commands above are faithful to the intended scripts but not a complete provenance record.
- This checked-out directory did not have Git metadata at handoff time.
