# AGENTS.md

## Purpose

This repo is a text-only, activation-accessible research scaffold for `google/gemma-4-E4B-it`.

Primary goals:

- direct access to the Hugging Face model object
- reproducible activation capture and intervention experiments
- no serving-first frameworks
- no quantization in v1
- GPU `bfloat16` execution
- Gemma thinking disabled for baseline interpretability work

## Ground rules

- Use direct `AutoModelForCausalLM.from_pretrained(...)` and `AutoProcessor.from_pretrained(...)`.
- Do not switch the main path to `pipeline(...)`.
- Do not introduce vLLM, TGI, SGLang, or other serving stacks unless explicitly requested.
- Keep the first-pass workflow text-only even though the checkpoint is multimodal.
- Keep `enable_thinking=False` in chat templating unless the experiment explicitly studies thinking.
- Prefer simple, inspectable code over abstraction-heavy wrappers.

## Where prior-run context lives

`AGENTS.md` is the canonical operating guide (env, policy, current judge state). Per-experiment records (completed run directories, vector norms, qualitative reads, findings, recommended next steps, and the long-form regenerate-commands tied to specific runs) live in `results/run_history.md`. Append new run summaries there rather than re-padding this file.

## Environment

Project root:

- `/home/kchafloq/gemma4-activation-lab`

Virtualenv:

- `/home/kchafloq/gemma4-activation-lab/.venv`

Create or refresh it:

```bash
cd /home/kchafloq/gemma4-activation-lab
bash env/create_env.sh
```

Manual equivalent:

```bash
cd /home/kchafloq/gemma4-activation-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip==26.0 setuptools==80.9.0 wheel==0.45.1
python -m pip install -r requirements.txt
```

Pinned packages in `requirements.txt` include:

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

Important:

- `torchvision` is required because Gemma 4 processor loading pulls in vision/video processor classes even for text-only use.
- The `torchvision` wheel must match the PyTorch CUDA stack. In this repo it is installed as `0.26.0+cu128`.

## Auth and model access

Authenticate with Hugging Face:

```bash
cd /home/kchafloq/gemma4-activation-lab
source .venv/bin/activate
hf auth login
hf auth whoami
```

The model was successfully loaded end to end in this environment.

Current local cache:

- `/home/kchafloq/.cache/huggingface/hub/models--google--gemma-4-E4B-it`

Approximate cache size at verification time:

- `15G`

## Known-good verification state

Verified in this repo:

- `torch.cuda.is_available() == True`
- `torch.cuda.is_bf16_supported() == True`
- visible device: `NVIDIA L40S`
- successful full load of `google/gemma-4-E4B-it`
- successful smoke-test generation on GPU in `bfloat16`

Observed loaded model details:

- class: `Gemma4ForConditionalGeneration`
- dtype: `torch.bfloat16`
- device: `cuda:0`
- parameter count estimate: about `7.94B`

## Main entry points

### 1. Smoke-test load

```bash
cd /home/kchafloq/gemma4-activation-lab
source .venv/bin/activate
PYTHONPATH=src python scripts/load_gemma.py --max-new-tokens 16
```

This prints:

- package versions
- model class
- dtype
- device
- parameter count estimate
- top-level module names
- smoke-test output text

### 2. List module names for hooks

```bash
PYTHONPATH=src python scripts/list_modules.py --limit 300
PYTHONPATH=src python scripts/list_modules.py --contains self_attn --limit 200
PYTHONPATH=src python scripts/list_modules.py --contains mlp --limit 200
PYTHONPATH=src python scripts/list_modules.py --show-default-hook-points
```

### 3. Traced generation

```bash
PYTHONPATH=src python scripts/generate_with_traces.py --max-new-tokens 48
```

Artifacts are written under `outputs/` and include:

- prompt text
- input ids
- generated ids
- decoded output
- per-step hidden state dumps
- metadata JSON

### 4. Probe and steering scaffold

```bash
PYTHONPATH=src python scripts/run_probe_experiment.py --layers 0,-1 --alpha 1.0
```

This:

- loads matched JSONL examples
- teacher-forces target assistant text
- captures selected activations
- averages by label
- computes simple difference vectors
- saves vectors to disk
- runs a small baseline vs intervened held-out test

## Source layout

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
- `src/gemma4_activation_lab/judge.py`
- `src/gemma4_activation_lab/experiments/probe.py`
- `src/gemma4_activation_lab/experiments/sweep.py`

Runnable scripts:

- `scripts/load_gemma.py`
- `scripts/list_modules.py`
- `scripts/generate_with_traces.py`
- `scripts/run_probe_experiment.py`
- `scripts/sweep_saved_steering.py`
- `scripts/extract_impactbench_autonomy_subsets.py`
- `scripts/build_judge_eval_set.py`
- `scripts/run_judge_eval.py`
- `scripts/validate_turn_localization.py`

Data scaffold:

- `data/matched_conversations.jsonl`
- `data/heldout_prompts.jsonl`
- `data/activation_steering_warm_boundary_training_100_matched.jsonl`
- `data/activation_steering_warm_boundary_heldout_40.jsonl`
- `data/warm_boundary_self_agency_contrast_raw.jsonl` (80 raw rows, kept for provenance)
- `data/warm_boundary_self_agency_contrast_pairs.jsonl` (80 repaired pair rows)
- `data/warm_boundary_self_agency_contrast_matched.jsonl` (160 probe-compatible matched rows)
- `data/impactbench_autonomy/autonomy_preservation_records.jsonl`
- `data/impactbench_autonomy/self_determination_records.jsonl`
- `data/impactbench_autonomy/manifest.json`
- `data/impactbench_autonomy/judge_eval/autonomy_preservation_eval_50.jsonl`
- `data/impactbench_autonomy/judge_eval/self_determination_eval_50.jsonl`
- `data/impactbench_autonomy/judge_eval/autonomy_preservation_eval_20.jsonl`
- `data/impactbench_autonomy/judge_eval/self_determination_eval_20.jsonl`
- `data/impactbench_autonomy/prompts/01_turn_localization_with_explicit_pooling.txt`

Local-only (gitignored) data:

- `activation_lab_dataset/`: local duplicate / import dump of the warm-boundary data. Canonical tracked copies live under `data/`. Do not write new artifacts here.

ImpactBench autonomy subsets:

```bash
python3 scripts/extract_impactbench_autonomy_subsets.py
```

This extracts literal full-record subsets from the local ImpactBench transcript archive:

- `Autonomy Preservation`: `7,560` full transcript records
- `Self-Determination`: `3,696` full transcript records

The extracted rows preserve the original ImpactBench shape, including:

- `scenario`
- `samples`
- `locations`
- `verdict`
- transcript metadata

Important:

- Do not treat ImpactBench verdicts as first-turn labels.
- Do not convert these files into the repo's single-turn probe schema unless explicitly building a multi-turn-aware extraction method.
- ImpactBench verdicts are conversation-level metric judgments; they may depend on later turns in `samples`.
- Keep these files separate from the warm-boundary/self-agency matched probe datasets.

## ImpactBench turn-localization prompt and validation

Use `data/impactbench_autonomy/prompts/01_turn_localization_with_explicit_pooling.txt` as the current reusable localization prompt template. It contains a `{{RECORD_JSON}}` placeholder; another agent or script can replace that placeholder with exactly one full ImpactBench record before sending it to the LLM.

The prompt asks the LLM to localize behavior to assistant turns and return an exact `assistant_pooling_text` plus character offsets into the source assistant turn. It also asks for:

- `activation_label`: `positive`, `negative`, or `discard`
- `activation_quality`: `strong`, `weak`, or `discard`
- `default_train_include`: `true` only for strong positive/negative rows that should enter first-pass vector training

Default vector construction should use only validated strong rows with `default_train_include: true`. Weak rows may be preserved for ablations, audits, and sensitivity tests, but they should not enter the default direction vector.

Validate LLM localization outputs before using them:

```bash
python3 scripts/validate_turn_localization.py \
  --annotation outputs/prompt_tests/turn_localization_self_determination_line560_prompt_v2_response.json \
  --prompt outputs/prompt_tests/turn_localization_self_determination_line560_prompt_v2.txt
```

The validator checks JSON/schema consistency, turn indices, label/quality/default-training consistency, exact span substring match, character offsets, evidence quotes, and best-index lists. Invalid strong spans are flagged as `review_required` for human inspection rather than silently dropped. Invalid weak spans are warnings and remain excluded from default training.

For bulk runs, `scripts/validate_turn_localization.py` can also take `--record-json path/to/record.json` instead of `--prompt` if prompts are not saved one-per-record.

## LLM judge (ImpactBench calibration)

`src/gemma4_activation_lab/judge.py` is a reusable yes/no LLM-judge harness. As of the `2026-05-25` head-to-head calibration (section `7` of `results/run_history.md`), the **recommended production backend is `claude -p` with `claude-opus-4-7`**. The `codex exec` backend with `gpt-5.4-mini` is still supported for cost-sensitive use (Codex runs are free via the ChatGPT subscription) and remains the current *code* default in `src/gemma4_activation_lab/judge.py`. Pass `--provider claude --model claude-opus-4-7` to use the recommended Opus path; the code default will be flipped in a follow-up change.

Record shape understanding (informs the prompt):

- Each ImpactBench row is `(scenario_id, transcript_model, metric_id) -> verdict.result`.
- The criterion is fixed per row in `metric_criterion`; the conversation in `samples[0]` was generated to probe that one criterion.
- A scenario_id is a probing setup, not a fixed transcript. The conversation differs across `transcript_model`s and even across metric rows for the same `(scenario, model)`.
- Therefore the judge grades per-`metric_criterion`, not per-subarea. There is no row-level subarea pass/fail in the data.

Build a blinded eval set:

```bash
.venv/bin/python scripts/build_judge_eval_set.py --input data/impactbench_autonomy/autonomy_preservation_records.jsonl --output data/impactbench_autonomy/judge_eval/autonomy_preservation_eval_50.jsonl --n-per-class 25 --seed 0
```

Sampling behavior:

- Stratified yes/no class balance (`--n-per-class`).
- Round-robin across `metric_id` for max criterion diversity within each class.
- Output rows have keys `rubric`, `conversation`, `ground_truth`, `metadata`. The judge code only ever reads `rubric` and `conversation`.

Run the judge (recommended Opus path):

```bash
PYTHONPATH=src .venv/bin/python scripts/run_judge_eval.py --eval-set data/impactbench_autonomy/judge_eval/autonomy_preservation_eval_50.jsonl --provider claude --model claude-opus-4-7
```

Cost-sensitive Codex path (free via ChatGPT subscription, lower accuracy on self-determination):

```bash
PYTHONPATH=src .venv/bin/python scripts/run_judge_eval.py --eval-set data/impactbench_autonomy/judge_eval/autonomy_preservation_eval_50.jsonl
```

Runner behavior:

- Code default today: `--provider codex --model gpt-5.4-mini` (1 worker). Recommended override: `--provider claude --model claude-opus-4-7` (4 workers).
- Smoke-test limit: `--max-rows 1`.
- Claude Sonnet comparison mode: `--provider claude --model claude-sonnet-4-6`.
- `--claude-opus-on-disagree --opus-model claude-opus-4-7` exists for *calibration* against labeled sets only. It does not transfer to grading new unlabeled generations because there is no ground-truth disagreement signal to trigger on — for production grading use Opus standalone instead.
- Output: `outputs/judge_eval/<eval-stem>_<timestamp>/{results.jsonl, summary.json}`.
- Each call enforces a JSON schema (`{"result": "yes"|"no", "justification": ...}`). Codex uses `codex exec --output-schema`; Claude uses `claude -p --json-schema`.

Judge prompt calibration:

- `yes` means the `metric_criterion` text is true of the assistant's overall behavior, not necessarily that the assistant behaved well.
- Do not mechanically invert based on `measurement=absence`; use the criterion wording itself.
- `behavior_type` is only a coarse benchmark family label and does not determine whether `yes` means good or bad behavior.
- Judge the whole conversation substantively: later corrections can repair early mistakes, and later contradictions can invalidate earlier correct statements.
- Current prompt includes extra calibration for urgency, endorsement, framework, counter-evidence, and factual-correction metrics.

Calibration state (`2026-05-25`, full arc in `results/run_history.md` section `7`):

- Codex `gpt-5.4-mini` (calibrated prompt) on `autonomy_preservation_eval_20`: `18/20` (`90%`).
- Codex `gpt-5.4-mini` (same prompt) on `self_determination_eval_20`: `13/20` (`65%`) — `25`-point generalization gap.
- Opus `claude-opus-4-7` standalone on `autonomy_preservation_eval_20`: `18/20` (`90%`) — ties Codex.
- Opus `claude-opus-4-7` standalone on `self_determination_eval_20`: `15/20` (`75%`) — `+2` over Codex; the lift catches real Codex errors (criterion-literalism on `metric_011`, final-turn-warning elision on `metric_017`).
- Combined head-to-head `n=40`: Codex `31/40` (`77.5%`), Opus `33/40` (`82.5%`). Opus wins `3` of `4` head-to-head disagreement rows on rater-match; on transcript-substance inspection, the one row where Codex matches the rater and Opus does not (`cog-bias_metric_030`) is a contestable rater verdict.
- A presence/sustained branching prompt revision was attempted and regressed both eval sets (`27/40` vs `31/40` baseline). Reverted. The structural finding: ImpactBench rater behavior is per-criterion-ID, not per-wording-pattern or per-subarea — global prompt tuning has plateaued.
- `75–82%` is plausibly the rater-noise ceiling, not the judge ceiling.
- Latest run artifacts under `outputs/judge_eval/` are `*_20260525T*` directories.

Important:

- Codex / ChatGPT subscription auth via `codex exec` is free for the user but slower and weaker; Opus 4.7 via `claude -p` is metered but more accurate and the recommended default for grading new generations.
- Subscription rate-limit windows can be exhausted by larger eval sets. If a Codex run hits the wall, drop `--n-per-class` (20-row eval sets are provided) or `--workers`.
- Do not invent your own subarea rubric — the verdicts in `verdict.result` are tied to the specific `metric_criterion` text on each row.
- Do not rely on the `--claude-opus-on-disagree` fallback as a production-grading protocol; it only fires when a labeled ground truth is available to disagree with.

Artifact layout:

- `outputs/`: ignored scratch run directories for traces, vectors, sweeps, and generated outputs
- `results/`: curated summaries/tables/notes promoted from `outputs/` when they are worth tracking

## Hooking policy

Default hook placement is implemented in `src/gemma4_activation_lab/hooks.py`.

For text-only Gemma 4 experiments, default discovery should target the language model, not the vision tower:

- transformer blocks: `model.language_model.layers.N`
- attention output projection: `model.language_model.layers.N.self_attn.o_proj`
- MLP output: `model.language_model.layers.N.mlp`
- final norm: `model.language_model.norm`

Recommended first-pass probe/steering layers:

- `0,21,41`

Rationale:

- layer `0` captures early prompt/token-format signal
- layer `21` gives a mid-model semantic/control point
- layer `41` is late and answer-style/readout-adjacent
- use transformer block outputs first because they are clean residual-stream boundaries
- use attention/MLP hooks later to decompose a layer after block-level effects are known
- inspect final norm for readout analysis, but do not use it as the first steering surface

Preferred hook surfaces:

- transformer block outputs
- attention output projection
- MLP output
- final norm

Why these are used:

- block outputs are a practical residual-stream boundary
- attention output projections isolate attention-written updates
- MLP outputs isolate feed-forward channel writes
- final norm is a late clean readout point before logits

Activation capture behavior:

- hooks fail loudly on missing modules
- activations are detached and moved to CPU immediately
- shape mismatches raise errors instead of silently passing

## Intervention policy

Interventions are implemented in `src/gemma4_activation_lab/intervene.py`.

Supported behavior:

- `forward_pre` or `forward` hooks
- additive vector interventions
- scalar coefficient `alpha`
- token targeting modes: `all`, `last`, `span`
- explicit on/off control

Use baseline and intervened runs from the same script path for comparisons.

## Gemma 4 specific caveats

- Even in text-only mode, the processor stack may pull in multimodal processor dependencies.
- The baseline path here uses `enable_thinking=False` in `apply_chat_template(...)`.
- Generation config on the model card is sampling-oriented; this repo overrides that for deterministic baseline experiments where needed.
- Gemma 4 E4B uses a hybrid local/global attention stack and per-layer embeddings, so block-level activations are useful approximations, not complete mechanistic decompositions.
- Long contexts can make trace saving very large very quickly. Start with short prompts and small `max_new_tokens`.

## Things not to change casually

- pinned package versions
- CUDA-matched Torch/Torchvision pairing
- `enable_thinking=False` default
- direct-model loading path
- CPU offloading behavior for captured activations

## Fastest sanity check

```bash
cd /home/kchafloq/gemma4-activation-lab
source .venv/bin/activate
PYTHONPATH=src python scripts/load_gemma.py --max-new-tokens 16
```
