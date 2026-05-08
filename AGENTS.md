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
- `src/gemma4_activation_lab/experiments/probe.py`
- `src/gemma4_activation_lab/experiments/sweep.py`

Runnable scripts:

- `scripts/load_gemma.py`
- `scripts/list_modules.py`
- `scripts/generate_with_traces.py`
- `scripts/run_probe_experiment.py`
- `scripts/sweep_saved_steering.py`

Data scaffold:

- `data/matched_conversations.jsonl`
- `data/heldout_prompts.jsonl`

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
