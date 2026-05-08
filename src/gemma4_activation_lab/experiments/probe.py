from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from gemma4_activation_lab.artifacts import save_direction_vectors, save_probe_artifacts
from gemma4_activation_lab.config import (
    DATA_DIR,
    DEFAULT_NEGATIVE_LABEL,
    DEFAULT_POSITIVE_LABEL,
    DEFAULT_PROBE_LAYERS,
    DEFAULT_STEERING_ALPHA,
    MODEL_ID,
    OUTPUT_DIR,
    SEED,
)
from gemma4_activation_lab.datasets import load_heldout_rows, load_matched_rows, require_labels
from gemma4_activation_lab.generation import generate_for_user
from gemma4_activation_lab.hooks import (
    ActivationStore,
    collect_default_hook_points,
    register_forward_hooks,
    remove_hooks,
)
from gemma4_activation_lab.intervene import AddVectorIntervention, register_additive_intervention
from gemma4_activation_lab.io_utils import make_run_dir
from gemma4_activation_lab.modeling import (
    load_processor_and_model,
    print_package_versions,
    set_reproducible_seed,
    tokenize_text,
)
from gemma4_activation_lab.prompting import assistant_span


@dataclass(frozen=True)
class ProbeExperimentConfig:
    model_id: str = MODEL_ID
    seed: int = SEED
    dataset: Path = DATA_DIR / "matched_conversations.jsonl"
    heldout_dataset: Path = DATA_DIR / "heldout_prompts.jsonl"
    layer_indices: tuple[int, ...] = tuple(DEFAULT_PROBE_LAYERS)
    positive_label: str = DEFAULT_POSITIVE_LABEL
    negative_label: str = DEFAULT_NEGATIVE_LABEL
    alpha: float = DEFAULT_STEERING_ALPHA
    max_new_tokens: int = 48
    output_dir: Path = OUTPUT_DIR
    run_name: str = "probe_experiment"
    print_versions: bool = True


@dataclass(frozen=True)
class ProbeExperimentResult:
    run_dir: Path
    selected_modules: list[str]
    steering_module: str
    summary: dict[str, Any]


def pool_activation(tensor: torch.Tensor, start: int, end: int) -> torch.Tensor:
    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise RuntimeError(f"Expected [1, seq, hidden] activation, got {tuple(tensor.shape)}.")
    if start < 0 or end > tensor.shape[1] or start >= end:
        raise RuntimeError(f"Bad token span {(start, end)} for activation shape {tuple(tensor.shape)}.")
    return tensor[0, start:end, :].float().mean(dim=0)


def capture_style_vectors(
    processor: Any,
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    selected_modules: list[str],
) -> tuple[dict[str, dict[str, torch.Tensor]], list[dict[str, Any]]]:
    store = ActivationStore()
    _, handles = register_forward_hooks(model, selected_modules, store=store)
    grouped_vectors: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
    records: list[dict[str, Any]] = []

    try:
        with torch.no_grad():
            for row in rows:
                label = row["response_style_label"]
                text, start, end = assistant_span(
                    processor,
                    row["user_message"],
                    row["assistant_target_text"],
                )
                inputs = tokenize_text(processor, text, next(model.parameters()).device)
                store.clear()
                model(**inputs, use_cache=False)
                for module_name in selected_modules:
                    pooled = pool_activation(store.latest(module_name), start, end)
                    grouped_vectors[label][module_name].append(pooled)
                    records.append(
                        {
                            "scenario_id": row["scenario_id"],
                            "response_style_label": label,
                            "module_name": module_name,
                            "token_span_start": start,
                            "token_span_end": end,
                            "vector_norm": float(torch.linalg.vector_norm(pooled).item()),
                        }
                    )
    finally:
        remove_hooks(handles)

    means: dict[str, dict[str, torch.Tensor]] = {}
    for label, per_module in grouped_vectors.items():
        means[label] = {}
        for module_name, vectors in per_module.items():
            means[label][module_name] = torch.stack(vectors).mean(dim=0)
    return means, records


def compute_direction_vectors(
    means: dict[str, dict[str, torch.Tensor]],
    selected_modules: list[str],
    *,
    positive_label: str,
    negative_label: str,
) -> dict[str, torch.Tensor]:
    directions: dict[str, torch.Tensor] = {}
    for module_name in selected_modules:
        positive = means[positive_label][module_name]
        negative = means[negative_label][module_name]
        directions[module_name] = positive - negative
    return directions


def run_probe_experiment(config: ProbeExperimentConfig) -> ProbeExperimentResult:
    set_reproducible_seed(config.seed)
    if config.print_versions:
        print_package_versions()

    dataset_rows = load_matched_rows(config.dataset)
    heldout_rows = load_heldout_rows(config.heldout_dataset)
    require_labels(dataset_rows, [config.positive_label, config.negative_label])
    processor, model = load_processor_and_model(model_id=config.model_id)

    hook_points = collect_default_hook_points(
        model,
        layer_indices=config.layer_indices,
        include_blocks=True,
        include_attn_output=False,
        include_mlp_output=False,
        include_final_norm=False,
    )
    selected_modules = hook_points["transformer_blocks"]

    run_dir = make_run_dir(config.output_dir, config.run_name)
    means, capture_records = capture_style_vectors(processor, model, dataset_rows, selected_modules)
    directions = compute_direction_vectors(
        means,
        selected_modules,
        positive_label=config.positive_label,
        negative_label=config.negative_label,
    )
    direction_metadata = save_direction_vectors(
        run_dir,
        directions,
        positive_label=config.positive_label,
        negative_label=config.negative_label,
    )

    steering_module = selected_modules[-1]
    steering_vector = directions[steering_module]
    steering_results: list[dict[str, Any]] = []
    for row in heldout_rows:
        baseline = generate_for_user(
            processor,
            model,
            row["user_message"],
            max_new_tokens=config.max_new_tokens,
        )

        controller = register_additive_intervention(
            model,
            AddVectorIntervention(
                module_name=steering_module,
                vector=steering_vector,
                alpha=config.alpha,
                position_mode="last",
                enabled=True,
            ),
            hook_mode="forward",
        )
        try:
            intervened = generate_for_user(
                processor,
                model,
                row["user_message"],
                max_new_tokens=config.max_new_tokens,
            )
        finally:
            controller.remove()

        steering_results.append(
            {
                "scenario_id": row["scenario_id"],
                "module_name": steering_module,
                "alpha": config.alpha,
                "baseline_output": baseline.decoded_output,
                "intervened_output": intervened.decoded_output,
                "user_message": row["user_message"],
            }
        )

    summary = {
        "model_id": config.model_id,
        "seed": config.seed,
        "selected_modules": selected_modules,
        "positive_label": config.positive_label,
        "negative_label": config.negative_label,
        "alpha": config.alpha,
        "direction_metadata": direction_metadata,
        "capture_records_count": len(capture_records),
        "steering_results_count": len(steering_results),
    }

    save_probe_artifacts(
        run_dir,
        summary=summary,
        capture_records=capture_records,
        steering_results=steering_results,
    )
    return ProbeExperimentResult(
        run_dir=run_dir,
        selected_modules=selected_modules,
        steering_module=steering_module,
        summary=summary,
    )
