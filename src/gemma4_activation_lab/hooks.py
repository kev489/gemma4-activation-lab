from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import torch


@dataclass
class ActivationStore:
    tensors: dict[str, list[torch.Tensor]] = field(default_factory=dict)

    def add(self, name: str, tensor: torch.Tensor) -> None:
        if tensor.ndim < 2:
            raise RuntimeError(f"Activation {name!r} has unexpected shape {tuple(tensor.shape)}")
        self.tensors.setdefault(name, []).append(tensor.detach().cpu())

    def clear(self) -> None:
        self.tensors.clear()

    def latest(self, name: str) -> torch.Tensor:
        if name not in self.tensors or not self.tensors[name]:
            raise KeyError(f"No activation captured for {name!r}.")
        return self.tensors[name][-1]


def enumerate_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    return list(model.named_modules())


def module_names(model: torch.nn.Module, contains: str | None = None) -> list[str]:
    names = [name for name, _ in model.named_modules()]
    if contains:
        names = [name for name in names if contains in name]
    return names


def get_module(model: torch.nn.Module, name: str) -> torch.nn.Module:
    modules = dict(model.named_modules())
    if name not in modules:
        candidates = [module_name for module_name in modules if name in module_name][:20]
        detail = f" Similar names: {candidates}" if candidates else ""
        raise KeyError(f"Module {name!r} not found.{detail}")
    return modules[name]


def first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return first_tensor(item)
            except TypeError:
                continue
    if isinstance(value, dict):
        for key in ("last_hidden_state", "hidden_states", "logits"):
            if key in value:
                return first_tensor(value[key])
        for item in value.values():
            try:
                return first_tensor(item)
            except TypeError:
                continue
    raise TypeError(f"No tensor found in hook value of type {type(value).__name__}.")


def register_forward_hooks(
    model: torch.nn.Module,
    selected_module_names: Iterable[str],
    store: ActivationStore | None = None,
) -> tuple[ActivationStore, list[torch.utils.hooks.RemovableHandle]]:
    store = store or ActivationStore()
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for name in selected_module_names:
        module = get_module(model, name)

        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any, *, hook_name: str = name) -> None:
            store.add(hook_name, first_tensor(output))

        handles.append(module.register_forward_hook(hook))
    return store, handles


def remove_hooks(handles: Iterable[torch.utils.hooks.RemovableHandle]) -> None:
    for handle in handles:
        handle.remove()


def find_transformer_blocks(model: torch.nn.Module) -> list[str]:
    candidates: list[tuple[str, int]] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and name.endswith("layers") and len(module) > 1:
            candidates.append((name, len(module)))
    if not candidates:
        raise RuntimeError("Could not find a ModuleList ending in 'layers'. Inspect scripts/list_modules.py output.")

    preferred_names = [
        "model.language_model.layers",
        "language_model.layers",
        "model.model.layers",
        "model.decoder.layers",
        "model.transformer.layers",
        "transformer.layers",
    ]
    candidate_names = {name for name, _ in candidates}
    for preferred_name in preferred_names:
        if preferred_name in candidate_names:
            name, count = next((candidate for candidate in candidates if candidate[0] == preferred_name))
            return [f"{name}.{idx}" if name else str(idx) for idx in range(count)]

    text_candidates = [
        candidate for candidate in candidates if any(marker in candidate[0] for marker in ("language_model", "decoder", "transformer"))
    ]
    name, count = max(text_candidates or candidates, key=lambda candidate: candidate[1])
    return [f"{name}.{idx}" if name else str(idx) for idx in range(count)]


def _first_existing(model: torch.nn.Module, names: Iterable[str]) -> str | None:
    modules = dict(model.named_modules())
    for name in names:
        if name in modules:
            return name
    return None


def resolve_layer_indices(blocks: list[str], layer_indices: Iterable[int]) -> list[str]:
    resolved: list[str] = []
    for index in layer_indices:
        actual = len(blocks) + index if index < 0 else index
        if actual < 0 or actual >= len(blocks):
            raise IndexError(f"Layer index {index} resolved to {actual}, but there are {len(blocks)} blocks.")
        resolved.append(blocks[actual])
    return resolved


def collect_default_hook_points(
    model: torch.nn.Module,
    layer_indices: Iterable[int] | None = None,
    *,
    include_blocks: bool = True,
    include_attn_output: bool = True,
    include_mlp_output: bool = True,
    include_final_norm: bool = True,
) -> dict[str, list[str]]:
    blocks = find_transformer_blocks(model)
    selected_blocks = resolve_layer_indices(blocks, layer_indices) if layer_indices is not None else blocks

    hook_points: dict[str, list[str]] = {
        "transformer_blocks": [],
        "attention_output_projection": [],
        "mlp_output": [],
        "final_norm": [],
    }

    for block_name in selected_blocks:
        if include_blocks:
            hook_points["transformer_blocks"].append(block_name)

        if include_attn_output:
            attn_name = _first_existing(
                model,
                [
                    f"{block_name}.self_attn.o_proj",
                    f"{block_name}.self_attn.out_proj",
                    f"{block_name}.attention.o_proj",
                    f"{block_name}.attn.o_proj",
                ],
            )
            if attn_name is None:
                raise RuntimeError(f"No attention output projection found under {block_name!r}.")
            hook_points["attention_output_projection"].append(attn_name)

        if include_mlp_output:
            mlp_name = _first_existing(
                model,
                [
                    f"{block_name}.mlp",
                    f"{block_name}.feed_forward",
                    f"{block_name}.ffn",
                ],
            )
            if mlp_name is None:
                raise RuntimeError(f"No MLP/feed-forward module found under {block_name!r}.")
            hook_points["mlp_output"].append(mlp_name)

    if include_final_norm:
        final_norm = _first_existing(
            model,
            [
                "model.norm",
                "model.final_layernorm",
                "model.language_model.norm",
                "language_model.model.norm",
                "language_model.norm",
                "transformer.ln_f",
            ],
        )
        if final_norm is None:
            raise RuntimeError("No final norm module found. Inspect module names and add the path explicitly.")
        hook_points["final_norm"].append(final_norm)

    return hook_points


def flatten_hook_points(hook_points: dict[str, list[str]]) -> list[str]:
    names: list[str] = []
    for group in hook_points.values():
        names.extend(group)
    return names
