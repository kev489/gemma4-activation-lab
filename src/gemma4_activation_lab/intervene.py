from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from .hooks import get_module

PositionMode = Literal["all", "last", "span"]
HookMode = Literal["forward_pre", "forward"]


@dataclass
class AddVectorIntervention:
    module_name: str
    vector: torch.Tensor
    alpha: float = 1.0
    position_mode: PositionMode = "last"
    token_span: tuple[int, int] | None = None
    enabled: bool = True

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return tensor
        if tensor.ndim != 3:
            raise RuntimeError(
                f"Intervention expects [batch, seq, hidden], got shape {tuple(tensor.shape)} "
                f"at {self.module_name!r}."
            )

        vector = self.vector.to(device=tensor.device, dtype=tensor.dtype)
        if vector.ndim != 1:
            raise RuntimeError(f"Steering vector must be 1D, got shape {tuple(vector.shape)}.")
        if vector.shape[0] != tensor.shape[-1]:
            raise RuntimeError(
                f"Vector width {vector.shape[0]} does not match activation width {tensor.shape[-1]} "
                f"at {self.module_name!r}."
            )

        updated = tensor.clone()
        delta = self.alpha * vector.view(1, 1, -1)
        if self.position_mode == "all":
            updated = updated + delta
        elif self.position_mode == "last":
            updated[:, -1:, :] = updated[:, -1:, :] + delta
        elif self.position_mode == "span":
            if self.token_span is None:
                raise RuntimeError("position_mode='span' requires token_span=(start, end).")
            start, end = self.token_span
            if start < 0 or end > updated.shape[1] or start >= end:
                raise RuntimeError(f"Invalid token span {(start, end)} for sequence length {updated.shape[1]}.")
            updated[:, start:end, :] = updated[:, start:end, :] + delta
        else:
            raise ValueError(f"Unknown position mode: {self.position_mode}")
        return updated


def _replace_first_tensor(value: Any, intervention: AddVectorIntervention) -> Any:
    if isinstance(value, torch.Tensor):
        return intervention.apply(value)
    if isinstance(value, tuple):
        items = list(value)
        for index, item in enumerate(items):
            try:
                items[index] = _replace_first_tensor(item, intervention)
                return tuple(items)
            except TypeError:
                continue
    if isinstance(value, list):
        items = list(value)
        for index, item in enumerate(items):
            try:
                items[index] = _replace_first_tensor(item, intervention)
                return items
            except TypeError:
                continue
    if isinstance(value, dict):
        items = dict(value)
        for key, item in items.items():
            try:
                items[key] = _replace_first_tensor(item, intervention)
                return items
            except TypeError:
                continue
    raise TypeError(f"No tensor found in value of type {type(value).__name__}.")


@dataclass
class InterventionController:
    specs: list[AddVectorIntervention]
    handles: list[torch.utils.hooks.RemovableHandle] = field(default_factory=list)

    def enable(self) -> None:
        for spec in self.specs:
            spec.enabled = True

    def disable(self) -> None:
        for spec in self.specs:
            spec.enabled = False

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def register_additive_intervention(
    model: torch.nn.Module,
    spec: AddVectorIntervention,
    *,
    hook_mode: HookMode = "forward",
) -> InterventionController:
    module = get_module(model, spec.module_name)

    if hook_mode == "forward_pre":
        def pre_hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
            return _replace_first_tensor(inputs, spec)

        handle = module.register_forward_pre_hook(pre_hook)
    elif hook_mode == "forward":
        def forward_hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            return _replace_first_tensor(output, spec)

        handle = module.register_forward_hook(forward_hook)
    else:
        raise ValueError(f"Unknown hook_mode={hook_mode!r}")

    return InterventionController(specs=[spec], handles=[handle])
