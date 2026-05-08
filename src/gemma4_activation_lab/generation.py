from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .modeling import decode_new_tokens, generation_pad_kwargs, tokenize_text
from .prompting import ChatMessage, prompt_for_user, render_messages_for_generation


@dataclass(frozen=True)
class GenerationResult:
    prompt_text: str
    decoded_output: str
    input_ids: torch.Tensor
    sequences: torch.Tensor
    new_token_ids: torch.Tensor
    raw_outputs: Any


def generate_from_prompt_text(
    processor: Any,
    model: torch.nn.Module,
    prompt_text: str,
    *,
    max_new_tokens: int,
    output_hidden_states: bool = False,
    output_attentions: bool = False,
) -> GenerationResult:
    inputs = tokenize_text(processor, prompt_text, next(model.parameters()).device)
    input_len = inputs["input_ids"].shape[-1]
    generation_settings = {
        "max_new_tokens": max_new_tokens,
        "return_dict_in_generate": True,
        "output_hidden_states": output_hidden_states,
        "output_attentions": output_attentions,
        "do_sample": False,
    }
    generation_settings.update(generation_pad_kwargs(processor))

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_settings,
        )

    sequences = outputs.sequences.detach().cpu()
    new_token_ids = sequences[:, input_len:]
    decoded_output = decode_new_tokens(processor, outputs.sequences, input_len)
    return GenerationResult(
        prompt_text=prompt_text,
        decoded_output=decoded_output,
        input_ids=inputs["input_ids"].detach().cpu(),
        sequences=sequences,
        new_token_ids=new_token_ids,
        raw_outputs=outputs,
    )


def generate_from_messages(
    processor: Any,
    model: torch.nn.Module,
    messages: list[ChatMessage],
    *,
    max_new_tokens: int,
    output_hidden_states: bool = False,
    output_attentions: bool = False,
) -> GenerationResult:
    return generate_from_prompt_text(
        processor,
        model,
        render_messages_for_generation(processor, messages),
        max_new_tokens=max_new_tokens,
        output_hidden_states=output_hidden_states,
        output_attentions=output_attentions,
    )


def generate_for_user(
    processor: Any,
    model: torch.nn.Module,
    user_message: str,
    *,
    max_new_tokens: int,
) -> GenerationResult:
    return generate_from_prompt_text(
        processor,
        model,
        prompt_for_user(processor, user_message),
        max_new_tokens=max_new_tokens,
    )


def save_generation_hidden_states(run_dir: Path, hidden_states: Any) -> list[str]:
    hidden_dir = run_dir / "hidden_states"
    hidden_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []
    if hidden_states is None:
        return saved_paths

    for step_index, layer_tuple in enumerate(hidden_states):
        serialized: list[torch.Tensor] = []
        for layer_tensor in layer_tuple:
            serialized.append(layer_tensor.detach().cpu())
        path = hidden_dir / f"step_{step_index:04d}.pt"
        torch.save(serialized, path)
        saved_paths.append(str(path))
    return saved_paths
