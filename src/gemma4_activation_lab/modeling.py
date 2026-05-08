from __future__ import annotations

import importlib.metadata as metadata
import random
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from .config import DEVICE, DTYPE_NAME, MODEL_ID, SEED


def set_reproducible_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def dtype_from_name(name: str = DTYPE_NAME) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype name: {name}")


def require_cuda(device: str = DEVICE) -> torch.device:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise RuntimeError(f"This scaffold is configured for GPU bfloat16, got device={device!r}.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch. Check GPU allocation and device visibility.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The visible GPU does not report bfloat16 support.")
    return resolved


def package_versions() -> dict[str, str]:
    packages = [
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "huggingface-hub",
        "safetensors",
        "sentencepiece",
        "numpy",
        "pandas",
        "matplotlib",
        "tqdm",
        "PyYAML",
    ]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def print_package_versions() -> None:
    for package, version in package_versions().items():
        print(f"{package}: {version}")


def load_processor_and_model(
    model_id: str = MODEL_ID,
    device: str = DEVICE,
    dtype: torch.dtype | None = None,
) -> tuple[Any, torch.nn.Module]:
    device_obj = require_cuda(device)
    dtype = dtype or dtype_from_name()

    processor = AutoProcessor.from_pretrained(model_id)

    kwargs = {
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        # Eager attention is slower but easier to reason about when requesting
        # attentions or debugging hooks. This is not a serving-oriented setup.
        "attn_implementation": "eager",
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except TypeError:
        kwargs.pop("attn_implementation", None)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        except TypeError:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
            model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    model.to(device_obj)
    model.eval()
    return processor, model


def apply_chat_template_text(
    processor: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool = False,
) -> str:
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
    except TypeError as exc:
        raise RuntimeError(
            "This Gemma 4 setup requires a processor with apply_chat_template(..., "
            "enable_thinking=False). Upgrade Transformers if this fails."
        ) from exc


def tokenize_text(processor: Any, text: str, device: torch.device | str) -> Any:
    inputs = processor(text=text, return_tensors="pt")
    return inputs.to(device)


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def first_parameter_dtype_device(model: torch.nn.Module) -> tuple[torch.dtype, torch.device]:
    parameter = next(model.parameters())
    return parameter.dtype, parameter.device


def generation_pad_kwargs(processor: Any) -> dict[str, int]:
    tokenizer = getattr(processor, "tokenizer", processor)
    kwargs: dict[str, int] = {}
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None and eos_token_id is not None:
        kwargs["pad_token_id"] = eos_token_id
    return kwargs


def decode_new_tokens(processor: Any, sequences: torch.Tensor, input_len: int) -> str:
    new_tokens = sequences[0, input_len:]
    return processor.decode(new_tokens, skip_special_tokens=False)
