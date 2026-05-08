from __future__ import annotations

from typing import Any

from .config import SYSTEM_PROMPT
from .modeling import apply_chat_template_text

ChatMessage = dict[str, str]


def user_prompt_messages(user_message: str, *, system_prompt: str = SYSTEM_PROMPT) -> list[ChatMessage]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]


def assistant_target_messages(
    user_message: str,
    assistant_target_text: str,
    *,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[ChatMessage]:
    return user_prompt_messages(user_message, system_prompt=system_prompt) + [
        {"role": "assistant", "content": assistant_target_text}
    ]


def prompt_for_user(
    processor: Any,
    user_message: str,
    *,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    return apply_chat_template_text(
        processor,
        user_prompt_messages(user_message, system_prompt=system_prompt),
        add_generation_prompt=True,
        enable_thinking=False,
    )


def render_messages_for_generation(processor: Any, messages: list[ChatMessage]) -> str:
    return apply_chat_template_text(
        processor,
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def assistant_span(
    processor: Any,
    user_message: str,
    assistant_target_text: str,
    *,
    system_prompt: str = SYSTEM_PROMPT,
) -> tuple[str, int, int]:
    prompt_messages = user_prompt_messages(user_message, system_prompt=system_prompt)
    full_messages = assistant_target_messages(
        user_message,
        assistant_target_text,
        system_prompt=system_prompt,
    )

    prompt_only_text = apply_chat_template_text(
        processor,
        prompt_messages,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full_text = apply_chat_template_text(
        processor,
        full_messages,
        add_generation_prompt=False,
        enable_thinking=False,
    )

    prompt_ids = processor(text=prompt_only_text, return_tensors="pt")["input_ids"]
    full_ids = processor(text=full_text, return_tensors="pt")["input_ids"]
    start = int(prompt_ids.shape[-1])
    end = int(full_ids.shape[-1])
    if start >= end:
        raise RuntimeError(f"Assistant token span is empty for user message: {user_message!r}")
    return full_text, start, end
