"""
Shared LiteLLM client helpers.

This module centralizes error handling, retry configuration, and reasoning
defaults for LiteLLM calls so that all LLM-using code paths behave
consistently.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

import litellm
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)

from llm_delusions_annotations.annotation_prompts import extract_first_choice_message

LITELLM_API_ERRORS = (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
    TimeoutError,
)

DEFAULT_CHAT_MODEL = "gpt-5.1-2025-11-13"
"""Default chat model identifier used across scripts."""

DEFAULT_NUM_RETRIES = 5
DEFAULT_RETRY_STRATEGY = "exponential_backoff_retry"

# Apply conservative safety overrides for Vertex AI calls to reduce empty responses.
litellm.vertex_ai_safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]


class LLMClientError(RuntimeError):
    """Raised when a LiteLLM request fails after applying retry policies.

    Parameters
    ----------
    message:
        High level description of the failure.
    inner:
        Optional underlying LiteLLM exception that triggered the failure.

    Attributes
    ----------
    inner:
        Saved underlying exception instance when available.
    """

    def __init__(self, message: str, *, inner: Optional[BaseException] = None) -> None:
        super().__init__(message)
        self.inner = inner


def apply_reasoning_defaults(
    model: str,
    params: dict[str, object],
    *,
    max_completion_tokens: Optional[int] = None,
) -> None:
    """Populate reasoning-related defaults into ``params`` for LiteLLM calls.

    For reasoning-capable models this sets temperature and reasoning_effort.
    For non-reasoning models this clamps temperature to 0 and optionally
    constrains max_tokens when a completion cap is provided.

    Parameters
    ----------
    model:
        LiteLLM model identifier to inspect.
    params:
        Mutable dictionary of request parameters to update in place.
    max_completion_tokens:
        Optional completion token cap applied for non-reasoning models when
        provided.
    """

    if litellm.supports_reasoning(model):
        params.setdefault("temperature", 1)
        model_lower = model.lower()
        level = "minimal"
        if "gpt-5.1" in model_lower:
            level = "none"
        elif "gemini-2.5-flash" in model_lower:
            level = "none"
        elif "gemini-3-flash" in model_lower:
            level = "minimal"
        params.setdefault("reasoning_effort", level)
    else:
        params.setdefault("temperature", 0)
        if max_completion_tokens is not None:
            params.setdefault("max_tokens", max_completion_tokens)


def completion(
    *,
    model: str,
    messages: Sequence[Mapping[str, object]],
    timeout: Optional[int] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    n: int = 1,
    num_retries: int = DEFAULT_NUM_RETRIES,
    retry_strategy: str = DEFAULT_RETRY_STRATEGY,
    enable_reasoning_defaults: bool = False,
    max_completion_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
) -> object:
    """Call :func:`litellm.completion` with shared retry and error handling.

    Parameters
    ----------
    model:
        LiteLLM model identifier to use.
    messages:
        Chat messages payload in LiteLLM-compatible dict form.
    timeout:
        Optional request timeout in seconds.
    max_tokens:
        Optional maximum number of completion tokens.
    temperature:
        Optional sampling temperature override.
    n:
        Number of completion variants to request.
    num_retries:
        Number of retries for transient errors.
    retry_strategy:
        LiteLLM retry strategy name.
    enable_reasoning_defaults:
        When True, apply reasoning defaults for the given model.
    max_completion_tokens:
        Optional completion token cap used when reasoning defaults are applied.
    reasoning_effort:
        Optional reasoning effort override passed to LiteLLM (for reasoning-capable
        models). When set, this overrides any default applied by
        ``enable_reasoning_defaults``.

    Returns
    -------
    object
        The raw LiteLLM completion response.

    Raises
    ------
    LLMClientError
        If the LiteLLM request ultimately fails.
    """

    request_kwargs: dict[str, object] = {
        "model": model,
        "messages": list(messages),
        "n": int(n),
        "num_retries": int(num_retries),
        "retry_strategy": retry_strategy,
    }
    if timeout is not None:
        request_kwargs["timeout"] = int(timeout)
    if max_tokens is not None:
        request_kwargs["max_tokens"] = int(max_tokens)
    if temperature is not None:
        request_kwargs["temperature"] = float(temperature)

    if enable_reasoning_defaults:
        apply_reasoning_defaults(
            model,
            request_kwargs,
            max_completion_tokens=max_completion_tokens,
        )
    if reasoning_effort is not None:
        request_kwargs["reasoning_effort"] = reasoning_effort

    try:
        return litellm.completion(**request_kwargs)
    except LITELLM_API_ERRORS as err:
        raise LLMClientError(f"LiteLLM completion failed: {err}", inner=err) from err


def batch_completion(
    messages: Iterable[Sequence[Mapping[str, object]]],
    *,
    model: str,
    timeout: int,
    max_workers: int,
    num_retries: int = DEFAULT_NUM_RETRIES,
    retry_strategy: str = DEFAULT_RETRY_STRATEGY,
    enable_reasoning_defaults: bool = False,
    max_completion_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
) -> list[object]:
    """Call :func:`litellm.batch_completion` with shared retry and error handling.

    Parameters
    ----------
    messages:
        Iterable of message sequences, one per request.
    model:
        LiteLLM model identifier to use.
    timeout:
        Request timeout in seconds.
    max_workers:
        Maximum number of concurrent worker threads.
    num_retries:
        Number of retries for transient errors.
    retry_strategy:
        LiteLLM retry strategy name.
    enable_reasoning_defaults:
        When True, apply reasoning defaults for the given model.
    max_completion_tokens:
        Optional completion token cap used when reasoning defaults are applied.
    reasoning_effort:
        Optional reasoning effort override passed to LiteLLM (for reasoning-capable
        models). When set, this overrides any default applied by
        ``enable_reasoning_defaults``.

    Returns
    -------
    list[object]
        List of LiteLLM batch completion responses, one per request.

    Raises
    ------
    LLMClientError
        If the batch request ultimately fails.
    """

    batch_kwargs: dict[str, object] = {
        "model": model,
        "timeout": int(timeout),
        "max_workers": int(max_workers),
        "num_retries": int(num_retries),
        "retry_strategy": retry_strategy,
    }

    if enable_reasoning_defaults:
        apply_reasoning_defaults(
            model,
            batch_kwargs,
            max_completion_tokens=max_completion_tokens,
        )
    if reasoning_effort is not None:
        batch_kwargs["reasoning_effort"] = reasoning_effort

    try:
        responses = litellm.batch_completion(
            messages=list(messages),
            **batch_kwargs,
        )
    except LITELLM_API_ERRORS as err:
        raise LLMClientError(f"LiteLLM batch request failed: {err}", inner=err) from err

    return list(responses)


def extract_reasoning_fields(
    response: object,
) -> tuple[Optional[str], Optional[list[dict[str, Any]]]]:
    """Extract reasoning content and thinking blocks from a LiteLLM response."""
    try:
        message, _finish_reason = extract_first_choice_message(response)
    except ValueError:
        return None, None

    if isinstance(message, dict):
        reasoning_content = message.get("reasoning_content")
        thinking_blocks = message.get("thinking_blocks")
    else:
        reasoning_content = getattr(message, "reasoning_content", None)
        thinking_blocks = getattr(message, "thinking_blocks", None)

    if isinstance(thinking_blocks, list):
        normalized_blocks: Optional[list[dict[str, Any]]] = []
        for item in thinking_blocks:
            if isinstance(item, dict):
                normalized_blocks.append(item)
        thinking_blocks = normalized_blocks
    else:
        thinking_blocks = None

    if isinstance(reasoning_content, str):
        reasoning_content = reasoning_content.strip() or None
    else:
        reasoning_content = None

    return reasoning_content, thinking_blocks
