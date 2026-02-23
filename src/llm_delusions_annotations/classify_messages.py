"""Helpers to classify individual chat messages using LiteLLM."""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

import json_repair

from llm_delusions_annotations.annotation_prompts import (
    ANNOTATION_SYSTEM_PROMPT,
    extract_first_choice_fields,
    split_thought_from_response,
)
from llm_delusions_annotations.chat.chat_utils import MessageContext
from llm_delusions_annotations.llm_utils.client import (
    LLMClientError,
    batch_completion,
    extract_reasoning_fields,
)
from llm_delusions_annotations.utils import MessagesPayload

MAX_CLASSIFICATION_TOKENS = 256


class ClassificationError(Exception):
    """Raised when a message cannot be classified successfully."""


ConversationKey = tuple[str, Path, str, int]

_WHITESPACE_PATTERN = re.compile(r"\s+")
_WORD_PATTERN = re.compile(r"[a-z0-9']+")


def parse_optional_int(value: object) -> Optional[int]:
    """Return an integer parsed from ``value`` or ``None`` when unusable."""

    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
    return None


@dataclass(frozen=True)
class ClassificationTask:
    """Single prompt submission bound to a message context and annotation."""

    context: MessageContext
    annotation: dict[str, object]
    prompt: str


@dataclass(frozen=True)
class ClassificationOutcome:
    """Result payload for a completed classification task."""

    task: ClassificationTask
    matches: List[str]
    error: Optional[str]
    thought: Optional[str] = None
    score: Optional[int] = None
    reasoning_content: Optional[str] = None
    thinking_blocks: Optional[List[dict[str, Any]]] = None


def build_completion_messages(
    prompt: str, *, system_prompt: str = ANNOTATION_SYSTEM_PROMPT
) -> MessagesPayload:
    """Construct the chat messages sent to LiteLLM."""

    augmented_system_prompt = (
        f"{system_prompt.rstrip()}\n\n"
        f"Do not use more than {MAX_CLASSIFICATION_TOKENS} tokens in your response."
    )
    return (
        {"role": "system", "content": augmented_system_prompt},
        {"role": "user", "content": prompt},
    )


def to_litellm_messages(
    messages: MessagesPayload,
) -> List[dict[str, str]]:
    """Convert completion messages to LiteLLM-compatible dictionaries."""

    return [dict(message) for message in messages]


def extract_matches_from_response(
    response: object,
) -> tuple[Optional[str], List[str], Optional[int]]:
    """Parse a LiteLLM completion response into a scratchpad, quotes, and score.

    Returns the extracted chain-of-thought text (when present), the parsed JSON
    list of quote strings, and an optional numeric score when provided.
    """
    try:
        content_raw, finish_reason = extract_first_choice_fields(response)
    except ValueError as err:
        raise ClassificationError(f"{err} Response: {response}.") from err

    content = str(content_raw or "").strip()
    if not content:
        if finish_reason == "stop":
            logging.warning(
                "Empty content with finish_reason='stop'; assuming empty matches list. "
                "Raw message content: %r",
                content_raw,
            )
            content = "[]"
        else:
            raise ClassificationError(
                f"Empty response from the LiteLLM API. Raw: {content_raw}"
            )
    return extract_matches_from_response_text(content)


def extract_matches_from_response_text(
    content: str,
) -> tuple[Optional[str], List[str], Optional[int]]:
    """Parse a LiteLLM completion response into a scratchpad, quotes, and score.

    Returns the extracted chain-of-thought text (when present), the parsed JSON
    list of quote strings, and an optional numeric score when provided.
    """
    thought, content_for_parsing = split_thought_from_response(content)

    try:
        parsed = json_repair.loads(content_for_parsing)
    except (json.JSONDecodeError, ValueError, TypeError, IndexError) as err:
        raise ClassificationError(
            f"Unable to parse model response as JSON: {err}"
        ) from err

    if not isinstance(parsed, dict):
        raise ClassificationError(
            "Model response must be a JSON object with either both "
            "'score' and 'quotes' fields or be an empty object {}. "
            f"Response: {content}"
        )

    # Empty object {} is allowed when the condition is clearly absent and there
    # are no supporting quotes.
    if not parsed:
        return thought, [], None

    matches: List[str] = []
    score_value: Optional[int] = None

    if "quotes" not in parsed or "score" not in parsed:
        raise ClassificationError(
            "Model response must include both 'score' and 'quotes' fields "
            "when it is not empty. "
            f"Response: {content}"
        )

    raw_quotes = parsed.get("quotes")
    if isinstance(raw_quotes, list):
        for item in raw_quotes:
            if isinstance(item, str) and item.strip():
                matches.append(item.strip())
    else:
        raise ClassificationError(
            "Model response 'quotes' field must be a JSON array of strings. "
            f"Response: {content}"
        )

    raw_score = parsed.get("score")
    if isinstance(raw_score, (int, float)):
        score_as_float = float(raw_score)
        rounded = int(round(score_as_float))
        if rounded < 0 or rounded > 10:
            logging.warning(
                "Score %r is outside [0, 10]; clamping into range.",
                raw_score,
            )
            rounded = max(0, min(10, rounded))
        score_value = rounded
    else:
        raise ClassificationError(
            "Model response 'score' field must be a number between 0 and 10. "
            f"Response: {content}"
        )

    return thought, matches, score_value


@dataclass
class ClassifyResult:
    """Parsed classification outputs for a single request."""

    matches: List[str]
    error: Optional[str]
    thought: Optional[str] = None
    score: Optional[int] = None
    reasoning_content: Optional[str] = None
    thinking_blocks: Optional[List[dict[str, Any]]] = None


def make_classify_requests(
    requests: List,
    *,
    model: str,
    timeout: int,
    max_workers: int,
) -> List[ClassifyResult]:
    """Submit batch completion requests and return parsed classification results."""
    if not requests:
        return []

    try:
        response_items = batch_completion(
            messages=requests,
            model=model,
            timeout=timeout,
            max_workers=max_workers,
            enable_reasoning_defaults=True,
            max_completion_tokens=MAX_CLASSIFICATION_TOKENS,
        )
    except LLMClientError as err:
        raise ClassificationError(f"LiteLLM batch request failed: {err}") from err

    if len(response_items) != len(requests):
        raise ClassificationError(
            "LiteLLM returned a different number of responses than requested."
        )

    results: List[ClassificationOutcome] = []
    for response in response_items:
        try:
            thought, matches, score = extract_matches_from_response(response)
            reasoning_content, thinking_blocks = extract_reasoning_fields(response)
            error_details: Optional[str] = None
        except ClassificationError as err:
            matches = []
            thought = None
            error_details = str(err)
            reasoning_content = None
            thinking_blocks = None
        results.append(
            ClassifyResult(
                matches=matches,
                error=error_details,
                thought=thought,
                score=score if error_details is None else None,
                reasoning_content=reasoning_content,
                thinking_blocks=thinking_blocks,
            )
        )
    return results


def classify_tasks_batch(
    tasks: Sequence[ClassificationTask],
    *,
    model: str,
    timeout: int,
    max_workers: int,
    system_prompt: str = ANNOTATION_SYSTEM_PROMPT,
) -> List[ClassificationOutcome]:
    """Submit tasks to LiteLLM using batch completion and return parsed outcomes."""
    requests = [
        to_litellm_messages(
            build_completion_messages(task.prompt, system_prompt=system_prompt)
        )
        for task in tasks
    ]
    classify_results = make_classify_requests(
        requests, model=model, timeout=timeout, max_workers=max_workers
    )

    outcomes: List[ClassificationOutcome] = []
    for task, classify_result in zip(tasks, classify_results):
        outcomes.append(
            ClassificationOutcome(
                task=task,
                matches=classify_result.matches,
                error=classify_result.error,
                thought=classify_result.thought,
                score=classify_result.score,
                reasoning_content=classify_result.reasoning_content,
                thinking_blocks=classify_result.thinking_blocks,
            )
        )
    return outcomes


def _normalize_for_quote_match(value: str) -> str:
    """Return a normalized representation of text for quote matching."""

    collapsed = _WHITESPACE_PATTERN.sub(" ", value)
    return collapsed.casefold().strip()


def _tokenize_for_quote_match(value: str) -> List[str]:
    """Return lowercased alphanumeric tokens for quote comparison."""

    return _WORD_PATTERN.findall(value.casefold())


def _tokens_form_subsequence(
    content_tokens: Sequence[str],
    candidate_tokens: Sequence[str],
) -> bool:
    """Return True when ``candidate_tokens`` appear contiguously in ``content_tokens``."""

    window = len(candidate_tokens)
    if window == 0 or window > len(content_tokens):
        return False
    for index in range(len(content_tokens) - window + 1):
        if list(content_tokens[index : index + window]) == list(candidate_tokens):
            return True
    return False


def find_unmatched_quotes(matches: Sequence[str], content: str) -> List[str]:
    """Return quotes that do not exist in ``content`` (case-insensitive)."""

    content_baseline = content.casefold()
    content_normalized = _normalize_for_quote_match(content)
    content_tokens = _tokenize_for_quote_match(content)
    unmatched: List[str] = []
    for match in matches:
        candidate = match.strip()
        if not candidate:
            continue
        candidate_casefold = candidate.casefold()
        if candidate_casefold and candidate_casefold in content_baseline:
            continue
        normalized = _normalize_for_quote_match(candidate)
        if normalized and normalized in content_normalized:
            continue
        candidate_tokens = _tokenize_for_quote_match(candidate)
        if candidate_tokens and _tokens_form_subsequence(
            content_tokens,
            candidate_tokens,
        ):
            continue
        unmatched.append(match)
    return unmatched


def filter_quotes_to_content(matches: Sequence[str], content: str) -> List[str]:
    """Return only quotes that appear within ``content``."""

    content_baseline = content.casefold()
    content_normalized = _normalize_for_quote_match(content)
    content_tokens = _tokenize_for_quote_match(content)
    filtered: List[str] = []
    for match in matches:
        candidate = match.strip()
        if not candidate:
            continue
        candidate_casefold = candidate.casefold()
        if candidate_casefold and candidate_casefold in content_baseline:
            filtered.append(match)
            continue
        normalized = _normalize_for_quote_match(candidate)
        if normalized and normalized in content_normalized:
            filtered.append(match)
            continue
        candidate_tokens = _tokenize_for_quote_match(candidate)
        if candidate_tokens and _tokens_form_subsequence(
            content_tokens,
            candidate_tokens,
        ):
            filtered.append(match)
            continue
    return filtered
