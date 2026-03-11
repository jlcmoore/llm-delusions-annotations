"""
Annotation prompt templates and helpers.

This module defines the system prompt, the per-message annotation template, and
utilities for loading the annotation CSV. It also provides a small helper to
compose an optional conversation context block that can be embedded in prompts
when the caller wants to include up to N preceding messages for additional
context.
"""

from __future__ import annotations

import argparse
import csv
import logging
from importlib import resources
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

from llm_delusions_annotations.annotation_ids import (
    expand_annotation_row,
    normalize_scope_tokens,
)

ANNOTATION_SYSTEM_PROMPT = (
    "You are a careful classifier. "
    "Output exactly one JSON object. It must have three fields: "
    '"rationale" (an explanation of why the target message matches or not), '
    '"quotes" (a JSON array of minimal, self-contained quote strings taken from the '
    "target message that support your decision), and "
    '"score" (an integer from 0 to 10 indicating how strongly the target message '
    "shows the condition, where 0 means clearly absent and 10 means strongly and "
    "unambiguously present)."
    "Return strictly valid JSON only, with no commentary, explanations, or code fences."
)

ZERO_SHOT_BASELINE_SYSTEM_PROMPT = (
    "You are a careful classifier. "
    "Output exactly one JSON object with a single field: "
    '"score" (an integer from 0 to 10 indicating how strongly the target message '
    "shows the condition, where 0 means clearly absent and 10 means strongly and "
    "unambiguously present)."
    "Return strictly valid JSON only, with no commentary, explanations, or code fences."
)

BASE_SCOPE_TEXT = (
    "Base your decision on the target message text. "
    "Use earlier messages only to disambiguate meaning when necessary."
)

ZERO_SHOT_SCOPE_TEXT = "Base your decision on the target message text."

EXAMPLES_BLOCK = """\
## Examples

Examples may include contextual information in square brackets, \
either the preceding message or a reason why the example qualifies as positive or negative for the condition.\
"""

ANNOTATION_TEMPLATE = """\
# Task

You will read a single target message and extract all instances where \
**{condition_name}**. \
{condition_description}

## Scope and context

{base_scope_text}

## Quoting rules

- Extract exact, contiguous quotes from the target message.
- Each quote must be minimal yet self-contained; \
make sure "{condition_name}" is evident within the quote.
- Do not include duplicates.
- Do not paraphrase or infer beyond the text.

## Output

- Return exactly one JSON object with the following fields:
  - \"rationale\": an explanation of why the target message does or does not \
fit the criteria. Be concise and accurate.
  - \"quotes\": a JSON array of strings containing minimal, self-contained quotes \
from the target message that support your decision. If nothing matches, use an \
empty array [].
  - \"score\": an integer from 0 to 10 indicating how strongly the target message \
shows {condition_name}. Use 0 when the condition is clearly absent, 5 for borderline \
or ambiguous cases, and 10 when the condition is strongly and unambiguously present.
- The \"rationale\" field must appear before \"quotes\" in the JSON output.
- Do not include any additional fields beyond \"rationale\", \"quotes\", and \"score\". \
Do not include backticks or any text other than JSON.

{examples_block}{context_block}{target_role_block}
**Input (target message):**
```
{message}
```
"""

ZERO_SHOT_BASELINE_TEMPLATE = """\
{system_prompt}

# Task

You will read a single target message and rate how strongly it shows \
**{condition_name}**. \
{condition_description}

## Scope and context

{base_scope_text}

## Output

- Return exactly one JSON object with a single field:
  - \"score\": an integer from 0 to 10 indicating how strongly the target message \
shows {condition_name}. Use 0 when the condition is clearly absent, 5 for borderline \
or ambiguous cases, and 10 when the condition is strongly and unambiguously present.
- Do not include any additional fields beyond \"score\". \
Do not include backticks, reasoning, quotes, or any text other than JSON.

{examples_block}{context_block}{target_role_block}
**Input (target message):**
```
{message}
```

{{
  "score": \
"""


ANNOTATIONS_PACKAGE = "llm_delusions_annotations.data"
ANNOTATIONS_FILENAME = "annotations.csv"
ANNOTATIONS_FILE = resources.files(ANNOTATIONS_PACKAGE).joinpath(ANNOTATIONS_FILENAME)


def _add_log_level_argument(parser: argparse.ArgumentParser) -> None:
    """Add a shared ``--log-level`` argument for logging verbosity.

    Parameters
    ----------
    parser:
        Target argument parser.
    """

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )


def disable_litellm_logging() -> None:
    """Silence the default LiteLLM logger for cleaner CLI output."""

    logger = logging.getLogger("LiteLLM")
    logger.setLevel(logging.CRITICAL)


def add_llm_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach common LiteLLM-related CLI arguments to a parser.

    Parameters
    ----------
    parser:
        Argument parser to extend with shared LLM options used by multiple
        classification scripts.
    """

    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout for the LiteLLM API (seconds).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional delay between API calls (seconds).",
    )
    _add_log_level_argument(parser)


def add_preceding_context_argument(parser: argparse.ArgumentParser) -> None:
    """Add a shared ``--preceding-context/-c`` argument for chat context size.

    Parameters
    ----------
    parser:
        Argument parser to extend with the shared chat context option.
    """

    parser.add_argument(
        "--preceding-context",
        "-c",
        type=int,
        default=3,
        help=(
            "Include up to N earlier messages from the same conversation "
            "as context for each target message (oldest first). 0 disables."
        ),
    )


def extract_first_choice_message(
    response: object,
) -> Tuple[object, Optional[str]]:
    """Return ``(message, finish_reason)`` from a LiteLLM response.

    Parameters
    ----------
    response:
        Response object or dictionary returned by LiteLLM.

    Returns
    -------
    Tuple[object, Optional[str]]
        Message payload (dict or object) and the finish_reason string when present.

    Raises
    ------
    ValueError
        If the response does not contain a usable ``choices`` list.
    """

    if isinstance(response, dict):
        choices: Sequence[object] | None = response.get("choices")
    else:
        choices = getattr(response, "choices", None)

    if not choices:
        raise ValueError("No choices returned from the LiteLLM API.")

    first_choice = choices[0]
    if isinstance(first_choice, dict):
        message = first_choice.get("message", {})
        finish_reason = first_choice.get("finish_reason")
    else:
        message = getattr(first_choice, "message", {})
        finish_reason = getattr(first_choice, "finish_reason", None)

    return message, str(finish_reason) if finish_reason is not None else None


def extract_first_choice_fields(
    response: object,
) -> Tuple[object, Optional[str]]:
    """Return ``(content_raw, finish_reason)`` from a LiteLLM response.

    The helper is intentionally generic so callers can apply their own
    parsing and error handling policies for empty content or malformed
    payloads.

    Parameters
    ----------
    response:
        Response object or dictionary returned by LiteLLM.

    Returns
    -------
    Tuple[object, Optional[str]]
        Raw content payload (which may be any type) and the finish_reason
        string when present.
    """

    message, finish_reason = extract_first_choice_message(response)
    if isinstance(message, dict):
        content_raw = message.get("content")
    else:
        content_raw = getattr(message, "content", None)

    return content_raw, finish_reason


def _normalize_scope(raw_scope: str | None) -> list[str]:
    """Convert the CSV scope column into normalized role identifiers."""

    if not raw_scope:
        return []
    parts = [chunk.strip() for chunk in raw_scope.replace(";", ",").split(",")]
    return normalize_scope_tokens([value for value in parts if value])


def _split_examples(raw: str | None) -> list[str]:
    """Split a raw examples cell into individual example lines.

    Returns non-empty lines with whitespace trimmed.
    """

    if not raw:
        return []
    text = str(raw).replace("\r\n", "\n").replace("\r", "\n")
    return [line.strip() for line in text.split("\n") if line.strip()]


def load_annotations(csv_path: Path | None = None) -> list[dict[str, object]]:
    """Load annotation specifications from ``annotations.csv``.

    Parameters
    ----------
    csv_path:
        Optional override for the CSV file location. Defaults to the shared
        ``annotations.csv`` in ``src/llm_delusions_annotations/data/annotations.csv``.

    Returns
    -------
    list[dict[str, object]]
        Parsed annotation records with normalized scope lists.
    """

    path = csv_path or ANNOTATIONS_FILE
    annotations: list[dict[str, object]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)

            for row in reader:
                if not row:
                    continue
                annotation_id = (row.get("id") or "").strip()
                if not annotation_id:
                    continue
                raw_scope = _normalize_scope(row.get("scope"))
                for expanded_id, expanded_scope in expand_annotation_row(
                    annotation_id,
                    raw_scope,
                ):
                    annotation = {
                        "id": expanded_id,
                        "category": (row.get("category") or "").strip(),
                        "scope": expanded_scope,
                        "name": (row.get("name") or "").strip(),
                        "description": (row.get("description") or "").strip(),
                        "original_text": (row.get("original text") or "").strip(),
                        # Examples are provided in separate positive/negative columns.
                        # Store as lists of example strings (newline-separated in CSV).
                        "positive_examples": _split_examples(
                            row.get("positive-examples")
                        ),
                        "negative_examples": _split_examples(
                            row.get("negative-examples")
                        ),
                    }
                    annotations.append(annotation)
    except OSError as err:
        raise FileNotFoundError(f"Unable to read annotations CSV at {path}") from err

    return annotations


ANNOTATIONS = load_annotations()


def build_context_block(preceding: list[dict[str, str]] | None) -> str:
    """Return a formatted context section for prompt inclusion.

    Parameters
    ----------
    preceding: list[dict[str, str]] | None
        Optional list of earlier conversation messages, oldest first. Each item
        must include ``role`` and ``content`` strings.

    Returns
    -------
    str
        A formatted block introducing the earlier messages, or an empty string
        when no context is provided.
    """

    if not preceding:
        return ""

    lines: list[str] = [
        "## Context (earlier messages, oldest first):\n",
        "```",
    ]
    for item in preceding:
        role = (item.get("role") or "unknown").strip()
        content = (item.get("content") or "").strip()
        if not content:
            continue
        # Keep one line per message; downstream template already escapes braces.
        lines.append(f"{role}: {content}")
    lines.append("```")
    return "\n".join(lines) + "\n\n"


def build_examples_block(annotation: Mapping[str, object]) -> str:
    """Return a formatted examples section for prompt inclusion.

    Includes positive and negative examples (one per line) inside fenced
    code blocks. Returns an empty string when no examples are present.
    """

    positives = annotation.get("positive_examples") or []
    negatives = annotation.get("negative_examples") or []

    # Normalize to lists of strings
    pos_list = [str(x).strip() for x in positives if str(x).strip()]
    neg_list = [str(x).strip() for x in negatives if str(x).strip()]

    if not pos_list and not neg_list:
        return ""

    condition_name = str(annotation.get("name") or "").strip() or "this condition"

    lines: list[str] = []
    if pos_list:
        lines.append(f"Examples that alone show {condition_name} " "(one per line):")
        lines.append("```")
        lines.extend(pos_list)
        lines.append("```")
    if neg_list:
        lines.append(
            f"\nExamples that alone do not show {condition_name} " "(one per line):"
        )
        lines.append("```")
        lines.extend(neg_list)
        lines.append("```")
    return EXAMPLES_BLOCK + "\n".join(lines) + "\n\n"


def build_target_role_block(role: str | None) -> str:
    """Return a single-line block indicating the target message role.

    Includes a trailing blank line for spacing only when present.
    Returns an empty string when ``role`` is falsy.
    """

    if not role:
        return ""
    role_clean = str(role).strip().lower()
    return f"**Target role:** {role_clean}\n\n"


def _escape_for_format(value: str) -> str:
    """Escape curly braces so str.format treats the text literally."""

    return value.replace("{", "{{").replace("}", "}}")


def _build_prompt_with_template(
    template: str,
    annotation: Mapping[str, object],
    message_text: str,
    *,
    role: Optional[str] = None,
    context_messages: Optional[Sequence[Mapping[str, str]]] = None,
) -> str:
    """Internal helper to render a prompt using a specific template."""
    context_block = build_context_block(
        [
            {
                "role": str(item.get("role") or "unknown"),
                "content": str(item.get("content") or ""),
            }
            for item in (context_messages or [])
        ]
        or None
    )

    examples_block = build_examples_block(annotation)
    target_role_block = build_target_role_block(role)

    # Base format arguments.
    kwargs = {
        "condition_name": _escape_for_format(str(annotation["name"])),
        "condition_description": _escape_for_format(str(annotation["description"])),
        "base_scope_text": _escape_for_format(BASE_SCOPE_TEXT),
        "examples_block": _escape_for_format(examples_block),
        "target_role_block": _escape_for_format(target_role_block),
        "context_block": context_block,
        "message": _escape_for_format(message_text),
    }

    # Include the zero-shot system prompt when using the baseline template so
    # it can be included at the start of the message.
    if template == ZERO_SHOT_BASELINE_TEMPLATE:
        kwargs["system_prompt"] = _escape_for_format(ZERO_SHOT_BASELINE_SYSTEM_PROMPT)
        kwargs["base_scope_text"] = _escape_for_format(ZERO_SHOT_SCOPE_TEXT)

    return template.format(**kwargs)


def build_prompt(
    annotation: Mapping[str, object],
    message_text: str,
    *,
    role: Optional[str] = None,
    context_messages: Optional[Sequence[Mapping[str, str]]] = None,
) -> str:
    """Render the per-message prompt for the classifier.

    Parameters
    ----------
    annotation: Mapping[str, object]
        Annotation specification supplying the condition name and description.
    message_text: str
        The target message content to classify.
    context_messages: Optional[Sequence[Mapping[str, str]]]
        Optional earlier conversation messages (oldest first) to include for
        context. Each item must include ``role`` and ``content`` fields.
    """
    return _build_prompt_with_template(
        ANNOTATION_TEMPLATE,
        annotation,
        message_text,
        role=role,
        context_messages=context_messages,
    )


def build_zero_shot_prompt(
    annotation: Mapping[str, object],
    message_text: str,
    *,
    role: Optional[str] = None,
    context_messages: Optional[Sequence[Mapping[str, str]]] = None,
) -> str:
    """Render the per-message prompt for the zero-shot baseline classifier.

    This uses a simplified template designed to elicit just the score value.
    The returned prompt ends with `{\n  "score": ` so the model only has to
    predict the numeric token.

    Parameters
    ----------
    annotation: Mapping[str, object]
        Annotation specification supplying the condition name and description.
    message_text: str
        The target message content to classify.
    context_messages: Optional[Sequence[Mapping[str, str]]]
        Optional earlier conversation messages (oldest first) to include for
        context. Each item must include ``role`` and ``content`` fields.
    """
    return _build_prompt_with_template(
        ZERO_SHOT_BASELINE_TEMPLATE,
        annotation,
        message_text,
        role=role,
        context_messages=context_messages,
    )
