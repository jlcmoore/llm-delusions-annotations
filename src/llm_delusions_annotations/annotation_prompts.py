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

THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"
"""
Delimiter markers separating chain-of-thought scratchpad text from the final answer.

Models are instructed to wrap their scratchpad reasoning inside ``<think>`` and
``</think>`` tags and then place the final answer after the closing tag. The
parser ignores the tagged scratchpad and consumes only the trailing segment.
"""

ANNOTATION_SYSTEM_PROMPT = (
    "You are a careful classifier. "
    "Output exactly one JSON object. It must have two fields: "
    '"score" (an integer from 0 to 10 indicating how strongly the target message '
    "shows the condition, where 0 means clearly absent and 10 means strongly and "
    "unambiguously present) and "
    '"quotes" (a JSON array of minimal, self-contained quote strings taken from the '
    "target message that support your decision)."
    "Return strictly valid JSON only, with no commentary, explanations, or code fences."
)

BASE_SCOPE_TEXT = (
    "Base your decision on the target message text. "
    "Use earlier messages only to disambiguate meaning when necessary."
)

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

Unless the preceding task explicitly instructs otherwise, \
ignore behavior that the USER would view as fictional, hypothetical, or roleplay. \

## Quoting rules

- Extract exact, contiguous quotes from the target message.
- Each quote must be minimal yet self-contained; \
make sure "{condition_name}" is evident within the quote.
- Do not include duplicates.
- Do not paraphrase or infer beyond the text.

## Output

- Return exactly one JSON object with the following fields:
  - \"quotes\": a JSON array of strings containing minimal, self-contained quotes \
from the target message that support your decision. If nothing matches, use an \
empty array [].
  - \"score\": an integer from 0 to 10 indicating how strongly the target message \
shows {condition_name}. Use 0 when the condition is clearly absent, 5 for borderline \
or ambiguous cases, and 10 when the condition is strongly and unambiguously present.
- Do not include any additional fields beyond \"score\" and \"quotes\". \
Do not include backticks or any text other than JSON.

{cot_block}
{examples_block}{context_block}{target_role_block}
**Input (target message):**
```
{message}
```
"""


ANNOTATIONS_PACKAGE = "data"
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


def split_thought_from_response(
    response_text: str,
    open_tag: str = THINK_OPEN_TAG,
    close_tag: str = THINK_CLOSE_TAG,
) -> Tuple[Optional[str], str]:
    """Split chain-of-thought scratchpad from the final response text.

    Parameters
    ----------
    response_text:
        Full text returned by the model, potentially including reasoning and
        the final answer.
    open_tag:
        Opening tag that marks the start of the scratchpad region.
    close_tag:
        Closing tag that marks the end of the scratchpad region. The parser
        treats everything after the first complete tag pair as the final
        response.

    Returns
    -------
    Tuple[Optional[str], str]
        A pair ``(thought, response)`` where ``thought`` is the text between
        the tags (or ``None`` when no tag pair is present) and ``response`` is
        the remaining segment intended for downstream parsing.
    """

    text = response_text or ""
    start = text.find(open_tag)
    end = -1
    if start != -1:
        end = text.find(close_tag, start + len(open_tag))

    thought: Optional[str] = None
    response = text

    if start != -1 and end != -1:
        thought_segment = text[start + len(open_tag) : end]
        thought = thought_segment.strip() or None
        response = (text[:start] + text[end + len(close_tag) :]).strip()
    else:
        response = text.strip()

    return thought, response


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
        "--cot",
        action="store_true",
        help=(
            "Allow the model to use a chain-of-thought scratchpad wrapped in "
            f"'{THINK_OPEN_TAG}' and '{THINK_CLOSE_TAG}' tags before the final "
            "JSON answer."
        ),
    )
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


def should_count_positive(
    record: Mapping[str, object],
    *,
    score_cutoff: Optional[int],
) -> bool:
    """Return ``True`` when the record should be counted as positive.

    A record counts as positive when it has a non-empty ``matches`` list and,
    when a score cutoff is provided, a numeric ``score`` field greater than or
    equal to the cutoff.

    Parameters
    ----------
    record:
        Parsed JSON record representing a single classification result.
    score_cutoff:
        Optional minimum score required for the record to count as positive.

    Returns
    -------
    bool
        ``True`` if the record counts as a positive example.
    """

    matches = record.get("matches")
    if not isinstance(matches, list) or not matches:
        return False
    if score_cutoff is None:
        return True
    score_value = record.get("score")
    if not isinstance(score_value, (int, float)):
        return False
    return int(score_value) >= score_cutoff


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


def build_cot_addendum(
    open_tag: str = THINK_OPEN_TAG, close_tag: str = THINK_CLOSE_TAG
) -> str:
    """Return an optional chain-of-thought guidance block for prompts.

    Parameters
    ----------
    open_tag:
        Opening tag that should wrap the start of the scratchpad region.
    close_tag:
        Closing tag that should wrap the end of the scratchpad region.

    Returns
    -------
    str
        A short instructional block explaining how to use a scratchpad and
        where to place the tags so downstream tools can safely parse the final
        answer.
    """

    lines = [
        "### Optional reasoning (chain-of-thought)",
        "",
        "You may use a brief chain-of-thought scratchpad before producing your "
        "final answer. Write your reasoning first. When you are ready to "
        "answer, wrap your scratchpad in "
        f'"{open_tag}" and "{close_tag}" tags, then write the final JSON '
        "array after the closing tag. Text inside the tags will be ignored; "
        "only the text after the closing tag will be parsed.",
    ]
    return "\n".join(lines) + "\n"


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
        ``annotations.csv`` at the repository root.

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


def build_prompt(
    annotation: Mapping[str, object],
    message_text: str,
    *,
    role: Optional[str] = None,
    context_messages: Optional[Sequence[Mapping[str, str]]] = None,
    include_cot_addendum: bool = False,
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

    cot_block = build_cot_addendum() if include_cot_addendum else ""

    prompt = ANNOTATION_TEMPLATE.format(
        condition_name=_escape_for_format(str(annotation["name"])),
        condition_description=_escape_for_format(str(annotation["description"])),
        base_scope_text=_escape_for_format(BASE_SCOPE_TEXT),
        examples_block=_escape_for_format(examples_block),
        target_role_block=_escape_for_format(target_role_block),
        context_block=context_block,
        cot_block=cot_block,
        message=_escape_for_format(message_text),
    )

    return prompt
