"""Shared helpers for annotation pipelines and tooling."""

from typing import List, Mapping, Sequence

ChatMessage = Mapping[str, str]
MessagesPayload = Sequence[ChatMessage]
AnnotationRequest = tuple[str, str, MessagesPayload]


def to_litellm_messages(messages: MessagesPayload) -> List[dict[str, object]]:
    """Return LiteLLM-compatible dictionaries for completion messages.

    Parameters
    ----------
    messages:
        Sequence of chat-style message mappings.

    Returns
    -------
    List[dict[str, object]]
        List of shallow-copied message dictionaries.
    """

    return [dict(message) for message in messages]


def is_positive_score(record: Mapping[str, object], cutoff: int) -> bool:
    """Return True when the record score meets or exceeds ``cutoff``.

    Parameters
    ----------
    record:
        Classification-like JSONL record containing a ``score`` field.
    cutoff:
        Minimum integer score required for a record to count as positive.

    Returns
    -------
    bool
        True when the record has a numeric score greater than or equal to the
        cutoff, otherwise False.
    """

    score_value = record.get("score")
    if not isinstance(score_value, (int, float)):
        return False
    return int(score_value) >= cutoff


def has_true_matches(record: Mapping[str, object], cutoff: int) -> bool:
    """Return True when a record has validated quote matches.

    A record has "true matches" when its score is at least ``cutoff``, the
    ``matches`` field is a non-empty list of strings, and every quoted string
    appears as a contiguous substring of the record's ``content`` field.

    Parameters
    ----------
    record:
        Classification-like JSONL record with ``score``, ``matches``, and
        ``content`` fields.
    cutoff:
        Minimum score required for the record to be considered positive.

    Returns
    -------
    bool
        True when all conditions for validated matches are satisfied.
    """

    if not is_positive_score(record, cutoff=cutoff):
        return False

    matches = record.get("matches")
    if not isinstance(matches, list) or not matches:
        return False

    content = record.get("content")
    content_text = str(content) if content is not None else ""
    if not content_text:
        return False

    for item in matches:
        if not isinstance(item, str):
            return False
        if item not in content_text:
            return False
    return True
