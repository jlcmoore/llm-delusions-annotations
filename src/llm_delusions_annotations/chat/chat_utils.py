"""Shared utilities for locating and loading chat JSON artefacts."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Mapping, Optional, Sequence, Tuple

from .chat_io import Chat, load_chats_for_file

BUCKET_PATTERN = re.compile(r"^(?:[a-z]+_[0-9]{2,}|[12][0-9]{2,})$", re.IGNORECASE)


@dataclass(frozen=True)
class MessageContext:
    """Metadata for a single message within a conversation.

    Parameters
    ----------
    participant:
        Participant identifier such as ``\"105\"`` or ``\"202\"``.
    source_path:
        Path to the transcript file relative to a transcripts root.
    chat_index:
        Zero-based index of the conversation within the transcript file.
    chat_key:
        Conversation title or key string.
    chat_date:
        Optional date label associated with the conversation.
    message_index:
        Zero-based index of the message within the conversation.
    role:
        Message role label such as ``\"user\"`` or ``\"assistant\"``.
    content:
        Message content text stripped of surrounding whitespace.
    timestamp:
        Optional timestamp label associated with the message.
    preceding:
        Optional list of preceding message dictionaries used for context.
    """

    participant: str
    source_path: Path
    chat_index: int
    chat_key: str
    chat_date: Optional[str]
    message_index: int
    role: str
    content: str
    timestamp: Optional[str]
    preceding: Optional[List[dict[str, str]]]


@dataclass(frozen=True)
class MessageContextData:
    """Data for a single message within a conversation.

    Parameters
    ----------
    message_index:
        Zero-based index of the message within the conversation.
    role:
        Message role label such as ``\"user\"`` or ``\"assistant\"``.
    content:
        Message content text stripped of surrounding whitespace.
    timestamp:
        Optional timestamp label associated with the message.
    preceding:
        Optional list of preceding message dictionaries used for context.
    """

    message_index: int
    role: str
    content: str
    timestamp: Optional[str]
    preceding: Optional[List[dict[str, str]]]


def iter_chat_json_files(root: Path, *, followlinks: bool = False) -> Iterator[Path]:
    """Yield JSON files beneath ``root`` that could contain chat data.

    Files are returned in a case-insensitive sorted order to provide
    deterministic processing regardless of filesystem specifics.
    """

    root_path = Path(root)
    if not root_path.exists():
        return

    candidates: List[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root_path, followlinks=followlinks):
        for name in filenames:
            if name.lower().endswith(".json"):
                candidates.append(Path(dirpath) / name)

    yield from sorted(candidates, key=lambda p: str(p).lower())


def iter_loaded_chats(
    root: Path, *, followlinks: bool = False
) -> Iterator[Tuple[Path, List[Chat]]]:
    """Iterate over chats grouped by their originating JSON file."""

    for file_path in iter_chat_json_files(root, followlinks=followlinks):
        chats = load_chats_for_file(file_path)
        if not chats:
            continue
        yield file_path, chats


def load_chats_from_directory(
    root: Path, *, followlinks: bool = False, limit: int | None = None
) -> List[Chat]:
    """Load chats from ``root`` while optionally capping the total count."""

    loaded: List[Chat] = []
    for _file_path, chats in iter_loaded_chats(root, followlinks=followlinks):
        loaded.extend(chats)
        if limit is not None and len(loaded) >= limit:
            return loaded[:limit]
    return loaded


def resolve_bucket_label(file_path: Path, root: Path) -> str | None:
    """Return the nearest ancestor directory name matching the bucket pattern."""

    resolved_file = file_path.resolve()
    resolved_root = root.resolve()

    for parent in resolved_file.parents:
        if parent == resolved_root.parent:
            break
        if parent.name and BUCKET_PATTERN.match(parent.name):
            return parent.name
    return None


def resolve_bucket_and_rel_path(
    file_path: Path,
    root: Path,
) -> Tuple[Optional[str], Path]:
    """Return a (bucket, rel_path) pair for a transcript file.

    The bucket is the nearest ancestor directory name matching the bucket
    pattern (for example, ``105`` or ``202``). When no such directory is
    found, the bucket component is ``None``. The rel_path component is the
    path to the file relative to ``root`` when possible, or the absolute path
    when the file is not under ``root``.
    """

    try:
        rel_path = file_path.relative_to(root)
    except ValueError:
        rel_path = file_path

    bucket = resolve_bucket_label(file_path, root)
    if bucket:
        bucket = bucket.strip()
    return (bucket or None, rel_path)


def normalize_optional_string(value: object) -> Optional[str]:
    """Return a stripped string value or ``None`` when unusable."""

    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def compute_previous_indices_skipping_roles(
    messages: Sequence[Mapping[str, object]],
    message_index: int,
    depth: int,
    *,
    skip_roles: Sequence[str] | None = None,
) -> List[int]:
    """Return indices for preceding messages, skipping some roles from the depth.

    Messages whose roles are listed in ``skip_roles`` are included in the
    returned indices but do not count toward the ``depth`` limit. This is
    useful when tool messages should appear in context while the budget is
    measured in user/assistant turns.

    Parameters
    ----------
    messages:
        Full message sequence for a single conversation.
    message_index:
        Zero-based index of the target message within ``messages``.
    depth:
        Maximum number of preceding messages to count for depth, excluding
        roles listed in ``skip_roles``.
    skip_roles:
        Optional collection of role names (case-insensitive) that should not
        count against ``depth``. These messages are still included between
        counted messages.

    Returns
    -------
    List[int]
        Indices of preceding messages in chronological order.

    Raises
    ------
    ValueError
        If ``message_index`` is out of range for ``messages``.
    """

    total = len(messages)
    if message_index < 0 or message_index >= total:
        raise ValueError(
            f"message_index {message_index} out of range for {total} messages"
        )

    limit = depth if depth > 0 else 0
    if limit == 0:
        return []

    skip_set = {
        str(role).strip().lower() for role in (skip_roles or []) if str(role).strip()
    }
    indices_rev: List[int] = []
    counted = 0

    for idx in range(message_index - 1, -1, -1):
        msg = messages[idx]
        role_raw = msg.get("role")
        role = str(role_raw).strip().lower() if role_raw is not None else ""

        indices_rev.append(idx)
        if role and role not in skip_set:
            counted += 1
            if counted >= limit:
                break

    indices_rev.reverse()
    return indices_rev


def build_preceding_entry(
    role: str,
    content: str,
    *,
    index: Optional[int] = None,
    timestamp: Optional[str] = None,
) -> dict[str, str]:
    """Return a normalized preceding message entry for context blocks.

    Parameters
    ----------
    role:
        Message role label such as ``\"user\"`` or ``\"assistant\"``.
    content:
        Message content text stripped of surrounding whitespace.
    index:
        Optional zero-based index of the message within its conversation.
    timestamp:
        Optional timestamp label associated with the message.

    Returns
    -------
    dict[str, str]
        Dictionary containing the preceding message fields. Always includes
        ``role`` and ``content``; includes ``index`` and ``timestamp`` when
        provided.
    """

    entry: dict[str, str] = {
        "role": role,
        "content": content,
    }
    if index is not None:
        entry["index"] = str(index)
    if timestamp is not None and timestamp.strip():
        entry["timestamp"] = timestamp.strip()
    return entry


def iter_chat_messages(
    chat_messages: Sequence[Mapping],
    *,
    allowed_roles: Optional[set[str]],
    reverse_conversations: bool,
    preceding_count: int = 0,
) -> Iterator[MessageContextData]:
    messages_sequence = list(chat_messages)
    if reverse_conversations:
        messages_sequence = list(messages_sequence)
        message_indices: Sequence[int] = range(len(messages_sequence) - 1, -1, -1)
    else:
        message_indices = range(len(messages_sequence))
    for message_index in message_indices:
        message = messages_sequence[message_index]
        content = message.get("content")
        if not isinstance(content, str):
            continue
        normalized = content.strip()
        if not normalized:
            continue
        role = str(message.get("role") or "unknown").lower()
        if allowed_roles is not None and role not in allowed_roles:
            continue
        timestamp = normalize_optional_string(message.get("timestamp"))
        preceding_messages: Optional[List[dict[str, str]]] = None
        if preceding_count and preceding_count > 0:
            indices = compute_previous_indices_skipping_roles(
                messages_sequence,
                message_index,
                preceding_count,
                skip_roles=("tool",),
            )
            if indices:
                preceding_messages = []
                for idx in indices:
                    prev = messages_sequence[idx]
                    prev_content = prev.get("content")
                    if not isinstance(prev_content, str):
                        continue
                    prev_text = prev_content.strip()
                    if not prev_text:
                        continue
                    prev_role = str(prev.get("role") or "unknown").lower()
                    prev_timestamp = normalize_optional_string(prev.get("timestamp"))
                    entry = build_preceding_entry(
                        prev_role,
                        prev_text,
                        index=idx,
                        timestamp=prev_timestamp,
                    )
                    preceding_messages.append(entry)
        yield MessageContextData(
            message_index=message_index,
            role=role,
            content=normalized,
            timestamp=timestamp,
            preceding=preceding_messages,
        )
