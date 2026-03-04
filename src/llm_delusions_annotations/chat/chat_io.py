"""Helpers to load parsed chat JSON into reusable Chat objects."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json_repair

from .timestamps import extract_best_timestamp_label


@dataclass
class Chat:
    """In-memory representation of a single chat transcript."""

    key: str
    notes: str
    messages: List[Dict[str, str]]
    date_label: Optional[str] = None


def load_chats_for_file(path: Path, strategy: str = "global_longest") -> List[Chat]:
    """Load zero, one, or many Chat objects from a parsed JSON file.

    Parameters
    ----------
    path:
        Path to the JSON transcript file.
    strategy:
        Conversation extraction strategy for ChatGPT export style data.
        Supported values are:

        - ``all_messages``: include every visible message from the mapping.
        - ``active_longest``: follow the behavior that prioritizes the path
          ending at ``current_node`` when available, otherwise the deepest leaf.
        - ``global_longest`` (default): choose the root-to-leaf path with the
          richest visible user/assistant dialogue.
        - Grok export conversations: responses are flattened into messages.

        For already-flattened ``messages`` lists, the strategy parameter has
        no effect.
    """

    chats: List[Chat] = []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as err:
        sys.stderr.write(f"[WARN] Failed to read {path}: {err}\n")
        return chats
    try:
        data = json_repair.loads(raw)
    except (JSONDecodeError, ValueError) as err:
        sys.stderr.write(f"[WARN] Invalid JSON {path}: {err}\n")
        return chats

    # Special case: raw ChatGPT export files often contain a list of conversation
    # objects with a graph-like "mapping". Handle those so downstream code can
    # render them without running the parser pipeline first.
    if isinstance(data, list) and data and isinstance(data[0], dict):
        for idx, conv in enumerate(data, start=1):
            if not isinstance(conv, dict):
                continue
            extracted = _extract_messages_from_chatgpt_export(conv, strategy)
            if not extracted:
                continue
            msgs = _normalize(extracted)
            title = conv.get("title")
            conv_key = (
                str(title).strip()
                if isinstance(title, str) and title.strip()
                else f"conversation {idx}"
            )
            chats.append(
                Chat(
                    key=conv_key,
                    notes="raw chatgpt export",
                    messages=msgs,
                    date_label=extract_best_timestamp_label(conv),
                )
            )
        return chats

    meta = data.get("meta", {}) if isinstance(data.get("meta", {}), dict) else {}
    base_key = meta.get("rel_path") or meta.get("filename") or path.name
    notes = data.get("notes", "")
    base_date = extract_best_timestamp_label(meta)

    messages = data.get("messages")
    if isinstance(messages, list):
        chats.append(
            Chat(
                key=base_key,
                notes=notes,
                messages=_normalize(messages),
                date_label=base_date,
            )
        )
        return chats

    conversations = data.get("conversations")
    if isinstance(conversations, list):
        if _looks_like_grok_export(conversations):
            for idx, conv in enumerate(conversations, start=1):
                if not isinstance(conv, dict):
                    continue
                raw_msgs = _extract_messages_from_grok_export(conv)
                if not raw_msgs:
                    continue
                title = _extract_grok_title(conv, idx, base_key)
                conv_date = _extract_grok_date_label(conv)
                chats.append(
                    Chat(
                        key=title,
                        notes="raw grok export",
                        messages=_normalize(raw_msgs),
                        date_label=conv_date or base_date,
                    )
                )
            return chats
        for idx, conv in enumerate(conversations, start=1):
            if not isinstance(conv, dict):
                continue
            raw_msgs = conv.get("messages")
            if isinstance(raw_msgs, list):
                msgs = _normalize(raw_msgs)
            else:
                extracted = _extract_messages_from_chatgpt_export(conv, strategy)
                if not extracted:
                    continue
                msgs = _normalize(extracted)
            conv_date = extract_best_timestamp_label(conv)
            title = conv.get("title")
            conv_key = (
                str(title).strip()
                if isinstance(title, str) and title.strip()
                else f"{base_key} [conv {idx}]"
            )
            chats.append(
                Chat(
                    key=conv_key,
                    notes=notes,
                    messages=msgs,
                    date_label=conv_date or base_date,
                )
            )
        return chats

    # Unsupported shape
    return chats


def _looks_like_grok_export(conversations: List[object]) -> bool:
    """Return True if the conversations list resembles a Grok export."""

    for entry in conversations:
        if not isinstance(entry, dict):
            continue
        has_conversation = isinstance(entry.get("conversation"), dict)
        has_responses = isinstance(entry.get("responses"), list)
        if has_conversation and has_responses:
            return True
    return False


def _extract_grok_title(
    conv: Dict[str, object],
    index: int,
    fallback_key: str,
) -> str:
    """Extract a display title for a Grok conversation.

    Parameters
    ----------
    conv:
        Grok conversation wrapper containing a ``conversation`` mapping.
    index:
        1-based fallback index for naming when title is missing.
    fallback_key:
        Baseline identifier derived from the source filename.

    Returns
    -------
    str
        Display title for the conversation.
    """

    conv_meta = conv.get("conversation")
    if isinstance(conv_meta, dict):
        title_value = conv_meta.get("title")
        if isinstance(title_value, str) and title_value.strip():
            return title_value.strip()
        id_value = conv_meta.get("id")
        if isinstance(id_value, str) and id_value.strip():
            return id_value.strip()
    return f"{fallback_key} [conv {index}]"


def _extract_grok_date_label(conv: Dict[str, object]) -> Optional[str]:
    """Extract the best date label from a Grok conversation wrapper.

    Parameters
    ----------
    conv:
        Grok conversation wrapper containing a ``conversation`` mapping.

    Returns
    -------
    Optional[str]
        Timestamp label string if detected, otherwise ``None``.
    """

    conv_meta = conv.get("conversation")
    if isinstance(conv_meta, dict):
        return extract_best_timestamp_label(conv_meta)
    return None


def _extract_messages_from_grok_export(conv: Dict[str, object]) -> List[Dict[str, str]]:
    """Extract message rows from a Grok export conversation object.

    Parameters
    ----------
    conv:
        Grok conversation wrapper containing a ``responses`` list.

    Returns
    -------
    List[Dict[str, str]]
        List of message dicts suitable for normalization.
    """

    responses = conv.get("responses")
    if not isinstance(responses, list):
        return []

    messages: List[Dict[str, str]] = []
    for response_entry in responses:
        if not isinstance(response_entry, dict):
            continue
        response = response_entry.get("response")
        if not isinstance(response, dict):
            continue
        text_value = response.get("message")
        if text_value is None:
            continue
        text = _coerce_text(text_value)
        if not text:
            continue
        sender_value = response.get("sender")
        role = _normalize_grok_sender(sender_value)
        model_slug = _extract_grok_model_slug(response)
        timestamp_label = _extract_grok_timestamp_label(response)
        messages.append(
            _build_message_entry(
                role=role,
                content=text,
                model_slug=model_slug,
                timestamp_label=timestamp_label,
            )
        )

    return messages


def _coerce_text(value: object) -> str:
    """Return a string representation of a Grok message value."""

    if isinstance(value, str):
        return value.strip()
    try:
        return json.dumps(value, ensure_ascii=False).strip()
    except (TypeError, ValueError):
        return str(value).strip()


def _normalize_grok_sender(sender_value: object) -> str:
    """Normalize Grok sender values into chat roles.

    Parameters
    ----------
    sender_value:
        Raw sender label from Grok response payloads.

    Returns
    -------
    str
        Normalized role string.
    """

    return _normalize_role(
        sender_value,
        aliases={
            "human": "user",
            "user": "user",
            "assistant": "assistant",
            "bot": "assistant",
        },
    )


def _extract_grok_model_slug(response: Dict[str, object]) -> Optional[str]:
    """Extract model info from a Grok response payload.

    Parameters
    ----------
    response:
        Grok response mapping.

    Returns
    -------
    Optional[str]
        Model slug when available, otherwise ``None``.
    """

    direct_model = response.get("model")
    if isinstance(direct_model, str) and direct_model.strip():
        return direct_model.strip()

    metadata = response.get("metadata")
    if isinstance(metadata, dict):
        request_model = metadata.get("requestModelDetails")
        if isinstance(request_model, dict):
            model_id = request_model.get("modelId")
            if isinstance(model_id, str) and model_id.strip():
                return model_id.strip()
        request_meta = metadata.get("request_metadata")
        if isinstance(request_meta, dict):
            model_name = request_meta.get("model")
            if isinstance(model_name, str) and model_name.strip():
                return model_name.strip()
    return None


def _extract_grok_timestamp_label(response: Dict[str, object]) -> Optional[str]:
    """Extract a timestamp label from a Grok response payload.

    Parameters
    ----------
    response:
        Grok response mapping.

    Returns
    -------
    Optional[str]
        Timestamp label string when available.
    """

    label = extract_best_timestamp_label(response)
    if label:
        return label

    create_time = response.get("create_time")
    if isinstance(create_time, dict):
        date_wrapper = create_time.get("$date")
        if isinstance(date_wrapper, dict):
            millis_value = date_wrapper.get("$numberLong")
            if isinstance(millis_value, str) and millis_value.isdigit():
                millis = int(millis_value)
                seconds = millis / 1000.0
                label = f"{seconds:.3f}".rstrip("0").rstrip(".")
                return label or None
            if isinstance(millis_value, (int, float)):
                seconds = float(millis_value) / 1000.0
                label = f"{seconds:.3f}".rstrip("0").rstrip(".")
                return label or None
    return None


def _normalize(messages: List[Dict[str, object]]) -> List[Dict[str, str]]:
    """Normalize raw message dicts into the Chat.messages shape.

    This helper preserves any model slug that upstream extractors attach to
    messages so that downstream analysis can attribute assistant turns to
    specific models when available.
    """

    out: List[Dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role_value = message.get("role", "")
        role = _normalize_role(role_value)
        content = message.get("content", "")
        if not isinstance(content, str):
            try:
                content = json.dumps(content, ensure_ascii=False)
            except (TypeError, ValueError):
                content = str(content)
        entry: Dict[str, str] = {"role": role, "content": content}

        # Preserve model information when present. ChatGPT export helpers
        # attach a top-level ``model_slug`` field to messages for assistant
        # turns, and some parsed transcripts may carry the same field.
        model_slug = message.get("model_slug")
        if isinstance(model_slug, str):
            slug = model_slug.strip()
            if slug:
                entry["model_slug"] = slug

        timestamp_label = extract_best_timestamp_label(message)
        if timestamp_label:
            entry["timestamp"] = timestamp_label
        out.append(entry)
    return out


def _extract_messages_from_chatgpt_export(
    conv: Dict[str, object],
    strategy: str,
) -> List[Dict[str, str]]:
    """Extract messages from a ChatGPT export conversation node.

    The ``strategy`` parameter controls how we walk the mapping:

    - ``active_longest``: previous behavior; follow the chain ending at
      ``current_node`` when present, otherwise the deepest leaf.
    - ``all_messages``: return all visible messages from the mapping.
    - ``global_longest``: follow the root-to-leaf path with the highest
      score based on visible user/assistant turns.
    """

    mapping = conv.get("mapping")
    if not isinstance(mapping, dict):
        return []

    id_to_node: Dict[str, Dict[str, object]] = mapping

    if strategy == "all_messages":
        return _extract_all_visible_messages(id_to_node)
    if strategy == "global_longest":
        path = _find_best_longest_chat_path(id_to_node)
        if path:
            return _extract_messages_along_path(id_to_node, path)
        # Fall back to active_longest behavior if scoring fails.

    path = _find_deepest_or_current_path(conv, id_to_node)
    if not path:
        return []
    return _extract_messages_along_path(id_to_node, path)


def _is_visually_hidden(node: Dict[str, object]) -> bool:
    """Return True if a ChatGPT export node is hidden from the conversation."""

    message = node.get("message") if isinstance(node.get("message"), dict) else {}
    meta = message.get("metadata") if isinstance(message, dict) else {}
    return bool(
        isinstance(meta, dict) and meta.get("is_visually_hidden_from_conversation")
    )


def _extract_text_from_node_message(node: Dict[str, object]) -> str:
    """Extract visible text from a ChatGPT export node message."""

    message = node.get("message") if isinstance(node.get("message"), dict) else {}
    content = message.get("content") if isinstance(message, dict) else {}
    parts = content.get("parts") if isinstance(content, dict) else None
    texts: List[str] = []
    if isinstance(parts, list):
        for part in parts:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict):
                value = part.get("text") or part.get("content")
                if isinstance(value, str):
                    texts.append(value)
    if not texts and isinstance(content, dict):
        text_value = content.get("text")
        if isinstance(text_value, str):
            texts.append(text_value)
    return "\n".join(fragment for fragment in texts if fragment).strip()


def _extract_all_visible_messages(
    id_to_node: Dict[str, Dict[str, object]],
) -> List[Dict[str, str]]:
    """Return every visible message from the mapping in insertion order."""

    messages: List[Dict[str, str]] = []
    for node in id_to_node.values():
        entry = _message_entry_from_node(node)
        if entry is not None:
            messages.append(entry)
    return messages


def _extract_messages_along_path(
    id_to_node: Dict[str, Dict[str, object]],
    path: List[str],
) -> List[Dict[str, str]]:
    """Extract visible messages for a specific node id path."""

    messages: List[Dict[str, str]] = []
    for node_id in path:
        node = id_to_node.get(node_id) or {}
        entry = _message_entry_from_node(node)
        if entry is not None:
            messages.append(entry)
    return messages


def _message_entry_from_node(node: Dict[str, object]) -> Dict[str, str] | None:
    """Build a normalized message entry from a ChatGPT export node."""

    if not isinstance(node, dict):
        return None
    if _is_visually_hidden(node):
        return None
    text = _extract_text_from_node_message(node)
    if not text:
        return None
    message = node.get("message") if isinstance(node.get("message"), dict) else {}
    author = message.get("author") if isinstance(message.get("author"), dict) else {}
    role = _normalize_role(author.get("role"))
    metadata = (
        message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
    )
    model_slug = metadata.get("model_slug")
    timestamp_label = extract_best_timestamp_label(message)
    return _build_message_entry(
        role=role,
        content=text,
        model_slug=model_slug,
        timestamp_label=timestamp_label,
    )


def _find_deepest_or_current_path(
    conv: Dict[str, object],
    id_to_node: Dict[str, Dict[str, object]],
) -> List[str]:
    """Return the chain ending at current_node or the deepest leaf."""

    current_node = conv.get("current_node")
    leaf_id: Optional[str] = None
    if isinstance(current_node, str) and current_node in id_to_node:
        leaf_id = current_node
    else:
        leaves = [
            nid for nid, node in id_to_node.items() if not (node.get("children") or [])
        ]

        def _depth(node_id: str) -> int:
            depth = 0
            seen: set[str] = set()
            cur = node_id
            while True:
                node = id_to_node.get(cur)
                if not isinstance(node, dict):
                    break
                parent = node.get("parent")
                if not isinstance(parent, str) or parent in seen:
                    break
                seen.add(parent)
                depth += 1
                cur = parent
            return depth

        if leaves:
            leaf_id = max(leaves, key=_depth)
        else:
            leaf_id = next(iter(id_to_node.keys()), None)

    if not leaf_id:
        return []

    chain: List[str] = []
    seen_ids: set[str] = set()
    cur = leaf_id
    while cur and cur not in seen_ids:
        seen_ids.add(cur)
        chain.append(cur)
        node = id_to_node.get(cur) or {}
        parent = node.get("parent") if isinstance(node, dict) else None
        cur = parent if isinstance(parent, str) else None
    chain.reverse()
    return chain


def _find_best_longest_chat_path(
    id_to_node: Dict[str, Dict[str, object]],
) -> List[str]:
    """Return the highest scoring root-to-leaf path for global_longest."""

    # Build children adjacency lists and identify roots.
    children: Dict[str, List[str]] = {}
    roots: List[str] = []
    for node_id, node in id_to_node.items():
        if not isinstance(node, dict):
            continue
        parent = node.get("parent")
        if not isinstance(parent, str):
            roots.append(node_id)
        node_children = node.get("children") or []
        node_children_ids: List[str] = []
        if isinstance(node_children, list):
            for child_id in node_children:
                if isinstance(child_id, str):
                    node_children_ids.append(child_id)
        children[node_id] = node_children_ids

    if not roots:
        roots = list(id_to_node.keys())

    # Dynamic programming: compute best score and best child per node without recursion.
    best_score: Dict[str, int] = {}
    best_child: Dict[str, Optional[str]] = {}
    visited: set[str] = set()
    processing: set[str] = set()

    for root_id in roots:
        if root_id in visited or root_id not in id_to_node:
            continue
        stack: List[Tuple[str, bool]] = [(root_id, False)]
        while stack:
            node_id, processed = stack.pop()
            if processed:
                processing.discard(node_id)
                node = id_to_node.get(node_id) or {}
                score_here = _score_node_for_longest_chat(node)
                best_child_id: Optional[str] = None
                best_child_score = 0
                for child_id in children.get(node_id, []):
                    child_score = best_score.get(child_id, 0)
                    if child_score > best_child_score:
                        best_child_score = child_score
                        best_child_id = child_id
                best_score[node_id] = score_here + best_child_score
                if best_child_id is not None:
                    best_child[node_id] = best_child_id
                visited.add(node_id)
                continue

            if node_id in visited:
                continue
            if node_id in processing:
                # Cycle guard: treat as already processed with zero score.
                visited.add(node_id)
                best_score.setdefault(node_id, 0)
                continue

            processing.add(node_id)
            stack.append((node_id, True))
            for child_id in children.get(node_id, []):
                if child_id not in id_to_node:
                    continue
                if child_id not in visited:
                    stack.append((child_id, False))

    global_best_score = 0
    global_best_path: List[str] = []
    for root_id in roots:
        if root_id not in id_to_node:
            continue
        score = best_score.get(root_id)
        if score is None:
            node = id_to_node.get(root_id) or {}
            score = _score_node_for_longest_chat(node)
        if score > global_best_score or not global_best_path:
            global_best_score = score
            # Reconstruct path by following best_child links.
            path: List[str] = []
            cur = root_id
            seen_on_path: set[str] = set()
            while cur in id_to_node and cur not in seen_on_path:
                seen_on_path.add(cur)
                path.append(cur)
                next_id = best_child.get(cur)
                if not next_id:
                    break
                cur = next_id
            global_best_path = path

    return global_best_path


def _score_node_for_longest_chat(node: Dict[str, object]) -> int:
    """Score a node for global_longest based on visible user/assistant turns."""

    if not isinstance(node, dict):
        return 0
    if _is_visually_hidden(node):
        return 0

    message = node.get("message") if isinstance(node.get("message"), dict) else {}
    if not isinstance(message, dict):
        return 0

    author = message.get("author") if isinstance(message.get("author"), dict) else {}
    role = _normalize_role(author.get("role"))
    if role not in ("user", "assistant"):
        return 0

    text = _extract_text_from_node_message(node)
    if not text:
        return 0
    return 1


def _normalize_role(
    role_value: object, aliases: Optional[Dict[str, str]] = None
) -> str:
    """Normalize a role value with optional alias mapping.

    Parameters
    ----------
    role_value:
        Raw role value to normalize.
    aliases:
        Optional mapping of alias values to normalized roles.

    Returns
    -------
    str
        Normalized role string, defaulting to ``"unknown"`` when empty.
    """

    role = str(role_value or "").strip().lower()
    if aliases and role in aliases:
        role = aliases[role]
    if role:
        return role
    return "unknown"


def _build_message_entry(
    role: str,
    content: str,
    model_slug: Optional[object],
    timestamp_label: Optional[str],
) -> Dict[str, str]:
    """Build a normalized message entry.

    Parameters
    ----------
    role:
        Normalized role label.
    content:
        Message content string.
    model_slug:
        Optional model identifier.
    timestamp_label:
        Optional timestamp label string.

    Returns
    -------
    Dict[str, str]
        Normalized message entry dictionary.
    """

    entry: Dict[str, str] = {"role": role, "content": content}
    if isinstance(model_slug, str):
        slug = model_slug.strip()
        if slug:
            entry["model_slug"] = slug
    if timestamp_label:
        entry["timestamp"] = timestamp_label
    return entry
