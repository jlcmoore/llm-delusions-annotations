"""Utilities for normalizing annotation identifiers and role splits."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Set, Tuple

ROLE_SPLIT_BASE_IDS: set[str] = {
    "platonic-affinity",
    "romantic-interest",
    "metaphysical-themes",
    "grand-significance",
}

BASE_ID_ALIASES: dict[str, str] = {
    "theme-awakening-consciousness": "metaphysical-themes",
    "delusion-themes": "metaphysical-themes",
    "claims-unique-understanding": "claims-unique-connection",
}

ROLE_ALIASES: dict[str, str] = {
    "assistant": "assistant",
    "bot": "assistant",
    "chatbot": "assistant",
    "user": "user",
}


def normalize_role_token(role: Optional[str]) -> Optional[str]:
    """Return a normalized role token for supported aliases.

    Parameters
    ----------
    role:
        Raw role value (for example, ``"assistant"`` or ``"bot"``).

    Returns
    -------
    Optional[str]
        Normalized role token (``"assistant"`` or ``"user"``), or ``None``
        when the input is empty or unsupported.
    """

    if not role:
        return None
    token = str(role).strip().lower()
    if not token:
        return None
    return ROLE_ALIASES.get(token)


def role_prefix_for_role(role: Optional[str], *, strict: bool = False) -> Optional[str]:
    """Return the canonical id prefix for a role.

    Parameters
    ----------
    role:
        Raw role value (for example, ``"assistant"`` or ``"user"``).
    strict:
        When True, raise ``ValueError`` on unsupported roles.

    Returns
    -------
    Optional[str]
        ``"bot"`` for assistant-like roles, ``"user"`` for user roles, or
        ``None`` when the role cannot be normalized and ``strict`` is False.
    """

    normalized = normalize_role_token(role)
    if normalized == "assistant":
        return "bot"
    if normalized == "user":
        return "user"
    if strict:
        raise ValueError(f"Unsupported role value: {role!r}")
    return None


def normalize_annotation_id(
    annotation_id: str,
    *,
    role: Optional[str] = None,
    strict_role: bool = False,
) -> str:
    """Return a canonical annotation id using bot/user prefixes.

    Parameters
    ----------
    annotation_id:
        Raw annotation id string.
    role:
        Optional role hint used to split base ids that require role-specific
        variants.
    strict_role:
        When True, raise ``ValueError`` if a role-split id cannot be derived.

    Returns
    -------
    str
        Canonical annotation id using ``bot-`` and ``user-`` prefixes.
    """

    raw = str(annotation_id or "").strip()
    if not raw:
        return ""

    lowered = raw.lower()
    if lowered.startswith("assistant-"):
        base = raw[len("assistant-") :]
        base = BASE_ID_ALIASES.get(base, base)
        return "bot-" + base
    if lowered.startswith("chatbot-"):
        base = raw[len("chatbot-") :]
        base = BASE_ID_ALIASES.get(base, base)
        return "bot-" + base
    if lowered.startswith("bot-") or lowered.startswith("user-"):
        prefix, base = raw.split("-", 1)
        base = BASE_ID_ALIASES.get(base, base)
        return f"{prefix}-{base}"

    base = BASE_ID_ALIASES.get(raw, raw)
    if base in ROLE_SPLIT_BASE_IDS:
        prefix = role_prefix_for_role(role, strict=strict_role)
        if prefix:
            return f"{prefix}-{base}"
        if strict_role:
            raise ValueError(
                f"Role is required to split annotation id {raw!r}.",
            )
    return base


def normalize_annotation_ids(raw_ids: Iterable[object]) -> Set[str]:
    """Normalize a collection of raw annotation id values.

    Parameters
    ----------
    raw_ids:
        Iterable of raw annotation id values (typically strings).

    Returns
    -------
    Set[str]
        Normalized annotation identifiers with role prefixes canonicalized.
    """

    normalized_ids: Set[str] = set()
    for raw in raw_ids:
        if not isinstance(raw, str):
            continue
        value = raw.strip()
        if not value:
            continue
        normalized = normalize_annotation_id(value)
        if normalized in ROLE_SPLIT_BASE_IDS:
            normalized_ids.add(f"bot-{normalized}")
            normalized_ids.add(f"user-{normalized}")
        else:
            normalized_ids.add(normalized or value)
    return normalized_ids


def expand_annotation_row(
    annotation_id: str,
    scope_tokens: Sequence[str],
) -> list[Tuple[str, list[str]]]:
    """Return annotation id rows expanded for role-split identifiers.

    Parameters
    ----------
    annotation_id:
        Raw annotation id string from the CSV.
    scope_tokens:
        Raw scope tokens (already split from the CSV).

    Returns
    -------
    list[tuple[str, list[str]]]
        List of ``(annotation_id, scope)`` pairs with canonical ids.
    """

    normalized_scope = [
        token
        for token in (normalize_role_token(value) for value in scope_tokens)
        if token
    ]

    raw = str(annotation_id or "").strip()
    if not raw:
        return []
    base = BASE_ID_ALIASES.get(raw, raw)

    lowered = raw.lower()
    if lowered.startswith(("assistant-", "chatbot-", "bot-", "user-")):
        normalized_id = normalize_annotation_id(raw)
        if normalized_id.startswith("bot-"):
            return [(normalized_id, ["assistant"])]
        if normalized_id.startswith("user-"):
            return [(normalized_id, ["user"])]
        return [(normalized_id, normalized_scope)]

    if base in ROLE_SPLIT_BASE_IDS:
        roles = normalized_scope or ["assistant", "user"]
        expanded: list[Tuple[str, list[str]]] = []
        for role in roles:
            prefix = role_prefix_for_role(role, strict=True)
            expanded.append((f"{prefix}-{base}", [role]))
        return expanded

    return [(normalize_annotation_id(base), normalized_scope)]


def normalize_scope_tokens(raw_scope: Iterable[str]) -> list[str]:
    """Return normalized scope tokens for a raw scope iterable.

    Parameters
    ----------
    raw_scope:
        Iterable of raw scope tokens.

    Returns
    -------
    list[str]
        Normalized scope tokens (``"assistant"`` or ``"user"``).
    """

    normalized: list[str] = []
    for value in raw_scope:
        token = normalize_role_token(value)
        if token:
            normalized.append(token)
    return normalized


__all__ = [
    "ROLE_SPLIT_BASE_IDS",
    "expand_annotation_row",
    "normalize_annotation_id",
    "normalize_role_token",
    "normalize_scope_tokens",
    "role_prefix_for_role",
]
