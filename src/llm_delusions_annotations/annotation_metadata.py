"""Helpers for loading annotation metadata and scopes.

This module centralises logic for reading
``src/llm_delusions_annotations/data/annotations.csv`` so that
scripts share consistent semantics for annotation ids, scopes, and harmful
flags and avoid duplicate code.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

from llm_delusions_annotations.annotation_ids import (
    ROLE_SPLIT_BASE_IDS,
    expand_annotation_row,
    normalize_role_token,
    normalize_scope_tokens,
)

HARMFUL_COLUMN = "Harmful (by itself)?"
SCOPE_COLUMN = "scope"
CATEGORY_COLUMN = "category"
ID_COLUMN = "id"


@dataclass
class AnnotationMetadata:
    """Metadata describing a single annotation id."""

    annotation_id: str
    category: str
    scope: Sequence[str]
    is_harmful: bool


def normalize_scope(raw_scope: str) -> list[str]:
    """Return a normalised scope list from a CSV scope string.

    Parameters
    ----------
    raw_scope:
        Raw scope value read from the CSV ``scope`` column.

    Returns
    -------
    list[str]
        List of lower-cased role identifiers. Empty when no scope is
        specified, which is treated as applying to all roles.
    """

    if not raw_scope:
        return []
    parts = [chunk.strip() for chunk in raw_scope.replace(";", ",").split(",")]
    return normalize_scope_tokens([value for value in parts if value])


def normalize_role_filter(raw: Optional[str]) -> Optional[str]:
    """Return a normalised role filter token or ``None``.

    Parameters
    ----------
    raw:
        Raw role filter string from the CLI, or ``None`` when not provided.

    Returns
    -------
    Optional[str]
        ``\"user\"`` or ``\"assistant\"`` when a specific role is requested,
        otherwise ``None`` to indicate that both roles should be included
        wherever annotations are in scope.

    Raises
    ------
    ValueError
        If the supplied value is not a supported role or synonym.
    """

    if raw is None:
        return None
    token = raw.strip().lower()
    if token in {"", "all", "any", "both", "auto"}:
        return None
    if normalize_role_token(token) is None:
        raise ValueError(
            f"Invalid role filter {raw!r}; expected 'user', 'assistant', or a "
            "synonym for both.",
        )
    return "assistant" if token in {"assistant", "bot", "chatbot"} else "user"


def is_role_in_scope(role: str, scope: Sequence[str]) -> bool:
    """Return True when a message role is compatible with the annotation scope.

    Parameters
    ----------
    role:
        Message role string such as ``\"user\"`` or ``\"assistant\"``.
    scope:
        Sequence of lower-cased role names for which the annotation is
        defined. An empty sequence means the annotation applies to all roles.

    Returns
    -------
    bool
        True when ``role`` is within ``scope`` or when ``scope`` is empty.
    """

    if not scope:
        return True
    normalized_role = normalize_role_token(role)
    if not normalized_role:
        return False
    normalized_scope = normalize_scope_tokens(scope)
    return normalized_role in normalized_scope


def _load_raw_annotation_metadata(csv_path: Path) -> Dict[str, AnnotationMetadata]:
    """Return annotation metadata keyed by canonical annotation id.

    This helper normalizes ids to canonical bot/user prefixes and expands
    role-split base ids into per-role variants when needed.
    """

    metadata: Dict[str, AnnotationMetadata] = {}
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            annotation_id = (row.get(ID_COLUMN) or "").strip()
            if not annotation_id:
                continue
            scope_raw = (row.get(SCOPE_COLUMN) or "").strip()
            scopes = normalize_scope(scope_raw)
            harmful_raw = (row.get(HARMFUL_COLUMN) or "").strip().lower()
            is_harmful = harmful_raw in {"harmful", "yes", "true", "1"}
            category = (row.get(CATEGORY_COLUMN) or "").strip()
            for expanded_id, expanded_scope in expand_annotation_row(
                annotation_id,
                scopes,
            ):
                if not expanded_id:
                    continue
                metadata[expanded_id] = AnnotationMetadata(
                    annotation_id=expanded_id,
                    category=category,
                    scope=expanded_scope,
                    is_harmful=is_harmful,
                )
    return metadata


def _iter_role_split_targets() -> Iterable[str]:
    """Yield base annotation ids that should be split by role.

    These identifiers correspond to relationship- and theme-oriented
    annotations where downstream analyses benefit from explicit
    user/assistant variants, for example:

    The concrete set of base identifiers is defined centrally in
    :data:`annotation_ids.ROLE_SPLIT_BASE_IDS` so that all
    consumers share the same split-target list.
    """

    yield from ROLE_SPLIT_BASE_IDS


def _augment_with_role_split_ids(
    metadata_by_id: Dict[str, AnnotationMetadata],
) -> Dict[str, AnnotationMetadata]:
    """Return metadata including synthetic role-specific annotation ids.

    This helper leaves the original annotation identifiers untouched while
    adding entries such as ``user-platonic-affinity`` and
    ``bot-platonic-affinity`` when the corresponding base id is in
    scope for that role. These synthetic ids share the same category and
    harmful flag as their base identifiers but expose a single-role scope.
    """

    augmented: Dict[str, AnnotationMetadata] = dict(metadata_by_id)

    for base_id in _iter_role_split_targets():
        base_meta = metadata_by_id.get(base_id)
        if base_meta is None:
            continue

        normalized_scopes = normalize_scope_tokens(base_meta.scope)
        has_user_scope = not normalized_scopes or "user" in normalized_scopes
        has_assistant_scope = not normalized_scopes or "assistant" in normalized_scopes

        if has_user_scope:
            user_id = f"user-{base_id}"
            if user_id not in augmented:
                augmented[user_id] = AnnotationMetadata(
                    annotation_id=user_id,
                    category=base_meta.category,
                    scope=["user"],
                    is_harmful=base_meta.is_harmful,
                )

        if has_assistant_scope:
            assistant_id = f"bot-{base_id}"
            if assistant_id not in augmented:
                augmented[assistant_id] = AnnotationMetadata(
                    annotation_id=assistant_id,
                    category=base_meta.category,
                    scope=["assistant"],
                    is_harmful=base_meta.is_harmful,
                )

        # For analysis-level mappings, drop the base id so that only
        # role-split variants are exposed to downstream tools. This ensures
        # that, for example, ``platonic-affinity`` and
        # ``theme-awakening-consciousness`` are always analysed via their
        # user/bot-specific forms.
        if base_id in augmented:
            augmented.pop(base_id, None)

    return augmented


def load_annotation_metadata(csv_path: Path) -> Dict[str, AnnotationMetadata]:
    """Return annotation metadata keyed by annotation id.

    The returned mapping normalizes ids to canonical bot/user prefixes and
    expands role-split base ids into per-role variants.
    """

    return _load_raw_annotation_metadata(csv_path)


def load_annotation_metadata_with_role_splits(
    csv_path: Path,
) -> Dict[str, AnnotationMetadata]:
    """Return annotation metadata including synthetic role-specific ids.

    This helper adds synthetic role-specific identifiers when a base id is
    still present in the CSV. For example, when the CSV defines
    ``platonic-affinity`` with scope including both roles, the mapping also
    exposes:

    * ``user-platonic-affinity`` with scope ``[\"user\"]``.
    * ``bot-platonic-affinity`` with scope ``[\"assistant\"]``.
    """

    base_metadata = _load_raw_annotation_metadata(csv_path)
    return _augment_with_role_split_ids(base_metadata)


def load_annotation_metadata_or_exit_code(
    csv_path: Path,
) -> tuple[Dict[str, AnnotationMetadata], int]:
    """Return annotation metadata and an exit-style status code.

    Parameters
    ----------
    csv_path:
        Path to the annotations metadata CSV file.

    Returns
    -------
    metadata_by_id:
        Mapping from annotation id to metadata objects. This mapping may be
        empty when loading fails.
    status:
        Zero when at least one annotation is loaded; ``2`` when no
        annotations could be loaded and an error message has been printed.
    """

    metadata_by_id = load_annotation_metadata_with_role_splits(csv_path)
    if not metadata_by_id:
        print(f"No annotations loaded from {csv_path}")
        return metadata_by_id, 2
    return metadata_by_id, 0


# Central blocklist of annotation identifiers that should be excluded from
# aggregate analyses even when present in the metadata table. This list
# complements category-based filters (for example, the ``test`` category)
# and is intended for synthetic or utility labels such as calibration codes.
EXCLUDED_ANNOTATION_IDS: set[str] = {
    "test",
    "bot-inchat-action",
    "user-bypass",
    "challenge-status-quo",
    "bot-offplatform-action",
    "bot-hyperbole",
    "user-reports-others-admire-self",
    "user-seeks-validation",
    "bot-extrapolates",
    "user-reports-followed-instruction",
    "user-misconstrues-lm-ability",
    "bot-validates-ideas",
    "message-outreach",
    "bot-message-outreach",
    "user-message-outreach",
    "assistant-urgency",
    "bot-urgency",
}


def filter_analysis_metadata(
    metadata_by_id: Mapping[str, AnnotationMetadata],
    *,
    exclude_categories: Sequence[str] | None = None,
    exclude_ids: Sequence[str] | None = None,
) -> Dict[str, AnnotationMetadata]:
    """Return metadata filtered for downstream analyses.

    Parameters
    ----------
    metadata_by_id:
        Mapping from annotation id to metadata objects.
    exclude_categories:
        Optional sequence of category names to exclude (case-insensitive).
        When omitted, ``[\"test\"]`` is used so that test-category labels
        are not included in aggregate summaries.
    exclude_ids:
        Optional sequence of annotation identifiers to exclude regardless
        of category. When omitted, :data:`EXCLUDED_ANNOTATION_IDS` is used.

    Returns
    -------
    Dict[str, AnnotationMetadata]
        Filtered mapping containing only annotations that are not excluded
        by id or category.
    """

    category_blocklist = {
        value.strip().lower()
        for value in (exclude_categories or ["test"])
        if value.strip()
    }
    id_blocklist = {
        value.strip()
        for value in (exclude_ids or EXCLUDED_ANNOTATION_IDS)
        if value.strip()
    }

    filtered: Dict[str, AnnotationMetadata] = {}
    for annotation_id, meta in metadata_by_id.items():
        if annotation_id in id_blocklist:
            continue
        category_value = meta.category.strip().lower()
        if category_value in category_blocklist:
            continue
        filtered[annotation_id] = meta
    return filtered


__all__ = [
    "AnnotationMetadata",
    "is_role_in_scope",
    "HARMFUL_COLUMN",
    "SCOPE_COLUMN",
    "CATEGORY_COLUMN",
    "ID_COLUMN",
    "normalize_scope",
    "normalize_role_filter",
    "load_annotation_metadata",
    "load_annotation_metadata_with_role_splits",
    "load_annotation_metadata_or_exit_code",
    "EXCLUDED_ANNOTATION_IDS",
    "filter_analysis_metadata",
]
