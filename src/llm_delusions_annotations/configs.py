"""Helpers for loading and scoping annotation configurations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Set

from llm_delusions_annotations.annotation_ids import normalize_role_token
from llm_delusions_annotations.annotation_prompts import ANNOTATIONS

LLM_SCORE_CUTOFF = 5


@dataclass(frozen=True)
class AnnotationConfig:
    """Selected annotation specification with derived scope metadata.

    Parameters
    ----------
    spec:
        Raw annotation specification dictionary loaded from prompts.
    allowed_roles:
        Optional set of message roles that this annotation applies to.
        When ``None``, the annotation is treated as applying to all roles.
    """

    spec: dict[str, object]
    allowed_roles: Optional[Set[str]]


def resolve_annotation(annotation_id: str) -> dict[str, object]:
    """Return the annotation spec matching ``annotation_id``.

    Parameters
    ----------
    annotation_id:
        Identifier of the annotation to resolve.

    Returns
    -------
    dict[str, object]
        Matching annotation specification.

    Raises
    ------
    ValueError
        If no annotation with ``annotation_id`` exists.
    """

    for spec in ANNOTATIONS:
        if spec.get("id") == annotation_id:
            return spec
    raise ValueError(f"Unknown annotation: {annotation_id}")


def parse_annotation_scope(annotation: Mapping[str, object]) -> Optional[Set[str]]:
    """Translate an annotation's scope into a set of allowed message roles.

    Parameters
    ----------
    annotation:
        Annotation specification dictionary that may contain a ``scope`` field.

    Returns
    -------
    Optional[Set[str]]
        Set of applicable roles or ``None`` when the annotation is not
        restricted to a specific subset of roles.
    """

    scope = annotation.get("scope")
    if not scope:
        return None
    if isinstance(scope, str):
        normalized_raw = {scope.lower()}
    else:
        normalized_raw = {str(item).lower() for item in scope}
    normalized = {
        role for role in (normalize_role_token(item) for item in normalized_raw) if role
    }
    if not normalized:
        return None
    if "both" in normalized or "all" in normalized:
        return None
    roles = {role for role in normalized if role in {"user", "assistant"}}
    return roles or None


def load_annotation_configs(
    annotation_ids: Optional[Sequence[str]],
) -> List[AnnotationConfig]:
    """Load selected annotations and compute their scoped roles.

    When no specific annotation is provided, defaults to all annotations except
    those whose CSV ``category`` is ``\"test\"``. Passing one or more specific
    annotation identifiers always includes those annotations even if they are
    in the test category.

    Parameters
    ----------
    annotation_ids:
        Optional collection of annotation identifiers to include.
    Returns
    -------
    List[AnnotationConfig]
        Loaded configurations with derived role scopes.
    """

    if annotation_ids:
        normalized_ids = [str(item) for item in annotation_ids if item]
        specs = [resolve_annotation(annotation_id) for annotation_id in normalized_ids]
    else:
        specs = [
            spec
            for spec in ANNOTATIONS
            if str(spec.get("category", "")).strip().lower() != "test"
        ]
    return [
        AnnotationConfig(spec=spec, allowed_roles=parse_annotation_scope(spec))
        for spec in specs
    ]


def derive_allowed_roles(configs: Sequence[AnnotationConfig]) -> Optional[Set[str]]:
    """Return the union of applicable roles across configs or None if unrestricted.

    Parameters
    ----------
    configs:
        Loaded annotation configurations for the current run.

    Returns
    -------
    Optional[Set[str]]
        Combined set of allowed roles or ``None`` when any configuration is
        unrestricted.
    """

    combined: Set[str] = set()
    for config in configs:
        if config.allowed_roles is None:
            return None
        combined.update(config.allowed_roles)
    return combined or None


__all__ = [
    "AnnotationConfig",
    "derive_allowed_roles",
    "load_annotation_configs",
    "parse_annotation_scope",
    "resolve_annotation",
]
