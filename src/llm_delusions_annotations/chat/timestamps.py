"""Timestamp parsing and normalization helpers for chat transcripts.

This module centralises small utilities related to timestamp handling so
that chat loading, export scripts, and analysis code share consistent
behaviour. It provides helpers for:

* extracting the most informative timestamp-like value from metadata
  dictionaries while preserving seconds-level precision, and
* parsing timestamp label strings into naive UTC ``datetime`` objects.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Sequence

TIMESTAMP_KEYS: Sequence[str] = (
    "date",
    "timestamp",
    "created_at",
    "create_time",
    "updated_at",
    "update_time",
)


def normalize_timestamp_value(value: object) -> Optional[str]:
    """Return a normalized timestamp label string or ``None``.

    This helper preserves seconds-level (and optional sub-second) precision
    for numeric epoch values while also passing through useful string
    representations. Values that are missing, empty, or non-positive are
    treated as unusable.

    Parameters
    ----------
    value:
        Raw timestamp-like value such as an integer epoch, float epoch,
        ISO 8601 string, or datetime object.

    Returns
    -------
    Optional[str]
        Normalized timestamp label string when usable, otherwise ``None``.
    """

    if value is None:
        return None

    if isinstance(value, str):
        text = value.strip()
        return text or None

    if isinstance(value, (int, float)):
        if value <= 0:
            return None
        return str(value)

    if isinstance(value, datetime):
        dt_val = value
        if dt_val.tzinfo is None:
            dt_val = dt_val.replace(tzinfo=timezone.utc)
        dt_val = dt_val.astimezone(timezone.utc)
        return dt_val.isoformat().replace("+00:00", "Z")

    return None


def extract_best_timestamp_label(candidate: object) -> Optional[str]:
    """Pick the first usable timestamp label from a metadata-like mapping.

    The search checks common timestamp fields such as ``\"date\"``,
    ``\"timestamp\"``, and ChatGPT-style ``\"create_time\"`` / ``\"updated_at\"``
    keys. When a field is present, its value is passed through
    :func:`normalize_timestamp_value`. If no direct field is usable, an
    optional nested ``\"metadata\"`` mapping is inspected recursively.

    Parameters
    ----------
    candidate:
        Object that may behave like a mapping containing timestamp-like fields.

    Returns
    -------
    Optional[str]
        Normalized timestamp label string when discovered, otherwise ``None``.
    """

    if not isinstance(candidate, dict):
        return None

    for key in TIMESTAMP_KEYS:
        if key in candidate:
            label = normalize_timestamp_value(candidate[key])
            if label:
                return label

    nested = candidate.get("metadata")
    if isinstance(nested, dict):
        nested_label = extract_best_timestamp_label(nested)
        if nested_label:
            return nested_label

    return None


def parse_date_label(label: Optional[str]) -> Optional[datetime]:
    """Parse a date label (various formats) into a naive UTC datetime.

    Supported formats include:

    * Epoch seconds represented as an integer or float string.
    * ``\"%Y-%m-%d %H:%M UTC\"``
    * ``\"%Y-%m-%d %H:%M:%S UTC\"``
    * ISO 8601 strings with or without a trailing ``\"Z\"``.
    * Fallbacks used previously: ``\"%Y-%m-%d %H:%M\"`` and ``\"%Y-%m-%d\"``,
      interpreted as UTC.

    Parameters
    ----------
    label:
        Raw timestamp label string to parse.

    Returns
    -------
    Optional[datetime]
        Naive UTC ``datetime`` value when parsing succeeds; otherwise ``None``.
    """

    if not label:
        return None
    text = label.strip()
    if not text:
        return None

    numeric_text = text.replace(".", "", 1)
    if numeric_text.isdigit():
        try:
            seconds = float(text)
            candidate = datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            candidate = None
        else:
            return candidate.replace(tzinfo=None)

    for fmt in ("%Y-%m-%d %H:%M UTC", "%Y-%m-%d %H:%M:%S UTC"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    try:
        candidate = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        candidate = None

    if candidate is None:
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                candidate = datetime.strptime(text, fmt)
                candidate = candidate.replace(tzinfo=timezone.utc)
                break
            except ValueError:
                continue

    if candidate is None:
        return None

    if candidate.tzinfo is not None:
        candidate = candidate.astimezone(timezone.utc).replace(tzinfo=None)
    return candidate


__all__ = [
    "TIMESTAMP_KEYS",
    "normalize_timestamp_value",
    "extract_best_timestamp_label",
    "parse_date_label",
]
