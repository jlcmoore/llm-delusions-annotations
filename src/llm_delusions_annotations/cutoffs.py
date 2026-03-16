"""Helpers for loading per-annotation LLM score cutoffs.

This module centralizes logic for interpreting cutoff mappings so that both
scripts and library code can share the same semantics when applying
per-annotation thresholds to LLM scores.
"""

from __future__ import annotations

import csv
import logging
from importlib import resources
from pathlib import Path
from typing import Dict, Optional

from llm_delusions_annotations.annotation_ids import (
    ROLE_SPLIT_BASE_IDS,
    normalize_annotation_id,
)

LOGGER = logging.getLogger(__name__)

CUTOFFS_PACKAGE = "llm_delusions_annotations.data"
CUTOFFS_FILENAME = "cutoffs.csv"
CUTOFFS_FILE = resources.files(CUTOFFS_PACKAGE).joinpath(CUTOFFS_FILENAME)


def load_llm_cutoffs_from_csv(
    csv_path: Optional[str | Path],
) -> Optional[Dict[str, int]]:
    """Return per-annotation LLM cutoffs loaded from a CSV file.

    Parameters
    ----------
    csv_path:
        Path to a CSV file with ``annotation_id`` and ``cutoff`` columns.
        Defaults to the shared ``cutoffs.csv`` in the package data.

    Returns
    -------
    Optional[Dict[str, int]]
        Mapping from annotation id to integer cutoff when the file could be
        read and parsed successfully; otherwise ``None``. Errors are logged
        using the module logger.
    """

    path = Path(str(csv_path)).expanduser() if csv_path else CUTOFFS_FILE
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            if "annotation_id" not in fieldnames or "cutoff" not in fieldnames:
                LOGGER.error(
                    "LLM cutoffs file %s must contain 'annotation_id' and "
                    "'cutoff' columns.",
                    path,
                )
                return None

            cutoffs: Dict[str, int] = {}
            for row in reader:
                annotation_id = str(row.get("annotation_id") or "").strip()
                if not annotation_id:
                    continue
                raw_value = row.get("cutoff")
                try:
                    cutoff_value = int(raw_value)
                except (TypeError, ValueError):
                    LOGGER.warning(
                        "Ignoring non-integer cutoff %r for annotation %r in "
                        "LLM cutoffs file %s",
                        raw_value,
                        annotation_id,
                        path,
                    )
                    continue
                cutoffs[annotation_id] = cutoff_value
    except OSError as err:
        LOGGER.error("Failed to read LLM cutoffs file %s: %s", path, err)
        return None

    if not cutoffs:
        LOGGER.warning(
            "LLM cutoffs file %s did not contain any usable "
            "annotation->cutoff mappings.",
            path,
        )
        return {}

    return cutoffs


def load_cutoffs_mapping(json_path: Optional[str | Path]) -> Dict[str, int]:
    """Return a mapping from annotation id to cutoff or an empty mapping.

    This thin wrapper avoids repeated boilerplate in analysis scripts and keeps
    fallback behaviour centralised.
    """

    cutoffs = load_llm_cutoffs_from_csv(json_path)
    if cutoffs is None and json_path is not None:
        return {}
    if cutoffs is None:
        # Fallback to internal data if no path was provided and loading failed
        # (though load_llm_cutoffs_from_csv handles the default path)
        return {}

    mapping: Dict[str, int] = {}
    for key, value in cutoffs.items():
        normalized_key = normalize_annotation_id(str(key or ""))
        if not normalized_key:
            continue
        mapping[normalized_key] = value

    # Provide synthetic role-specific cutoffs for selected annotations so
    # that downstream analyses can safely consume derived score columns
    # (for example, ``score__user-platonic-affinity``) without requiring
    # separate entries in the cutoffs CSV. These synthetic ids inherit
    # the same cutoff as their base annotation.
    for base_id in ROLE_SPLIT_BASE_IDS:
        if base_id not in mapping:
            continue
        base_cutoff = mapping[base_id]
        for role_prefix in ("user", "bot"):
            synthetic_id = f"{role_prefix}-{base_id}"
            mapping.setdefault(synthetic_id, base_cutoff)

    return mapping


__all__ = ["load_llm_cutoffs_from_csv", "load_cutoffs_mapping", "CUTOFFS_FILE"]
