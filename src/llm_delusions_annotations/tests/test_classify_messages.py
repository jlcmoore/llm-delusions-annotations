"""Tests for JSON parsing in ``classify_messages``."""

import pytest

from llm_delusions_annotations.classify_messages import (
    ClassificationError,
    extract_matches_from_response_text,
)


def test_extract_matches_recovers_over_escaped_quotes_array():
    """Parser should recover fields from over-escaped quote arrays."""
    content = r'{"rationale":"x","quotes":[\\"a\\",\\"b\\"],"score":1}'

    rationale, matches, score = extract_matches_from_response_text(content)

    assert rationale == "x"
    assert matches == ["a", "b"]
    assert score == 1


def test_extract_matches_recovers_doubly_escaped_quotes_array():
    """Parser should recover fields when escaping is applied multiple times."""
    content = r'{"rationale":"x","quotes":[\\\\"a\\\\",\\\\"b\\\\"],"score":1}'

    rationale, matches, score = extract_matches_from_response_text(content)

    assert rationale == "x"
    assert matches == ["a", "b"]
    assert score == 1


def test_extract_matches_does_not_require_field_order():
    """Parser should accept valid JSON regardless of key order."""
    content = '{"quotes":["a"],"rationale":"x","score":2}'

    rationale, matches, score = extract_matches_from_response_text(content)

    assert rationale == "x"
    assert matches == ["a"]
    assert score == 2


def test_extract_matches_rejects_missing_required_fields():
    """Parser should still fail when required fields are truly missing."""
    content = '{"rationale":"x","quotes":["a"]}'

    with pytest.raises(ClassificationError, match="must include"):
        extract_matches_from_response_text(content)
