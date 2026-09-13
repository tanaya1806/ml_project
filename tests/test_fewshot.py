"""Unit tests for Phase 1E few-shot support-set sampling and recurrence."""

import pytest

from src.fewshot.support import sample_support_set, assert_subject_isolation, compute_recurrence


POOL = [f"SUBJ_{i:03d}" for i in range(50)]


def test_sample_support_set_shot_count():
    support = sample_support_set(POOL, shot=5, seed=1)
    assert len(support) == 5
    assert len(set(support)) == 5
    assert all(s in POOL for s in support)


def test_sample_support_set_reproducible():
    a = sample_support_set(POOL, shot=10, seed=42)
    b = sample_support_set(POOL, shot=10, seed=42)
    assert a == b


def test_sample_support_set_different_seeds_differ():
    a = sample_support_set(POOL, shot=10, seed=1)
    b = sample_support_set(POOL, shot=10, seed=2)
    assert a != b


def test_sample_support_set_too_large_raises():
    with pytest.raises(ValueError):
        sample_support_set(POOL, shot=len(POOL) + 1, seed=1)


def test_subject_isolation_raises_on_overlap():
    support = ["A", "B", "C"]
    with pytest.raises(ValueError):
        assert_subject_isolation(support, forbidden_ids=["B", "Z"])


def test_subject_isolation_passes_when_disjoint():
    support = ["A", "B", "C"]
    assert_subject_isolation(support, forbidden_ids=["X", "Y"])  # should not raise


def test_compute_recurrence():
    flags = {
        "s1": {10: True, 20: False},
        "s2": {10: True, 20: True},
        "s3": {10: False, 20: False},
    }
    recurrence = compute_recurrence(["s1", "s2", "s3"], flags)
    assert recurrence[10] == pytest.approx(2 / 3)
    assert recurrence[20] == pytest.approx(1 / 3)


def test_compute_recurrence_empty_support():
    assert compute_recurrence([], {}) == {}
