"""
Unit tests for the Phase 3 few-shot comparison pipeline's own new logic (classifier
fit/eval, summary aggregation). GIN/GRU scoring and the generator itself are already
covered by test_embedding_sequence.py, test_gru_temporal.py, and test_generator.py --
these tests target what Phase 3 adds on top: fit_and_evaluate and summarize.
"""

import numpy as np
import pytest

from src.evaluate.phase3_final_comparison import fit_and_evaluate, summarize


CLASSIFIER_CFG = {"C": 1.0, "class_weight": "balanced"}


def test_fit_and_evaluate_returns_expected_keys():
    rng = np.random.default_rng(0)
    pos = [rng.normal(5, 1, size=4) for _ in range(5)]
    neg = [rng.normal(0, 1, size=4) for _ in range(50)]
    test_vectors = {f"t{i}": rng.normal(2.5, 1, size=4) for i in range(10)}
    test_labels = {sid: i % 2 for i, sid in enumerate(test_vectors)}

    metrics = fit_and_evaluate(pos, neg, test_vectors, test_labels, CLASSIFIER_CFG)
    for key in ["precision", "recall", "f1", "auroc", "auprc", "num_positive_train", "num_negative_train"]:
        assert key in metrics
    assert metrics["num_positive_train"] == 5
    assert metrics["num_negative_train"] == 50


def test_fit_and_evaluate_deterministic():
    rng = np.random.default_rng(1)
    pos = [rng.normal(5, 1, size=3) for _ in range(3)]
    neg = [rng.normal(0, 1, size=3) for _ in range(20)]
    test_vectors = {f"t{i}": rng.normal(2.5, 1, size=3) for i in range(8)}
    test_labels = {sid: i % 2 for i, sid in enumerate(test_vectors)}

    m1 = fit_and_evaluate(pos, neg, test_vectors, test_labels, CLASSIFIER_CFG)
    m2 = fit_and_evaluate(pos, neg, test_vectors, test_labels, CLASSIFIER_CFG)
    assert m1 == m2


def test_fit_and_evaluate_separable_classes_score_well():
    """Sanity check: if the feature perfectly separates classes, AUROC should be ~1.0."""
    pos = [np.array([10.0]) for _ in range(5)]
    neg = [np.array([0.0]) for _ in range(30)]
    test_vectors = {"a": np.array([10.0]), "b": np.array([0.0]), "c": np.array([9.0]), "d": np.array([0.5])}
    test_labels = {"a": 1, "b": 0, "c": 1, "d": 0}

    metrics = fit_and_evaluate(pos, neg, test_vectors, test_labels, CLASSIFIER_CFG)
    assert metrics["auroc"] == pytest.approx(1.0)


def test_summarize_computes_mean_std_and_deltas():
    fake_results = {
        "1-shot": {
            "few_shot_only": [
                {"precision": 0.5, "recall": 0.2, "f1": 0.28, "auroc": 0.5, "auprc": 0.4},
                {"precision": 0.6, "recall": 0.3, "f1": 0.40, "auroc": 0.6, "auprc": 0.5},
            ],
            "few_shot_plus_structure_aware": [
                {"precision": 0.6, "recall": 0.4, "f1": 0.48, "auroc": 0.55, "auprc": 0.45},
                {"precision": 0.7, "recall": 0.5, "f1": 0.58, "auroc": 0.65, "auprc": 0.55},
            ],
            "few_shot_plus_random": [
                {"precision": 0.5, "recall": 0.2, "f1": 0.28, "auroc": 0.5, "auprc": 0.4},
                {"precision": 0.5, "recall": 0.2, "f1": 0.28, "auroc": 0.5, "auprc": 0.4},
            ],
        }
    }
    summary = summarize(fake_results)
    a = summary["1-shot"]["few_shot_only"]
    assert a["f1_mean"] == pytest.approx((0.28 + 0.40) / 2)

    b = summary["1-shot"]["few_shot_plus_structure_aware"]
    assert b["deltas_vs_few_shot_only"]["delta_f1"] == pytest.approx(b["f1_mean"] - a["f1_mean"])
    assert b["deltas_vs_few_shot_only"]["delta_auroc"] == pytest.approx(b["auroc_mean"] - a["auroc_mean"])
