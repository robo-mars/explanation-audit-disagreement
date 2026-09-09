from __future__ import annotations

import numpy as np

from src.explanations import aggregate_shap_by_original, explanation_disagreement, topk_feature_disagreement


def test_aggregate_shap_by_original_sums_one_hot_columns() -> None:
    shap_values = np.array([[0.3, 0.2, -0.1], [0.1, -0.2, 0.5]])
    grouped = aggregate_shap_by_original(
        shap_values=shap_values,
        transformed_feature_names=["COW=1", "COW=2", "AGEP"],
        transformed_to_original=["COW", "COW", "AGEP"],
        original_feature_names=["AGEP", "COW"],
    )

    assert grouped["COW"].tolist() == [0.5, -0.1]
    assert grouped["AGEP"].tolist() == [-0.1, 0.5]


def test_explanation_disagreement_is_zero_for_identical_members() -> None:
    member_values = [
        np.array([[1.0, 0.0], [0.5, 0.5]]),
        np.array([[1.0, 0.0], [0.5, 0.5]]),
    ]

    disagreement, magnitudes = explanation_disagreement(member_values, epsilon=1e-8)

    assert np.allclose(disagreement, np.zeros(2))
    assert np.allclose(magnitudes, np.array([1.0, 1.0]))


def test_topk_feature_disagreement_uses_overlap_fraction() -> None:
    member_values = [
        np.array([[5.0, 4.0, 3.0, 0.0]]),
        np.array([[5.0, 4.0, 0.0, 3.0]]),
    ]

    disagreement = topk_feature_disagreement(member_values, feature_names=["A", "B", "C", "D"], topk=3)

    # Top-3 overlap is {A, B}, so disagreement is 1 - 2/3.
    assert np.allclose(disagreement, np.array([1.0 / 3.0]))
