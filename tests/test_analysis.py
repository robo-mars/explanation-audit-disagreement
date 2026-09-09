from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis import BootstrapView, evaluate_policy_metrics, referral_mask, validation_threshold, weighted_selective_risk_from_view


def test_referral_mask_uses_record_key_tie_break() -> None:
    frame = pd.DataFrame(
        {
            "task_name": ["ACSIncome"] * 4,
            "seed": [11] * 4,
            "state": ["TX"] * 4,
            "partition": ["test"] * 4,
            "dataset_name": ["TX_test"] * 4,
            "record_key": ["b", "a", "d", "c"],
            "error_indicator": [0, 1, 0, 1],
            "score": [0.9, 0.9, 0.2, 0.2],
        }
    )

    referred = referral_mask(frame, "score", review_budget=0.5)
    assert referred.tolist() == [True, True, False, False]


def test_evaluate_policy_metrics_computes_selective_risk() -> None:
    frame = pd.DataFrame(
        {
            "task_name": ["ACSIncome"] * 4,
            "seed": [11] * 4,
            "state": ["NY"] * 4,
            "partition": ["test"] * 4,
            "dataset_name": ["NY_test"] * 4,
            "record_key": ["a", "b", "c", "d"],
            "error_indicator": [1, 0, 1, 0],
            "policy_a": [0.9, 0.8, 0.1, 0.0],
        }
    )

    metrics = evaluate_policy_metrics(frame, ["policy_a"], (0.5,))
    assert metrics.loc[0, "n_referred"] == 2
    assert metrics.loc[0, "n_retained"] == 2
    assert metrics.loc[0, "selective_risk"] == 0.5
    assert metrics.loc[0, "error_capture"] == 0.5


def test_validation_threshold_returns_cutoff_score() -> None:
    frame = pd.DataFrame(
        {
            "record_key": ["a", "b", "c", "d"],
            "policy": [0.9, 0.7, 0.7, 0.1],
        }
    )

    threshold = validation_threshold(frame, "policy", 0.5)
    assert threshold == 0.7


def test_weighted_selective_risk_matches_expanded_sample() -> None:
    view = BootstrapView(
        household_codes=np.array([0, 0, 1, 2]),
        errors=np.array([1.0, 0.0, 1.0, 0.0]),
    )
    state_household_counts = np.array([2, 1, 1])

    risk = weighted_selective_risk_from_view(view, state_household_counts, review_budget=0.5)

    # Expanded sample after weighting is [rec1, rec1, rec2, rec2, rec3, rec4].
    # Top 3 weighted copies are referred, leaving errors [1, 0, 0].
    assert risk == 1.0 / 3.0
