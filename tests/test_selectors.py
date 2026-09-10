from __future__ import annotations

import pandas as pd

from src.config import ExperimentConfig, SHAPConfig, SelectorGridConfig, XGBoostConfig
from src.review_selectors import score_policies, train_selectors


def make_config() -> ExperimentConfig:
    return ExperimentConfig(
        name="test",
        data_year=2018,
        source_state="CA",
        target_states=("TX", "NY"),
        tasks=("ACSIncome",),
        seeds=(11,),
        n_members=5,
        train_limit=10,
        explanation_limit=10,
        sh_train_background=5,
        sh_pilot_records=5,
        coverage_primary=0.2,
        coverage_secondary=(0.1,),
        bootstrap_reps=10,
        split_seed=123,
        sampling_seed=456,
        bootstrap_seed=789,
        random_policy_seed=314159,
        robustness_background_limit=5,
        fairness_min_group_size=100,
        vote_threshold=0.5,
        review_rounding="half_up",
        review_tie_breaker="record_key_ascending",
        xgb=XGBoostConfig(
            n_estimators=10,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state_offset=1000,
        ),
        selector_grid=SelectorGridConfig(C=(0.1, 1.0)),
        shap=SHAPConfig(
            feature_perturbation="interventional",
            model_output="probability",
            grouping="original_variable",
            l1_epsilon=1e-8,
            reconstruction_tolerance=1e-4,
        ),
    )


def make_summary_frame(dataset_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "task_name": ["ACSIncome"] * 6,
            "seed": [11] * 6,
            "state": ["CA"] * 6,
            "partition": ["selector_fit"] * 6,
            "dataset_name": [dataset_name] * 6,
            "record_key": [f"row-{index}" for index in range(6)],
            "error_indicator": [1, 0, 1, 0, 1, 0],
            "p_bar": [0.9, 0.2, 0.8, 0.1, 0.7, 0.3],
            "entropy": [0.3, 0.6, 0.35, 0.7, 0.4, 0.65],
            "variance": [0.2, 0.1, 0.18, 0.08, 0.16, 0.12],
            "vote_disagreement": [0.4, 0.0, 0.2, 0.0, 0.2, 0.0],
            "explanation_disagreement": [0.8, 0.1, 0.7, 0.2, 0.65, 0.15],
            "top3_feature_disagreement": [0.7, 0.2, 0.65, 0.25, 0.6, 0.3],
            "low_confidence": [0.1, 0.2, 0.2, 0.1, 0.3, 0.3],
        }
    )


def test_train_selectors_and_score_policies() -> None:
    config = make_config()
    selector_fit_summary = make_summary_frame("selector_fit")
    validation_summary = make_summary_frame("validation")
    selector_fit_features = pd.DataFrame(
        {
            "AGEP": [0.1, -0.2, 0.3, -0.1, 0.5, -0.4],
            "WKHP": [1.0, 0.2, 0.8, 0.1, 0.9, 0.3],
        }
    )
    validation_features = selector_fit_features.iloc[::-1].reset_index(drop=True)

    bundle = train_selectors(
        selector_fit_summary=selector_fit_summary,
        selector_fit_features=selector_fit_features,
        validation_summary=validation_summary,
        validation_features=validation_features,
        config=config,
        seed=11,
    )
    scores = score_policies(
        summary=validation_summary,
        transformed_features=validation_features,
        selectors=bundle,
        config=config,
        seed=11,
    )

    assert {"random", "confidence", "ensemble_variance", "uncertainty_selector", "augmented_selector"}.issubset(scores.columns)
    assert len(bundle.candidate_metrics) == 14
