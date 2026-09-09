from __future__ import annotations

import pandas as pd

from src.config import ExperimentConfig, SHAPConfig, SelectorGridConfig, XGBoostConfig
from src.data import build_source_household_split, sample_households_up_to_limit


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


def test_build_source_household_split_assigns_every_household_once() -> None:
    config = make_config()
    frame = pd.DataFrame({"SERIALNO": [f"{idx:05d}" for idx in range(20)]})

    split_manifest = build_source_household_split(frame, config)

    assert split_manifest["household_id"].nunique() == 20
    assert split_manifest["partition"].isin(["train", "selector_fit", "validation", "test"]).all()
    assert len(split_manifest) == 20


def test_sample_households_up_to_limit_preserves_household_integrity() -> None:
    frame = pd.DataFrame(
        {
            "record_key": ["a1", "a2", "b1", "c1", "c2"],
            "household_id": ["A", "A", "B", "C", "C"],
        }
    )

    sampled = sample_households_up_to_limit(frame, limit=3, seed=7)

    sampled_households = sampled.groupby("household_id").size().to_dict()
    assert sampled_households in ({"A": 2, "B": 1}, {"B": 1, "C": 2})
    assert len(sampled) <= 3
