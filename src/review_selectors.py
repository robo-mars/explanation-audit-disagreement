from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import ExperimentConfig
from src.utils import referral_count_from_budget, stable_uniform


UNCERTAINTY_FEATURES = ["p_bar", "entropy", "variance", "vote_disagreement"]


@dataclass(frozen=True)
class SelectorFeatureSpec:
    policy_name: str
    use_uncertainty: bool = True
    disagreement_column: str | None = None
    include_transformed_features: bool = False
    include_random_feature: bool = False


@dataclass
class TrainedSelector:
    spec: SelectorFeatureSpec
    chosen_c: float
    pipeline: Pipeline
    validation_selective_risk: float


@dataclass
class SelectorBundle:
    trained_selectors: dict[str, TrainedSelector]
    candidate_metrics: pd.DataFrame


def primary_selector_specs() -> list[SelectorFeatureSpec]:
    return [
        SelectorFeatureSpec(policy_name="uncertainty_selector"),
        SelectorFeatureSpec(policy_name="explanation_only", use_uncertainty=False, disagreement_column="explanation_disagreement"),
        SelectorFeatureSpec(policy_name="augmented_selector", disagreement_column="explanation_disagreement"),
        SelectorFeatureSpec(policy_name="raw_feature_selector", include_transformed_features=True),
        SelectorFeatureSpec(policy_name="random_feature_control", include_random_feature=True),
        SelectorFeatureSpec(policy_name="explanation_top3", use_uncertainty=False, disagreement_column="top3_feature_disagreement"),
        SelectorFeatureSpec(policy_name="augmented_top3", disagreement_column="top3_feature_disagreement"),
    ]


def train_selectors(
    selector_fit_summary: pd.DataFrame,
    selector_fit_features: pd.DataFrame,
    validation_summary: pd.DataFrame,
    validation_features: pd.DataFrame,
    config: ExperimentConfig,
    seed: int,
) -> SelectorBundle:
    trained: dict[str, TrainedSelector] = {}
    candidate_rows = []

    y_train = selector_fit_summary["error_indicator"].to_numpy(dtype=int)
    y_validation = validation_summary["error_indicator"].to_numpy(dtype=int)

    for spec in primary_selector_specs():
        train_matrix = build_selector_matrix(selector_fit_summary, selector_fit_features, spec, config, seed)
        validation_matrix = build_selector_matrix(validation_summary, validation_features, spec, config, seed)

        best_model: Pipeline | None = None
        best_c: float | None = None
        best_risk = float("inf")

        for c_value in sorted(config.selector_grid.C):
            pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "logistic",
                        LogisticRegression(
                            C=c_value,
                            penalty="l2",
                            solver="liblinear",
                            max_iter=4000,
                            random_state=seed,
                        ),
                    ),
                ]
            )
            pipeline.fit(train_matrix, y_train)
            validation_scores = pipeline.predict_proba(validation_matrix)[:, 1]
            validation_risk = selective_risk_from_scores(
                errors=y_validation,
                scores=validation_scores,
                record_keys=validation_summary["record_key"].astype(str),
                review_budget=config.coverage_primary,
            )

            candidate_rows.append(
                {
                    "task_name": selector_fit_summary["task_name"].iloc[0],
                    "seed": seed,
                    "policy_name": spec.policy_name,
                    "C": c_value,
                    "validation_selective_risk": validation_risk,
                }
            )

            if (
                validation_risk < best_risk - 1e-12
                or (np.isclose(validation_risk, best_risk, equal_nan=True) and (best_c is None or c_value < best_c))
            ):
                best_risk = validation_risk
                best_c = c_value
                best_model = pipeline

        if best_model is None or best_c is None:
            raise ValueError(f"Failed to train selector {spec.policy_name}")

        trained[spec.policy_name] = TrainedSelector(
            spec=spec,
            chosen_c=best_c,
            pipeline=best_model,
            validation_selective_risk=best_risk,
        )

    return SelectorBundle(
        trained_selectors=trained,
        candidate_metrics=pd.DataFrame(candidate_rows).sort_values(["policy_name", "C"]).reset_index(drop=True),
    )


def build_selector_matrix(
    summary: pd.DataFrame,
    transformed_features: pd.DataFrame,
    spec: SelectorFeatureSpec,
    config: ExperimentConfig,
    seed: int,
) -> pd.DataFrame:
    components: list[pd.DataFrame] = []
    if spec.use_uncertainty:
        components.append(summary[UNCERTAINTY_FEATURES].copy())
    if spec.disagreement_column is not None:
        components.append(summary[[spec.disagreement_column]].rename(columns={spec.disagreement_column: "disagreement_feature"}))
    if spec.include_transformed_features:
        renamed = transformed_features.copy()
        renamed.columns = [f"raw::{column}" for column in renamed.columns]
        components.append(renamed)
    if spec.include_random_feature:
        random_feature = summary["record_key"].astype(str).map(
            lambda value: stable_uniform(value, seed + config.random_policy_seed)
        )
        components.append(pd.DataFrame({"random_control_feature": random_feature.to_numpy()}, index=summary.index))

    if not components:
        raise ValueError(f"Selector {spec.policy_name} has no feature components")
    return pd.concat(components, axis=1)


def score_policies(
    summary: pd.DataFrame,
    transformed_features: pd.DataFrame,
    selectors: SelectorBundle,
    config: ExperimentConfig,
    seed: int,
) -> pd.DataFrame:
    policy_scores = pd.DataFrame(index=summary.index)
    policy_scores["random"] = summary["record_key"].astype(str).map(
        lambda value: stable_uniform(value, seed + config.random_policy_seed)
    )
    policy_scores["confidence"] = summary["low_confidence"].to_numpy()
    policy_scores["ensemble_variance"] = summary["variance"].to_numpy()

    for policy_name, selector in selectors.trained_selectors.items():
        matrix = build_selector_matrix(summary, transformed_features, selector.spec, config, seed)
        policy_scores[policy_name] = selector.pipeline.predict_proba(matrix)[:, 1]

    return policy_scores.reset_index(drop=True)


def selective_risk_from_scores(
    errors: np.ndarray,
    scores: np.ndarray,
    record_keys: pd.Series,
    review_budget: float,
) -> float:
    if len(errors) == 0:
        return float("nan")
    ordered = pd.DataFrame(
        {
            "score": scores,
            "record_key": record_keys.to_numpy(),
            "error_indicator": errors,
        }
    ).sort_values(["score", "record_key"], ascending=[False, True], kind="mergesort")

    review_count = referral_count_from_budget(len(ordered), review_budget)
    retained = ordered.iloc[review_count:]
    if retained.empty:
        return float("nan")
    return float(retained["error_indicator"].mean())
