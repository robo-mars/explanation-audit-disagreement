from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.config import ExperimentConfig
from src.explanations import ReconstructionCheck, aggregate_shap_by_original, explanation_disagreement, topk_feature_disagreement
from src.preprocessing import PreprocessorBundle
from src.utils import timed_stage


@dataclass
class EnsembleMember:
    member_id: int
    model: Any
    bootstrap_manifest: pd.DataFrame


@dataclass
class EnsembleBundle:
    task_name: str
    seed: int
    preprocessor: PreprocessorBundle
    background_matrix: pd.DataFrame
    members: list[EnsembleMember]


@dataclass
class ScoredPartition:
    dataset_name: str
    state: str
    partition: str
    summary: pd.DataFrame
    transformed_features: pd.DataFrame
    grouped_attributions: pd.DataFrame
    reconstruction_checks: pd.DataFrame


def fit_ensemble(
    task_name: str,
    seed: int,
    training_frame: pd.DataFrame,
    background_frame: pd.DataFrame,
    model_features: list[str],
    preprocessor: PreprocessorBundle,
    config: ExperimentConfig,
    runtime_sink: list[dict[str, object]],
) -> EnsembleBundle:
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        if "libomp" in str(exc):
            raise RuntimeError(
                "xgboost could not load its OpenMP runtime on macOS. "
                "Install libomp with `brew install libomp` and rerun the experiment."
            ) from exc
        raise

    transformed_training = preprocessor.transform(training_frame[model_features])
    training_labels = training_frame["label"].to_numpy(dtype=int)
    background_matrix = preprocessor.transform(background_frame[model_features])

    members: list[EnsembleMember] = []
    for member_id in range(1, config.n_members + 1):
        bootstrap_weights, bootstrap_manifest = household_bootstrap_weights(
            training_frame["household_id"],
            seed=(config.xgb.random_state_offset + seed * 100 + member_id),
        )
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=config.xgb.n_estimators,
            max_depth=config.xgb.max_depth,
            learning_rate=config.xgb.learning_rate,
            subsample=config.xgb.subsample,
            colsample_bytree=config.xgb.colsample_bytree,
            reg_lambda=config.xgb.reg_lambda,
            random_state=config.xgb.random_state_offset + seed * 100 + member_id,
            tree_method="hist",
        )
        with timed_stage(
            "fit_member_model",
            runtime_sink,
            task_name=task_name,
            seed=seed,
            member_id=member_id,
        ):
            model.fit(transformed_training, training_labels, sample_weight=bootstrap_weights)
        members.append(
            EnsembleMember(
                member_id=member_id,
                model=model,
                bootstrap_manifest=bootstrap_manifest,
            )
        )

    return EnsembleBundle(
        task_name=task_name,
        seed=seed,
        preprocessor=preprocessor,
        background_matrix=background_matrix,
        members=members,
    )


def household_bootstrap_weights(household_ids: pd.Series, seed: int) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    unique_households = np.array(sorted(household_ids.astype(str).unique()))
    sampled_households = rng.choice(unique_households, size=len(unique_households), replace=True)
    bootstrap_manifest = (
        pd.Series(sampled_households, name="household_id")
        .value_counts()
        .rename_axis("household_id")
        .reset_index(name="bootstrap_count")
    )
    weight_lookup = bootstrap_manifest.set_index("household_id")["bootstrap_count"]
    weights = household_ids.astype(str).map(weight_lookup).fillna(0).to_numpy(dtype=float)
    return weights, bootstrap_manifest.sort_values("household_id").reset_index(drop=True)


def score_partition(
    ensemble: EnsembleBundle,
    frame: pd.DataFrame,
    model_features: list[str],
    config: ExperimentConfig,
    runtime_sink: list[dict[str, object]],
    dataset_name: str,
    background_override: pd.DataFrame | None = None,
) -> ScoredPartition:
    import shap

    transformed_features = ensemble.preprocessor.transform(frame[model_features])
    background_matrix = (
        background_override
        if background_override is not None
        else ensemble.background_matrix
    )

    probability_columns: dict[str, np.ndarray] = {}
    member_probabilities: list[np.ndarray] = []
    with timed_stage(
        "predict_partition",
        runtime_sink,
        task_name=ensemble.task_name,
        seed=ensemble.seed,
        dataset_name=dataset_name,
        n_records=len(frame),
    ):
        for member in ensemble.members:
            probabilities = member.model.predict_proba(transformed_features)[:, 1]
            probability_columns[f"member_{member.member_id}_probability"] = probabilities
            member_probabilities.append(probabilities)

    member_probability_matrix = np.column_stack(member_probabilities)
    p_bar = member_probability_matrix.mean(axis=1)
    ensemble_class = (p_bar >= config.vote_threshold).astype(int)
    member_classes = (member_probability_matrix >= config.vote_threshold).astype(int)
    vote_disagreement = (member_classes != ensemble_class[:, None]).mean(axis=1)
    variance = member_probability_matrix.var(axis=1)
    entropy = binary_entropy(p_bar)
    confidence = np.maximum(p_bar, 1.0 - p_bar)
    low_confidence = 1.0 - confidence
    error_indicator = (ensemble_class != frame["label"].to_numpy(dtype=int)).astype(int)

    grouped_attribution_tables: list[pd.DataFrame] = []
    grouped_arrays: list[np.ndarray] = []
    l1_magnitudes: list[np.ndarray] = []
    reconstruction_checks: list[ReconstructionCheck] = []

    for member in ensemble.members:
        with timed_stage(
            "score_member_shap",
            runtime_sink,
            task_name=ensemble.task_name,
            seed=ensemble.seed,
            dataset_name=dataset_name,
            member_id=member.member_id,
            n_records=len(frame),
        ):
            explainer = shap.TreeExplainer(
                member.model,
                data=background_matrix,
                feature_perturbation=config.shap.feature_perturbation,
                model_output=config.shap.model_output,
            )
            explanation = explainer(transformed_features, check_additivity=False)

        shap_values = np.asarray(explanation.values)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, -1]

        base_values = np.asarray(explanation.base_values)
        if base_values.ndim == 0:
            base_values = np.full(len(frame), float(base_values))
        elif base_values.ndim > 1:
            base_values = base_values.reshape(len(frame), -1)[:, -1]

        grouped = aggregate_shap_by_original(
            shap_values=shap_values,
            transformed_feature_names=ensemble.preprocessor.transformed_feature_names,
            transformed_to_original=ensemble.preprocessor.transformed_to_original,
            original_feature_names=ensemble.preprocessor.original_feature_names,
        )
        grouped_arrays.append(grouped.to_numpy())
        l1_values = np.abs(grouped.to_numpy()).sum(axis=1)
        l1_magnitudes.append(l1_values)

        reconstruction = base_values + grouped.sum(axis=1).to_numpy()
        probabilities = probability_columns[f"member_{member.member_id}_probability"]
        abs_error = np.abs(reconstruction - probabilities)
        max_abs_error = float(abs_error.max()) if len(abs_error) else 0.0
        if max_abs_error > config.shap.reconstruction_tolerance:
            raise ValueError(
                f"SHAP reconstruction exceeded tolerance for {dataset_name}, "
                f"member {member.member_id}: {max_abs_error:.6g}"
            )

        reconstruction_checks.append(
            ReconstructionCheck(
                member_id=member.member_id,
                base_value=float(np.mean(base_values)),
                max_abs_error=max_abs_error,
                mean_abs_error=float(abs_error.mean()) if len(abs_error) else 0.0,
                tolerance=config.shap.reconstruction_tolerance,
            )
        )

        grouped_table = grouped.copy()
        grouped_table.insert(0, "record_key", frame["record_key"].to_numpy())
        grouped_table.insert(0, "member_id", member.member_id)
        grouped_table.insert(0, "dataset_name", dataset_name)
        grouped_table.insert(0, "partition", frame["partition"].iloc[0])
        grouped_table.insert(0, "state", frame["state"].iloc[0])
        grouped_table.insert(0, "seed", ensemble.seed)
        grouped_table.insert(0, "task_name", ensemble.task_name)
        grouped_attribution_tables.append(
            grouped_table.melt(
                id_vars=["task_name", "seed", "state", "partition", "dataset_name", "member_id", "record_key"],
                var_name="feature_name",
                value_name="shap_value",
            )
        )

    explanation_scores, attribution_l1_mean = explanation_disagreement(
        grouped_member_values=grouped_arrays,
        epsilon=config.shap.l1_epsilon,
    )
    top3_disagreement = topk_feature_disagreement(
        grouped_member_values=grouped_arrays,
        feature_names=ensemble.preprocessor.original_feature_names,
        topk=3,
    )
    l1_stack = np.column_stack(l1_magnitudes)

    summary = frame[
        [
            "task_name",
            "state",
            "partition",
            "record_key",
            "household_id",
            "row_id",
            "person_order",
            "survey_weight",
            "sex_code",
            "sex_label",
            "race_code",
            "race_label",
            "label",
        ]
    ].copy()
    summary["dataset_name"] = dataset_name
    summary["seed"] = ensemble.seed
    summary["p_bar"] = p_bar
    summary["ensemble_prediction"] = ensemble_class
    summary["error_indicator"] = error_indicator
    summary["entropy"] = entropy
    summary["variance"] = variance
    summary["vote_disagreement"] = vote_disagreement
    summary["ensemble_confidence"] = confidence
    summary["low_confidence"] = low_confidence
    summary["explanation_disagreement"] = explanation_scores
    summary["top3_feature_disagreement"] = top3_disagreement
    summary["attribution_l1_mean"] = attribution_l1_mean
    summary["attribution_l1_min"] = l1_stack.min(axis=1)
    summary["attribution_l1_max"] = l1_stack.max(axis=1)
    for column_name, values in probability_columns.items():
        summary[column_name] = values

    return ScoredPartition(
        dataset_name=dataset_name,
        state=str(frame["state"].iloc[0]),
        partition=str(frame["partition"].iloc[0]),
        summary=summary.reset_index(drop=True),
        transformed_features=transformed_features.reset_index(drop=True),
        grouped_attributions=pd.concat(grouped_attribution_tables, ignore_index=True),
        reconstruction_checks=pd.DataFrame([check.__dict__ for check in reconstruction_checks]),
    )


def binary_entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    entropy = -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))
    entropy[(probabilities == 0.0) | (probabilities == 1.0)] = 0.0
    return entropy
