from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReconstructionCheck:
    member_id: int
    base_value: float
    max_abs_error: float
    mean_abs_error: float
    tolerance: float


def aggregate_shap_by_original(
    shap_values: np.ndarray,
    transformed_feature_names: list[str],
    transformed_to_original: list[str],
    original_feature_names: list[str],
) -> pd.DataFrame:
    if shap_values.ndim != 2:
        raise ValueError("shap_values must be a 2D array")
    if shap_values.shape[1] != len(transformed_feature_names):
        raise ValueError("shap_values and transformed_feature_names length mismatch")
    if len(transformed_to_original) != len(transformed_feature_names):
        raise ValueError("transformed_to_original and transformed_feature_names length mismatch")

    grouped = pd.DataFrame(0.0, index=np.arange(shap_values.shape[0]), columns=original_feature_names)
    transformed_df = pd.DataFrame(shap_values, columns=transformed_feature_names)
    for transformed_name, original_name in zip(transformed_feature_names, transformed_to_original, strict=True):
        grouped[original_name] += transformed_df[transformed_name]
    return grouped


def explanation_disagreement(
    grouped_member_values: list[np.ndarray],
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(grouped_member_values) < 2:
        raise ValueError("At least two ensemble members are required")
    member_stack = np.stack(grouped_member_values, axis=0)
    l1_norms = np.abs(member_stack).sum(axis=2)
    normalized = member_stack / np.maximum(l1_norms[..., None], epsilon)

    pairwise_distances = []
    for left in range(normalized.shape[0]):
        for right in range(left + 1, normalized.shape[0]):
            pairwise = np.abs(normalized[left] - normalized[right]).sum(axis=1) / 2.0
            pairwise_distances.append(pairwise)

    disagreement = np.mean(np.stack(pairwise_distances, axis=0), axis=0)
    attribution_magnitudes = l1_norms.mean(axis=0)
    return disagreement, attribution_magnitudes


def topk_feature_disagreement(
    grouped_member_values: list[np.ndarray],
    feature_names: list[str],
    topk: int = 3,
) -> np.ndarray:
    if len(grouped_member_values) < 2:
        raise ValueError("At least two ensemble members are required")
    if topk <= 0:
        raise ValueError("topk must be positive")

    topk_feature_sets: list[list[set[str]]] = []
    for values in grouped_member_values:
        top_sets = []
        for row in values:
            order = sorted(
                zip(np.abs(row), feature_names, strict=True),
                key=lambda item: (-item[0], item[1]),
            )
            top_sets.append({feature for _, feature in order[:topk]})
        topk_feature_sets.append(top_sets)

    pairwise = []
    n_records = len(topk_feature_sets[0])
    for left in range(len(topk_feature_sets)):
        for right in range(left + 1, len(topk_feature_sets)):
            row_scores = np.zeros(n_records)
            for index in range(n_records):
                overlap = len(topk_feature_sets[left][index] & topk_feature_sets[right][index])
                row_scores[index] = 1.0 - (overlap / topk)
            pairwise.append(row_scores)

    return np.mean(np.stack(pairwise, axis=0), axis=0)
