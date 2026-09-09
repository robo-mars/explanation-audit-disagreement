from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import ExperimentConfig
from src.utils import referral_count_from_budget


PRIMARY_POLICY_ORDER = [
    "random",
    "confidence",
    "ensemble_variance",
    "uncertainty_selector",
    "explanation_only",
    "augmented_selector",
    "raw_feature_selector",
]

ROBUSTNESS_POLICY_ORDER = [
    "random_feature_control",
    "explanation_top3",
    "augmented_top3",
]


@dataclass(frozen=True)
class BootstrapView:
    household_codes: np.ndarray
    errors: np.ndarray


def evaluate_policy_metrics(
    scored_frame: pd.DataFrame,
    policies: list[str],
    review_budgets: tuple[float, ...],
) -> pd.DataFrame:
    rows = []
    errors = scored_frame["error_indicator"].to_numpy(dtype=int)

    for policy_name in policies:
        auroc = safe_roc_auc(errors, scored_frame[policy_name].to_numpy())
        average_precision = safe_average_precision(errors, scored_frame[policy_name].to_numpy())
        for review_budget in review_budgets:
            referred_mask = referral_mask(scored_frame, policy_name, review_budget)
            retained_mask = ~referred_mask

            retained_errors = errors[retained_mask]
            selective_risk = float(retained_errors.mean()) if retained_errors.size else float("nan")
            total_errors = int(errors.sum())
            error_capture = (
                float(errors[referred_mask].sum() / total_errors)
                if total_errors > 0
                else float("nan")
            )

            rows.append(
                {
                    "task_name": scored_frame["task_name"].iloc[0],
                    "seed": int(scored_frame["seed"].iloc[0]),
                    "state": scored_frame["state"].iloc[0],
                    "partition": scored_frame["partition"].iloc[0],
                    "dataset_name": scored_frame["dataset_name"].iloc[0],
                    "policy_name": policy_name,
                    "review_budget": review_budget,
                    "retained_coverage": 1.0 - review_budget,
                    "n_records": len(scored_frame),
                    "n_referred": int(referred_mask.sum()),
                    "n_retained": int(retained_mask.sum()),
                    "selective_risk": selective_risk,
                    "error_capture": error_capture,
                    "auroc": auroc,
                    "average_precision": average_precision,
                }
            )

    return pd.DataFrame(rows).sort_values(["policy_name", "review_budget"]).reset_index(drop=True)


def compute_risk_coverage_curves(scored_frame: pd.DataFrame, policies: list[str]) -> pd.DataFrame:
    rows = []
    base_columns = {
        "task_name": scored_frame["task_name"].iloc[0],
        "seed": int(scored_frame["seed"].iloc[0]),
        "state": scored_frame["state"].iloc[0],
        "partition": scored_frame["partition"].iloc[0],
        "dataset_name": scored_frame["dataset_name"].iloc[0],
    }

    for policy_name in policies:
        ordered = order_for_policy(scored_frame, policy_name)
        errors = ordered["error_indicator"].to_numpy(dtype=float)
        n_records = len(ordered)
        total_errors = float(errors.sum())
        cumulative_errors = np.concatenate(([0.0], np.cumsum(errors)))

        for referred_count in range(0, n_records + 1):
            retained_count = n_records - referred_count
            if retained_count > 0:
                retained_error_count = total_errors - cumulative_errors[referred_count]
                selective_risk = retained_error_count / retained_count
            else:
                selective_risk = float("nan")
            error_capture = cumulative_errors[referred_count] / total_errors if total_errors > 0 else float("nan")

            rows.append(
                {
                    **base_columns,
                    "policy_name": policy_name,
                    "n_records": n_records,
                    "n_referred": referred_count,
                    "n_retained": retained_count,
                    "review_budget": referred_count / n_records if n_records else float("nan"),
                    "retained_coverage": retained_count / n_records if n_records else float("nan"),
                    "selective_risk": float(selective_risk),
                    "error_capture": float(error_capture),
                }
            )

    return pd.DataFrame(rows)


def compute_group_audit(
    scored_frame: pd.DataFrame,
    policies: list[str],
    review_budget: float,
    min_group_size: int,
) -> pd.DataFrame:
    rows = []
    group_specs = [("sex", "sex_code", "sex_label"), ("race", "race_code", "race_label")]

    for policy_name in policies:
        referred_mask = referral_mask(scored_frame, policy_name, review_budget)
        retained_mask = ~referred_mask
        for group_name, code_column, label_column in group_specs:
            for group_value, group_frame in scored_frame.groupby(code_column, dropna=False, sort=False):
                group_index = group_frame.index
                group_referred = referred_mask[group_index]
                group_retained = retained_mask[group_index]
                group_errors = group_frame["error_indicator"].to_numpy(dtype=int)

                n_total = len(group_frame)
                n_retained = int(group_retained.sum())
                referral_rate = float(group_referred.mean()) if n_total else float("nan")
                retained_error = (
                    float(group_errors[group_retained].mean())
                    if n_retained > 0
                    else float("nan")
                )
                referral_rate_se = (
                    float(np.sqrt(referral_rate * (1.0 - referral_rate) / n_total))
                    if n_total > 0 and not np.isnan(referral_rate)
                    else float("nan")
                )
                retained_error_se = (
                    float(np.sqrt(retained_error * (1.0 - retained_error) / n_retained))
                    if n_retained > 0 and not np.isnan(retained_error)
                    else float("nan")
                )

                rows.append(
                    {
                        "task_name": scored_frame["task_name"].iloc[0],
                        "seed": int(scored_frame["seed"].iloc[0]),
                        "state": scored_frame["state"].iloc[0],
                        "dataset_name": scored_frame["dataset_name"].iloc[0],
                        "policy_name": policy_name,
                        "review_budget": review_budget,
                        "grouping": group_name,
                        "group_code": group_value,
                        "group_label": group_frame[label_column].iloc[0],
                        "total_count": n_total,
                        "retained_count": n_retained,
                        "referral_rate": referral_rate,
                        "referral_rate_se": referral_rate_se,
                        "retained_error": retained_error,
                        "retained_error_se": retained_error_se,
                        "mean_uncertainty": float(group_frame["entropy"].mean()),
                        "flag_small_group": n_total < min_group_size,
                    }
                )

    return pd.DataFrame(rows)


def compute_weighted_sensitivity(
    scored_frame: pd.DataFrame,
    policies: list[str],
    review_budget: float,
) -> pd.DataFrame:
    rows = []
    weights = scored_frame["survey_weight"].fillna(0.0).to_numpy(dtype=float)
    errors = scored_frame["error_indicator"].to_numpy(dtype=float)

    for policy_name in policies:
        referred_mask = referral_mask(scored_frame, policy_name, review_budget)
        retained_mask = ~referred_mask
        retained_weights = weights[retained_mask]
        retained_errors = errors[retained_mask]

        weighted_selective_risk = (
            float(np.average(retained_errors, weights=retained_weights))
            if retained_weights.sum() > 0
            else float("nan")
        )
        weighted_error_capture = (
            float(np.sum(weights[referred_mask] * errors[referred_mask]) / np.sum(weights * errors))
            if np.sum(weights * errors) > 0
            else float("nan")
        )

        rows.append(
            {
                "task_name": scored_frame["task_name"].iloc[0],
                "seed": int(scored_frame["seed"].iloc[0]),
                "state": scored_frame["state"].iloc[0],
                "dataset_name": scored_frame["dataset_name"].iloc[0],
                "policy_name": policy_name,
                "review_budget": review_budget,
                "weighting": "survey_weight_sensitivity",
                "count_based_review_budget": review_budget,
                "weighted_selective_risk": weighted_selective_risk,
                "weighted_error_capture": weighted_error_capture,
                "weighted_retained_coverage": float(retained_weights.sum() / weights.sum()) if weights.sum() > 0 else float("nan"),
            }
        )

    return pd.DataFrame(rows)


def compute_deployment_thresholds(
    validation_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    policies: list[str],
    review_budget: float,
) -> pd.DataFrame:
    rows = []
    for policy_name in policies:
        threshold = validation_threshold(validation_frame, policy_name, review_budget)
        referred_mask = target_frame[policy_name].to_numpy(dtype=float) >= threshold
        errors = target_frame["error_indicator"].to_numpy(dtype=int)
        retained_mask = ~referred_mask

        rows.append(
            {
                "task_name": target_frame["task_name"].iloc[0],
                "seed": int(target_frame["seed"].iloc[0]),
                "state": target_frame["state"].iloc[0],
                "dataset_name": target_frame["dataset_name"].iloc[0],
                "policy_name": policy_name,
                "source_validation_threshold": float(threshold),
                "realized_review_budget": float(referred_mask.mean()),
                "realized_retained_coverage": float(retained_mask.mean()),
                "realized_selective_risk": float(errors[retained_mask].mean()) if retained_mask.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def compare_uncertainty_baselines(metrics: pd.DataFrame, primary_budget: float) -> pd.DataFrame:
    filtered = metrics[metrics["review_budget"] == primary_budget]
    pivot = filtered.pivot_table(
        index=["task_name", "seed", "state", "dataset_name"],
        columns="policy_name",
        values="selective_risk",
    )
    rows = []
    for baseline_name in ("confidence", "ensemble_variance"):
        if baseline_name not in pivot.columns or "uncertainty_selector" not in pivot.columns:
            continue
        better = pivot[baseline_name] < pivot["uncertainty_selector"]
        for index in pivot[better].index:
            rows.append(
                {
                    "task_name": index[0],
                    "seed": index[1],
                    "state": index[2],
                    "dataset_name": index[3],
                    "baseline_policy": baseline_name,
                    "baseline_selective_risk": float(pivot.loc[index, baseline_name]),
                    "uncertainty_selective_risk": float(pivot.loc[index, "uncertainty_selector"]),
                    "improvement_pp": float(100.0 * (pivot.loc[index, "uncertainty_selector"] - pivot.loc[index, baseline_name])),
                }
            )
    return pd.DataFrame(rows)


def primary_effect_table(
    metrics: pd.DataFrame,
    bootstrap_replicates: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    primary_metrics = metrics[
        (metrics["review_budget"] == config.coverage_primary)
        & (metrics["policy_name"].isin(["uncertainty_selector", "augmented_selector"]))
    ]
    pivot = primary_metrics.pivot_table(
        index=["task_name", "seed", "state", "dataset_name"],
        columns="policy_name",
        values="selective_risk",
    ).reset_index()
    pivot["delta_selective_risk"] = pivot["augmented_selector"] - pivot["uncertainty_selector"]
    pivot["level"] = "task_state_seed"

    seed_level = pivot[
        ["level", "task_name", "seed", "state", "dataset_name", "uncertainty_selector", "augmented_selector", "delta_selective_risk"]
    ].copy()
    seed_level["seed_label"] = seed_level["seed"].astype(str)

    task_state = (
        seed_level.groupby(["task_name", "state", "dataset_name"], as_index=False)
        .agg(
            uncertainty_selector=("uncertainty_selector", "mean"),
            augmented_selector=("augmented_selector", "mean"),
            delta_selective_risk=("delta_selective_risk", "mean"),
        )
        .assign(level="task_state", seed="ALL", seed_label="ALL")
    )

    shifted_seed_summary = (
        seed_level[seed_level["state"].isin(config.target_states)]
        .groupby("seed", as_index=False)
        .agg(
            uncertainty_selector=("uncertainty_selector", "mean"),
            augmented_selector=("augmented_selector", "mean"),
            delta_selective_risk=("delta_selective_risk", "mean"),
        )
        .assign(level="seed_summary", task_name="ALL", state="SHIFTED", dataset_name="SHIFTED", seed_label=lambda frame: frame["seed"].astype(str))
    )

    shifted_summary = pd.DataFrame(
        [
            {
                "level": "shifted_summary",
                "task_name": "ALL",
                "seed": "ALL",
                "state": "SHIFTED",
                "dataset_name": "SHIFTED",
                "seed_label": "ALL",
                "uncertainty_selector": float(seed_level[seed_level["state"].isin(config.target_states)]["uncertainty_selector"].mean()),
                "augmented_selector": float(seed_level[seed_level["state"].isin(config.target_states)]["augmented_selector"].mean()),
                "delta_selective_risk": float(seed_level[seed_level["state"].isin(config.target_states)]["delta_selective_risk"].mean()),
            }
        ]
    )

    summary = pd.concat([seed_level, task_state, shifted_seed_summary, shifted_summary], ignore_index=True, sort=False)
    bootstrap_summary = summarize_bootstrap_replicates(bootstrap_replicates)
    merged = summary.merge(
        bootstrap_summary,
        on=["level", "task_name", "seed", "state", "dataset_name"],
        how="left",
    )
    merged["delta_selective_risk_pp"] = 100.0 * merged["delta_selective_risk"]
    merged["ci_low_pp"] = 100.0 * merged["ci_low"]
    merged["ci_high_pp"] = 100.0 * merged["ci_high"]
    return merged.sort_values(["level", "task_name", "state", "seed_label"]).reset_index(drop=True)


def paired_household_bootstrap(test_records: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    target_policies = ["uncertainty_selector", "augmented_selector"]
    state_households = {
        state: pd.Index(sorted(test_records.loc[test_records["state"] == state, "household_id"].astype(str).unique()))
        for state in sorted(test_records["state"].unique())
    }

    dataset_views: dict[tuple[str, str, int, str], BootstrapView] = {}
    dataset_keys = (
        test_records[["task_name", "state", "seed", "dataset_name"]]
        .drop_duplicates()
        .sort_values(["task_name", "state", "seed", "dataset_name"])
        .itertuples(index=False, name=None)
    )
    for task_name, state, seed, dataset_name in dataset_keys:
        dataset_frame = test_records[
            (test_records["task_name"] == task_name)
            & (test_records["state"] == state)
            & (test_records["seed"] == seed)
            & (test_records["dataset_name"] == dataset_name)
        ].copy()
        state_index = state_households[state]
        for policy_name in target_policies:
            ordered = order_for_policy(dataset_frame, policy_name)
            household_codes = state_index.get_indexer(ordered["household_id"].astype(str))
            dataset_views[(task_name, state, seed, policy_name)] = BootstrapView(
                household_codes=household_codes,
                errors=ordered["error_indicator"].to_numpy(dtype=float),
            )

    rng = np.random.default_rng(config.bootstrap_seed)
    rows = []
    states = sorted(state_households)
    seed_values = sorted(test_records["seed"].unique())
    task_values = sorted(test_records["task_name"].unique())

    for replicate_id in range(config.bootstrap_reps):
        state_counts = {
            state: np.bincount(
                rng.integers(0, len(state_households[state]), size=len(state_households[state])),
                minlength=len(state_households[state]),
            )
            for state in states
        }

        replicate_seed_rows = []
        for task_name in task_values:
            for state in states:
                for seed in seed_values:
                    uncertainty_view = dataset_views[(task_name, state, seed, "uncertainty_selector")]
                    augmented_view = dataset_views[(task_name, state, seed, "augmented_selector")]
                    counts = state_counts[state]
                    uncertainty_risk = weighted_selective_risk_from_view(
                        uncertainty_view,
                        counts,
                        review_budget=config.coverage_primary,
                    )
                    augmented_risk = weighted_selective_risk_from_view(
                        augmented_view,
                        counts,
                        review_budget=config.coverage_primary,
                    )
                    delta = augmented_risk - uncertainty_risk
                    rows.append(
                        {
                            "replicate_id": replicate_id,
                            "level": "task_state_seed",
                            "task_name": task_name,
                            "seed": seed,
                            "state": state,
                            "dataset_name": f"{state}_test",
                            "delta_selective_risk": delta,
                        }
                    )
                    replicate_seed_rows.append((task_name, state, seed, delta))

        replicate_seed_df = pd.DataFrame(
            replicate_seed_rows,
            columns=["task_name", "state", "seed", "delta_selective_risk"],
        )

        task_state_rows = (
            replicate_seed_df.groupby(["task_name", "state"], as_index=False)["delta_selective_risk"]
            .mean()
            .assign(level="task_state", seed="ALL")
        )
        for row in task_state_rows.itertuples(index=False):
            rows.append(
                {
                    "replicate_id": replicate_id,
                    "level": row.level,
                    "task_name": row.task_name,
                    "seed": row.seed,
                    "state": row.state,
                    "dataset_name": f"{row.state}_test",
                    "delta_selective_risk": row.delta_selective_risk,
                }
            )

        shifted_seed_rows = (
            replicate_seed_df[replicate_seed_df["state"].isin(config.target_states)]
            .groupby("seed", as_index=False)["delta_selective_risk"]
            .mean()
        )
        for row in shifted_seed_rows.itertuples(index=False):
            rows.append(
                {
                    "replicate_id": replicate_id,
                    "level": "seed_summary",
                    "task_name": "ALL",
                    "seed": row.seed,
                    "state": "SHIFTED",
                    "dataset_name": "SHIFTED",
                    "delta_selective_risk": row.delta_selective_risk,
                }
            )

        shifted_values = replicate_seed_df[replicate_seed_df["state"].isin(config.target_states)]["delta_selective_risk"]
        rows.append(
            {
                "replicate_id": replicate_id,
                "level": "shifted_summary",
                "task_name": "ALL",
                "seed": "ALL",
                "state": "SHIFTED",
                "dataset_name": "SHIFTED",
                "delta_selective_risk": float(shifted_values.mean()),
            }
        )

    return pd.DataFrame(rows)


def summarize_bootstrap_replicates(bootstrap_replicates: pd.DataFrame) -> pd.DataFrame:
    summary = (
        bootstrap_replicates.groupby(["level", "task_name", "seed", "state", "dataset_name"], dropna=False)["delta_selective_risk"]
        .agg(
            bootstrap_mean="mean",
            ci_low=lambda values: np.quantile(values, 0.025),
            ci_high=lambda values: np.quantile(values, 0.975),
        )
        .reset_index()
    )
    return summary


def weighted_selective_risk_from_view(
    view: BootstrapView,
    state_household_counts: np.ndarray,
    review_budget: float,
) -> float:
    weights = state_household_counts[view.household_codes]
    total_weight = int(weights.sum())
    if total_weight == 0:
        return float("nan")

    review_count = referral_count_from_budget(total_weight, review_budget)
    cumulative_weights = np.cumsum(weights)
    prior_weights = cumulative_weights - weights
    referred_weights = np.clip(review_count - prior_weights, 0, weights)
    retained_weights = weights - referred_weights
    retained_total = retained_weights.sum()
    if retained_total == 0:
        return float("nan")
    return float(np.dot(view.errors, retained_weights) / retained_total)


def plot_risk_coverage_curves(risk_curves: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotted = (
        risk_curves[risk_curves["policy_name"].isin(PRIMARY_POLICY_ORDER)]
        .groupby(["task_name", "state", "retained_coverage", "policy_name"], as_index=False)
        .agg(selective_risk=("selective_risk", "mean"))
    )
    if plotted.empty:
        return

    tasks = sorted(plotted["task_name"].unique())
    states = sorted(plotted["state"].unique())
    fig, axes = plt.subplots(len(tasks), len(states), figsize=(5 * len(states), 4 * len(tasks)), squeeze=False, sharey=True)

    for row_index, task_name in enumerate(tasks):
        for col_index, state in enumerate(states):
            axis = axes[row_index][col_index]
            subset = plotted[(plotted["task_name"] == task_name) & (plotted["state"] == state)]
            if subset.empty:
                axis.set_axis_off()
                continue
            sns.lineplot(data=subset, x="retained_coverage", y="selective_risk", hue="policy_name", ax=axis)
            axis.set_title(f"{task_name} — {state}")
            axis.set_xlabel("Retained coverage")
            axis.set_ylabel("Selective risk")
            if row_index != 0 or col_index != 0:
                legend = axis.get_legend()
                if legend is not None:
                    legend.remove()

    handles: list[object] = []
    labels: list[str] = []
    for row_axes in axes:
        for axis in row_axes:
            current_handles, current_labels = axis.get_legend_handles_labels()
            if current_handles:
                handles = current_handles
                labels = current_labels
                break
        if handles:
            break
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_group_audit_charts(group_audit: pd.DataFrame, output_dir: Path, review_budget: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    filtered = group_audit[
        (group_audit["policy_name"].isin(["uncertainty_selector", "augmented_selector"]))
        & (group_audit["review_budget"] == review_budget)
    ]
    if filtered.empty:
        return

    for grouping in ("sex", "race"):
        plotted = (
            filtered[filtered["grouping"] == grouping]
            .groupby(["task_name", "state", "group_label", "policy_name"], as_index=False)
            .agg(referral_rate=("referral_rate", "mean"))
        )
        if plotted.empty:
            continue

        tasks = sorted(plotted["task_name"].unique())
        states = sorted(plotted["state"].unique())
        fig, axes = plt.subplots(len(tasks), len(states), figsize=(5 * len(states), 4 * len(tasks)), squeeze=False, sharey=True)

        for row_index, task_name in enumerate(tasks):
            for col_index, state in enumerate(states):
                axis = axes[row_index][col_index]
                subset = plotted[(plotted["task_name"] == task_name) & (plotted["state"] == state)]
                if subset.empty:
                    axis.set_axis_off()
                    continue
                sns.barplot(data=subset, x="group_label", y="referral_rate", hue="policy_name", ax=axis)
                axis.set_title(f"{task_name} — {state}")
                axis.set_xlabel(grouping.capitalize())
                axis.set_ylabel("Referral rate")
                axis.tick_params(axis="x", rotation=45)
                if row_index != 0 or col_index != 0:
                    legend = axis.get_legend()
                    if legend is not None:
                        legend.remove()

        handles = []
        labels = []
        for row_axes in axes:
            for axis in row_axes:
                current_handles, current_labels = axis.get_legend_handles_labels()
                if current_handles:
                    handles = current_handles
                    labels = current_labels
                    break
            if handles:
                break
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=2)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(output_dir / f"group_audit_{grouping}.png", dpi=200)
        plt.close(fig)


def order_for_policy(frame: pd.DataFrame, policy_name: str) -> pd.DataFrame:
    return frame.sort_values([policy_name, "record_key"], ascending=[False, True], kind="mergesort").reset_index(drop=True)


def referral_mask(frame: pd.DataFrame, policy_name: str, review_budget: float) -> np.ndarray:
    ordered = order_for_policy(frame, policy_name)
    review_count = referral_count_from_budget(len(ordered), review_budget)
    referred_keys = set(ordered.head(review_count)["record_key"])
    return frame["record_key"].isin(referred_keys).to_numpy()


def validation_threshold(frame: pd.DataFrame, policy_name: str, review_budget: float) -> float:
    ordered = order_for_policy(frame, policy_name)
    review_count = referral_count_from_budget(len(ordered), review_budget)
    if review_count == 0:
        return float("inf")
    return float(ordered.iloc[review_count - 1][policy_name])


def safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def safe_average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, scores))
