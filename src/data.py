from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import ExperimentConfig
from src.utils import round_half_up, sha256_file, stable_hash_int


PROTECTED_FEATURES = {"SEX", "RAC1P"}


@dataclass
class ProtocolAssets:
    raw_state_frames: dict[str, pd.DataFrame]
    definition_df: pd.DataFrame
    file_manifest: pd.DataFrame


@dataclass
class PreparedTaskData:
    task_name: str
    model_features: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    sampled_partitions: dict[str, pd.DataFrame]
    background_sample: pd.DataFrame
    pilot_sample: pd.DataFrame
    partition_summary: pd.DataFrame
    missingness_summary: pd.DataFrame
    sample_manifest: pd.DataFrame
    demographic_labels: dict[str, dict[object, str]]
    task_metadata: dict[str, Any]


def load_protocol_assets(base_data_path: Path, config: ExperimentConfig) -> ProtocolAssets:
    from folktables import ACSDataSource
    from folktables.load_acs import _STATE_CODES  # type: ignore[attr-defined]

    data_source = ACSDataSource(
        survey_year=str(config.data_year),
        horizon="1-Year",
        survey="person",
        root_dir=str(base_data_path),
    )

    raw_state_frames: dict[str, pd.DataFrame] = {}
    for state in config.all_states:
        raw_frame = data_source.get_data(states=[state], download=True).copy()
        raw_frame["__state_abbr__"] = state
        raw_frame["__raw_row_id__"] = np.arange(len(raw_frame), dtype=int)
        raw_state_frames[state] = raw_frame

    definition_df = data_source.get_definitions(download=True)

    base_dir = base_data_path / str(config.data_year) / "1-Year"
    manifest_rows = []
    for state in config.all_states:
        file_name = f"psam_p{_STATE_CODES[state]}.csv"
        file_path = base_dir / file_name
        manifest_rows.append(
            {
                "file_kind": "person_records",
                "state": state,
                "path": str(file_path),
                "sha256": sha256_file(file_path),
            }
        )
    definition_path = base_dir / "definition.csv"
    manifest_rows.append(
        {
            "file_kind": "definitions",
            "state": "ALL",
            "path": str(definition_path),
            "sha256": sha256_file(definition_path),
        }
    )

    return ProtocolAssets(
        raw_state_frames=raw_state_frames,
        definition_df=definition_df,
        file_manifest=pd.DataFrame(manifest_rows),
    )


def build_source_household_split(raw_source_frame: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    households = (
        raw_source_frame["SERIALNO"]
        .astype(str)
        .map(lambda serial: f"{config.source_state}:{serial}")
        .drop_duplicates()
        .to_frame(name="household_id")
    )
    households["sort_key"] = households["household_id"].map(
        lambda household_id: stable_hash_int(household_id, config.split_seed)
    )
    households = households.sort_values("sort_key", kind="mergesort").reset_index(drop=True)

    n_households = len(households)
    train_end = round_half_up(0.60 * n_households)
    selector_end = round_half_up(0.75 * n_households)
    validation_end = round_half_up(0.85 * n_households)

    partitions = np.full(n_households, "test", dtype=object)
    partitions[:train_end] = "train"
    partitions[train_end:selector_end] = "selector_fit"
    partitions[selector_end:validation_end] = "validation"
    households["partition"] = partitions

    return households.drop(columns=["sort_key"])


def prepare_task_data(
    task_name: str,
    assets: ProtocolAssets,
    split_manifest: pd.DataFrame,
    config: ExperimentConfig,
) -> PreparedTaskData:
    problem = get_task_problem(task_name)
    official_features = list(problem.features)
    model_features = [feature for feature in official_features if feature not in PROTECTED_FEATURES]
    numeric_features, categorical_features = infer_feature_types(model_features, assets.definition_df)

    demographic_labels = {
        "sex_code": category_labels(assets.definition_df, "SEX"),
        "race_code": category_labels(assets.definition_df, "RAC1P"),
    }

    state_task_frames = {
        state: build_task_frame(
            raw_frame=assets.raw_state_frames[state],
            task_name=task_name,
            state=state,
            data_year=config.data_year,
            problem=problem,
            demographic_labels=demographic_labels,
            model_features=model_features,
            official_features=official_features,
        )
        for state in config.all_states
    }

    source_full = state_task_frames[config.source_state].merge(split_manifest, on="household_id", how="left")
    if source_full["partition"].isna().any():
        raise ValueError(f"Missing split assignment for task {task_name}")

    full_partitions: dict[str, pd.DataFrame] = {
        "train": source_full[source_full["partition"] == "train"].reset_index(drop=True),
        "selector_fit": source_full[source_full["partition"] == "selector_fit"].reset_index(drop=True),
        "validation": source_full[source_full["partition"] == "validation"].reset_index(drop=True),
        f"{config.source_state}_test": source_full[source_full["partition"] == "test"].reset_index(drop=True),
    }
    for state in config.target_states:
        target_frame = state_task_frames[state].copy()
        target_frame["partition"] = "test"
        full_partitions[f"{state}_test"] = target_frame.reset_index(drop=True)

    sampled_partitions = {
        "train": sample_households_up_to_limit(
            full_partitions["train"],
            limit=config.train_limit,
            seed=stable_hash_int(f"{task_name}:train", config.sampling_seed),
        ),
        "selector_fit": sample_records_up_to_limit(
            full_partitions["selector_fit"],
            limit=config.explanation_limit,
            seed=stable_hash_int(f"{task_name}:selector_fit", config.sampling_seed),
        ),
        "validation": sample_records_up_to_limit(
            full_partitions["validation"],
            limit=config.explanation_limit,
            seed=stable_hash_int(f"{task_name}:validation", config.sampling_seed),
        ),
        f"{config.source_state}_test": sample_records_up_to_limit(
            full_partitions[f"{config.source_state}_test"],
            limit=config.explanation_limit,
            seed=stable_hash_int(f"{task_name}:{config.source_state}_test", config.sampling_seed),
        ),
    }
    for state in config.target_states:
        sampled_partitions[f"{state}_test"] = sample_records_up_to_limit(
            full_partitions[f"{state}_test"],
            limit=config.explanation_limit,
            seed=stable_hash_int(f"{task_name}:{state}_test", config.sampling_seed),
        )

    background_sample = sample_records_up_to_limit(
        sampled_partitions["train"],
        limit=config.sh_train_background,
        seed=stable_hash_int(f"{task_name}:background", config.sampling_seed),
    )
    pilot_sample = sample_records_up_to_limit(
        sampled_partitions["selector_fit"],
        limit=config.sh_pilot_records,
        seed=stable_hash_int(f"{task_name}:pilot", config.sampling_seed),
    )

    partition_summary = build_partition_summary(task_name, full_partitions, sampled_partitions)
    missingness_summary = build_missingness_summary(task_name, full_partitions, model_features)
    sample_manifest = build_sample_manifest(task_name, sampled_partitions)

    task_metadata = {
        "task_name": task_name,
        "official_features": official_features,
        "model_features": model_features,
        "excluded_protected_features": sorted(PROTECTED_FEATURES & set(official_features)),
        "target_column": problem.target,
        "target_transform": getattr(problem.target_transform, "__name__", "callable"),
        "official_filter": getattr(problem, "_preprocess").__name__,
    }

    return PreparedTaskData(
        task_name=task_name,
        model_features=model_features,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        sampled_partitions=sampled_partitions,
        background_sample=background_sample,
        pilot_sample=pilot_sample,
        partition_summary=partition_summary,
        missingness_summary=missingness_summary,
        sample_manifest=sample_manifest,
        demographic_labels=demographic_labels,
        task_metadata=task_metadata,
    )


def get_task_problem(task_name: str) -> Any:
    from folktables import ACSIncome, ACSPublicCoverage

    tasks = {
        "ACSIncome": ACSIncome,
        "ACSPublicCoverage": ACSPublicCoverage,
    }
    if task_name not in tasks:
        raise KeyError(f"Unsupported task: {task_name}")
    return tasks[task_name]


def infer_feature_types(features: list[str], definition_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_features: list[str] = []
    categorical_features: list[str] = []
    for feature in features:
        definition_rows = definition_df[(definition_df[0] == "VAL") & (definition_df[1] == feature)]
        if definition_rows.empty:
            numeric_features.append(feature)
            continue
        feature_type = str(definition_rows.iloc[0][2]).strip()
        if feature_type == "N":
            numeric_features.append(feature)
        else:
            categorical_features.append(feature)
    return numeric_features, categorical_features


def category_labels(definition_df: pd.DataFrame, feature_name: str) -> dict[object, str]:
    definition_rows = definition_df[(definition_df[0] == "VAL") & (definition_df[1] == feature_name)]
    labels: dict[object, str] = {}
    for _, row in definition_rows.iterrows():
        raw_value = pd.to_numeric(pd.Series([row[4]]), errors="coerce").iloc[0]
        if pd.isna(raw_value):
            continue
        key: object = int(raw_value) if float(raw_value).is_integer() else float(raw_value)
        labels[key] = str(row[6])
    return labels


def build_task_frame(
    raw_frame: pd.DataFrame,
    task_name: str,
    state: str,
    data_year: int,
    problem: Any,
    demographic_labels: dict[str, dict[object, str]],
    model_features: list[str],
    official_features: list[str],
) -> pd.DataFrame:
    preprocess = getattr(problem, "_preprocess", None)
    if preprocess is None:
        raise ValueError(f"Task {task_name} does not expose a preprocess function")
    filtered = preprocess(raw_frame.copy()).reset_index(drop=True)

    raw_target = filtered[problem.target]
    if problem.target_transform is None:
        label = raw_target.astype(int)
    else:
        label = problem.target_transform(raw_target).astype(int)

    row_id = filtered["__raw_row_id__"].astype(int)
    person_order = (
        pd.to_numeric(filtered["SPORDER"], errors="coerce")
        if "SPORDER" in filtered.columns
        else pd.Series(np.nan, index=filtered.index)
    )
    household_id = filtered["SERIALNO"].astype(str).map(lambda serial: f"{state}:{serial}")
    person_token = person_order.fillna(-1).astype(int).map(lambda value: f"{value:04d}")
    record_key = (
        state
        + ":"
        + filtered["SERIALNO"].astype(str)
        + ":"
        + person_token.astype(str)
        + ":"
        + row_id.map(lambda value: f"{value:07d}")
    )

    sex_code = pd.to_numeric(filtered["SEX"], errors="coerce")
    race_code = pd.to_numeric(filtered["RAC1P"], errors="coerce")

    frame = filtered[official_features].copy()
    frame["task_name"] = task_name
    frame["survey_year"] = data_year
    frame["state"] = state
    frame["household_id"] = household_id
    frame["record_key"] = record_key
    frame["row_id"] = row_id
    frame["person_order"] = person_order
    frame["survey_weight"] = pd.to_numeric(filtered["PWGTP"], errors="coerce")
    frame["label"] = label
    frame["sex_code"] = sex_code
    frame["race_code"] = race_code
    frame["sex_label"] = sex_code.map(lambda value: demographic_labels["sex_code"].get(_canonical_code(value), "Unknown"))
    frame["race_label"] = race_code.map(lambda value: demographic_labels["race_code"].get(_canonical_code(value), "Unknown"))
    frame["model_row_missing_count"] = frame[model_features].isna().sum(axis=1)
    return frame


def sample_households_up_to_limit(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame.sort_values("record_key", kind="mergesort").reset_index(drop=True)

    household_counts = (
        frame.groupby("household_id", as_index=False)
        .size()
        .rename(columns={"size": "n_records"})
    )
    household_counts["sort_key"] = household_counts["household_id"].map(lambda value: stable_hash_int(value, seed))
    household_counts = household_counts.sort_values("sort_key", kind="mergesort").reset_index(drop=True)
    selected_households: list[str] = []
    running_total = 0
    for row in household_counts.itertuples(index=False):
        if running_total + row.n_records <= limit or not selected_households:
            selected_households.append(row.household_id)
            running_total += row.n_records

    sampled = frame[frame["household_id"].isin(selected_households)]
    return sampled.sort_values("record_key", kind="mergesort").reset_index(drop=True)


def sample_records_up_to_limit(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame.sort_values("record_key", kind="mergesort").reset_index(drop=True)

    sampled = frame.assign(sort_key=frame["record_key"].map(lambda value: stable_hash_int(value, seed)))
    sampled = sampled.sort_values(["sort_key", "record_key"], kind="mergesort").head(limit)
    return sampled.drop(columns=["sort_key"]).reset_index(drop=True)


def build_partition_summary(
    task_name: str,
    full_partitions: dict[str, pd.DataFrame],
    sampled_partitions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for partition_name, full_frame in full_partitions.items():
        sampled_frame = sampled_partitions.get(partition_name, full_frame)
        state = (
            sampled_frame["state"].iloc[0]
            if not sampled_frame.empty
            else (full_frame["state"].iloc[0] if not full_frame.empty else partition_name.split("_")[0])
        )
        partition = (
            sampled_frame["partition"].iloc[0]
            if not sampled_frame.empty
            else (full_frame["partition"].iloc[0] if not full_frame.empty else partition_name)
        )
        rows.append(
            {
                "task_name": task_name,
                "dataset_name": partition_name,
                "state": state,
                "partition": partition,
                "n_records_full": len(full_frame),
                "n_records_sampled": len(sampled_frame),
                "n_households_full": full_frame["household_id"].nunique(),
                "prevalence_full": float(full_frame["label"].mean()) if not full_frame.empty else float("nan"),
                "prevalence_sampled": float(sampled_frame["label"].mean()) if not sampled_frame.empty else float("nan"),
                "mean_row_missing_features_full": float(full_frame["model_row_missing_count"].mean()) if not full_frame.empty else float("nan"),
                "mean_row_missing_features_sampled": float(sampled_frame["model_row_missing_count"].mean()) if not sampled_frame.empty else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["state", "dataset_name"]).reset_index(drop=True)


def build_missingness_summary(
    task_name: str,
    full_partitions: dict[str, pd.DataFrame],
    model_features: list[str],
) -> pd.DataFrame:
    rows = []
    for partition_name, frame in full_partitions.items():
        if frame.empty:
            continue
        feature_missingness = frame[model_features].isna().mean()
        for feature_name, missing_rate in feature_missingness.items():
            rows.append(
                {
                    "task_name": task_name,
                    "dataset_name": partition_name,
                    "state": frame["state"].iloc[0],
                    "partition": frame["partition"].iloc[0],
                    "feature_name": feature_name,
                    "missing_rate": float(missing_rate),
                }
            )
    return pd.DataFrame(rows).sort_values(["state", "dataset_name", "feature_name"]).reset_index(drop=True)


def build_sample_manifest(task_name: str, sampled_partitions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    manifest_rows = []
    for partition_name, frame in sampled_partitions.items():
        for _, row in frame.iterrows():
            manifest_rows.append(
                {
                    "task_name": task_name,
                    "dataset_name": partition_name,
                    "state": row["state"],
                    "partition": row["partition"],
                    "record_key": row["record_key"],
                    "household_id": row["household_id"],
                }
            )
    return pd.DataFrame(manifest_rows).sort_values(["dataset_name", "record_key"]).reset_index(drop=True)


def _canonical_code(value: object) -> object:
    if pd.isna(value):
        return value
    numeric_value = float(value)
    if numeric_value.is_integer():
        return int(numeric_value)
    return numeric_value
