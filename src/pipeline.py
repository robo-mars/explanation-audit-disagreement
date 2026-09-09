from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.analysis import (
    PRIMARY_POLICY_ORDER,
    ROBUSTNESS_POLICY_ORDER,
    compare_uncertainty_baselines,
    compute_deployment_thresholds,
    compute_group_audit,
    compute_risk_coverage_curves,
    compute_weighted_sensitivity,
    evaluate_policy_metrics,
    paired_household_bootstrap,
    plot_group_audit_charts,
    plot_risk_coverage_curves,
    primary_effect_table,
)
from src.config import ExperimentConfig, load_experiment_config
from src.data import PreparedTaskData, ProtocolAssets, build_source_household_split, load_protocol_assets, prepare_task_data, sample_records_up_to_limit
from src.modeling import EnsembleBundle, ScoredPartition, fit_ensemble, score_partition
from src.preprocessing import fit_preprocessor
from src.selectors import SelectorBundle, score_policies, train_selectors
from src.utils import stable_hash_int, timed_stage, utc_now_iso, write_json


@dataclass(frozen=True)
class Paths:
    base: Path
    configs: Path
    data: Path
    splits: Path
    models: Path
    explanations: Path
    results: Path
    logs: Path

    @classmethod
    def from_base(cls, base: str | Path) -> "Paths":
        base = Path(base)
        return cls(
            base=base,
            configs=base / "configs",
            data=base / "data",
            splits=base / "splits",
            models=base / "models",
            explanations=base / "explanations",
            results=base / "results",
            logs=base / "logs",
        )

    def ensure(self) -> None:
        for path in [
            self.data,
            self.splits,
            self.models,
            self.explanations,
            self.results,
            self.logs,
        ]:
            path.mkdir(parents=True, exist_ok=True)


class ExperimentRunner:
    def __init__(self, base: str | Path, config_path: str | Path | None = None) -> None:
        self.paths = Paths.from_base(base)
        config_file = Path(config_path) if config_path is not None else self.paths.configs / "experiment.yaml"
        self.config: ExperimentConfig = load_experiment_config(config_file)
        self._assets: ProtocolAssets | None = None
        self._split_manifest: pd.DataFrame | None = None

    def run_pilot(self, task_name: str | None = None, seed: int | None = None) -> dict[str, Any]:
        self.paths.ensure()
        runtime_rows: list[dict[str, object]] = []
        assets, split_manifest = self._prepare_protocol_assets(runtime_rows)

        chosen_task = task_name or self.config.tasks[0]
        chosen_seed = seed or self.config.seeds[0]
        task_data = prepare_task_data(chosen_task, assets, split_manifest, self.config)
        self._persist_task_preparation(task_data)

        with timed_stage("fit_preprocessor", runtime_rows, task_name=chosen_task):
            preprocessor = fit_preprocessor(
                task_data.sampled_partitions["train"][task_data.model_features],
                numeric_features=task_data.numeric_features,
                categorical_features=task_data.categorical_features,
            )

        pilot_model_dir = self.paths.models / chosen_task
        pilot_model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, pilot_model_dir / "preprocessor.joblib")
        ensemble = fit_ensemble(
            task_name=chosen_task,
            seed=chosen_seed,
            training_frame=task_data.sampled_partitions["train"],
            background_frame=task_data.background_sample,
            model_features=task_data.model_features,
            preprocessor=preprocessor,
            config=self.config,
            runtime_sink=runtime_rows,
        )
        self._persist_ensemble_artifacts(ensemble, chosen_task, chosen_seed)

        pilot_partition = score_partition(
            ensemble=ensemble,
            frame=task_data.pilot_sample,
            model_features=task_data.model_features,
            config=self.config,
            runtime_sink=runtime_rows,
            dataset_name="pilot_selector_fit",
        )
        self._persist_scored_partition(pilot_partition, chosen_task, chosen_seed)

        correlation_d_variance = float(
            pilot_partition.summary["explanation_disagreement"].corr(pilot_partition.summary["variance"])
        )
        correlation_d_entropy = float(
            pilot_partition.summary["explanation_disagreement"].corr(pilot_partition.summary["entropy"])
        )

        shap_runtime = sum(
            row["duration_s"]
            for row in runtime_rows
            if row["stage"] == "score_member_shap"
        )
        member_record_ops = sum(
            int(row.get("n_records", 0))
            for row in runtime_rows
            if row["stage"] == "score_member_shap"
        )
        seconds_per_member_record = shap_runtime / member_record_ops if member_record_ops else float("nan")
        projected_test_only_seconds = (
            seconds_per_member_record
            * self.config.n_members
            * self.config.explanation_limit
            * len(self.config.tasks)
            * len(self.config.all_states)
            * len(self.config.seeds)
        )
        projected_all_scored_partitions_seconds = (
            seconds_per_member_record
            * self.config.n_members
            * self.config.explanation_limit
            * (3 + len(self.config.target_states))
            * len(self.config.tasks)
            * len(self.config.seeds)
        )

        subgroup_projection = self._pilot_group_projection(task_data)
        subgroup_projection.to_csv(self.paths.results / "pilot_group_cell_sizes.csv", index=False)

        summary = {
            "timestamp_utc": utc_now_iso(),
            "task_name": chosen_task,
            "seed": chosen_seed,
            "pilot_records": int(len(pilot_partition.summary)),
            "correlation_explanation_variance": correlation_d_variance,
            "correlation_explanation_entropy": correlation_d_entropy,
            "max_reconstruction_error": float(pilot_partition.reconstruction_checks["max_abs_error"].max()),
            "reconstruction_tolerance": float(self.config.shap.reconstruction_tolerance),
            "seconds_per_member_record": seconds_per_member_record,
            "projected_test_only_hours": projected_test_only_seconds / 3600.0,
            "projected_all_scored_partitions_hours": projected_all_scored_partitions_seconds / 3600.0,
        }

        write_json(self.paths.results / f"pilot_summary_{chosen_task}_{chosen_seed}.json", summary)
        pd.DataFrame(runtime_rows).to_csv(self.paths.logs / "pilot_runtime.csv", index=False)
        return summary

    def run_full(self) -> dict[str, Any]:
        self.paths.ensure()
        runtime_rows: list[dict[str, object]] = []
        assets, split_manifest = self._prepare_protocol_assets(runtime_rows)

        partition_summaries: list[pd.DataFrame] = []
        missingness_tables: list[pd.DataFrame] = []
        sample_manifests: list[pd.DataFrame] = []
        selector_candidates: list[pd.DataFrame] = []
        selector_configs: list[pd.DataFrame] = []
        base_metrics: list[pd.DataFrame] = []
        robustness_metrics: list[pd.DataFrame] = []
        risk_curves: list[pd.DataFrame] = []
        group_audits: list[pd.DataFrame] = []
        weighted_sensitivities: list[pd.DataFrame] = []
        deployment_thresholds: list[pd.DataFrame] = []
        diagnostic_correlations: list[pd.DataFrame] = []
        all_test_records: list[pd.DataFrame] = []

        for task_name in self.config.tasks:
            task_data = prepare_task_data(task_name, assets, split_manifest, self.config)
            self._persist_task_preparation(task_data)
            partition_summaries.append(task_data.partition_summary)
            missingness_tables.append(task_data.missingness_summary)
            sample_manifests.append(task_data.sample_manifest)

            with timed_stage("fit_preprocessor", runtime_rows, task_name=task_name):
                preprocessor = fit_preprocessor(
                    task_data.sampled_partitions["train"][task_data.model_features],
                    numeric_features=task_data.numeric_features,
                    categorical_features=task_data.categorical_features,
                )
            task_model_dir = self.paths.models / task_name
            task_model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(preprocessor, task_model_dir / "preprocessor.joblib")

            for seed in self.config.seeds:
                ensemble = fit_ensemble(
                    task_name=task_name,
                    seed=seed,
                    training_frame=task_data.sampled_partitions["train"],
                    background_frame=task_data.background_sample,
                    model_features=task_data.model_features,
                    preprocessor=preprocessor,
                    config=self.config,
                    runtime_sink=runtime_rows,
                )
                self._persist_ensemble_artifacts(ensemble, task_name, seed)

                scored_partitions = self._score_all_partitions(task_data, ensemble, runtime_rows)
                selectors = train_selectors(
                    selector_fit_summary=scored_partitions["selector_fit"].summary,
                    selector_fit_features=scored_partitions["selector_fit"].transformed_features,
                    validation_summary=scored_partitions["validation"].summary,
                    validation_features=scored_partitions["validation"].transformed_features,
                    config=self.config,
                    seed=seed,
                )
                selector_candidates.append(selectors.candidate_metrics)
                selector_configs.append(self._persist_selector_artifacts(selectors, task_name, seed))

                scored_frames = self._attach_policy_scores(scored_partitions, selectors, seed)
                correlation_rows = [
                    self._diagnostic_correlation_row(frame)
                    for dataset_name, frame in scored_frames.items()
                    if dataset_name.endswith("_test")
                ]
                diagnostic_correlations.append(pd.DataFrame(correlation_rows))

                validation_frame = scored_frames["validation"]
                primary_and_robustness_policies = PRIMARY_POLICY_ORDER + ROBUSTNESS_POLICY_ORDER

                for dataset_name, scored_frame in scored_frames.items():
                    if dataset_name.endswith("_test"):
                        all_test_records.append(scored_frame)

                        primary_metric_frame = evaluate_policy_metrics(
                            scored_frame,
                            PRIMARY_POLICY_ORDER,
                            self.config.review_budgets,
                        )
                        primary_metric_frame["analysis_scope"] = "primary"
                        base_metrics.append(primary_metric_frame)

                        robustness_metric_frame = evaluate_policy_metrics(
                            scored_frame,
                            ROBUSTNESS_POLICY_ORDER,
                            self.config.review_budgets,
                        )
                        robustness_metric_frame["analysis_scope"] = robustness_metric_frame["policy_name"].map(
                            {
                                "random_feature_control": "added_random_feature_control",
                                "explanation_top3": "top3_feature_disagreement",
                                "augmented_top3": "top3_feature_disagreement",
                            }
                        )
                        robustness_metrics.append(robustness_metric_frame)

                        risk_curves.append(compute_risk_coverage_curves(scored_frame, PRIMARY_POLICY_ORDER))
                        group_audits.append(
                            compute_group_audit(
                                scored_frame,
                                PRIMARY_POLICY_ORDER,
                                review_budget=self.config.coverage_primary,
                                min_group_size=self.config.fairness_min_group_size,
                            )
                        )
                        weighted_sensitivities.append(
                            compute_weighted_sensitivity(
                                scored_frame,
                                PRIMARY_POLICY_ORDER,
                                review_budget=self.config.coverage_primary,
                            )
                        )
                        deployment_thresholds.append(
                            compute_deployment_thresholds(
                                validation_frame,
                                scored_frame,
                                primary_and_robustness_policies,
                                review_budget=self.config.coverage_primary,
                            )
                        )

                        unanimous_subset = scored_frame[scored_frame["vote_disagreement"] == 0].reset_index(drop=True)
                        if not unanimous_subset.empty:
                            unanimous_metrics = evaluate_policy_metrics(
                                unanimous_subset,
                                PRIMARY_POLICY_ORDER,
                                self.config.review_budgets,
                            )
                            unanimous_metrics["analysis_scope"] = "unanimous_vote_subset"
                            robustness_metrics.append(unanimous_metrics)

                second_background_metrics = self._run_second_background_robustness(
                    task_data=task_data,
                    ensemble=ensemble,
                    seed=seed,
                    runtime_rows=runtime_rows,
                )
                robustness_metrics.extend(second_background_metrics)

        self._persist_aggregates(
            partition_summaries=partition_summaries,
            missingness_tables=missingness_tables,
            sample_manifests=sample_manifests,
            selector_candidates=selector_candidates,
            selector_configs=selector_configs,
            base_metrics=base_metrics,
            robustness_metrics=robustness_metrics,
            risk_curves=risk_curves,
            group_audits=group_audits,
            weighted_sensitivities=weighted_sensitivities,
            deployment_thresholds=deployment_thresholds,
            diagnostic_correlations=diagnostic_correlations,
            all_test_records=all_test_records,
            runtime_rows=runtime_rows,
        )

        return {
            "tasks": list(self.config.tasks),
            "seeds": list(self.config.seeds),
            "results_dir": str(self.paths.results),
        }

    def _prepare_protocol_assets(self, runtime_rows: list[dict[str, object]]) -> tuple[ProtocolAssets, pd.DataFrame]:
        if self._assets is None or self._split_manifest is None:
            with timed_stage("load_protocol_assets", runtime_rows):
                self._assets = load_protocol_assets(self.paths.data, self.config)
            self._split_manifest = build_source_household_split(
                self._assets.raw_state_frames[self.config.source_state],
                self.config,
            )
            self._persist_protocol_assets(self._assets, self._split_manifest)
        return self._assets, self._split_manifest

    def _persist_protocol_assets(self, assets: ProtocolAssets, split_manifest: pd.DataFrame) -> None:
        write_json(self.paths.results / "frozen_protocol.json", self.config.to_dict())
        write_json(self.paths.results / "environment_manifest.json", self._environment_manifest())
        assets.file_manifest.to_csv(self.paths.results / "data_file_manifest.csv", index=False)
        split_manifest.to_csv(self.paths.splits / "source_household_split_manifest.csv", index=False)
        self._ensure_analysis_change_log()

    def _persist_task_preparation(self, task_data: PreparedTaskData) -> None:
        task_split_dir = self.paths.splits / task_data.task_name
        task_split_dir.mkdir(parents=True, exist_ok=True)
        task_data.partition_summary.to_csv(task_split_dir / "partition_summary.csv", index=False)
        task_data.missingness_summary.to_csv(task_split_dir / "missingness_summary.csv", index=False)
        task_data.sample_manifest.to_csv(task_split_dir / "sample_manifest.csv", index=False)
        task_data.background_sample[
            ["record_key", "household_id", "state", "partition"]
        ].to_csv(task_split_dir / "background_sample_manifest.csv", index=False)
        task_data.pilot_sample[
            ["record_key", "household_id", "state", "partition"]
        ].to_csv(task_split_dir / "pilot_sample_manifest.csv", index=False)
        write_json(task_split_dir / "task_metadata.json", task_data.task_metadata)

    def _persist_ensemble_artifacts(self, ensemble: EnsembleBundle, task_name: str, seed: int) -> None:
        seed_dir = self.paths.models / task_name / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for member in ensemble.members:
            joblib.dump(member.model, seed_dir / f"member_{member.member_id}.joblib")
            member.bootstrap_manifest.to_csv(
                seed_dir / f"member_{member.member_id}_bootstrap_manifest.csv",
                index=False,
            )
        ensemble.background_matrix.to_csv(seed_dir / "background_matrix.csv.gz", index=False)

    def _score_all_partitions(
        self,
        task_data: PreparedTaskData,
        ensemble: EnsembleBundle,
        runtime_rows: list[dict[str, object]],
    ) -> dict[str, ScoredPartition]:
        scored: dict[str, ScoredPartition] = {}
        for dataset_name, frame in task_data.sampled_partitions.items():
            if dataset_name == "train":
                continue
            partition = score_partition(
                ensemble=ensemble,
                frame=frame,
                model_features=task_data.model_features,
                config=self.config,
                runtime_sink=runtime_rows,
                dataset_name=dataset_name,
            )
            self._persist_scored_partition(partition, task_data.task_name, ensemble.seed)
            scored[dataset_name] = partition
        return scored

    def _persist_scored_partition(self, partition: ScoredPartition, task_name: str, seed: int) -> None:
        explanation_dir = self.paths.explanations / task_name / f"seed_{seed}"
        result_dir = self.paths.results / "record_scores" / task_name / f"seed_{seed}"
        explanation_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        partition.grouped_attributions.to_csv(
            explanation_dir / f"{partition.dataset_name}_grouped_attributions.csv.gz",
            index=False,
        )
        partition.reconstruction_checks.to_csv(
            explanation_dir / f"{partition.dataset_name}_reconstruction_checks.csv",
            index=False,
        )
        partition.summary.to_csv(result_dir / f"{partition.dataset_name}_base_scores.csv.gz", index=False)

    def _persist_selector_artifacts(self, selectors: SelectorBundle, task_name: str, seed: int) -> pd.DataFrame:
        selector_dir = self.paths.models / task_name / f"seed_{seed}" / "selectors"
        selector_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for policy_name, selector in selectors.trained_selectors.items():
            joblib.dump(selector.pipeline, selector_dir / f"{policy_name}.joblib")
            rows.append(
                {
                    "task_name": task_name,
                    "seed": seed,
                    "policy_name": policy_name,
                    "chosen_c": selector.chosen_c,
                    "validation_selective_risk": selector.validation_selective_risk,
                }
            )
        config_frame = pd.DataFrame(rows).sort_values("policy_name").reset_index(drop=True)
        config_frame.to_csv(selector_dir / "selector_configurations.csv", index=False)
        selectors.candidate_metrics.to_csv(selector_dir / "selector_candidate_metrics.csv", index=False)
        return config_frame

    def _attach_policy_scores(
        self,
        scored_partitions: dict[str, ScoredPartition],
        selectors: SelectorBundle,
        seed: int,
    ) -> dict[str, pd.DataFrame]:
        scored_frames: dict[str, pd.DataFrame] = {}
        for dataset_name, partition in scored_partitions.items():
            policy_scores = score_policies(
                summary=partition.summary,
                transformed_features=partition.transformed_features,
                selectors=selectors,
                config=self.config,
                seed=seed,
            )
            scored_frame = pd.concat([partition.summary.reset_index(drop=True), policy_scores], axis=1)
            scored_frames[dataset_name] = scored_frame
            saved_name = str(scored_frame["dataset_name"].iloc[0])
            scored_frame.to_csv(
                self.paths.results / "record_scores" / partition.summary["task_name"].iloc[0] / f"seed_{seed}" / f"{saved_name}_record_scores.csv.gz",
                index=False,
            )
        return scored_frames

    def _run_second_background_robustness(
        self,
        task_data: PreparedTaskData,
        ensemble: EnsembleBundle,
        seed: int,
        runtime_rows: list[dict[str, object]],
    ) -> list[pd.DataFrame]:
        alternative_background = sample_records_up_to_limit(
            task_data.sampled_partitions["train"],
            limit=self.config.sh_train_background,
            seed=stable_hash_int(f"{task_data.task_name}:background_alt", self.config.sampling_seed),
        )
        alternative_background_matrix = ensemble.preprocessor.transform(alternative_background[task_data.model_features])

        subset_partitions = {}
        for dataset_name, frame in task_data.sampled_partitions.items():
            if dataset_name == "train":
                continue
            subset_partitions[dataset_name] = sample_records_up_to_limit(
                frame,
                limit=self.config.robustness_background_limit,
                seed=stable_hash_int(f"{task_data.task_name}:{dataset_name}:background_subset", self.config.sampling_seed),
            )

        rescored = {
            dataset_name: score_partition(
                ensemble=ensemble,
                frame=subset_frame,
                model_features=task_data.model_features,
                config=self.config,
                runtime_sink=runtime_rows,
                dataset_name=f"{dataset_name}_background_alt",
                background_override=alternative_background_matrix,
            )
            for dataset_name, subset_frame in subset_partitions.items()
        }
        for partition in rescored.values():
            self._persist_scored_partition(partition, task_data.task_name, seed)

        selectors = train_selectors(
            selector_fit_summary=rescored["selector_fit"].summary,
            selector_fit_features=rescored["selector_fit"].transformed_features,
            validation_summary=rescored["validation"].summary,
            validation_features=rescored["validation"].transformed_features,
            config=self.config,
            seed=seed,
        )
        scored_frames = self._attach_policy_scores(rescored, selectors, seed)

        robustness_frames: list[pd.DataFrame] = []
        for dataset_name, frame in scored_frames.items():
            if not dataset_name.endswith("_test"):
                continue
            metrics = evaluate_policy_metrics(
                frame,
                ["uncertainty_selector", "explanation_only", "augmented_selector"],
                self.config.review_budgets,
            )
            metrics["analysis_scope"] = "second_background_subset"
            robustness_frames.append(metrics)
        return robustness_frames

    def _persist_aggregates(
        self,
        *,
        partition_summaries: list[pd.DataFrame],
        missingness_tables: list[pd.DataFrame],
        sample_manifests: list[pd.DataFrame],
        selector_candidates: list[pd.DataFrame],
        selector_configs: list[pd.DataFrame],
        base_metrics: list[pd.DataFrame],
        robustness_metrics: list[pd.DataFrame],
        risk_curves: list[pd.DataFrame],
        group_audits: list[pd.DataFrame],
        weighted_sensitivities: list[pd.DataFrame],
        deployment_thresholds: list[pd.DataFrame],
        diagnostic_correlations: list[pd.DataFrame],
        all_test_records: list[pd.DataFrame],
        runtime_rows: list[dict[str, object]],
    ) -> None:
        partition_summary_frame = pd.concat(partition_summaries, ignore_index=True)
        missingness_frame = pd.concat(missingness_tables, ignore_index=True)
        sample_manifest_frame = pd.concat(sample_manifests, ignore_index=True)
        selector_candidate_frame = pd.concat(selector_candidates, ignore_index=True)
        selector_config_frame = pd.concat(selector_configs, ignore_index=True)
        base_metric_frame = pd.concat(base_metrics, ignore_index=True)
        robustness_metric_frame = pd.concat(robustness_metrics, ignore_index=True) if robustness_metrics else pd.DataFrame()
        risk_curve_frame = pd.concat(risk_curves, ignore_index=True)
        group_audit_frame = pd.concat(group_audits, ignore_index=True)
        weighted_sensitivity_frame = pd.concat(weighted_sensitivities, ignore_index=True)
        deployment_threshold_frame = pd.concat(deployment_thresholds, ignore_index=True)
        baseline_flag_frame = compare_uncertainty_baselines(base_metric_frame, self.config.coverage_primary)
        diagnostic_correlation_frame = pd.concat(diagnostic_correlations, ignore_index=True)
        all_test_records_frame = pd.concat(all_test_records, ignore_index=True)

        partition_summary_frame.to_csv(self.paths.results / "partition_summary.csv", index=False)
        missingness_frame.to_csv(self.paths.results / "missingness_summary.csv", index=False)
        sample_manifest_frame.to_csv(self.paths.results / "sample_manifest.csv.gz", index=False)
        selector_candidate_frame.to_csv(self.paths.results / "selector_candidate_metrics.csv", index=False)
        selector_config_frame.to_csv(self.paths.results / "selector_configurations.csv", index=False)
        base_metric_frame.to_csv(self.paths.results / "policy_metrics.csv", index=False)
        if not robustness_metric_frame.empty:
            robustness_metric_frame.to_csv(self.paths.results / "robustness_metrics.csv", index=False)
        risk_curve_frame.to_csv(self.paths.results / "risk_coverage_curves.csv.gz", index=False)
        group_audit_frame.to_csv(self.paths.results / "group_audit.csv", index=False)
        weighted_sensitivity_frame.to_csv(self.paths.results / "survey_weight_sensitivity.csv", index=False)
        deployment_threshold_frame.to_csv(self.paths.results / "deployment_thresholds.csv", index=False)
        if not baseline_flag_frame.empty:
            baseline_flag_frame.to_csv(self.paths.results / "uncertainty_baseline_flags.csv", index=False)
        diagnostic_correlation_frame.to_csv(self.paths.results / "diagnostic_correlations.csv", index=False)
        all_test_records_frame.to_csv(self.paths.results / "all_test_record_scores.csv.gz", index=False)

        bootstrap_replicates = paired_household_bootstrap(all_test_records_frame, self.config)
        bootstrap_replicates.to_csv(self.paths.results / "bootstrap_replicates.csv.gz", index=False)
        effect_table = primary_effect_table(base_metric_frame, bootstrap_replicates, self.config)
        effect_table.to_csv(self.paths.results / "primary_effects.csv", index=False)

        plot_risk_coverage_curves(risk_curve_frame, self.paths.results / "risk_coverage_curves.png")
        plot_group_audit_charts(group_audit_frame, self.paths.results / "group_audit_charts", self.config.coverage_primary)

        pd.DataFrame(runtime_rows).to_csv(self.paths.logs / "runtime_log.csv", index=False)

    def _pilot_group_projection(self, task_data: PreparedTaskData) -> pd.DataFrame:
        rows = []
        for dataset_name, frame in task_data.sampled_partitions.items():
            if not dataset_name.endswith("_test"):
                continue
            grouped = (
                frame.groupby(["state", "sex_label", "race_label"], dropna=False)
                .size()
                .reset_index(name="n_records")
            )
            grouped["task_name"] = task_data.task_name
            grouped["dataset_name"] = dataset_name
            grouped["projected_referred_at_20pct"] = grouped["n_records"] * self.config.coverage_primary
            grouped["below_100_threshold"] = grouped["n_records"] < self.config.fairness_min_group_size
            rows.append(grouped)
        return pd.concat(rows, ignore_index=True)

    def _diagnostic_correlation_row(self, scored_frame: pd.DataFrame) -> dict[str, Any]:
        return {
            "task_name": scored_frame["task_name"].iloc[0],
            "seed": int(scored_frame["seed"].iloc[0]),
            "state": scored_frame["state"].iloc[0],
            "dataset_name": scored_frame["dataset_name"].iloc[0],
            "corr_explanation_variance": float(scored_frame["explanation_disagreement"].corr(scored_frame["variance"])),
            "corr_explanation_entropy": float(scored_frame["explanation_disagreement"].corr(scored_frame["entropy"])),
        }

    def _environment_manifest(self) -> dict[str, Any]:
        distributions = {
            "numpy": "numpy",
            "pandas": "pandas",
            "scikit_learn": "scikit-learn",
            "xgboost": "xgboost",
            "shap": "shap",
            "folktables": "folktables",
            "scipy": "scipy",
            "matplotlib": "matplotlib",
            "seaborn": "seaborn",
            "pyyaml": "PyYAML",
            "joblib": "joblib",
        }
        versions = {}
        for alias, distribution in distributions.items():
            try:
                versions[alias] = metadata.version(distribution)
            except metadata.PackageNotFoundError:
                versions[alias] = "not-installed"
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "dependencies": versions,
        }

    def _ensure_analysis_change_log(self) -> None:
        change_log = self.paths.logs / "analysis_changes.csv"
        if change_log.exists():
            return
        pd.DataFrame(
            columns=[
                "timestamp_utc",
                "change_summary",
                "reason",
                "results_seen_before_change",
                "inspected_artifacts",
            ]
        ).to_csv(change_log, index=False)
