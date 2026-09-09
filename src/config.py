from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class XGBoostConfig:
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    reg_lambda: float
    random_state_offset: int


@dataclass(frozen=True)
class SelectorGridConfig:
    C: tuple[float, ...]


@dataclass(frozen=True)
class SHAPConfig:
    feature_perturbation: str
    model_output: str
    grouping: str
    l1_epsilon: float
    reconstruction_tolerance: float


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    data_year: int
    source_state: str
    target_states: tuple[str, ...]
    tasks: tuple[str, ...]
    seeds: tuple[int, ...]
    n_members: int
    train_limit: int
    explanation_limit: int
    sh_train_background: int
    sh_pilot_records: int
    coverage_primary: float
    coverage_secondary: tuple[float, ...]
    bootstrap_reps: int
    split_seed: int
    sampling_seed: int
    bootstrap_seed: int
    random_policy_seed: int
    robustness_background_limit: int
    fairness_min_group_size: int
    vote_threshold: float
    review_rounding: str
    review_tie_breaker: str
    xgb: XGBoostConfig
    selector_grid: SelectorGridConfig
    shap: SHAPConfig

    def __post_init__(self) -> None:
        budgets = self.review_budgets
        if self.source_state in self.target_states:
            raise ValueError("source_state must not also appear in target_states")
        if not self.tasks:
            raise ValueError("At least one task is required")
        if self.n_members < 2:
            raise ValueError("n_members must be at least 2 to compute disagreement")
        if self.train_limit <= 0 or self.explanation_limit <= 0:
            raise ValueError("Record limits must be positive")
        if self.sh_train_background <= 0 or self.sh_pilot_records <= 0:
            raise ValueError("SHAP limits must be positive")
        if self.review_rounding != "half_up":
            raise ValueError("Only the half_up review rounding rule is supported")
        if self.review_tie_breaker != "record_key_ascending":
            raise ValueError("Only the record_key_ascending tie-breaker is supported")
        if any(not 0.0 <= budget <= 1.0 for budget in budgets):
            raise ValueError("Review budgets must be between 0 and 1")
        if self.bootstrap_reps <= 0:
            raise ValueError("bootstrap_reps must be positive")

    @property
    def review_budgets(self) -> tuple[float, ...]:
        return tuple(sorted({self.coverage_primary, *self.coverage_secondary}))

    @property
    def retained_coverages(self) -> tuple[float, ...]:
        return tuple(1.0 - budget for budget in self.review_budgets)

    @property
    def all_states(self) -> tuple[str, ...]:
        return (self.source_state, *self.target_states)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    project = payload["project"]
    return ExperimentConfig(
        name=project["name"],
        data_year=int(project["data_year"]),
        source_state=project["source_state"],
        target_states=tuple(project["target_states"]),
        tasks=tuple(project["tasks"]),
        seeds=tuple(int(seed) for seed in project["seeds"]),
        n_members=int(project["n_members"]),
        train_limit=int(project["train_limit"]),
        explanation_limit=int(project["explanation_limit"]),
        sh_train_background=int(project["sh_train_background"]),
        sh_pilot_records=int(project["sh_pilot_records"]),
        coverage_primary=float(project["coverage_primary"]),
        coverage_secondary=tuple(float(value) for value in project["coverage_secondary"]),
        bootstrap_reps=int(project["bootstrap_reps"]),
        split_seed=int(project["split_seed"]),
        sampling_seed=int(project["sampling_seed"]),
        bootstrap_seed=int(project["bootstrap_seed"]),
        random_policy_seed=int(project["random_policy_seed"]),
        robustness_background_limit=int(project["robustness_background_limit"]),
        fairness_min_group_size=int(project["fairness_min_group_size"]),
        vote_threshold=float(project["vote_threshold"]),
        review_rounding=project["review_rounding"],
        review_tie_breaker=project["review_tie_breaker"],
        xgb=XGBoostConfig(**project["xgb"]),
        selector_grid=SelectorGridConfig(C=tuple(float(value) for value in project["selector_grid"]["C"])),
        shap=SHAPConfig(
            feature_perturbation=project["shap"]["feature_perturbation"],
            model_output=project["shap"]["model_output"],
            grouping=project["shap"]["grouping"],
            l1_epsilon=float(project["shap"]["l1_epsilon"]),
            reconstruction_tolerance=float(project["shap"]["reconstruction_tolerance"]),
        ),
    )
