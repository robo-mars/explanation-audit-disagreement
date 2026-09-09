# Explanation Disagreement Audit

This repository now contains a full local CPU experiment for the attached protocol on explanation disagreement, selective prediction, and demographic auditing under geographic shift.

## What it runs

The implementation follows the frozen protocol for:

- **Data**: Folktables ACS 2018 one-year person records
- **Tasks**: `ACSIncome` and `ACSPublicCoverage`
- **Domains**: California source split, Texas and New York held-out targets
- **Predictor**: 5-member XGBoost ensemble with household bootstraps
- **Explanations**: probability-space interventional TreeSHAP with a fixed 100-record source-training background
- **Selectors**: random, confidence, ensemble variance, uncertainty, explanation-only, augmented, raw-feature, plus robustness controls
- **Analysis**: 20% primary review budget, 5/10/30% secondary budgets, full risk-coverage curves, 1,000 household-cluster bootstrap replicates, demographic audit, survey-weight sensitivity, and robustness checks

## Install

1. Create and activate a Python 3.11 environment.
2. On **macOS**, install the OpenMP runtime that `xgboost` needs:

```bash
brew install libomp
```

3. Install the project and dev dependencies:

```bash
python3 -m pip install -e '.[dev]'
```

## Run

Day 1 feasibility pilot:

```bash
python3 scripts/pilot_day1.py
```

Full experiment:

```bash
python3 scripts/run_full_experiment.py
```

Both scripts accept `--config` to point at a different frozen configuration.

## Key protocol guarantees baked into the code

- Uses Folktables' **official task filters** instead of reimplementing them manually
- Builds a stable `record_key` and preserves `household_id`, `row_id`, `survey_weight`, `sex`, and `race`
- Splits California by **household** into train / selector-fit / validation / test
- Excludes recorded sex and race from predictive inputs while retaining them for audit
- Uses deterministic sampling, deterministic review tie-breaking, and deterministic review-count rounding
- Verifies SHAP reconstruction against the model probability before continuing
- Keeps selector fitting, selector validation, and final evaluation separated

## Output layout

- `splits/`: household split manifest, sampled-record manifests, partition summaries, missingness, task metadata
- `models/`: fitted preprocessors, XGBoost members, selector models, bootstrap manifests
- `explanations/`: grouped TreeSHAP values and reconstruction checks
- `results/`: per-record predictions and selector scores, policy metrics, primary effects, bootstrap replicates, risk-coverage data, group audits, robustness tables, deployment-threshold analysis, and figures
- `logs/`: runtime log and analysis-change log template

## Primary result artifacts

After a full run, the main paper-facing outputs are:

- `results/primary_effects.csv`
- `results/policy_metrics.csv`
- `results/risk_coverage_curves.csv.gz`
- `results/group_audit.csv`
- `results/robustness_metrics.csv`
- `results/bootstrap_replicates.csv.gz`
- `results/risk_coverage_curves.png`
- `results/group_audit_charts/`

## Notes

- The pipeline is intentionally **CPU-first** and favors reproducibility over aggressive parallel orchestration.
- Survey-weighted outputs are reported as sensitivity analyses only; the primary benchmark remains unweighted and count-budgeted.
- No illustrative numbers are written anywhere by default: result files are produced only from actual runs.
