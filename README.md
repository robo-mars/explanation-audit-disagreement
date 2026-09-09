# Explanation Disagreement Audit

Reproducible local CPU experiment scaffold for the protocol on explanation disagreement and AI error detection.

## Quick start

1. Create and activate a Python 3.11 environment.
2. Install dependencies from `pyproject.toml`.
3. Run the Day 1 pilot script, then the full experiment driver.

## Project layout

- `src/`: pipeline code
- `configs/`: frozen experiment configuration
- `scripts/`: runnable entrypoints
- `data/`, `splits/`, `models/`, `explanations/`, `results/`, `logs/`: generated artifacts
- `.vscode/`: editor and task configuration

## Notes

- The protocol is frozen before final evaluation.
- All generated outputs stay inside this folder.
