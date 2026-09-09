from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline import ExperimentRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Day 1 feasibility pilot.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "experiment.yaml",
        help="Path to the frozen experiment configuration.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Optional task override (defaults to the first configured task).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed override (defaults to the first configured seed).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runner = ExperimentRunner(Path(__file__).resolve().parents[1], config_path=args.config)
    summary = runner.run_pilot(task_name=args.task, seed=args.seed)
    print(f"Pilot complete for {summary['task_name']} (seed={summary['seed']}).")
    print(f"Summary written to results/pilot_summary_{summary['task_name']}_{summary['seed']}.json")
