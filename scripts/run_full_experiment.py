from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline import ExperimentRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full explanation disagreement audit.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "experiment.yaml",
        help="Path to the frozen experiment configuration.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runner = ExperimentRunner(Path(__file__).resolve().parents[1], config_path=args.config)
    result = runner.run_full()
    print(
        "Full experiment complete for tasks "
        + ", ".join(result["tasks"])
        + f" across seeds {result['seeds']}."
    )
    print(f"Artifacts written under {result['results_dir']}")
