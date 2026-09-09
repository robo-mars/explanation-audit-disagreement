from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    base: Path
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
            data=base / "data",
            splits=base / "splits",
            models=base / "models",
            explanations=base / "explanations",
            results=base / "results",
            logs=base / "logs",
        )

    def ensure(self) -> None:
        for path in [self.data, self.splits, self.models, self.explanations, self.results, self.logs]:
            path.mkdir(parents=True, exist_ok=True)
