from __future__ import annotations

import hashlib
import json
import math
import resource
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def stable_hash_int(token: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}|{token}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def stable_uniform(token: str, seed: int) -> float:
    return stable_hash_int(token, seed) / float((1 << 64) - 1)


def round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def referral_count_from_budget(n_records: int, review_budget: float) -> int:
    if n_records < 0:
        raise ValueError("n_records must be non-negative")
    if not 0.0 <= review_budget <= 1.0:
        raise ValueError("review_budget must be between 0 and 1")
    return min(n_records, max(0, round_half_up(review_budget * n_records)))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rss_mb() -> float | None:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    max_rss = getattr(usage, "ru_maxrss", None)
    if max_rss is None:
        return None
    # macOS reports bytes, Linux reports kilobytes.
    if max_rss > 10_000_000:
        return max_rss / (1024 * 1024)
    return max_rss / 1024


@contextmanager
def timed_stage(stage: str, sink: list[dict[str, Any]], **metadata: Any) -> Iterator[None]:
    started_at = utc_now_iso()
    started_rss_mb = rss_mb()
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_s = time.perf_counter() - start
        sink.append(
            {
                "stage": stage,
                "started_at": started_at,
                "duration_s": duration_s,
                "rss_mb": rss_mb(),
                "started_rss_mb": started_rss_mb,
                **metadata,
            }
        )
