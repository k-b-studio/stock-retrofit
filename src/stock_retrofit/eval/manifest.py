"""Run manifests: config + git SHA + data hash, written beside every result (R17).

A number in `results/` is worthless if you cannot say which data and which code
produced it. Every run writes one of these.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..paths import RESULTS_DIR, ROOT


def git_sha(short: bool = False) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short" if short else "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        sha = out.stdout.strip()
        if out.returncode != 0 or not sha:
            return "unknown"
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, timeout=10
        ).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}"
    except Exception:
        return "unknown"


@dataclass
class RunManifest:
    run_id: str
    kind: str  # "evaluate" | "backtest" | "report"
    created_at: str
    git_sha: str
    python: str
    config: dict[str, Any]
    data: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    notes: str = ""

    @classmethod
    def create(
        cls, kind: str, config: dict, *, data: dict | None = None, seed: int = 0, notes: str = ""
    ) -> RunManifest:
        now = datetime.now(UTC)
        return cls(
            run_id=f"{kind}-{now.strftime('%Y%m%dT%H%M%S')}",
            kind=kind,
            created_at=now.isoformat(timespec="seconds"),
            git_sha=git_sha(),
            python=platform.python_version(),
            config=config,
            data=data or {},
            seed=seed,
            notes=notes,
        )

    def write(self, directory: Path | None = None) -> Path:
        directory = directory or RESULTS_DIR
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.run_id}.manifest.json"
        path.write_text(json.dumps(asdict(self), indent=2, default=str) + "\n")
        return path


def data_fingerprint(symbols: list[str]) -> dict[str, Any]:
    """Content hashes of the cached bars a run consumed."""
    from ..data import read_meta

    out: dict[str, Any] = {}
    for s in symbols:
        meta = read_meta(s)
        if meta is None:
            out[s] = {"status": "uncached"}
        else:
            out[s] = {
                "source": meta.source,
                "rows": meta.rows,
                "range": [meta.start, meta.end],
                "content_hash": meta.content_hash,
                "repairs": meta.repairs.get("count", 0),
            }
    return out
