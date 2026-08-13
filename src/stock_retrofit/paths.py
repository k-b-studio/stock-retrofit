"""Canonical on-disk locations. Everything resolves off the repo root."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = ROOT / "configs"
MODEL_CONFIG_DIR = CONFIG_DIR / "models"
AGENT_CONFIG_DIR = CONFIG_DIR / "agents"

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RESULTS_DIR = ROOT / "results"
DOCS_DIR = ROOT / "docs"


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
