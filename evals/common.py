"""Shared helpers for eval suites."""

from __future__ import annotations

import json

from src.config import ROOT

EVAL_DIR = ROOT / "data" / "evals"
RESULTS_DIR = ROOT / "evals" / "results"


def load_jsonl(name: str) -> list[dict]:
    path = EVAL_DIR / name
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def parse_judge_json(raw: str) -> dict:
    """Judges sometimes wrap JSON in code fences; strip and parse."""
    return json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
