"""Load and validate config/config.yaml."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from halo import METRICS, REPO_ROOT

DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    """Read the YAML config, apply nested overrides, and validate it."""
    with open(path or DEFAULT_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg = deep_update(cfg, overrides)
    validate_config(cfg)
    return cfg


def deep_update(base: dict, overrides: dict) -> dict:
    """Return a copy of base with overrides merged in, recursing into nested dicts."""
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def validate_config(cfg: dict) -> None:
    """Fail loudly on settings the spec does not allow."""
    metrics = cfg["score"]["metrics"]
    unknown = set(metrics) - set(METRICS)
    assert not unknown, f"unknown metrics in config: {sorted(unknown)}"
    assert 1 <= cfg["score"]["min_metrics"] <= len(metrics), "min_metrics out of range"
    assert cfg["fundamentals"]["ppe"] in ("ppent", "ppegt"), "fundamentals.ppe must be ppent|ppegt"
    assert cfg["fundamentals"]["nonpositive_ebitda"] in ("nan", "max_rank"), (
        "fundamentals.nonpositive_ebitda must be 'nan' or 'max_rank'"
    )
    assert 0 < cfg["fundamentals"]["lag_months"] < cfg["fundamentals"]["expiry_months"], (
        "fundamentals lag must be positive and shorter than expiry"
    )
    assert cfg["portfolio"]["basket"] in ("quintile", "tercile", "half")
    assert cfg["portfolio"]["weighting"] in ("equal", "value")
    first_train_end = cfg["tuning"]["first_train_end"]
    holdout = cfg["holdout"]["start"]
    assert first_train_end < holdout, "tuning window must end before the holdout"
    last_wf_year = cfg["tuning"]["walkforward_years"][1]
    assert f"{last_wf_year}-12" < holdout, "walk-forward years must end before the holdout"


def config_hash(cfg: dict) -> str:
    """Stable short hash of a config, used to tag trials in results/trials.csv."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]
