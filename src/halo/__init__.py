"""HALO capital intensity score: US-equity replication and assessment."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

METRICS = ("tai", "fai", "fas", "clr", "cxi", "cxl")

__all__ = ["METRICS", "REPO_ROOT"]
