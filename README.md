# halo-strategy

A US-equity version of Goldman Sachs' HALO (Heavy Assets, Low Obsolescence) Capital Intensity Score, built from WRDS CRSP (CIZ format) and Compustat, with an out-of-sample test of whether tuning the blend of its six metrics adds anything. The design is in [HALO_STRATEGY_SPEC.md](HALO_STRATEGY_SPEC.md). Commands to run are in [RUNBOOK.md](RUNBOOK.md).

## Status

| Milestone | Deliverable | Status |
|---|---|---|
| M0 | Scaffold, config, data guard, CI, synthetic fixture | done |
| M1 | WRDS schema discovery | done; see docs/schema_discovery.md and docs/schema_notes.md |
| M2 | Pull scripts | done; both pulls passed (2026-10-27) |
| M3 | Panel build | script built; waiting for the local run |
| M4 | Unit tests (look-ahead, ranks, turnover) | |
| M5 | Tuning and walk-forward | |
| M6 | Evaluation and report | |
| M7 | Holdout | |

## Decisions

Confirmed with the user on 2026-10-27 (details in the spec, section 10):

- Universe: top 1000 by market cap, non-financial, price at least $5.
- Standard ranking is primary; sector-neutral is a reported variant.
- Holdout starts 2023-01 and is sealed until M7.
- Obsolescence extension off for now; revisit after M6.
- Added: leverage control and term/default spread factors. No credit ratings (Compustat coverage ends 2017).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # add ,wrds on the machine that pulls data
pre-commit install               # blocks data files and credentials from commits
pytest -q
```

## Data

WRDS data is licensed and never leaves the user's machine. `data/` and `results/` are gitignored and a pre-commit hook blocks parquet files, files over 5 MB, and credentials. Tests and CI run only on the synthetic fixture in `src/halo/fixtures.py`, which produces tables with the same column names, dtypes and CIZ conventions as the real pulls.

## Layout

```
config/       config.yaml (all research settings), schema_map.yaml (from M1)
docs/         schema_discovery.md (from M1)
src/halo/     library code
scripts/      numbered pipeline scripts, run in order
tools/        repo hygiene (data-file guard)
tests/        synthetic-data tests only
```
