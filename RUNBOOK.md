# RUNBOOK

Commands in the order to run them. Steps marked **WRDS** run on your machine, with your WRDS account. Paste the printed report back into the conversation after each WRDS step.

## One-time setup

```bash
git clone <repo-url> halo-strategy && cd halo-strategy
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,wrds]"
pre-commit install
export WRDS_USERNAME=<your username>
```

The WRDS password goes in `~/.pgpass` (the `wrds` package offers to create it on first connect). Never put it in the repo or in `.env` files inside the repo.

Check the install without touching WRDS:

```bash
pytest -q        # about 10 seconds; all tests use synthetic data
```

## Pipeline

| Step | Command | Needs WRDS | Status |
|---|---|---|---|
| 0 | `python scripts/00_discover_schema.py` | yes | M1, not built yet |
| 1 | `python scripts/01_pull_crsp.py` | yes | M2, not built yet |
| 2 | `python scripts/02_pull_compustat.py` | yes | M2, not built yet |
| 3 | `python scripts/03_build_panel.py` | no | M3, not built yet |
| 4 | `python scripts/04_tune_walkforward.py` | no | M5, not built yet |
| 5 | `python scripts/05_evaluate.py` | no | M6, not built yet |
| 6 | `python scripts/06_holdout.py` | no | M7, run once only, after asking |

Each step's expected runtime and printed output will be documented here as it is built.
