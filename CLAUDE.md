# CLAUDE.md

The full build spec is [HALO_STRATEGY_SPEC.md](HALO_STRATEGY_SPEC.md). Read it before any work. It wins over this file if they disagree.

## Rules that apply to every change

- WRDS is not reachable from Claude Code. Every WRDS pull is a script the user runs locally. Everything else is built and tested on synthetic data from `src/halo/fixtures.py`.
- Never commit data, results or credentials. `tools/check_staged_files.py` runs as a pre-commit hook and in CI and blocks `.parquet`, files over 5 MB, `.pgpass`, `.env`, and anything under `data/` or `results/`.
- Never guess a WRDS table or column name. After M1, read names from `config/schema_map.yaml`. Downstream code uses the standard CIZ names that the fixture uses.
- Every WRDS-dependent script ends with assertion-based checks (spec 3.9) and prints a report the user can paste back.
- No look-ahead. Every signal function takes an explicit as-of date.
- Monthly dates become `pandas.Period("M")` right after loading. Join only on that period.
- CIZ format: delisting returns are already in `mthret`; prices are positive; `mthcap` is in thousands of dollars; missing returns are NaN and are never filled.
- `gvkey` is a 6-character string. Never cast it to int.
- Plain, direct prose in docs and comments. No em dashes.
- Ask the user before changing the holdout dates, running the holdout, dropping a metric, or changing the tuning objective.

## Workflow

- One milestone per pull request (spec section 10). Current status is in README.md.
- Before pushing: `ruff check .` and `pytest -q`.
- Settings live in `config/config.yaml` and are loaded with `halo.config.load_config`.
