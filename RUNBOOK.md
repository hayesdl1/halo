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
| 0 | `python scripts/00_discover_schema.py` | yes | M1, ready to run |
| 1 | `python scripts/01_pull_crsp.py` | yes | M2, not built yet |
| 2 | `python scripts/02_pull_compustat.py` | yes | M2, not built yet |
| 3 | `python scripts/03_build_panel.py` | no | M3, not built yet |
| 4 | `python scripts/04_tune_walkforward.py` | no | M5, not built yet |
| 5 | `python scripts/05_evaluate.py` | no | M6, not built yet |
| 6 | `python scripts/06_holdout.py` | no | M7, run once only, after asking |

Each step's expected runtime and printed output are documented below as they are built.

## Step 0: schema discovery (WRDS)

```bash
export WRDS_USERNAME=<your username>
python scripts/00_discover_schema.py
# if both the standard crsp tables and crsp_a_stock62 exist and you want the latter:
python scripts/00_discover_schema.py --prefer-library crsp_a_stock62
```

**Runtime:** a few minutes. Listing tables and describing them is quick. The slow part is a handful of aggregate queries over the monthly stock file (about 5 million rows).

**What it does:** finds the WRDS libraries matching `crsp*`, `comp*` and `ff*`, picks the table for each concept (monthly stock file, security info, CCM link, Compustat annual, factors), maps column names to the canonical names the code uses, and tests every `[VERIFY]` claim in spec section 3. It reads only names, types, counts, date ranges and flag-value frequencies. It never saves rows of data.

**Writes:**
- `config/schema_map.yaml`: the table and column names every later script reads.
- `docs/schema_discovery.md`: the evidence, with a CONFIRMED / REFUTED / UNKNOWN / ERROR row per claim.

Both files are metadata only and are meant to be committed.

**Prints:** a few progress lines, then a line `PASTE EVERYTHING BELOW THIS LINE`, then the full contents of both files, then `Discovery passed.` or `DISCOVERY FAILED:` with a list of reasons. It exits with status 1 if a needed table is missing or a critical check is refuted (for example, a negative price or a market cap that is not in thousands).

**Send back:** everything below the paste line. Or commit the two files on a branch and push:

```bash
git checkout -b m1-schema-results
git add config/schema_map.yaml docs/schema_discovery.md
git commit -m "M1: schema discovery results"
git push -u origin m1-schema-results
```

**If it fails to connect:** check `WRDS_USERNAME`, that `~/.pgpass` exists, and whether WRDS needs you on a university network or VPN. Approve the Duo push if one appears.
