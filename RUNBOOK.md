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
| 1 | `python scripts/01_pull_crsp.py` | yes | M2, ready to run |
| 2 | `python scripts/02_pull_compustat.py` | yes (and FRED) | M2, ready to run |
| 3 | `python scripts/03_build_panel.py` | no | M3, ready to run |
| 4 | `python scripts/04_tune_walkforward.py` | no | M5, ready to run |
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

## Steps 1 and 2: pull the data (WRDS)

Run from the `halo` folder with the venv active (`(.venv)` at the start of the prompt) and `WRDS_USERNAME` set. On Windows PowerShell:

```
git pull
$env:WRDS_USERNAME="hayesdl"
python scripts/01_pull_crsp.py
```

then, once it finishes:

```
python scripts/02_pull_compustat.py
```

**Runtime:** step 1 about 5 to 15 minutes (the monthly file comes down in 3-year chunks, one progress line each). Step 2 about 2 to 5 minutes.

**What they pull** (names come from `config/schema_map.yaml`, settings from `config/config.yaml`):

| Step | Data | File in `data/raw/` |
|---|---|---|
| 1 | CRSP monthly, 1994 to today, common stock only (exchange filter applied later, so delisting-month returns are kept) | `crsp_msf_<start>_<end>.parquet` |
| 1 | Security info history | `crsp_secinfo.parquet` |
| 1 | CCM links, types LC, LU, LS and primacy P, C | `ccm_link.parquet` |
| 2 | Compustat annual, standard filters, datadate from 1994 | `comp_funda.parquet` |
| 2 | Fama-French 5 factors and momentum, decimals | `ff_factors.parquet` |
| 2 | FRED GS10, TB3MS, BAA, AAA, with term and default spreads | `fred_rates.parquet` |

Everything in `data/` is gitignored and blocked by the pre-commit hook. Never commit or share it.

**Prints:** progress lines, then `PASTE EVERYTHING BELOW THIS LINE`, then for each table its row count, date range, distinct ids, NULL share per column and schema names used, then the checks as `[PASS]`, `[FAIL]`, `[FLAG]` or `[INFO]` lines. FLAG means worth a look but not an error. The script exits with status 1 if any check FAILs.

**Checks in step 1:** one row per stock-month, prices positive, returns at least -100%, share of returns above 300% (FLAG above 0.01%, FAIL above 1%), NULL returns by return flag, delisting months carry a return, `mthcap` units, 12 months in every year, stocks passing the screen each December, and link-table sanity.

**Checks in step 2:** one Compustat record per firm and date, input coverage by year, factor units and `rf` range, no missing factor months, yield ranges, link coverage of the top 1000 each December (must be above 90%), fundamentals coverage of that universe, and the 6-month lag rule.

**Send back:** everything below the paste line from each step. If FRED is blocked on your network, step 2 reports `RATE0` as FAIL; rerun with `--skip-fred` and tell me.

## Step 3: build the panel (local, no WRDS)

```
git pull
python scripts/03_build_panel.py
```

**Runtime:** about 1 minute; peak memory about 2 GB.

**What it does:** reads the files from steps 1 and 2, and for every stock-month: links to Compustat as of that month, attaches the latest annual report that is at least 6 and at most 18 months old, decides eligibility (exchange filter, price at least $5, non-financial, top 1000 by market cap, linked, has fundamentals), computes the six metrics, ranks them within the eligible universe each month, and computes the equal-weight score (Goldman replica) plus a sector-neutral version. Writes `data/processed/panel.parquet` (gitignored).

**Checks printed:** one row per stock-month, eligible counts each December, link coverage, metric coverage and how often `intan` was treated as 0, the lag rule on every eligible row, share of eligible stocks with a score, missing next-month returns (must be under 0.5%), SIC source, and the Goldman Exhibit 6 check: median score by sector on the last month, with utilities in the top 3 and software in the bottom 3. If Exhibit 6 fails, the metrics are built wrong and we stop to investigate.

**Send back:** everything below the paste line.

## Step 4: tune the blend (local, no WRDS)

```
git pull
python scripts/03_build_panel.py
python scripts/04_tune_walkforward.py
```

Rerun step 3 first: it now also prints check P10 (what happened to stocks with a missing next-month return).

**Runtime:** about 2 to 5 minutes.

**What it does:** evaluates all 3,003 blends of the six metric ranks (weights in steps of 0.1 that sum to 1) on every month, using the objective fixed in the config: net Sharpe of the equal-weight Q5 minus Q1 portfolio, 10 bps per dollar traded on both sides. For each year from 2008 to 2022 it picks the blend with the best smoothed Sharpe (averaged with its neighbours) using only earlier months, then applies it to that year. It also runs the blend halfway back to equal weight, Goldman's equal-weight replica, and each metric on its own. Months from the holdout start (2023-01) are dropped before anything is computed.

**Writes** (gitignored): `results/weights_path.csv`, `oos_returns.csv`, `oos_summary.csv`, and `trials.csv` (every blend and configuration tried, for the multiple-testing count).

**Prints:** the chosen weights each year, how stable they are, and the out-of-sample Sharpe of every strategy. Send back everything below the paste line.
