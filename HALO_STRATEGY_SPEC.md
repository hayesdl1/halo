# HALO strategy build spec (for Claude Code)

Repo goal: build and honestly assess a US-equity version of Goldman Sachs' HALO (Heavy Assets, Low Obsolescence) Capital Intensity Score, using WRDS CRSP for prices and returns and Compustat (via the CCM link) for fundamentals, then tune the blend of the six score metrics without overfitting.

Save this file in the repo root and reference it from `CLAUDE.md`. Work milestone by milestone (section 10), one pull request per milestone.

---

## 0. Working rules

1. **Claude Code cannot assume it can reach WRDS.** WRDS needs the user's credentials and often a university network or two-factor login. Build every WRDS pull as a script the user runs locally. Build and test everything else on small synthetic fixtures so CI and your own runs never touch WRDS.
2. **Never commit data or credentials.** WRDS data is licensed. `.gitignore` must cover `data/`, `results/`, `*.parquet`, `*.csv` under `data/`, `.pgpass`, `.env`. Add a pre-commit check that fails if a file over 5 MB or any `.parquet` is staged.
3. **Never guess a WRDS table or column name.** Section 3 separates verified names from names to confirm. Milestone 1 discovers the real schema, writes `docs/schema_discovery.md` and `config/schema_map.yaml`, and all later code reads names from that map.
4. **Every WRDS-dependent script ends with assertion-based validation checks** (section 3.9) and prints a short report the user can paste back to you. You cannot verify WRDS outputs yourself, so make failures loud and specific.
5. **No look-ahead anywhere.** Every function that builds a signal takes an explicit "as of" date. Add a test for this (section 10, M4).
6. Plain, direct prose in docs and comments. No em dashes.
7. Stop and ask the user before: changing the holdout dates, running the holdout, dropping a metric from the score, or changing the tuning objective.

---

## 1. Background: what HALO is

Source: Jaisson, Oppenheimer, Bell and Ferrannini, "The HALO Effect: Heavy Assets, Low Obsolescence in the AI Era," Goldman Sachs Research, 24 Feb 2026 (redacted public version, 19 pages).

- Thesis: higher real yields, fragmentation of supply chains, and AI disruption have moved equity leadership toward firms with physical assets that are costly to replicate and slow to become obsolete (grids, pipelines, utilities, transport, long-cycle industrial capacity), and away from capital-light software and services.
- Goldman's Capital Intensity Score is the **average percentile rank (0 to 100%) of six metrics, measured against the constituents of the index (STOXX 600)**:

| # | Metric | Goldman definition |
|---|---|---|
| 1 | Tangible Asset Intensity | (Assets - Cash - Intangibles) / Sales |
| 2 | Fixed Asset Intensity | PPE / Sales |
| 3 | Fixed Asset Share | PPE / Assets |
| 4 | Capital-Labour Ratio | Tangible Assets / Employees |
| 5 | Capex Intensity | Capex / Sales |
| 6 | Capex Load | Capex / EBITDA |

- Goldman forms two baskets: Capital Intensive, centred near the 75th percentile of the score, and Capital Light, near the 25th. It reports Capital Intensive beating Capital Light by about 35% since Jan 2025 (Europe).
- The score equal-weights the six metrics. **Tuning that blend is the new part of this project.**

Weaknesses in the Goldman note that this project must address, not repeat:

| Weakness | How the project addresses it |
|---|---|
| Short window (2025 onward), no out-of-sample test | Walk-forward tuning over decades plus a sealed holdout (section 5.4) |
| Overlaps with Value and likely with the investment factor (high capex firms tend to have low future returns in the older literature) | Spanning regressions on FF5 + momentum, double sorts on book-to-market and investment (section 7) |
| All six metrics measure asset heaviness; none measures obsolescence | Optional obsolescence extension, off by default (section 4.4) |
| Sector discretion; classification cuts across sectors | Sector-neutral variant and basket composition reports |
| Hindsight construction, no costs | Point-in-time data, costs, turnover |

---

## 2. Research design (defaults; all live in `config/config.yaml`)

| Item | Default | Note |
|---|---|---|
| Market | US common stocks on NYSE, AMEX, NASDAQ | CRSP CIZ filters, section 3.3 |
| Sample | Signals from 1996-01; returns from 1996-02 | Needs Compustat fiscal years from 1995 |
| Universe at each month-end | Top 1000 by market cap, price >= $5, non-financial (SIC outside 6000-6999) | Goldman uses STOXX 600, a large and mid cap universe. Variants: top 500, top 1500, include financials |
| Rebalance | Monthly | Score changes only when new fundamentals arrive or the universe changes |
| Fundamentals lag | Available 6 months after `datadate`, valid for 18 months after `datadate` | Conservative reporting lag, section 3.6 |
| Baskets | Quintiles of the score: Q5 (most capital intensive) and Q1 | Variants: terciles; top half vs bottom half (closer to Goldman's 75th/25th percentile baskets) |
| Weighting | Equal-weighted (primary), value-weighted (secondary) | Equal weight avoids mega-cap domination of 200-stock baskets |
| Long-short | Q5 minus Q1 | Also report Q5 minus universe equal-weight, since Goldman's claim is relative |
| Costs | 10 bps per dollar traded one way | Sensitivity at 0, 5, 10, 25 bps |
| Direction | Hypothesis: high score outperforms | Do not force the sign. Report a negative result as a result |

---

## 3. WRDS data specification

### 3.1 Connection

The reference script the user supplied (`wrds` + `crsp_a_stock62`) never defines `db`. Use:

```python
import wrds
db = wrds.Connection(wrds_username=os.environ["WRDS_USERNAME"])  # password comes from ~/.pgpass, never from the repo
```

Run queries with `db.raw_sql(sql, date_cols=[...])`. Pass `date_cols` for every date column so pandas gets `datetime64`. Close the connection at the end of each script.

### 3.2 CRSP format: this is the CIZ ("v2") format, not the legacy format

Facts below are tagged **[script]** if they appear in the user's working-style reference script, **[docs]** if taken from WRDS/CRSP documentation, **[VERIFY]** if they must be confirmed in Milestone 1. Sources are listed at the end.

**Table and column names.** The user's script uses a schema variant whose names differ from the standard WRDS `crsp` schema. Support both through `config/schema_map.yaml`, and pick whichever exists.

| Concept | Standard WRDS CIZ (`crsp` schema) | User's schema variant (`crsp_a_stock62`) |
|---|---|---|
| Daily stock table | `crsp.dsf_v2` [docs] | `wrds_dsf62v2_query` [script] |
| Monthly stock table | `crsp.msf_v2` [docs] | not in script: discover (a name like `wrds_msf62v2_query` is a guess, [VERIFY]) |
| Security info history | `crsp.stksecurityinfohist` or `crsp.stocknames_v2` [docs] | `stksecurityinfohist62` [script] |
| Security id | `permno` [docs] | `permno` [script] |
| Daily date | `dlycaldt` [docs] | `dlycaldt` [script] |
| Daily return | `dlyret` [docs] | `dlyret` [script] |
| Monthly date / return | `mthcaldt`, `mthret` [docs] | [VERIFY] |
| Monthly price, market cap | `mthprc`, `mthcap` [docs] | [VERIFY] |
| Info validity window | `secinfostartdt`, `secinfoenddt` [docs] | `securitybegdt`, `securityenddt` [script] |
| Ticker | `ticker` [docs] | `ticker` [script] |
| CCM link table | `crsp.ccmxpf_lnkhist` [docs] | [VERIFY] (probably the same table in a related schema) |

**Format rules that differ from legacy CRSP (all [docs] unless noted; confirm each in M1 with the listed test):**

1. **Delisting returns are already inside `mthret` and `dlyret`.** Do not add a separate delisting return and do not look for `dlret` or `dlretx`; they do not exist in CIZ. Adding one would double count. Test: pull a known delisted stock and confirm its last month has a non-null return.
2. **Prices are always positive.** Legacy CRSP flagged bid-ask averages with a negative price. In CIZ the flag is a separate column (`mthprcflg` / `dlyprcflg`; values include `BA`, `TR`, `DA`). Do not wrap prices in `abs()` and do not treat negative prices as meaningful. Assert `price > 0`.
3. **Market cap is `mthcap` / `dlycap`, in thousands of dollars.** Use it directly. Do not rebuild it as price x shares. Test: the largest firm's cap should be in the trillions of dollars after multiplying by 1000.
4. **`mthret` compounds daily returns** with dividends reinvested on ex-dates, so it can differ slightly from legacy `ret`. Returns are decimals (0.01 = 1%).
5. **Missing returns are NULL**, not legacy codes like -66, -77, -88, -99. Never `fillna(0)`. See 3.4.
6. **Legacy `shrcd` and `exchcd` do not exist.** Replace them with the five-field common-stock filter and the exchange filter in 3.3.
7. **Legacy tables (`crsp.dsf`, `crsp.msf`, `crsp.stocknames`) were frozen after 2024-12-31** [docs, secondary source, VERIFY]. All 2025 data, which is Goldman's regime window, lives only in the v2 tables. Never mix legacy and v2 tables.
8. **`mthcaldt` is the last trading day of the month, not the calendar month end.** Convert every monthly date to `pandas.Period("M")` immediately after loading and join only on that period, never on raw dates. This avoids silent misalignment with the Fama-French factor dates.

### 3.3 Universe filters (point in time)

Join the stock table to security info on `permno` and `date BETWEEN secinfostartdt AND secinfoenddt` so classifications are as of each date, not as of today.

Common stock (all five must hold) [docs]:

```sql
sharetype = 'NS' AND securitytype = 'EQTY' AND securitysubtype = 'COM'
AND usincflg = 'Y' AND issuertype IN ('ACOR','CORP')
```

Exchange, equivalent to legacy `exchcd IN (1,2,3)` [docs]:

```sql
primaryexch IN ('N','A','Q') AND conditionaltype = 'RW' AND tradingstatusflg = 'A'
```

Notes: `securitysubtype = 'COM'` alone is not enough, and `sharetype = 'COM'` returns nothing. If the user's schema variant lacks some of these columns, M1 records which are present, and the pull script applies every filter that exists and logs those that do not.

Also pull point-in-time SIC (`siccd` in security info [VERIFY]) for the financials exclusion and the sector-neutral variant. Fall back to Compustat `sich`, then `sic`.

### 3.4 Return handling rules

- Return for holding month `t+1` is `mthret` of month `t+1`, applied to the portfolio formed with information at the end of month `t`.
- If a held stock has NULL `mthret` and was not delisted, drop it from that month's average, count it, and assert such drops are under 0.5% of observations.
- Do not winsorize returns. Do not fill missing returns.
- Keep a `retflag` column if present (`mthretflg`, [VERIFY]) so delisting months can be identified in diagnostics.

### 3.5 CRSP/Compustat link (CCM)

CRSP alone has no fundamentals. The six metrics need Compustat, joined through the CCM link table `ccmxpf_lnkhist` [docs].

```sql
SELECT gvkey, lpermno AS permno, linktype, linkprim, linkdt, linkenddt
FROM <ccm link table from schema_map>
WHERE linktype IN ('LC','LU') AND linkprim IN ('P','C')
```

- Link validity: `linkdt <= date AND (linkenddt IS NULL OR date <= linkenddt)`. A NULL `linkenddt` means the link is still active.
- `LC` and `LU` is the default. Keep `LS` as a config option [VERIFY which link types the user's school licence includes].
- `gvkey` is a 6-character string. Keep it as string and zero-pad. Never cast to int.
- Assert that no `(permno, month)` maps to more than one `gvkey` after applying link dates. If duplicates remain, keep the `P` link, then the most recent `linkdt`, and log the count.

### 3.6 Compustat fundamentals

Library `comp`, table `funda` (annual). Standard industrial filters:

```sql
indfmt = 'INDL' AND datafmt = 'STD' AND popsrc = 'D' AND consol = 'C'
```

Columns: `gvkey, datadate, fyear, at, che, intan, sale, ppent, ppegt, emp, capx, ebitda, oibdp, dp, xrd, sich`. Confirm names in M1 [VERIFY]; `ebitda` may need `oibdp` as a fallback.

Units: dollar items are in millions; `emp` is in thousands of employees. Capital-labour ratio is therefore in millions of dollars per thousand employees. Units do not matter for percentile ranks, so do not rescale, but document it.

Timing: a fiscal year record with `datadate = D` becomes usable at month-end `D + 6 months` and expires after 18 months. Implement with a per-firm `merge_asof` on an `available_date` column. Never use the record on its own `datadate`. Make the lag a config value and run the 3-month and 9-month versions as robustness.

Annual data only: `emp` exists only annually, so quarterly fundamentals are out of scope for v1.

### 3.7 Factor data

Prefer WRDS Fama-French library tables (`ff.factors_monthly`, `ff.fivefactors_monthly`, with momentum `umd` in the monthly file) [VERIFY names and units; WRDS factors are typically in decimals, Ken French's site is in percent]. Fall back to Ken French's library CSVs only if WRDS lacks them. Convert to decimals, convert dates to `Period("M")`, and assert `rf` is between 0 and 0.02 monthly.

### 3.8 Panel construction

Write `data/processed/panel.parquet`, one row per `(permno, month)`, with:
`permno, gvkey, month, mthret, mthprc, mthcap, siccd, exchange flags, eligible, the six raw metrics, the six percentile ranks, n_metrics_available, score, fundamentals_datadate, fundamentals_age_months`.

Eligibility at month-end `t`: passes 3.3 filters at `t`, price >= $5, SIC not 6000-6999, top `N_UNIVERSE` by `mthcap` among all filtered stocks, valid CCM link, fundamentals record available and not expired.

### 3.9 Validation checks (assert, and print a report)

Raw CRSP: no duplicate `(permno, month)`; return range sanity (monthly returns > -1; share of returns > 300% under 0.01%); count of eligible stocks per year is stable (print the series; flag any year with a drop over 30%); date range printed per table.
CCM and Compustat: link coverage of the universe above 90% by count in every year after 1996 (print by year); share of universe with each of the six raw inputs non-missing, by year; the 6-month lag rule holds for every row (`month >= datadate + 6 months`).
Score sanity (Goldman Exhibit 6 reproduction): on the last month-end, print median score by sector. Utilities (SIC 4900-4949) must rank in the top 3 sectors and software and data processing (SIC 7370-7379) in the bottom 3. If this fails, the metric construction is wrong. Stop and report.

---

## 4. Metric definitions (Compustat names; direction: higher means more capital intensive)

| Metric | Formula | Rule for invalid inputs |
|---|---|---|
| `tai` | (`at` - `che` - `intan`) / `sale` | `sale` <= 0 or `at` <= 0 gives NaN. `intan` missing is treated as 0 (log the count). `che` missing gives NaN |
| `fai` | `ppent` / `sale` | `sale` <= 0 gives NaN |
| `fas` | `ppent` / `at` | `at` <= 0 gives NaN |
| `clr` | (`at` - `che` - `intan`) / `emp` | `emp` <= 0 or missing gives NaN |
| `cxi` | `capx` / `sale` | `sale` <= 0 gives NaN. `capx` missing gives NaN (never 0) |
| `cxl` | `capx` / `ebitda` | `ebitda` <= 0 gives NaN by default. Config option `nonpositive_ebitda: max_rank` assigns the top rank. Test both |

Config option `ppe: ppent | ppegt` (default `ppent`).

**Percentile rank:** at each month-end, within the eligible universe only, `pct = rank(method="average") / n_non_missing`, giving values in (0, 1]. Tied values share a rank. Compute ranks on the full eligible universe before assigning baskets.

**Score:** with weights `w` on the six metrics, `score_i = sum_k w_k * pct_ik / sum_k w_k` over the metrics available for stock `i`. Require at least 4 of 6 metrics available, otherwise the stock is dropped that month. Goldman's replica is `w = (1/6, ..., 1/6)`.

**Sector-neutral variant:** compute percentile ranks within 2-digit SIC groups (merge groups smaller than 10 into their 1-digit parent), then compute the score. Keep as a config flag; report both.

### 4.4 Optional obsolescence extension (off by default, build after M6)

Goldman never measures obsolescence. Test three added metrics, each ranked so that higher means more obsolescence risk, entered with a negative sign: depreciation rate `dp / ppegt` (faster write-off), R&D intensity `xrd / sale` (technology disruption), intangible share `intan / at`. These get their own weights in the tuning search. Report whether any survives out of sample.

---

## 5. Tuning the blend

### 5.1 What is tuned

Weights `w` on the six percentile-rank metrics: `w_k >= 0`, `sum w_k = 1`. Nothing else is tuned. Universe, basket definition, lag, and costs are fixed by config before tuning starts and varied only in the robustness section.

### 5.2 Search space

Simplex grid with step 0.1 (3003 points for six metrics). Precompute the six percentile-rank matrices once as arrays of shape (months, stocks). A weight vector then costs one weighted sum, one quantile cut, and one equal-weight return calculation per month, so the full grid is cheap. Vectorize it; do not loop over stocks. Optional refinement: step 0.05 in a neighbourhood of the best 20 grid points.

### 5.3 Objective and regularisation

- Objective: annualised Sharpe ratio of the **net** Q5-minus-Q1 equal-weighted monthly long-short return (signed, high score minus low score). One objective only. Other statistics are for evaluation, not selection.
- Smoothed objective: the score of a grid point is the mean Sharpe over that point and its simplex neighbours at L1 distance up to 0.2. This prefers broad plateaus over isolated spikes.
- Shrinkage family reported side by side: `w_final = lambda * w_tuned + (1 - lambda) * (1/6)` for lambda in {0, 0.5, 1}. Lambda = 0 is the Goldman replica.

### 5.4 Protocol: expanding-window walk-forward

1. First tuning window: 1996-02 to 2007-12 (minimum 10 years).
2. For each year `Y` from 2008 to 2022: tune on all months up to Dec of `Y-1`, freeze the weights, and apply them to months in `Y`. Because holding periods are one month, no embargo is needed.
3. Concatenate the out-of-sample months. This 2008-2022 series is the headline result.
4. Log the chosen weights each year to `results/weights_path.csv`. If weights jump between corners year to year, the tuning is fitting noise. Report this.
5. **Holdout: 2023-01 onward, sealed.** Tuning code asserts `max(month) < HOLDOUT_START`. A separate script `scripts/06_holdout.py` runs the single final evaluation using weights frozen on all data before the holdout, writes `results/holdout_evaluated.lock`, and refuses to run again without `--force` (which logs a warning in the report). Also report the 2025-01 to latest sub-window, which is Goldman's regime.

### 5.5 Comparisons the tuned result must beat to count

| Benchmark | Purpose |
|---|---|
| Equal-weight six metrics (Goldman replica) | Does tuning add anything |
| Each metric alone | Which metrics matter |
| Random weights: 1000 Dirichlet(1) vectors, fixed (not tuned), evaluated over the same out-of-sample months | Where does tuned performance sit in the distribution of arbitrary blends |
| Value (top vs bottom quintile on book-to-market) and Investment (low vs high asset growth) long-short | Is HALO a repackaged known factor |

### 5.6 Multiple-testing discipline

Log every weight vector and configuration evaluated to `results/trials.csv` (config hash, objective, count). Report the total number of trials. For the in-sample best-weight Sharpe, report a Reality Check style p-value using a stationary bootstrap over the max Sharpe across the grid (or the deflated Sharpe ratio). The walk-forward out-of-sample series needs no deflation for the weights, but configuration choices (basket, universe, lag) do count as trials.

---

## 6. Portfolio construction

- At each month-end `t`: form baskets from scores, hold through `t+1`.
- Equal weight within basket. Weight drift within the month is tracked so turnover is measured against drifted weights: `turnover_t = 0.5 * sum_i |w_target_i - w_drifted_i|`, summed over both legs for the long-short.
- Cost per month = `turnover x cost_bps / 10000`. Report gross and net.
- Basket composition report each year: number of stocks, share of weight by 2-digit SIC and by Fama-French 12 industry, average market cap. The point is to show how much of the result is a sector bet.

---

## 7. Evaluation outputs

For every strategy variant (tuned, replica, single metrics, benchmarks), over the walk-forward out-of-sample window, the holdout, and the Goldman-regime window:

1. Mean, geometric return, volatility, Sharpe, Newey-West t of the mean (6 lags), max drawdown, hit rate, worst month, turnover, net-of-cost versions.
2. Factor regressions of the long-short: CAPM, FF3, Carhart, and **FF5 + momentum** (MKT, SMB, HML, RMW, CMA, UMD). Report alpha (annualised), its Newey-West t, and betas. Expect a negative CMA loading, since capex heavy firms invest a lot. The alpha after CMA is the real HALO test.
3. Spanning tests: regress the HALO long-short on HML and on CMA separately, and report the correlation with HML (Goldman's Exhibit 12 claims HALO tracks Value versus Growth).
4. Fama-MacBeth: monthly cross-sectional regression of next-month return on the standardised score, controlling for log size, book-to-market, profitability, asset growth, and 12-1 month momentum. Report the mean slope and Newey-West t.
5. Double sorts: score quintiles within book-to-market terciles, and within asset-growth terciles.
6. Sub-period table: by decade, by rate regime (months when the 10-year yield rose versus fell, if a yield series is available on WRDS or FRED; skip if not), and the 2025-onward window.
7. Goldman claim check: cumulative Q5 versus Q1 relative performance from Jan 2025 to the last available month, on the same axes as Goldman's headline (+35% in Europe).
8. Robustness grid: universe size (500, 1000, 1500), include financials, tercile and top-half/bottom-half baskets, value-weighted, lag 3 and 9 months, `ppegt`, `nonpositive_ebitda: max_rank`, sector-neutral, cost levels.
9. Charts (thin lines, one-hue ordered ramp for the five quintiles, no dual axes): quintile growth of $1, long-short growth gross and net, weights path, tuned versus random-weight distribution, annual long-short bars.

Write `results/REPORT.md` automatically: design, data checks, headline table, the comparisons from 5.5, limitations. Every number in the report must come from a CSV in `results/`.

---

## 8. Repo layout

```
halo-strategy/
  README.md  CLAUDE.md  HALO_STRATEGY_SPEC.md  RUNBOOK.md
  pyproject.toml          # python>=3.11; pandas numpy scipy statsmodels pyarrow wrds matplotlib pyyaml pytest ruff
  .gitignore  .pre-commit-config.yaml
  config/config.yaml  config/schema_map.yaml   # schema_map generated in M1
  docs/schema_discovery.md                     # generated in M1
  src/halo/  wrds_io.py  crsp.py  compustat.py  link.py  metrics.py  score.py
             portfolio.py  tuning.py  evaluate.py  report.py  fixtures.py
  scripts/   00_discover_schema.py  01_pull_crsp.py  02_pull_compustat.py
             03_build_panel.py  04_tune_walkforward.py  05_evaluate.py  06_holdout.py
  tests/     (synthetic fixtures only)
  data/raw  data/processed  results/            # gitignored
  .github/workflows/ci.yml                      # ruff + pytest on synthetic data, no WRDS
```

`RUNBOOK.md` lists the exact commands in order, expected runtimes, and what each script prints, so the user can run the WRDS steps locally and paste outputs back.

---

## 9. Pull scripts: format requirements

- Pull CRSP in year chunks (for example 3 years at a time) to avoid timeouts. Save each chunk to `data/raw/crsp_msf_{start}_{end}.parquet` with explicit dtypes: `permno` int64, dates as `datetime64[ns]`, returns and prices as float64, flags as category.
- Pull monthly data as the primary source. Pull daily only if a diagnostic needs it (the daily table is large).
- Never build `IN (...)` lists with thousands of permnos. Filter in SQL with the 3.3 conditions instead, and join security info inside the query.
- Use parameterised dates, not string concatenation of user input.
- `gvkey` as string, `datadate` as date. Compustat pull restricted to `datadate >= '1994-01-01'`.
- Every pull script prints: row count, date range, distinct permno or gvkey count, share of NULLs per column, and which schema map entries it used.

---

## 10. Milestones and acceptance criteria

| M | Deliverable | Done when |
|---|---|---|
| M0 | Scaffold, config, `.gitignore`, pre-commit, CI, `fixtures.py` building a synthetic 200-stock, 20-year panel in the real column format | CI passes; staging a parquet is blocked |
| M1 | `00_discover_schema.py`: lists WRDS libraries matching `crsp*`, `comp*`, `ff*`; for every table used, prints columns, dtypes, min and max dates and row counts; writes `docs/schema_discovery.md` and `config/schema_map.yaml`; confirms or refutes every `[VERIFY]` item in section 3 | User has run it and pasted output; schema map committed |
| M2 | Pull scripts for CRSP monthly, security info, CCM link, Compustat annual, factors | User ran them; validation report (3.9) passes or lists exact failures |
| M3 | `03_build_panel.py`: universe, link, lag, metrics, ranks, score | Score sanity check (3.9) passes; panel has no look-ahead (test below) |
| M4 | Unit tests: hand-computed metric values; ranking with ties; `merge_asof` lag test (no row uses fundamentals before `datadate + 6m`); a test that shifts all fundamentals dates forward and confirms results do not improve; basket formation; turnover with drift | All tests pass on fixtures |
| M5 | `tuning.py` and `04_tune_walkforward.py`: vectorised grid, smoothed objective, walk-forward, trial log, holdout guard | On a fixture with a planted weighting (for example only `clr` predicts returns), tuning recovers weights near the planted vector and the equal-weight replica does worse |
| M6 | `evaluate.py`, `05_evaluate.py`, `REPORT.md`, charts | Report reproduces from CSVs; null fixture produces no false positive at |t| > 3 |
| M7 | `06_holdout.py` with lock file; final report with holdout and 2025 window | Run once by the user |

Open decisions to confirm with the user before M5 (defaults are in config): universe size, sector-neutral as primary or variant, holdout start date, whether the obsolescence extension is in scope.

---

## 11. Sources for the CRSP format claims

- WRDS, CRSP CIZ to SIZ macro, variable and `shrcd`/`exchcd` mapping: https://wrds-www.wharton.upenn.edu/pages/wrds-research/macros/crsp-ciztosiz-macro/
- WRDS, Flat File Format 2.0 (CIZ) webinar, table names, delisting returns, positive prices with price flag: https://wrds-www.wharton.upenn.edu/documents/2084/Webinar.pdf
- Secondary summary (not independently verified; hence the `[VERIFY]` tags): common-stock and exchange filter values, units of `mthcap`, legacy table freeze date, CCM table name: https://www.skillsdirectory.com/skills/edwinhu-crsp-v2
- Goldman Sachs Research, The HALO Effect, 24 Feb 2026 (redacted public version).
