# Schema notes for M2

Hand-written interpretation of `docs/schema_discovery.md` (run of 2026-10-27 17:11 UTC). The generated file is the evidence; this file says what later code must do about it.

## Access

- This account cannot read the `crsp_a_stock` schema, so `crsp.msf_v2`, `crsp.dsf_v2`, `crsp.stksecurityinfohist`, `crsp.stocknames_v2` and `crsp.msf` fail on query. The readable copy is `crsp_a_stock62`.
- Compustat (`comp.funda`), the CCM link (`crsp.ccmxpf_lnkhist`) and Fama-French (`ff`) are readable.

## Monthly stock file

- Source: `crsp_a_stock62.wrds_msf62v2_query`, 1962-07 to 2025-12. It already carries the eight spec 3.3 filter columns and point-in-time `siccd`, so the pull can filter without a separate security info join.
- **Likely duplicate rows.** The view also carries distribution columns (`disexdt`, `disseqnbr`, `distype` and others) and share columns (`shrstartdt`, `shrenddt`). A month with two distributions can then appear twice. The view has about 55,000 more rows than the base table `msf62_v2`. The pull must select only the columns it needs with `SELECT DISTINCT`, then assert one row per `(permno, month)` and fail loudly if not.
- `mthretflg` values: CR (4.73m), NT (81,915), NS (36,871), DE (35,378), MP, IP, GP. NT matches the 81,915 NULL returns exactly. DE is very likely the delisting month. M2 keeps `mthretflg` for diagnostics (spec 3.4) and reports counts by flag rather than assuming meanings.
- `mthprcflg` values: TR, BA, MP. MP matches the 80,068 NULL prices exactly. No DA appears.
- Filter values present: `issuertype` has CORP, ACOR and REIT; `primaryexch` has Q, N, A plus R, X, B, I (excluded by the filter); `conditionaltype` has RW, NT, NW; `tradingstatusflg` has A, X, D, H, S. 3,495,858 rows and 26,373 permnos pass all eight filters over 1962 to 2025.
- 709 returns above 300% (0.0144%), slightly over the spec 3.9 limit of 0.01% for the full 1962 to 2025 history. The M2 check applies to the 1996+ pull and reports the number either way. Returns are never winsorized.

## Security info

- `crsp_a_stock62.stksecurityinfohist62` uses `secinfostartdt` / `secinfoenddt` for each row's validity window (not the `securitybegdt` / `securityenddt` used in the earlier reference script, which span the security's whole life). Windows do not overlap (S1).

## CCM link

- `lpermno` is DOUBLE PRECISION and NULL in 65.5% of rows (links to non-CRSP securities). The pull drops NULL `lpermno` and casts to int64.
- `LS` links exist (7,159), so the `LS` variant in `config.yaml` is usable.

## Compustat

- `gvkey` is VARCHAR(7) in `funda` and VARCHAR(6) in the link table; values are 6 characters in both. Keep as string.
- `ebitda` and `oibdp` are identical wherever both exist, and never differ in NULL-ness, so the `oibdp` fallback is a no-op. Kept for safety.

## Factors

- Decimal units. The `date` column is the first of the month (`dateff` is the month end). Convert `date` to `Period("M")` directly.
- `rf` is slightly negative (-0.0006) in the 1930s. From 1996 it is between 0 and 0.0056.
