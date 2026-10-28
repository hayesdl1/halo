# Findings: the HALO capital intensity score in US equities, 1996 to 2025

Hand-written summary of the results produced by `scripts/05_evaluate.py` (run 2026-10-27) and `scripts/06_holdout.py` (run once, 2026-10-28). Every number below is copied from a CSV in `results/` on the user's machine; the file is named next to each claim. `results/` is not committed (it is derived from licensed WRDS data), so the CSVs are the source of truth and this file is a summary of them.

## Question

Goldman Sachs (The HALO Effect, 24 February 2026) reports that European stocks with heavy, hard-to-replicate physical assets beat capital-light stocks by about 35% from January 2025, using an equal-weight average of six capital-intensity ranks. This project asked whether that score earns a premium in US large and mid caps, whether a tuned blend of the six ranks does better than equal weights, and whether any result survives out of sample.

## Design in one paragraph

Top 1,000 US non-financial common stocks by market cap, price at least $5, each month from 1996. Six metrics from Compustat annual data, usable 6 to 18 months after the fiscal year-end; percentile ranks within the universe; quintile baskets, equal weight, long the most capital-intensive fifth and short the least, rebalanced monthly; 10 bps per dollar traded on buys and sells. Blend weights tuned on a 0.1 grid with an expanding walk-forward (2008 to 2022 out of sample), then frozen and tested once on a sealed holdout (2023 to 2025). Full design: `HALO_STRATEGY_SPEC.md`; settings: `results/settings.csv`.

## Results

1. **No lasting premium.** Goldman's equal-weight score returned -1.6% a year net over 1996 to 2022 (t = -0.70) and -3.2% a year over 2008 to 2022 (t = -1.02) (`perf_summary.csv`). In the holdout it returned -6.0% a year (t = -1.11) (`holdout_perf.csv`). None of these is statistically significant.

2. **Not explained by known factors, in either direction.** The long-short has almost no market beta (-0.01 to -0.13), loads positively on Value (HML about +0.3) and on conservative investment (CMA about +0.2 to +0.4), and its alpha after Fama-French five factors and momentum is -1.6% a year (t = -0.73) over 1996 to 2022 (`factor_regressions.csv`). Adding changes in the term and default spreads does not change the conclusion; the strategy loses when credit spreads widen (default-spread beta about -2.7). Fama-MacBeth regressions give the score no explanatory power once size, value, profitability, investment, momentum and leverage are controlled (t = -0.57) (`fama_macbeth.csv`).

3. **Robust to design choices.** All 16 variants (universe 500 and 1,500, financials included, terciles, halves, value weighting, 3- and 9-month lags, gross PP&E, EBITDA rule, sector-neutral ranking, costs 0 to 25 bps, buys-only costs) give a non-positive equal-weight result; the only positive number in the grid is +0.2% a year for the tuned blend under sector-neutral ranking (`robustness.csv`).

4. **Mostly a sector bet.** The top basket is 40% to 65% utilities and energy; the bottom is led by business equipment (`composition.csv`). Ranking within industries moves results toward zero.

5. **Episodic, not persistent.** The equal-weight score lost 6.0% a year in 2010 to 2019 (t = -2.03) and gained 8.0% a year in 2020 to 2022 (`subperiods.csv`). In 2025 the capital-intensive quintile beat the capital-light quintile by 8.0% (gross, top relative to bottom), the direction Goldman reports but about a quarter of its European figure, over 12 months and not significant (t = 0.94) (`holdout_goldman.csv`, `holdout_perf.csv`).

6. **Tuning: a consistent tilt, no edge.** The walk-forward settled on about 75% fixed-asset share (PP&E / assets) and 25% capital per employee every year. That blend beat Goldman's equal weights in every test: out of sample 2008 to 2022 (Sharpe -0.16 vs -0.30, 99.9th percentile of 1,000 random blends, `comparisons.csv`), in the holdout (-0.13 vs -0.66), and in 2025 (1.10 vs 0.71). It was still never reliably profitable, and the best in-sample blend is indistinguishable from luck after searching 3,003 blends (Reality Check p = 0.71, `reality_check.csv`).

## Conclusion

In US large and mid caps from 1996 to 2025, the HALO capital intensity score did not earn a persistent premium, before or after controlling for known factors, under any of 16 design variants, or in a sealed holdout. The pattern Goldman describes does show up in some periods, including 2025, but it reverses in others, which fits a sector rotation rather than a durable factor. Of the six metrics, fixed-asset share is the most useful, but no blend makes the strategy reliably profitable.

## Caveats

- One country and one universe; Goldman's study is European.
- Annual fundamentals with a conservative lag; missing intangibles treated as zero (about 19% of firms before 2002).
- 0.4% of scored stock-months have no next-month return; their final-month returns averaged +5.6% and none reappear, consistent with takeovers. Not resolved, and small enough not to affect the conclusions (`data_checks.csv`; panel check P10).
- The obsolescence extension was not run; with the holdout used, any test of it would be exploratory.
- The holdout config hash differs from the tuning hash only by Goldman's reference figure added to the config for the report.
