"""Cross-table checks from spec 3.9 that need CRSP, the link and Compustat together.

These run on a preliminary universe: the CRSP screen (exchange filter, price, non-financial,
top N by market cap) at each December. M3 builds the real monthly panel and repeats them.
"""

from __future__ import annotations

import pandas as pd

from halo.checks import INFO, Check, check
from halo.compustat import METRIC_INPUTS, attach_fundamentals
from halo.crsp import crsp_screen
from halo.link import link_gvkey


def cross_checks(msf: pd.DataFrame, ccm: pd.DataFrame, funda: pd.DataFrame,
                 cfg: dict) -> list[Check]:
    start = pd.Period(cfg["sample"]["signal_start"], "M").year
    s = crsp_screen(msf, cfg)
    top = s[s["in_top_n"] & (s["month"].dt.month == 12) & (s["month"].dt.year >= start)]
    top = top[["permno", "month"]].reset_index(drop=True)
    linked, n_multi = link_gvkey(top, ccm, cfg)
    linked["has_link"] = linked["gvkey"].notna()
    cov = linked.groupby(linked["month"].dt.year)["has_link"].mean()
    # The first sample year needs a full year of links behind it; years after must hold 90%.
    bad = cov[(cov < 0.90) & (cov.index > start)]
    out = [
        check("X1", f"CCM link coverage of the top {cfg['universe']['n_stocks']} each December "
              "is above 90% in every year after the first", bad.empty,
              ", ".join(f"{y}: {v:.1%}" for y, v in cov.items())
              + f". Below 90%: {bad.round(3).to_dict() or 'none'}"),
        Check("X2", "(permno, December) pairs with more than one valid gvkey, resolved by the "
              "P-then-latest-linkdt rule", INFO, f"{n_multi:,}"),
    ]
    f = cfg["fundamentals"]
    att = attach_fundamentals(linked, funda, f["lag_months"], f["expiry_months"])
    att["has_fund"] = att["fundamentals_datadate"].notna()
    by_year = att.groupby(att["month"].dt.year)
    lines = []
    for y, g in by_year:
        parts = [f"any {g['has_fund'].mean():.0%}"]
        parts += [f"{c} {g[c].notna().mean():.0%}" for c in METRIC_INPUTS if c in g]
        lines.append(f"{y}: " + ", ".join(parts))
    out.append(Check("X3", "Share of the top-N universe each December with a usable "
                     "fundamentals record, and with each metric input non-missing", INFO,
                     " | ".join(lines)))
    has = att[att["has_fund"]]
    lag_month = has["fundamentals_datadate"].dt.to_period("M") + f["lag_months"]
    early = int((has["month"] < lag_month).sum())
    age = has["fundamentals_age_months"]
    out.append(check("X4", f"Lag rule: no row uses fundamentals before datadate + "
                     f"{f['lag_months']} months, or after {f['expiry_months']} months",
                     early == 0 and bool(age.between(f["lag_months"], f["expiry_months"]).all()),
                     f"{early:,} early rows; age {age.min():.0f} to {age.max():.0f} months"))
    return out
