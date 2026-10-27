"""Cross-table checks from spec 3.9 that need CRSP, the link and Compustat together.

These run on a preliminary universe: the CRSP screen (exchange filter, price, non-financial,
top N by market cap) at each December. M3 builds the real monthly panel and repeats them.
"""

from __future__ import annotations

import pandas as pd

from halo import METRICS
from halo.checks import FLAG, INFO, Check, check
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


def exhibit6(panel: pd.DataFrame, min_stocks: int = 5) -> pd.DataFrame:
    """Median score by sector on the last month (Goldman Exhibit 6 reproduction)."""
    last = panel[panel["scored"] & (panel["month"] == panel.loc[panel["scored"], "month"].max())]
    t = last.groupby("sector")["score"].agg(median="median", n="size")
    t = t[(t["n"] >= min_stocks) & ~t.index.isin(["Unknown", "Money"])]
    return t.sort_values("median", ascending=False)


def check_panel(p: pd.DataFrame, cfg: dict) -> list[Check]:
    start = pd.Period(cfg["sample"]["signal_start"], "M")
    s = p[p["month"] >= start]
    out = []
    dup = int(p.duplicated(["permno", "month"]).sum())
    out.append(check("P1", "One row per (permno, month)", dup == 0, f"{dup:,} duplicates"))

    dec = s[s["month"].dt.month == 12]
    n_elig = dec[dec["eligible"]].groupby(dec["month"].dt.year).size()
    drops = n_elig.pct_change()
    bad = drops[drops < -0.30]
    out.append(check("P2", "Eligible stocks each December are stable (no drop over 30%) and at "
                     f"most {cfg['universe']['n_stocks']}",
                     bad.empty and int(n_elig.max()) <= cfg["universe"]["n_stocks"],
                     ", ".join(f"{y}: {n:,}" for y, n in n_elig.items())))

    top = s[s["in_top_n"]]
    cov = top.groupby(top["month"].dt.year)["has_link"].mean()
    low = cov[(cov < 0.90) & (cov.index > start.year)]
    out.append(check("P3", "Link coverage of the top N is above 90% every year after the first",
                     low.empty, ", ".join(f"{y}: {v:.1%}" for y, v in cov.items())))

    e = s[s["eligible"]]
    ey = e.groupby(e["month"].dt.year)
    lines = []
    for y, g in ey:
        parts = [f"{c} {g[c].notna().mean():.0%}" for c in METRICS]
        parts.append(f"intan filled {g['intan_filled'].mean():.1%}")
        parts.append(f"EBITDA<=0 {g['ebitda_nonpos'].mean():.1%}")
        lines.append(f"{y}: " + ", ".join(parts))
    out.append(Check("P4", "Eligible universe: share with each metric, share with intan "
                     "treated as 0, share with EBITDA <= 0 (cxl missing)", INFO, " | ".join(lines)))

    lag, exp = cfg["fundamentals"]["lag_months"], cfg["fundamentals"]["expiry_months"]
    early = int((e["month"] < e["fundamentals_datadate"].dt.to_period("M") + lag).sum())
    age = e["fundamentals_age_months"]
    out.append(check("P5", f"Lag rule on every eligible row: fundamentals at least {lag} and at "
                     f"most {exp} months old", early == 0 and bool(age.between(lag, exp).all()),
                     f"{early:,} early; age {age.min():.0f} to {age.max():.0f}"))

    scored = ey["scored"].mean()
    low_sc = scored[scored < 0.90]
    out.append(check("P6", f"At least 90% of eligible stocks have {cfg['score']['min_metrics']}+ "
                     "metrics and get a score, every year", low_sc.empty,
                     ", ".join(f"{y}: {v:.1%}" for y, v in scored.items()), FLAG))

    held = e[e["scored"] & (e["month"] < s["month"].max())]
    no_row = int((~held["has_next_row"]).sum())
    null_ret = int((held["has_next_row"] & held["ret_next"].isna()).sum())
    share = (no_row + null_ret) / max(len(held), 1)
    out.append(check("P7", "Scored stock-months whose next-month return is missing are under "
                     "0.5% (spec 3.4)", share < 0.005,
                     f"{null_ret:,} NULL next return, {no_row:,} with no next row, of "
                     f"{len(held):,} ({share:.3%})"))

    src = e.groupby(e["month"].dt.year)["sic_source"].value_counts(normalize=True).unstack()
    out.append(Check("P8", "SIC source among eligible rows (CRSP siccd, else Compustat sich)",
                     INFO, " | ".join(f"{y}: " + ", ".join(f"{k} {v:.1%}" for k, v in r.dropna()
                                                            .items()) for y, r in src.iterrows())))

    t = exhibit6(p)
    names = list(t.index)
    ok = "Utils" in names[:3] and "Software" in names[-3:]
    table = ", ".join(f"{k} {r['median']:.2f} (n={int(r['n'])})" for k, r in t.iterrows())
    last = p.loc[p["scored"], "month"].max()
    out.append(check("P9", f"Goldman Exhibit 6 sanity check on {last}: Utils in the top 3 sectors "
                     "and Software in the bottom 3 by median score", ok,
                     f"{table}. If this fails, the metric construction is wrong: stop and report"))
    return out
