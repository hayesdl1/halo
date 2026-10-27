"""Build the monthly panel (spec 3.8): universe, link, lag, metrics, ranks, score.

One row per (permno, month) for every stock-month in the CRSP pull. A row is
``eligible`` at month-end t when, at t, it passes the exchange filter, has price at
least $5, is non-financial, ranks in the top N by market cap among such stocks, has a
valid CCM link, and has a usable fundamentals record (at least ``lag_months`` and at
most ``expiry_months`` after its datadate). Ranks and scores use eligible rows only.

``ret_next`` is the stock's ``mthret`` in month t+1: the return earned by a portfolio
formed at the end of t (spec 3.4).

No look-ahead: everything in a row uses data dated at or before its month. Passing
``as_of`` first discards every input that was not knowable at that month-end, and the
rows up to ``as_of`` must come out identical to the full build (tests/test_panel.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from halo import METRICS
from halo.compustat import attach_fundamentals
from halo.crsp import crsp_screen
from halo.industries import ff12, sector
from halo.link import link_gvkey
from halo.metrics import compute_metrics
from halo.score import equal_weights, pct_ranks, rank_inputs, sector_groups, weighted_score

KEEP_FUNDA = ["at", "che", "intan", "sale", "ppent", "ppegt", "emp", "capx", "ebitda", "oibdp",
              "dp", "xrd", "dltt", "dlc", "ceq", "seq", "pstk", "pstkrv", "pstkl", "txditc",
              "revt", "cogs", "xsga", "xint"]


def add_lagged_assets(funda: pd.DataFrame) -> pd.DataFrame:
    """``at_lag``: total assets from the same firm's previous fiscal year, if that record
    is 9 to 15 months older (so a fiscal year-end change does not give a bogus growth)."""
    f = funda.sort_values(["gvkey", "datadate"]).copy()
    prev_at = f.groupby("gvkey")["at"].shift(1)
    prev_dd = f.groupby("gvkey")["datadate"].shift(1)
    gap = (f["datadate"].dt.year - prev_dd.dt.year) * 12 + (f["datadate"].dt.month
                                                          - prev_dd.dt.month)
    f["at_lag"] = prev_at.where(gap.between(9, 15))
    return f


def truncate_inputs(msf: pd.DataFrame, ccm: pd.DataFrame, funda: pd.DataFrame, cfg: dict,
                    as_of: pd.Period):
    """Inputs as they could have been known at the end of month ``as_of``."""
    lag = cfg["fundamentals"]["lag_months"]
    end = as_of.to_timestamp(how="end").normalize()
    msf = msf[msf["month"] <= as_of]
    ccm = ccm[ccm["linkdt"] <= end]
    funda = funda[funda["datadate"].dt.to_period("M") + lag <= as_of]
    return msf, ccm, funda


def build_panel(msf: pd.DataFrame, ccm: pd.DataFrame, funda: pd.DataFrame, cfg: dict,
                as_of: str | pd.Period | None = None, log=print) -> pd.DataFrame:
    if as_of is not None:
        msf, ccm, funda = truncate_inputs(msf, ccm, funda, cfg, pd.Period(as_of, "M"))
    f = cfg["fundamentals"]
    p = msf.sort_values(["permno", "month"]).reset_index(drop=True)

    # Next month's return, only if the next row really is month t+1.
    nxt = p.groupby("permno")[["month", "mthret", "mthretflg"]].shift(-1)
    follows = nxt["month"] == p["month"] + 1
    p["ret_next"] = nxt["mthret"].where(follows)
    p["retflg_next"] = nxt["mthretflg"].where(follows)
    p["has_next_row"] = follows.fillna(False).astype(bool)

    # 12-1 momentum: compounded return over months t-11 to t-1, all 11 present and
    # consecutive. Uses only returns up to t-1.
    lr = np.log1p(p["mthret"])
    g = p.groupby("permno")
    lagged = lr.groupby(p["permno"]).shift(1)
    p["_lr1"] = lagged
    mi = p["month"].dt.year * 12 + p["month"].dt.month
    span_ok = (g["month"].shift(1).dt.year * 12 + g["month"].shift(1).dt.month
               - (g["month"].shift(11).dt.year * 12 + g["month"].shift(11).dt.month)) == 10
    roll = p.groupby("permno")["_lr1"].rolling(11, min_periods=11).sum()
    p["mom_12_1"] = np.expm1(roll.reset_index(level=0, drop=True)).where(span_ok & mi.notna())
    p = p.drop(columns="_lr1")

    linked, n_multi = link_gvkey(p[["permno", "month"]], ccm, cfg)
    p["gvkey"] = linked["gvkey"].to_numpy()
    p["has_link"] = p["gvkey"].notna()
    log(f"  linked {p['has_link'].mean():.1%} of stock-months; {n_multi:,} needed the "
        "P-then-latest tie-break")

    # Fundamentals only matter for rows that can be eligible: linked rows passing the
    # exchange, price and market-cap screen. Skipping the rest keeps memory down.
    pre = crsp_screen(p.assign(_nosic=float("nan")), cfg, sic_col="_nosic")["crsp_ok"]
    rows = p.index[pre & p["has_link"]]
    funda = add_lagged_assets(funda)
    keep = ["gvkey", "datadate", *[c for c in KEEP_FUNDA if c in funda], "sich", "at_lag"]
    att = attach_fundamentals(p.loc[rows, ["gvkey", "month"]],
                              funda[[c for c in keep if c in funda]],
                              f["lag_months"], f["expiry_months"])
    for c in att.columns:
        if c not in ("gvkey", "month"):
            p[c] = pd.Series(att[c].to_numpy(), index=rows).reindex(p.index)
    p["has_fund"] = p["fundamentals_datadate"].notna()

    # SIC: CRSP point-in-time siccd, falling back to Compustat sich (spec 3.3).
    crsp_sic = pd.to_numeric(p["siccd"], errors="coerce").astype("float64")
    use_crsp = crsp_sic > 0
    p["sic"] = crsp_sic.where(use_crsp, pd.to_numeric(p.get("sich"), errors="coerce"))
    p["sic_source"] = pd.Series("crsp", index=p.index).where(use_crsp, "compustat")
    p.loc[p["sic"].isna(), "sic_source"] = "none"
    p["ff12"] = ff12(p["sic"])
    p["sector"] = sector(p["sic"])

    p = crsp_screen(p, cfg, sic_col="sic")
    p["eligible"] = p["in_top_n"] & p["has_link"] & p["has_fund"]

    m = compute_metrics(p, ppe=f["ppe"])
    for c in m.columns:
        p[c] = m[c].where(p["has_fund"])
    p["intan_filled"] = p["intan_filled"].fillna(False).astype(bool)
    p["ebitda_nonpos"] = p["ebitda_nonpos"].fillna(False).astype(bool)

    e = p.index[p["eligible"]]
    vals = rank_inputs(p.loc[e], f["nonpositive_ebitda"])
    pct = pct_ranks(vals, p.loc[e, "month"])
    groups = sector_groups(p.loc[e, "sic"], p.loc[e, "month"], cfg["score"]["sector_min_group"])
    pct_sn = pct_ranks(vals, [p.loc[e, "month"], groups])
    w, k = equal_weights(), cfg["score"]["min_metrics"]
    score_std, n_avail = weighted_score(pct, w, k)
    score_sn, _ = weighted_score(pct_sn, w, k)
    for col in METRICS:
        p[f"pct_{col}"] = pct[col].reindex(p.index)
        p[f"pct_sn_{col}"] = pct_sn[col].reindex(p.index)
    p["n_metrics_available"] = n_avail.reindex(p.index).fillna(0).astype(int)
    p["score_std"] = score_std.reindex(p.index)
    p["score_sn"] = score_sn.reindex(p.index)
    primary = "score_sn" if cfg["score"]["sector_neutral"] else "score_std"
    p["score"] = p[primary]
    p["scored"] = p["score"].notna()
    return p


PANEL_FILE = "panel.parquet"


def run_build(raw_dir, out_dir, cfg: dict, log=print):
    """Load the M2 pulls, build the panel, write it, and return (report, checks)."""
    from pathlib import Path

    from halo.checks import render
    from halo.compustat import load_funda
    from halo.crsp import load_msf
    from halo.link import load_ccm
    from halo.validate import check_panel

    log("Loading raw files:")
    msf, ccm, funda = load_msf(raw_dir), load_ccm(raw_dir), load_funda(raw_dir)
    log(f"  msf {len(msf):,} rows, ccm {len(ccm):,}, funda {len(funda):,}")
    log("Building panel:")
    p = build_panel(msf, ccm, funda, cfg, log=log)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p.to_parquet(out / PANEL_FILE, index=False)
    log(f"  wrote {out / PANEL_FILE}")
    checks = check_panel(p, cfg)
    e = p[p["eligible"] & (p["month"] >= pd.Period(cfg["sample"]["signal_start"], "M"))]
    per_month = e.groupby("month").size()
    head = [
        f"== Panel: {out / PANEL_FILE} ==",
        f"rows: {len(p):,}   columns: {p.shape[1]}   months: {p['month'].min()} to "
        f"{p['month'].max()}",
        f"eligible stocks per month from {cfg['sample']['signal_start']}: mean "
        f"{per_month.mean():,.0f}, min {per_month.min():,} ({per_month.idxmin()}), max "
        f"{per_month.max():,}",
        f"scored stock-months: {int(e['scored'].sum()):,} of {len(e):,} eligible",
        f"settings: universe top {cfg['universe']['n_stocks']}, lag "
        f"{cfg['fundamentals']['lag_months']}m, expiry {cfg['fundamentals']['expiry_months']}m, "
        f"ppe {cfg['fundamentals']['ppe']}, nonpositive_ebitda "
        f"{cfg['fundamentals']['nonpositive_ebitda']}, primary score "
        f"{'sector-neutral' if cfg['score']['sector_neutral'] else 'standard'}",
    ]
    return "\n".join(head) + "\n\n" + render("Checks: panel", checks), checks
