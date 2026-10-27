"""Baskets, returns, turnover and costs (spec sections 2 and 6).

At each month-end t, eligible stocks with a score are sorted into baskets. A basket
is held through month t+1 and earns the stocks' ``ret_next``. Results are indexed by
the formation month t; ``hold_month`` = t + 1 is the month the return is earned.

Turnover is measured against drifted weights: the weights a basket formed at t-1
has at the end of month t after its stocks' returns, rescaled to sum to one. Stocks
that left (no return) are treated as sold with no cost. ``turnover = 0.5 * sum |target
- drifted|`` per basket; the long-short sums both legs. A basket built from nothing has
turnover 0.5 under this formula.

This is the clear, loop-free reference implementation. The tuning code in M5 uses
array shortcuts and is tested against it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_BASKETS = {"quintile": 5, "tercile": 3, "half": 2}


def assign_baskets(df: pd.DataFrame, score_col: str, scheme: str) -> pd.Series:
    """Basket number 1 (lowest score) to k (highest) within each month.

    Equal-count buckets: the i-th of n stocks by score goes to basket ceil(i * k / n).
    Ties in score break by permno, so the result never depends on row order.
    """
    k = N_BASKETS[scheme]
    d = df[["month", "permno", score_col]].dropna(subset=[score_col])
    d = d.sort_values(["month", score_col, "permno"])
    pos = d.groupby("month").cumcount() + 1
    n = d.groupby("month")["permno"].transform("size")
    basket = np.ceil(pos * k / n).astype(int)
    return basket.reindex(df.index)


def holdings(panel: pd.DataFrame, cfg: dict, score_col: str = "score",
             scheme: str | None = None, weighting: str | None = None) -> pd.DataFrame:
    """One row per (formation month, stock) with its basket and target weight."""
    scheme = scheme or cfg["portfolio"]["basket"]
    weighting = weighting or cfg["portfolio"]["weighting"]
    h = panel.loc[panel["eligible"] & panel[score_col].notna(),
                  ["month", "permno", score_col, "mthcap", "ret_next", "has_next_row"]].copy()
    h["basket"] = assign_baskets(h, score_col, scheme)
    if weighting == "equal":
        h["w"] = 1.0 / h.groupby(["month", "basket"])["permno"].transform("size")
    else:
        h["w"] = h["mthcap"] / h.groupby(["month", "basket"])["mthcap"].transform("sum")
    return h.reset_index(drop=True)


def basket_returns(h: pd.DataFrame) -> pd.DataFrame:
    """Return of each basket held over the next month, dropping stocks with no return
    (spec 3.4) and rescaling the rest. Columns: ret, n, n_missing."""
    ok = h["ret_next"].notna()
    g = h[ok].groupby(["month", "basket"])
    wsum = g["w"].sum()
    ret = (h.loc[ok, "w"] * h.loc[ok, "ret_next"]).groupby([h.loc[ok, "month"],
                                                            h.loc[ok, "basket"]]).sum() / wsum
    n = h.groupby(["month", "basket"]).size()
    miss = (~ok).groupby([h["month"], h["basket"]]).sum()
    return pd.DataFrame({"ret": ret, "n": n, "n_missing": miss})


def basket_turnover(h: pd.DataFrame) -> pd.Series:
    """0.5 * sum |target - drifted| per (formation month, basket)."""
    prev = h[h["ret_next"].notna()].copy()
    prev["grown"] = prev["w"] * (1 + prev["ret_next"])
    prev["wd"] = prev["grown"] / prev.groupby(["month", "basket"])["grown"].transform("sum")
    prev["month"] = prev["month"] + 1           # drifted weights as of the next formation
    m = h[["month", "basket", "permno", "w"]].merge(
        prev[["month", "basket", "permno", "wd"]], on=["month", "basket", "permno"],
        how="outer")
    m[["w", "wd"]] = m[["w", "wd"]].fillna(0.0)
    m = m[m.set_index(["month", "basket"]).index.isin(
        h.set_index(["month", "basket"]).index.unique())]
    return 0.5 * (m["w"] - m["wd"]).abs().groupby([m["month"], m["basket"]]).sum()


def long_short(panel: pd.DataFrame, cfg: dict, score_col: str = "score",
               scheme: str | None = None, weighting: str | None = None,
               cost_bps: float | None = None, cost_basis: str | None = None) -> pd.DataFrame:
    """Monthly top-minus-bottom basket returns, gross and net of costs.

    Also reports each basket's return, the equal-weight return of all scored eligible
    stocks (``univ``), and top minus universe, since Goldman's claim is relative.
    """
    p = cfg["portfolio"]
    scheme = scheme or p["basket"]
    cost_bps = p["cost_bps"] if cost_bps is None else cost_bps
    cost_mult = 2.0 if (cost_basis or p.get("cost_basis", "spec")) == "both_sides" else 1.0
    k = N_BASKETS[scheme]
    h = holdings(panel, cfg, score_col, scheme, weighting)
    br = basket_returns(h)
    to = basket_turnover(h)
    ret = br["ret"].unstack()
    out = pd.DataFrame(index=ret.index)
    for b in range(1, k + 1):
        out[f"q{b}"] = ret.get(b)
    out["univ"] = h[h["ret_next"].notna()].groupby("month")["ret_next"].mean()
    out["to_long"] = to.xs(k, level="basket")
    out["to_short"] = to.xs(1, level="basket")
    out["cost"] = (out["to_long"] + out["to_short"]) * cost_bps / 1e4 * cost_mult
    out["ls_gross"] = out[f"q{k}"] - out["q1"]
    out["ls_net"] = out["ls_gross"] - out["cost"]
    out["top_minus_univ"] = out[f"q{k}"] - out["univ"]
    out["n_long"] = br["n"].xs(k, level="basket")
    out["n_short"] = br["n"].xs(1, level="basket")
    out["n_missing"] = br["n_missing"].groupby(level="month").sum()
    out["n_held"] = br["n"].groupby(level="month").sum()
    out = out[out[["q1", f"q{k}"]].notna().all(axis=1)]
    out.index.name = "month"
    out.insert(0, "hold_month", out.index + 1)
    return out


def missing_share(ls: pd.DataFrame) -> float:
    """Share of held stock-months dropped for a missing next-month return (spec 3.4)."""
    return float(ls["n_missing"].sum() / max(ls["n_held"].sum(), 1))


def annualised_sharpe(r: pd.Series, periods: int = 12) -> float:
    r = r.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods))
