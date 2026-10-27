"""Evaluation (spec sections 5.5, 5.6 and 7). Everything here is pre-holdout only.

All functions take plain series or the panel and return DataFrames, which
scripts/05_evaluate.py writes to results/*.csv. REPORT.md is built from those CSVs
alone (halo/report.py).

Return series are indexed by holding month (``Period[M]``): the month the return is
earned. Factor and rate data are keyed the same way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from halo import METRICS
from halo.portfolio import assign_baskets, long_short

MODELS = {
    "CAPM": ["mktrf"],
    "FF3": ["mktrf", "smb", "hml"],
    "Carhart": ["mktrf", "smb", "hml", "umd"],
    "FF5+UMD": ["mktrf", "smb", "hml", "rmw", "cma", "umd"],
    "FF5+UMD+rates": ["mktrf", "smb", "hml", "rmw", "cma", "umd", "d_term", "d_def"],
}
CONTROLS = ["log_size", "log_bm", "profitability", "asset_growth", "momentum_12_1", "leverage"]


# ---- performance ---------------------------------------------------------------------

def nw_t(x: pd.Series, lags: int = 6) -> float:
    """Newey-West t statistic of the mean."""
    x = pd.Series(x).dropna()
    if len(x) < 3 or x.std() == 0:
        return float("nan")
    fit = sm.OLS(x.to_numpy(), np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(fit.tvalues[0])


def perf_stats(r: pd.Series, turnover: pd.Series | None = None, lags: int = 6) -> dict:
    r = pd.Series(r).dropna()
    n = len(r)
    if n == 0:
        return {"months": 0}
    wealth = (1 + r).cumprod()
    dd = wealth / wealth.cummax() - 1
    sd = r.std()
    return {
        "months": n,
        "mean_ann": 12 * r.mean(),
        "geo_ann": float(wealth.iloc[-1] ** (12 / n) - 1),
        "vol_ann": float(sd * np.sqrt(12)),
        "sharpe": float(r.mean() / sd * np.sqrt(12)) if sd > 0 else float("nan"),
        "t_nw": nw_t(r, lags),
        "max_drawdown": float(dd.min()),
        "hit_rate": float((r > 0).mean()),
        "worst_month": float(r.min()),
        "turnover_month": float(turnover.reindex(r.index).mean()) if turnover is not None
        else float("nan"),
    }


def window(s: pd.Series | pd.DataFrame, start=None, end=None):
    idx = s.index
    m = np.ones(len(idx), dtype=bool)
    if start is not None:
        m &= idx >= pd.Period(start, "M")
    if end is not None:
        m &= idx <= pd.Period(end, "M")
    return s[m]


# ---- factor regressions ---------------------------------------------------------------

def factor_frame(ff: pd.DataFrame, rates: pd.DataFrame | None) -> pd.DataFrame:
    f = ff.set_index("month")
    if rates is not None:
        f = f.join(rates.set_index("month")[["d_term", "d_def", "gs10"]], how="left")
    return f


def regress(y: pd.Series, factors: pd.DataFrame, cols: list[str], lags: int = 6) -> dict:
    d = pd.concat([y.rename("y"), factors[cols]], axis=1, join="inner").dropna()
    X = sm.add_constant(d[cols], has_constant="add")
    if len(d) < len(cols) + 12 or np.linalg.matrix_rank(X.to_numpy()) < X.shape[1]:
        return {"months": len(d)}             # too short, or a factor never varies
    fit = sm.OLS(d["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    out = {"months": len(d), "alpha_ann": 12 * fit.params["const"],
           "alpha_t": fit.tvalues["const"], "r2": fit.rsquared}
    for c in cols:
        out[f"b_{c}"] = fit.params[c]
        out[f"t_{c}"] = fit.tvalues[c]
    return out


def factor_table(series: dict[str, pd.Series], factors: pd.DataFrame, lags: int = 6):
    rows = []
    for name, y in series.items():
        for model, cols in MODELS.items():
            if all(c in factors for c in cols):
                rows.append({"strategy": name, "model": model, **regress(y, factors, cols, lags)})
    return pd.DataFrame(rows)


def spanning_table(series: dict[str, pd.Series], factors: pd.DataFrame, lags: int = 6):
    rows = []
    for name, y in series.items():
        d = pd.concat([y.rename("y"), factors[["hml", "cma"]]], axis=1, join="inner").dropna()
        row = {"strategy": name, "months": len(d), "corr_hml": d["y"].corr(d["hml"]),
               "corr_cma": d["y"].corr(d["cma"])}
        for f in ("hml", "cma"):
            r = regress(y, factors, [f], lags)
            row.update({f"alpha_ann_on_{f}": r.get("alpha_ann"), f"alpha_t_on_{f}":
                        r.get("alpha_t"), f"beta_{f}": r.get(f"b_{f}"), f"t_{f}": r.get(f"t_{f}"),
                        f"r2_{f}": r.get("r2")})
        rows.append(row)
    return pd.DataFrame(rows)


# ---- characteristics for Fama-MacBeth and sorts ---------------------------------------

def add_controls(panel: pd.DataFrame) -> pd.DataFrame:
    """Firm characteristics from the panel, all known at the formation month-end.

    Book equity follows Fama and French: stockholders' equity (seq, else ceq + pstk)
    plus deferred taxes (txditc, 0 if missing) minus preferred stock (pstkrv, else
    pstkl, else pstk, else 0); non-positive book equity is missing. Market equity is
    mthcap in $ millions. Profitability is (revt - cogs - xsga - xint) / book equity
    with missing xsga or xint as 0. Asset growth is at / at_lag - 1. Leverage is
    (dltt + dlc) / at with a missing part as 0, missing if both are missing.
    """
    p = panel
    se = p["seq"].fillna(p["ceq"] + p["pstk"].fillna(0))
    ps = p["pstkrv"].fillna(p["pstkl"]).fillna(p["pstk"]).fillna(0)
    be = (se + p["txditc"].fillna(0) - ps).where(lambda x: x > 0)
    me = p["mthcap"] / 1000
    debt = p[["dltt", "dlc"]].sum(axis=1, min_count=1)
    at = p["at"].where(p["at"] > 0)
    out = pd.DataFrame(index=p.index)
    out["log_size"] = np.log(p["mthcap"].where(p["mthcap"] > 0))
    out["book_to_market"] = be / me
    out["log_bm"] = np.log(out["book_to_market"])
    out["profitability"] = (p["revt"] - p["cogs"] - p["xsga"].fillna(0)
                            - p["xint"].fillna(0)) / be
    out["asset_growth"] = at / p["at_lag"].where(p["at_lag"] > 0) - 1
    out["momentum_12_1"] = p["mom_12_1"]
    out["leverage"] = debt / at
    return out


def _winsor(x: pd.Series, groups: pd.Series, q: float = 0.01) -> pd.Series:
    lo = x.groupby(groups).transform(lambda s: s.quantile(q))
    hi = x.groupby(groups).transform(lambda s: s.quantile(1 - q))
    return x.clip(lo, hi)


def fama_macbeth(panel: pd.DataFrame, start: str, end: str, score_col: str = "score",
                 lags: int = 6) -> pd.DataFrame:
    """Monthly cross-sectional OLS of next-month return on the standardised score, alone
    and with the six controls (winsorised at 1% and 99% each month). Mean slopes with
    Newey-West t statistics."""
    p = panel[panel["eligible"] & panel[score_col].notna() & panel["ret_next"].notna()]
    hold = p["month"] + 1
    p = p[(hold >= pd.Period(start, "M")) & (hold <= pd.Period(end, "M"))]
    c = add_controls(p)
    d = pd.DataFrame({"month": p["month"], "y": p["ret_next"], "score": p[score_col]})
    d["score"] = d.groupby("month")["score"].transform(lambda s: (s - s.mean()) / s.std())
    for col in CONTROLS:
        d[col] = _winsor(c[col].replace([np.inf, -np.inf], np.nan), d["month"])
    rows = []
    for spec, cols in [("score only", ["score"]), ("score + controls", ["score", *CONTROLS])]:
        slopes, n_obs = [], []
        for month, g in d.dropna(subset=["y", *cols]).groupby("month"):
            if len(g) < len(cols) + 20:
                continue
            X = np.column_stack([np.ones(len(g)), g[cols].to_numpy()])
            beta, *_ = np.linalg.lstsq(X, g["y"].to_numpy(), rcond=None)
            slopes.append(pd.Series(beta, index=["const", *cols], name=month))
            n_obs.append(len(g))
        s = pd.DataFrame(slopes)
        for var in ["const", *cols]:
            rows.append({"spec": spec, "variable": var, "mean_slope": s[var].mean(),
                         "t_nw": nw_t(s[var], lags), "months": len(s),
                         "avg_stocks": float(np.mean(n_obs)) if n_obs else 0.0})
    return pd.DataFrame(rows)


def double_sorts(panel: pd.DataFrame, on: str, start: str, end: str, scheme: str = "quintile",
                 lags: int = 6) -> pd.DataFrame:
    """Score quintiles within terciles of a control. Gross equal-weight Q5 minus Q1 in
    each tercile, and the average across terciles."""
    p = panel[panel["eligible"] & panel["score"].notna()].copy()
    hold = p["month"] + 1
    p = p[(hold >= pd.Period(start, "M")) & (hold <= pd.Period(end, "M"))]
    p[on] = add_controls(p)[on].replace([np.inf, -np.inf], np.nan)
    p = p.dropna(subset=[on])
    p["ctl_t"] = assign_baskets(p.assign(_c=p[on]), "_c", "tercile")
    k = 5 if scheme == "quintile" else 3
    out = {}
    for t, g in p.groupby("ctl_t"):
        g = g.copy()
        g["b"] = assign_baskets(g, "score", scheme)
        r = g.dropna(subset=["ret_next"]).groupby(["month", "b"])["ret_next"].mean().unstack()
        out[int(t)] = (r[k] - r[1]).rename(int(t))
    ls = pd.DataFrame(out)
    ls["average"] = ls.mean(axis=1)
    rows = []
    for col in ls.columns:
        s = ls[col].dropna()
        rows.append({"control": on, "tercile": col, "mean_ann": 12 * s.mean(),
                     "t_nw": nw_t(s, lags), "months": len(s)})
    return pd.DataFrame(rows)


def characteristic_long_short(panel: pd.DataFrame, cfg: dict, char: str, sign: int = 1):
    """Benchmark factor in the same universe: top minus bottom quintile on a control,
    equal weight, net of the same costs. ``sign=-1`` sorts on the negative (low minus
    high), as for investment."""
    p = panel.copy()
    p["_char"] = sign * add_controls(p)[char].replace([np.inf, -np.inf], np.nan)
    p.loc[~p["eligible"], "_char"] = np.nan
    ls = long_short(p, cfg, score_col="_char", scheme="quintile", weighting="equal")
    return ls.set_index("hold_month")


# ---- tuning diagnostics -----------------------------------------------------------------

def random_weight_sharpes(engine, n: int, seed: int, hold_mask: np.ndarray) -> pd.DataFrame:
    """Fixed Dirichlet(1) weight vectors evaluated on the masked holding months."""
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.ones(len(METRICS)), size=n)
    net = engine.run(w)["net"][hold_mask]
    mu, sd = np.nanmean(net, axis=0), np.nanstd(net, axis=0, ddof=1)
    out = pd.DataFrame(w, columns=[f"w_{m}" for m in METRICS])
    out["sharpe_net"] = mu / sd * np.sqrt(12)
    return out


def stationary_bootstrap_index(n: int, mean_block: float, rng) -> np.ndarray:
    """Politis and Romano stationary bootstrap indices with geometric block lengths."""
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    new_block = rng.random(n) < 1 / mean_block
    starts = rng.integers(n, size=n)
    for t in range(1, n):
        idx[t] = starts[t] if new_block[t] else (idx[t - 1] + 1) % n
    return idx


def reality_check(net: np.ndarray, n_boot: int = 500, mean_block: float = 6.0,
                  seed: int = 0) -> dict:
    """White's Reality Check on the best Sharpe across all grid points.

    ``net`` is (months, grid points). Under the null every strategy has zero mean. The
    statistic is the largest annualised Sharpe; each bootstrap resamples months in
    blocks, re-centres every column at its sample mean, and records the largest Sharpe.
    Reports one-sided (best positive) and two-sided (largest absolute) p-values.
    """
    r = net[~np.isnan(net).any(axis=1)]
    T = len(r)
    mu, sd = r.mean(axis=0), r.std(axis=0, ddof=1)
    sr = mu / sd * np.sqrt(12)
    rng = np.random.default_rng(seed)
    mx, mx_abs = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        rb = r[stationary_bootstrap_index(T, mean_block, rng)]
        sb = (rb.mean(axis=0) - mu) / rb.std(axis=0, ddof=1) * np.sqrt(12)
        mx[b], mx_abs[b] = np.nanmax(sb), np.nanmax(np.abs(sb))
    best = int(np.nanargmax(sr))
    return {"months": T, "grid_points": r.shape[1], "best_index": best,
            "best_sharpe": float(sr[best]), "p_one_sided": float((mx >= sr[best]).mean()),
            "max_abs_sharpe": float(np.nanmax(np.abs(sr))),
            "p_two_sided": float((mx_abs >= np.nanmax(np.abs(sr))).mean()),
            "bootstraps": n_boot, "mean_block": mean_block}


# ---- sub-periods and composition -------------------------------------------------------

def subperiods(series: dict[str, pd.Series], factors: pd.DataFrame, lags: int = 6):
    windows = [("1996-2000", "1996-02", "2000-12"), ("2001-2009", "2001-01", "2009-12"),
               ("2010-2019", "2010-01", "2019-12"), ("2020-2022", "2020-01", "2022-12")]
    rows = []
    rose = factors["gs10"].diff() > 0 if "gs10" in factors else None
    for name, s in series.items():
        for label, a, b in windows:
            w = window(s, a, b).dropna()
            if len(w) >= 12:
                rows.append({"strategy": name, "period": label, **perf_stats(w, lags=lags)})
        if rose is not None:
            for label, mask in [("10y yield rose", rose), ("10y yield fell", ~rose)]:
                m = mask.reindex(s.index).fillna(False).astype(bool)
                w = s[m].dropna()
                if len(w) >= 12:
                    rows.append({"strategy": name, "period": label, **perf_stats(w, lags=lags)})
    return pd.DataFrame(rows)


def composition(panel: pd.DataFrame, cfg: dict, start: str, end: str) -> pd.DataFrame:
    """Each December: top and bottom basket of the primary score by Fama-French 12
    industry (share of stocks) with stock count and average market cap."""
    p = panel[panel["eligible"] & panel["score"].notna() & (panel["month"].dt.month == 12)]
    p = p[(p["month"] >= pd.Period(start, "M")) & (p["month"] <= pd.Period(end, "M"))].copy()
    k = 5 if cfg["portfolio"]["basket"] == "quintile" else 3
    p["b"] = assign_baskets(p, "score", cfg["portfolio"]["basket"])
    p = p[p["b"].isin([1, k])]
    p["basket"] = np.where(p["b"] == k, "top", "bottom")
    rows = []
    for (m, b), g in p.groupby(["month", "basket"]):
        base = {"year": m.year, "basket": b, "n_stocks": len(g),
                "avg_cap_bn": g["mthcap"].mean() / 1e6}
        for ind, share in g["ff12"].value_counts(normalize=True).items():
            rows.append({**base, "group": "ff12", "industry": ind, "share": share})
        sic2 = (g["sic"] // 100).astype("Int64").astype(str)
        for ind, share in sic2.value_counts(normalize=True).items():
            rows.append({**base, "group": "sic2", "industry": ind, "share": share})
    return pd.DataFrame(rows)
