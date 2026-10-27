"""Tune the blend of the six metric ranks (spec section 5).

The engine stores, for each formation month, the eligible stocks (sorted by permno)
with their six percentile ranks and next-month return. Any set of weight vectors is
then scored in one pass: a weighted mean of the available ranks, equal-count baskets,
equal-weight returns, turnover against drifted weights, and costs. Results for every
grid point and every month are kept, so a walk-forward window is just a slice.

``Engine.run`` must reproduce ``halo.portfolio.long_short`` exactly for the same
weights; tests/test_tuning.py checks this.

Nothing here may see the holdout. ``Engine.from_panel`` drops every formation month
whose holding month is at or after ``holdout.start`` and the walk-forward asserts it.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from halo import METRICS
from halo.portfolio import N_BASKETS

K = len(METRICS)


def simplex_grid(step: float = 0.1, k: int = K) -> np.ndarray:
    """All weight vectors with k non-negative entries in multiples of ``step`` summing to 1."""
    n = round(1 / step)
    pts = [c for c in itertools.product(range(n + 1), repeat=k) if sum(c) == n]
    return np.array(pts, dtype="float64") / n


def neighbours(grid: np.ndarray, radius: float) -> list[np.ndarray]:
    """Indices of grid points within L1 distance ``radius`` of each point (itself included)."""
    d = np.abs(grid[:, None, :] - grid[None, :, :]).sum(axis=2)
    return [np.flatnonzero(row <= radius + 1e-9) for row in d]


def sharpe_cols(r: np.ndarray, periods: int = 12) -> np.ndarray:
    """Annualised Sharpe of each column, ignoring NaN months."""
    mu = np.nanmean(r, axis=0)
    sd = np.nanstd(r, axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sd > 0, mu / sd * np.sqrt(periods), np.nan)


@dataclass
class Month:
    pct: np.ndarray          # (n, K) ranks, NaN where missing
    ret: np.ndarray          # (n,) next-month return, NaN where missing
    n_avail: np.ndarray      # (n,) number of non-missing ranks
    prev_pos: np.ndarray     # (n_prev,) position of each previous-month stock now, -1 if gone


@dataclass
class Engine:
    months: pd.PeriodIndex                   # formation months
    data: list[Month]
    min_metrics: int
    scheme: str
    cost_bps: float
    cost_mult: float
    rank_prefix: str = "pct_"
    meta: dict = field(default_factory=dict)

    @property
    def hold_months(self) -> pd.PeriodIndex:
        return self.months + 1

    @classmethod
    def from_panel(cls, panel: pd.DataFrame, cfg: dict, sector_neutral: bool | None = None,
                   holdout_guard: bool = True) -> Engine:
        sn = cfg["score"]["sector_neutral"] if sector_neutral is None else sector_neutral
        prefix = "pct_sn_" if sn else "pct_"
        cols = [f"{prefix}{m}" for m in METRICS]
        e = panel.loc[panel["eligible"], ["month", "permno", "ret_next", *cols]]
        if holdout_guard:
            holdout = pd.Period(cfg["holdout"]["start"], "M")
            e = e[e["month"] + 1 < holdout]
        e = e.sort_values(["month", "permno"])
        months, data, prev_ids = [], [], None
        for month, g in e.groupby("month", sort=True):
            ids = g["permno"].to_numpy()
            pct = g[cols].to_numpy(dtype="float64")
            if prev_ids is None or months[-1] + 1 != month:
                prev_pos = np.array([], dtype=int)
            else:
                pos = np.searchsorted(ids, prev_ids)
                pos = np.minimum(pos, len(ids) - 1)
                prev_pos = np.where(ids[pos] == prev_ids, pos, -1)
            data.append(Month(pct=pct, ret=g["ret_next"].to_numpy(dtype="float64"),
                              n_avail=(~np.isnan(pct)).sum(axis=1), prev_pos=prev_pos))
            months.append(month)
            prev_ids = ids
        p = cfg["portfolio"]
        mult = 2.0 if p.get("cost_basis", "spec") == "both_sides" else 1.0
        return cls(pd.PeriodIndex(months, freq="M"), data, cfg["score"]["min_metrics"],
                   p["basket"], p["cost_bps"], mult, prefix)

    def run(self, weights: np.ndarray, cost_bps: float | None = None) -> dict[str, np.ndarray]:
        """Monthly results for G weight vectors.

        ``weights`` is (G, K) for fixed weights, or (T, G, K) for weights that change by
        formation month (a walk-forward path). Returns arrays of shape (T, G): ``top``,
        ``bottom``, ``gross`` (top minus bottom), ``net``, ``to_long``, ``to_short``.
        """
        w_all = np.asarray(weights, dtype="float64")
        fixed = w_all.ndim == 2
        T = len(self.data)
        G = w_all.shape[0] if fixed else w_all.shape[1]
        k = N_BASKETS[self.scheme]
        bps = self.cost_bps if cost_bps is None else cost_bps
        out = {name: np.full((T, G), np.nan) for name in
               ("top", "bottom", "gross", "net", "to_long", "to_short")}
        prev = None
        for t, m in enumerate(self.data):
            w = w_all if fixed else w_all[t]
            avail = ~np.isnan(m.pct)
            num = np.where(avail, m.pct, 0.0) @ w.T              # (n, G)
            den = avail.astype("float64") @ w.T
            with np.errstate(invalid="ignore", divide="ignore"):
                score = np.where((m.n_avail[:, None] >= self.min_metrics) & (den > 0),
                                 num / den, np.nan)
            order = np.argsort(score, axis=0, kind="stable")     # NaN last, ties by permno
            n_ok = (~np.isnan(score)).sum(axis=0)                 # (G,)
            pos = np.empty_like(order)
            np.put_along_axis(pos, order, np.arange(1, len(score) + 1)[:, None], axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                basket = np.ceil(pos * k / n_ok[None, :])
            valid = ~np.isnan(score)
            legs = {}
            for name, b in (("top", k), ("bottom", 1)):
                mem = valid & (basket == b)
                w_t = mem / np.maximum(mem.sum(axis=0), 1)
                has_r = ~np.isnan(m.ret)
                r = np.where(has_r, m.ret, 0.0)[:, None]
                held = mem & has_r[:, None]
                cnt = held.sum(axis=0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    out[name][t] = np.where(cnt > 0, (held * r).sum(axis=0) / cnt, np.nan)
                to = self._turnover(w_t, prev[name] if prev else None, m.prev_pos)
                out["to_long" if name == "top" else "to_short"][t] = to
                legs[name] = (w_t, held, r)
            out["gross"][t] = out["top"][t] - out["bottom"][t]
            cost = (out["to_long"][t] + out["to_short"][t]) * bps / 1e4 * self.cost_mult
            out["net"][t] = out["gross"][t] - cost
            prev = {}
            for name, (w_t, held, r) in legs.items():
                grown = np.where(held, w_t * (1 + r), 0.0)
                tot = grown.sum(axis=0)
                prev[name] = np.divide(grown, tot, out=np.zeros_like(grown), where=tot > 0)
        return out

    @staticmethod
    def _turnover(w_t: np.ndarray, wd_prev: np.ndarray | None, prev_pos: np.ndarray):
        if wd_prev is None or len(prev_pos) == 0:
            return 0.5 * np.abs(w_t).sum(axis=0)
        wd = np.zeros_like(w_t)
        here = prev_pos >= 0
        wd[prev_pos[here]] = wd_prev[here]
        gone = wd_prev[~here].sum(axis=0)                         # held, now out of universe
        return 0.5 * (np.abs(w_t - wd).sum(axis=0) + gone)


@dataclass
class WalkForward:
    weights_path: pd.DataFrame
    oos: pd.DataFrame
    trials: pd.DataFrame
    grid: np.ndarray


def tune_window(sharpe: np.ndarray, nbrs: list[np.ndarray]) -> tuple[int, np.ndarray]:
    """Index of the grid point with the best smoothed Sharpe, and all smoothed values."""
    smooth = np.array([np.nanmean(sharpe[ix]) for ix in nbrs])
    best = int(np.nanargmax(smooth))
    return best, smooth


def walk_forward(engine: Engine, cfg: dict, config_hash: str = "") -> WalkForward:
    """Expanding-window walk-forward (spec 5.4). Returns the weights path, the
    out-of-sample monthly returns of every strategy, and the trial log."""
    t = cfg["tuning"]
    holdout = pd.Period(cfg["holdout"]["start"], "M")
    hold = engine.hold_months
    assert hold.max() < holdout, f"engine reaches into the holdout ({hold.max()} >= {holdout})"
    grid = simplex_grid(t["grid_step"])
    nbrs = neighbours(grid, t["smoothing_l1_radius"])
    res = engine.run(grid)
    net = res["net"]
    first_end = pd.Period(t["first_train_end"], "M")
    y0, y1 = t["walkforward_years"]
    start = pd.Period(cfg["sample"]["return_start"], "M")
    lams = t["shrinkage_lambdas"]
    eq = np.full(K, 1 / K)

    rows, trials = [], []
    path = {lam: np.tile(eq, (len(hold), 1)) for lam in lams}
    oos_mask = np.zeros(len(hold), dtype=bool)
    for year in range(y0, y1 + 1):
        train_end = pd.Period(f"{year - 1}-12", "M")
        assert train_end >= first_end, "first tuning window is shorter than configured"
        assert train_end < holdout
        train = (hold >= start) & (hold <= train_end)
        apply = (hold.year == year)
        if not apply.any():
            break
        assert hold[apply].max() < holdout
        sr = sharpe_cols(net[train])
        best, smooth = tune_window(sr, nbrs)
        w_best = grid[best]
        trials.append(pd.DataFrame({
            "config_hash": config_hash, "year": year, "grid_index": np.arange(len(grid)),
            **{f"w_{m}": grid[:, j] for j, m in enumerate(METRICS)},
            "sharpe_net": sr, "sharpe_smoothed": smooth,
        }))
        rows.append({"year": year, "train_start": str(start), "train_end": str(train_end),
                     "train_months": int(train.sum()),
                     **{f"w_{m}": w_best[j] for j, m in enumerate(METRICS)},
                     "best_sharpe_smoothed": smooth[best], "best_sharpe_raw": sr[best],
                     "max_sharpe_raw": float(np.nanmax(sr))})
        for lam in lams:
            path[lam][apply] = lam * w_best + (1 - lam) * eq
        oos_mask |= apply

    wp = pd.DataFrame(rows)
    wcols = [f"w_{m}" for m in METRICS]
    jumps = wp[wcols].diff().abs().sum(axis=1)
    wp["l1_change"] = jumps

    # Out-of-sample series: walk-forward paths (weights change each January, with the
    # turnover that implies), plus fixed benchmarks.
    lam_w = np.stack([path[lam] for lam in lams], axis=1)            # (T, L, K)
    p_res = engine.run(lam_w)
    singles = np.eye(K)
    f_res = engine.run(np.vstack([eq, singles]))
    names = [f"tuned_l{lam:g}" for lam in lams]
    fixed_names = ["replica_eq", *[f"only_{m}" for m in METRICS]]
    oos = pd.DataFrame(index=pd.PeriodIndex(hold[oos_mask], freq="M", name="hold_month"))
    for j, nm in enumerate(names):
        for kind in ("gross", "net"):
            oos[f"{nm}_{kind}"] = p_res[kind][oos_mask, j]
        oos[f"{nm}_turnover"] = (p_res["to_long"] + p_res["to_short"])[oos_mask, j]
    for j, nm in enumerate(fixed_names):
        for kind in ("gross", "net"):
            oos[f"{nm}_{kind}"] = f_res[kind][oos_mask, j]
    return WalkForward(wp, oos, pd.concat(trials, ignore_index=True), grid)


def oos_summary(oos: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in [c for c in oos.columns if c.endswith("_net")]:
        name = col[:-4]
        net, gross = oos[col], oos[f"{name}_gross"]
        rows.append({"strategy": name, "months": int(net.notna().sum()),
                     "mean_net_pct_month": 100 * net.mean(), "vol_net_pct_year":
                     100 * net.std() * np.sqrt(12),
                     "sharpe_net": float(sharpe_cols(net.to_numpy()[:, None])[0]),
                     "sharpe_gross": float(sharpe_cols(gross.to_numpy()[:, None])[0])})
    return pd.DataFrame(rows).sort_values("sharpe_net", ascending=False)


def run_tuning(panel_path, results_dir, cfg: dict, log=print) -> str:
    """Load the panel, run the walk-forward, write results/*.csv, return the report."""
    from pathlib import Path

    from halo.config import config_hash

    results = Path(results_dir)
    results.mkdir(parents=True, exist_ok=True)
    holdout = pd.Period(cfg["holdout"]["start"], "M")
    panel = pd.read_parquet(panel_path)
    log(f"Loaded panel: {len(panel):,} rows, {panel['month'].min()} to {panel['month'].max()}")
    engine = Engine.from_panel(panel, cfg)
    del panel
    assert engine.hold_months.max() < holdout
    log(f"Engine: {len(engine.months)} formation months, holding months "
        f"{engine.hold_months.min()} to {engine.hold_months.max()} (holdout {holdout} sealed)")
    h = config_hash(cfg)
    log("Evaluating the weight grid and walking forward:")
    wf = walk_forward(engine, cfg, config_hash=h)

    wf.weights_path.to_csv(results / "weights_path.csv", index=False)
    wf.oos.to_csv(results / "oos_returns.csv")
    summary = oos_summary(wf.oos)
    summary.to_csv(results / "oos_summary.csv", index=False)
    trials_path = results / "trials.csv"
    new = wf.trials.assign(kind="grid")
    if trials_path.exists():
        old = pd.read_csv(trials_path, dtype={"config_hash": str})
        kind = old["kind"] if "kind" in old else pd.Series("grid", index=old.index)
        old = old[~((old["config_hash"] == h) & (kind == "grid"))]   # a rerun replaces itself
        trials = pd.concat([old, new], ignore_index=True)
    else:
        trials = new
    trials.to_csv(trials_path, index=False)

    wcols = [f"w_{m}" for m in METRICS]
    wp = wf.weights_path
    corners = int((wp[wcols].max(axis=1) >= 0.999).sum())
    big_jumps = int((wp["l1_change"] >= 1.0).sum())
    n_configs = trials.loc[trials["kind"] == "grid", "config_hash"].nunique()
    lines = [
        "== Walk-forward tuning ==",
        f"config hash {h}; grid step {cfg['tuning']['grid_step']} ({len(wf.grid)} weight "
        f"vectors); smoothing radius {cfg['tuning']['smoothing_l1_radius']} (L1); objective "
        f"net Sharpe of Q5 minus Q1, equal weight, {cfg['portfolio']['cost_bps']} bps "
        f"{cfg['portfolio']['cost_basis']}",
        f"first training window {cfg['sample']['return_start']} to "
        f"{cfg['tuning']['first_train_end']}; out-of-sample years "
        f"{wp['year'].min()} to {wp['year'].max()}; holdout from {holdout} not touched",
        "",
        "Chosen weights by year (tuned on all months before that year):",
        wp[["year", "train_months", *wcols, "best_sharpe_smoothed", "best_sharpe_raw",
            "l1_change"]].round(3).to_string(index=False),
        "",
        f"Stability: {corners} of {len(wp)} years pick a single-metric corner; {big_jumps} "
        "year-to-year changes of L1 >= 1.0 (half the weight moves). Frequent large jumps mean "
        "the tuning is fitting noise.",
        "",
        "Out-of-sample results (monthly, net and gross of costs). tuned_l1 = tuned weights, "
        "tuned_l0.5 = halfway to equal weight, tuned_l0 = replica_eq = Goldman's equal weight:",
        summary.round(3).to_string(index=False),
        "",
        f"Trials logged: {len(trials):,} rows in results/trials.csv from {n_configs} "
        "configuration(s). Each configuration tried later counts as more trials (spec 5.6).",
        "Files: results/weights_path.csv, oos_returns.csv, oos_summary.csv, trials.csv",
        "These are out-of-sample walk-forward numbers only. Significance tests, the random-"
        "weights comparison and factor regressions come in M6.",
    ]
    return "\n".join(lines)
