"""M7: the single, final evaluation on the sealed holdout (spec 5.4 step 5).

Order matters and is enforced here:

1. Refuse to run if ``results/holdout_evaluated.lock`` exists, unless forced; a forced
   rerun is recorded and printed in the report.
2. Freeze the weights by tuning once on every month before ``holdout.start`` (same
   grid, smoothing and objective as the walk-forward). The engine used for this cannot
   see holdout months, and the weights are written to disk before any holdout return
   is computed.
3. Evaluate the frozen weights, the halfway blend, the equal-weight replica, each metric
   alone, and the Value and Investment benchmarks on the holdout months, plus
   Goldman's regime window (from ``holdout.goldman_regime_start``).
4. Write the lock file, the holdout CSVs and charts, and rebuild REPORT.md.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from halo import METRICS, charts
from halo.config import config_hash
from halo.eval_steps import PANEL_COLS
from halo.evaluate import (
    characteristic_long_short,
    factor_frame,
    factor_table,
    perf_stats,
    window,
)
from halo.portfolio import long_short
from halo.tuning import Engine, neighbours, sharpe_cols, simplex_grid, tune_window

LOCK = "holdout_evaluated.lock"


class HoldoutLocked(RuntimeError):
    pass


def freeze_weights(panel: pd.DataFrame, cfg: dict) -> dict:
    """Tune once on all pre-holdout months. Returns the frozen weights and diagnostics."""
    t = cfg["tuning"]
    holdout = pd.Period(cfg["holdout"]["start"], "M")
    eng = Engine.from_panel(panel, cfg, holdout_guard=True)
    assert eng.hold_months.max() < holdout
    grid = simplex_grid(t["grid_step"])
    net = eng.run(grid)["net"]
    train = eng.hold_months >= pd.Period(cfg["sample"]["return_start"], "M")
    sr = sharpe_cols(net[np.asarray(train)])
    best, smooth = tune_window(sr, neighbours(grid, t["smoothing_l1_radius"]))
    return {"weights": grid[best], "train_start": cfg["sample"]["return_start"],
            "train_end": str(eng.hold_months.max()), "train_months": int(train.sum()),
            "sharpe_smoothed": float(smooth[best]), "sharpe_raw": float(sr[best])}


def run_holdout(raw_dir, panel_path, results_dir, cfg: dict, force: bool = False,
                log=print) -> Path:
    from halo.factors import load_factors, load_rates
    from halo.report import build_report

    res = Path(results_dir)
    lock = res / LOCK
    if lock.exists() and not force:
        raise HoldoutLocked(
            f"The holdout has already been evaluated ({lock}). It is meant to be run once. "
            "Rerunning with --force is possible but is recorded in the report.")
    res.mkdir(parents=True, exist_ok=True)
    (res / "charts").mkdir(exist_ok=True)
    holdout = pd.Period(cfg["holdout"]["start"], "M")
    regime = pd.Period(cfg["holdout"]["goldman_regime_start"], "M")
    lags = cfg["evaluation"]["newey_west_lags"]
    lams = cfg["tuning"]["shrinkage_lambdas"]
    eq = np.full(len(METRICS), 1 / len(METRICS))

    log("Loading panel:")
    panel = pd.read_parquet(panel_path, columns=PANEL_COLS)

    log("Freezing weights on pre-holdout data only:")
    pre = panel[panel["month"] + 1 < holdout]
    fz = freeze_weights(pre, cfg)
    del pre
    w_rows = [{"lambda": lam, **{f"w_{m}": lam * fz["weights"][j] + (1 - lam) * eq[j]
                                  for j, m in enumerate(METRICS)}} for lam in lams]
    wdf = pd.DataFrame(w_rows).assign(train_start=fz["train_start"], train_end=fz["train_end"],
                                      train_months=fz["train_months"],
                                      in_sample_sharpe_smoothed=fz["sharpe_smoothed"],
                                      in_sample_sharpe_raw=fz["sharpe_raw"])
    wdf.to_csv(res / "holdout_weights.csv", index=False)
    log("  " + " ".join(f"{m}={w:.1f}" for m, w in zip(METRICS, fz["weights"], strict=True)))

    log("Evaluating the holdout (once):")
    eng = Engine.from_panel(panel, cfg, holdout_guard=False)
    W = np.vstack([np.array([r[f"w_{m}"] for m in METRICS]) for r in w_rows] + [eq] +
                  list(np.eye(len(METRICS))))
    names = [f"tuned_l{lam:g}" for lam in lams] + ["replica_eq_engine"] + \
        [f"only_{m}" for m in METRICS]
    out = eng.run(W)
    hold = eng.hold_months
    series = {}
    for j, nm in enumerate(names):
        for kind in ("net", "gross", "top", "bottom"):
            series[(nm, kind)] = pd.Series(out[kind][:, j], index=hold)
        series[(nm, "turnover")] = pd.Series(out["to_long"][:, j] + out["to_short"][:, j],
                                             index=hold)
    rep = long_short(panel, cfg).set_index("hold_month")
    value = characteristic_long_short(panel, cfg, "book_to_market")
    invest = characteristic_long_short(panel, cfg, "asset_growth", sign=-1)
    last = rep["ls_net"].dropna().index.max()
    assert last >= holdout, "the panel has no holdout months with returns yet"

    strategies = {
        **{nm: (series[(nm, "net")], series[(nm, "gross")], series[(nm, "turnover")])
           for nm in names if nm != "replica_eq_engine"},
        "replica_eq": (rep["ls_net"], rep["ls_gross"], rep["to_long"] + rep["to_short"]),
        "value_hml": (value["ls_net"], value["ls_gross"], value["to_long"] + value["to_short"]),
        "investment_low_minus_high": (invest["ls_net"], invest["ls_gross"],
                                      invest["to_long"] + invest["to_short"]),
        "replica_top_minus_universe": (rep["top_minus_univ"], rep["top_minus_univ"], None),
    }
    windows = [(f"holdout {holdout} to {last}", holdout, last),
               (f"Goldman regime {regime} to {last}", regime, last)]
    rows = []
    for wname, a, b in windows:
        for nm, (net, gross, to) in strategies.items():
            for kind, s in (("net", net), ("gross", gross)):
                rows.append({"window": wname, "strategy": nm, "returns": kind,
                             **perf_stats(window(s, a, b), to, lags)})
    pd.DataFrame(rows).to_csv(res / "holdout_perf.csv", index=False)

    ff = load_factors(raw_dir)
    try:
        rates = load_rates(raw_dir)
    except FileNotFoundError:
        rates = None
    F = factor_frame(ff, rates)
    factor_table({"tuned_l1": window(series[("tuned_l1", "net")], holdout, last),
                  "replica_eq": window(rep["ls_net"], holdout, last)}, F, lags).to_csv(
        res / "holdout_factor_regressions.csv", index=False)

    # Goldman claim check: cumulative top-basket vs bottom-basket relative performance.
    goldman = []
    for nm, top, bottom, univ in [
        ("replica_eq", rep[f"q{5 if cfg['portfolio']['basket'] == 'quintile' else 3}"],
         rep["q1"], rep["univ"]),
        ("tuned_l1", series[("tuned_l1", "top")], series[("tuned_l1", "bottom")], rep["univ"]),
    ]:
        t, b, u = (window(x, regime, last).dropna() for x in (top, bottom, univ))
        common = t.index.intersection(b.index)
        goldman.append({"strategy": nm, "start": str(regime), "end": str(last),
                        "months": len(common),
                        "top_cumulative": float((1 + t[common]).prod() - 1),
                        "bottom_cumulative": float((1 + b[common]).prod() - 1),
                        "top_vs_bottom_relative": float((1 + t[common]).prod()
                                                        / (1 + b[common]).prod() - 1),
                        "top_vs_universe_relative": float((1 + t[common]).prod()
                                                          / (1 + u.reindex(common)).prod() - 1),
                        "goldman_reported_europe": cfg["holdout"]["goldman_reported_relative"]})
    pd.DataFrame(goldman).to_csv(res / "holdout_goldman.csv", index=False)

    log("Charts:")
    q = window(rep[[f"q{i}" for i in range(1, 6)]], holdout, last)
    charts.quintile_growth(q, res / "charts" / "holdout_quintiles.png",
                           f"Holdout {holdout} to {last}: equal-weight score quintiles, gross")
    charts.two_line_growth(window(rep["ls_net"], holdout, last),
                           window(series[("tuned_l1", "net")], holdout, last),
                           ("equal weight", "frozen tuned"),
                           res / "charts" / "holdout_long_short.png",
                           f"Holdout {holdout} to {last}: Q5 minus Q1, net")
    # Goldman's axes: value of the top basket relative to the bottom basket since the
    # regime start (monthly relative return (1 + top) / (1 + bottom) - 1, compounded).
    rel = {nm: window((1 + top) / (1 + bottom) - 1, regime, last) for nm, top, bottom in [
        ("equal weight", rep["q5"], rep["q1"]),
        ("frozen tuned", series[("tuned_l1", "top")], series[("tuned_l1", "bottom")])]}
    charts.two_line_growth(rel["equal weight"], rel["frozen tuned"],
                           ("equal weight", "frozen tuned"),
                           res / "charts" / "holdout_goldman_window.png",
                           f"Capital intensive vs capital light since {regime}, gross "
                           "(top quintile value relative to bottom)")

    meta = {"evaluated_at_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M"),
            "forced_rerun": bool(lock.exists()), "config_hash": config_hash(cfg),
            "holdout_start": str(holdout), "last_month": str(last)}
    pd.DataFrame([meta]).to_csv(res / "holdout_meta.csv", index=False)
    with open(lock, "a") as f:
        f.write(f"{meta['evaluated_at_utc']} config {meta['config_hash']} "
                f"forced={meta['forced_rerun']}\n")
    (res / "REPORT.md").write_text(build_report(res))
    return res / "REPORT.md"
