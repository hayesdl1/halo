"""M6 orchestration: run every evaluation on pre-holdout data and write results/*.csv.

Inputs: data/processed/panel.parquet (M3), results/weights_path.csv and
results/oos_returns.csv (M5), data/raw/ for factors, rates and the robustness rebuilds.
Outputs: one CSV per analysis, charts in results/charts/, and results/REPORT.md built
from the CSVs alone.

The holdout is never touched: every series stops at the last holding month before
``holdout.start``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from halo import METRICS, charts
from halo.config import config_hash, deep_update
from halo.evaluate import (
    characteristic_long_short,
    composition,
    double_sorts,
    factor_frame,
    factor_table,
    fama_macbeth,
    perf_stats,
    random_weight_sharpes,
    reality_check,
    regress,
    spanning_table,
    subperiods,
    window,
)
from halo.portfolio import long_short
from halo.score import weighted_score
from halo.tuning import Engine, simplex_grid

PANEL_COLS = ["month", "permno", "eligible", "in_top_n", "has_link", "has_next_row",
              "score", "score_std", "score_sn", "scored", "ret_next", "mthcap", "sic", "ff12",
              *[f"pct_{m}" for m in METRICS], *[f"pct_sn_{m}" for m in METRICS],
              "seq", "ceq", "pstk", "pstkrv", "pstkl", "txditc", "revt", "cogs", "xsga", "xint",
              "dltt", "dlc", "at", "at_lag", "mom_12_1"]


def load_panel(path, holdout: pd.Period) -> pd.DataFrame:
    p = pd.read_parquet(path, columns=PANEL_COLS)
    return p[p["month"] + 1 < holdout].reset_index(drop=True)


def tuned_score(panel: pd.DataFrame, wp: pd.DataFrame, prefix: str = "pct_") -> pd.Series:
    """Score from the walk-forward weights: each formation month uses the weights tuned
    for its holding year. NaN outside the out-of-sample years."""
    pct = panel[[f"{prefix}{m}" for m in METRICS]].set_axis(list(METRICS), axis=1)
    hold_year = (panel["month"] + 1).dt.year
    out = pd.Series(np.nan, index=panel.index)
    for _, row in wp.iterrows():
        rows = (hold_year == row["year"]) & panel["eligible"]
        if rows.any():
            w = row[[f"w_{m}" for m in METRICS]].to_numpy(dtype=float)
            out[rows] = weighted_score(pct[rows], w, 4)[0]
    return out


def run_evaluation(raw_dir, panel_path, results_dir, cfg: dict, n_boot: int = 500,
                   n_random: int | None = None, robustness: bool = True, log=print):
    from halo.factors import load_factors, load_rates

    res = Path(results_dir)
    (res / "charts").mkdir(parents=True, exist_ok=True)
    holdout = pd.Period(cfg["holdout"]["start"], "M")
    start = cfg["sample"]["return_start"]
    last = str(holdout - 1)
    lags = cfg["evaluation"]["newey_west_lags"]
    wp = pd.read_csv(res / "weights_path.csv")
    oos = pd.read_csv(res / "oos_returns.csv")
    oos["hold_month"] = pd.PeriodIndex(oos["hold_month"], freq="M")
    oos = oos.set_index("hold_month")
    assert oos.index.max() < holdout
    oos_start, oos_end = str(oos.index.min()), str(oos.index.max())

    log("Loading panel and factors:")
    panel = load_panel(panel_path, holdout)
    ff = load_factors(raw_dir)
    try:
        rates = load_rates(raw_dir)
    except FileNotFoundError:
        rates = None
    F = factor_frame(ff, rates)

    log("Equal-weight replica and consistency check against the tuning run:")
    rep = long_short(panel, cfg).set_index("hold_month")
    diff = (rep["ls_net"].reindex(oos.index) - oos["replica_eq_net"]).abs().max()
    if not diff < 1e-9:
        raise RuntimeError("results/oos_returns.csv does not match the current panel "
                           f"(max difference {diff}). Rerun scripts/04_tune_walkforward.py.")

    log("Benchmarks: value and investment long-shorts in the same universe:")
    value = characteristic_long_short(panel, cfg, "book_to_market")
    invest = characteristic_long_short(panel, cfg, "asset_growth", sign=-1)

    log("Engine for single metrics, random weights and the reality check:")
    eng = Engine.from_panel(panel, cfg)
    hold = eng.hold_months
    singles = eng.run(np.eye(len(METRICS)))
    single_net = {f"only_{m}": pd.Series(singles["net"][:, j], index=hold)
                  for j, m in enumerate(METRICS)}

    # ---- performance tables ---------------------------------------------------------
    series_oos = {
        "tuned_l1": (oos["tuned_l1_net"], oos["tuned_l1_gross"], oos["tuned_l1_turnover"]),
        "tuned_l0.5": (oos["tuned_l0.5_net"], oos["tuned_l0.5_gross"],
                       oos["tuned_l0.5_turnover"]),
        "replica_eq": (rep["ls_net"], rep["ls_gross"], rep["to_long"] + rep["to_short"]),
        **{f"only_{m}": (oos[f"only_{m}_net"], oos[f"only_{m}_gross"], None) for m in METRICS},
        "value_hml": (value["ls_net"], value["ls_gross"], value["to_long"] + value["to_short"]),
        "investment_low_minus_high": (invest["ls_net"], invest["ls_gross"],
                                      invest["to_long"] + invest["to_short"]),
        "replica_top_minus_universe": (rep["top_minus_univ"], rep["top_minus_univ"], None),
    }
    rows = []
    for name, (net, gross, to) in series_oos.items():
        for kind, s in (("net", net), ("gross", gross)):
            rows.append({"window": f"oos {oos_start} to {oos_end}", "strategy": name,
                         "returns": kind, **perf_stats(window(s, oos_start, oos_end), to, lags)})
    full = {"replica_eq": (rep["ls_net"], rep["ls_gross"], rep["to_long"] + rep["to_short"]),
            **{k: (v, None, None) for k, v in single_net.items()},
            "value_hml": series_oos["value_hml"],
            "investment_low_minus_high": series_oos["investment_low_minus_high"]}
    for name, (net, gross, to) in full.items():
        for kind, s in (("net", net), ("gross", gross)):
            if s is not None:
                rows.append({"window": f"full {start} to {last}", "strategy": name,
                             "returns": kind, **perf_stats(window(s, start, last), to, lags)})
    perf = pd.DataFrame(rows)
    perf.to_csv(res / "perf_summary.csv", index=False)

    q = rep[[f"q{i}" for i in range(1, 6)]]
    q.to_csv(res / "quintile_returns.csv")

    # ---- factor regressions and spanning --------------------------------------------
    log("Factor regressions, spanning, Fama-MacBeth, double sorts:")
    reg_series = {"tuned_l1": oos["tuned_l1_net"], "replica_eq (oos)":
                  window(rep["ls_net"], oos_start, oos_end),
                  "replica_eq (full)": window(rep["ls_net"], start, last),
                  "value_hml (oos)": window(value["ls_net"], oos_start, oos_end),
                  "investment (oos)": window(invest["ls_net"], oos_start, oos_end)}
    fr = factor_table(reg_series, F, lags)
    fr.insert(1, "returns", "net")
    gross_series = {"tuned_l1": oos["tuned_l1_gross"],
                    "replica_eq (oos)": window(rep["ls_gross"], oos_start, oos_end),
                    "replica_eq (full)": window(rep["ls_gross"], start, last)}
    fg = factor_table(gross_series, F, lags)
    fg.insert(1, "returns", "gross")
    pd.concat([fr, fg], ignore_index=True).to_csv(res / "factor_regressions.csv", index=False)
    spanning_table(reg_series, F, lags).to_csv(res / "spanning.csv", index=False)

    fm = pd.concat([fama_macbeth(panel, start, last, lags=lags).assign(window="full"),
                    fama_macbeth(panel, oos_start, oos_end, lags=lags).assign(window="oos")])
    fm.to_csv(res / "fama_macbeth.csv", index=False)

    ds = []
    for on in ["book_to_market", "asset_growth", "leverage"]:
        for lab, a, b in (("full", start, last), ("oos", oos_start, oos_end)):
            ds.append(double_sorts(panel, on, a, b, lags=lags).assign(window=lab))
    pd.concat(ds).to_csv(res / "double_sorts.csv", index=False)

    subperiods({"replica_eq": window(rep["ls_net"], start, last), "tuned_l1":
                oos["tuned_l1_net"]}, F, lags).to_csv(res / "subperiods.csv", index=False)

    # ---- tuning diagnostics -----------------------------------------------------------
    log("Random weights and the reality check:")
    oos_mask = (hold >= pd.Period(oos_start, "M")) & (hold <= pd.Period(oos_end, "M"))
    rw = random_weight_sharpes(eng, n_random or cfg["tuning"]["n_random_weights"],
                               cfg["tuning"]["random_seed"], np.asarray(oos_mask))
    rw.to_csv(res / "random_weights.csv", index=False)
    tuned_sr = perf.query("strategy == 'tuned_l1' and returns == 'net'")["sharpe"].iloc[0]
    rep_sr = perf.query("strategy == 'replica_eq' and returns == 'net' and "
                        "window.str.startswith('oos')", engine="python")["sharpe"].iloc[0]
    comp = pd.DataFrame([
        {"comparison": "tuned_l1 percentile among random weightings",
         "value": float((rw["sharpe_net"] < tuned_sr).mean())},
        {"comparison": "replica_eq percentile among random weightings",
         "value": float((rw["sharpe_net"] < rep_sr).mean())},
        {"comparison": "random weightings median sharpe",
         "value": float(rw["sharpe_net"].median())},
        {"comparison": "random weightings 5th percentile sharpe",
         "value": float(rw["sharpe_net"].quantile(0.05))},
        {"comparison": "random weightings 95th percentile sharpe",
         "value": float(rw["sharpe_net"].quantile(0.95))},
        {"comparison": "tuned_l1 minus replica_eq sharpe", "value": tuned_sr - rep_sr},
    ])
    comp.to_csv(res / "comparisons.csv", index=False)

    full_mask = (hold >= pd.Period(start, "M")) & (hold <= pd.Period(last, "M"))
    grid_net = eng.run(simplex_grid(cfg["tuning"]["grid_step"]))["net"][np.asarray(full_mask)]
    rc = reality_check(grid_net, n_boot=n_boot, seed=cfg["tuning"]["random_seed"])
    rc["best_weights"] = " ".join(f"{m}={w:.1f}" for m, w in
                                  zip(METRICS, simplex_grid(cfg["tuning"]["grid_step"])[
                                      rc["best_index"]], strict=True))
    pd.DataFrame([rc]).to_csv(res / "reality_check.csv", index=False)
    del grid_net

    composition(panel, cfg, f"{pd.Period(start, 'M').year}-01", last).to_csv(
        res / "composition.csv", index=False)
    data_checks(panel, holdout).to_csv(res / "data_checks.csv", index=False)

    # ---- charts -----------------------------------------------------------------------
    log("Charts:")
    qq = window(q, start, last)
    charts.quintile_growth(qq, res / "charts" / "quintile_growth.png",
                           f"Equal-weight score quintiles, gross, {start} to {last}")
    charts.two_line_growth(window(rep["ls_gross"], start, last),
                           window(rep["ls_net"], start, last), ("gross", "net"),
                           res / "charts" / "replica_long_short.png",
                           "Equal-weight replica: Q5 minus Q1, gross and net of costs")
    charts.two_line_growth(window(rep["ls_net"], oos_start, oos_end), oos["tuned_l1_net"],
                           ("equal weight", "tuned"), res / "charts" / "tuned_vs_replica.png",
                           f"Out of sample {oos_start} to {oos_end}: Q5 minus Q1, net")
    charts.weights_path(wp, METRICS, res / "charts" / "weights_path.png")
    charts.random_vs_tuned(rw["sharpe_net"], tuned_sr, rep_sr,
                           res / "charts" / "random_vs_tuned.png")
    charts.annual_bars(window(rep["ls_net"], start, last), oos["tuned_l1_net"].reindex(
        window(rep["ls_net"], start, last).index), ("equal weight", "tuned (2008 on)"),
        res / "charts" / "annual_long_short.png", "Q5 minus Q1 by calendar year, net")

    settings(cfg, oos_start, oos_end).to_csv(res / "settings.csv", index=False)

    if robustness:
        log("Robustness grid (rebuilds the panel for some variants):")
        rob = robustness_grid(raw_dir, panel, cfg, wp, oos_start, oos_end, F, lags, log)
        rob.to_csv(res / "robustness.csv", index=False)
        log_config_trials(res, cfg, rob)
    from halo.report import build_report
    (res / "REPORT.md").write_text(build_report(res))
    return res / "REPORT.md"


def data_checks(panel: pd.DataFrame, holdout: pd.Period) -> pd.DataFrame:
    p = panel[panel["month"].dt.month == 12]
    rows = []
    for m, g in p.groupby("month"):
        e = g[g["eligible"]]
        top = g[g["in_top_n"]]
        held = e[e["scored"]]
        rows.append({"year": m.year, "eligible": len(e), "scored_share": e["scored"].mean(),
                     "link_coverage_top_n": top["has_link"].mean(),
                     "missing_next_return_share": held["ret_next"].isna().mean()})
    return pd.DataFrame(rows)


def settings(cfg: dict, oos_start: str, oos_end: str) -> pd.DataFrame:
    u, f, p, t = cfg["universe"], cfg["fundamentals"], cfg["portfolio"], cfg["tuning"]
    items = [
        ("universe", f"top {u['n_stocks']} by market cap, price >= ${u['min_price']:g}, "
         f"{'non-financial' if u['exclude_financials'] else 'including financials'}"),
        ("fundamentals lag", f"{f['lag_months']} months after datadate, expire after "
         f"{f['expiry_months']}"),
        ("baskets", f"{p['basket']}s, {p['weighting']} weight, long top minus short bottom"),
        ("costs", f"{p['cost_bps']} bps per dollar traded, {p['cost_basis']}"),
        ("tuning objective", "net Sharpe of equal-weight Q5 minus Q1"),
        ("grid", f"step {t['grid_step']}, smoothing radius {t['smoothing_l1_radius']} (L1)"),
        ("walk-forward", f"first window {cfg['sample']['return_start']} to "
         f"{t['first_train_end']}; out of sample {oos_start} to {oos_end}"),
        ("holdout", f"from {cfg['holdout']['start']}, sealed"),
        ("config hash", config_hash(cfg)),
    ]
    return pd.DataFrame(items, columns=["setting", "value"])


ROBUSTNESS = [
    ("base", {}, False, {}),
    ("universe top 500", {"universe": {"n_stocks": 500}}, True, {}),
    ("universe top 1500", {"universe": {"n_stocks": 1500}}, True, {}),
    ("include financials", {"universe": {"exclude_financials": False}}, True, {}),
    ("terciles", {}, False, {"scheme": "tercile"}),
    ("top half vs bottom half", {}, False, {"scheme": "half"}),
    ("value weighted", {}, False, {"weighting": "value"}),
    ("lag 3 months", {"fundamentals": {"lag_months": 3}}, True, {}),
    ("lag 9 months", {"fundamentals": {"lag_months": 9}}, True, {}),
    ("ppe = ppegt", {"fundamentals": {"ppe": "ppegt"}}, True, {}),
    ("EBITDA <= 0 ranked top", {"fundamentals": {"nonpositive_ebitda": "max_rank"}}, True, {}),
    ("sector neutral", {"score": {"sector_neutral": True}}, False, {"sector_neutral": True}),
    ("cost 0 bps", {"portfolio": {"cost_bps": 0}}, False, {"cost_bps": 0}),
    ("cost 5 bps", {"portfolio": {"cost_bps": 5}}, False, {"cost_bps": 5}),
    ("cost 25 bps", {"portfolio": {"cost_bps": 25}}, False, {"cost_bps": 25}),
    ("cost basis: buys only (spec formula)", {"portfolio": {"cost_basis": "spec"}}, False,
     {"cost_basis": "spec"}),
]


def robustness_grid(raw_dir, base: pd.DataFrame, cfg: dict, wp: pd.DataFrame, oos_start,
                    oos_end, F, lags, log=print) -> pd.DataFrame:
    """Replica and walk-forward weights under each variant. Weights are those tuned in
    the main run, applied unchanged, so no variant is re-tuned."""
    from halo.compustat import load_funda
    from halo.crsp import load_msf
    from halo.link import load_ccm
    from halo.panel import build_panel

    start = cfg["sample"]["return_start"]
    last = str(pd.Period(cfg["holdout"]["start"], "M") - 1)
    holdout = pd.Period(cfg["holdout"]["start"], "M")
    rows = []

    def evaluate(name, cfg_v, panel, kw):
        sn = kw.get("sector_neutral", False)
        ls_kw = {k: v for k, v in kw.items() if k != "sector_neutral"}
        p = panel.assign(_tuned=tuned_score(panel, wp, "pct_sn_" if sn else "pct_"))
        rep = long_short(p, cfg_v, score_col="score_sn" if sn else "score",
                         **ls_kw).set_index("hold_month")
        tun = long_short(p, cfg_v, score_col="_tuned", **ls_kw).set_index("hold_month")
        for strat, s, a, b in [("replica_eq", rep["ls_net"], oos_start, oos_end),
                               ("replica_eq", rep["ls_net"], start, last),
                               ("tuned_l1", tun["ls_net"], oos_start, oos_end)]:
            w = window(s, a, b)
            st = perf_stats(w, lags=lags)
            al = regress(w, F, ["mktrf", "smb", "hml", "rmw", "cma", "umd"], lags)
            rows.append({"variant": name, "strategy": strat, "window": f"{a} to {b}",
                         "months": st["months"], "mean_ann": st.get("mean_ann"),
                         "sharpe": st.get("sharpe"), "t_nw": st.get("t_nw"),
                         "alpha_ff5umd_ann": al.get("alpha_ann"),
                         "alpha_ff5umd_t": al.get("alpha_t"),
                         "config_hash": config_hash(cfg_v)})

    for name, over, rebuild, kw in ROBUSTNESS:
        if not rebuild:
            log(f"  {name}")
            evaluate(name, deep_update(cfg, over), base, kw)
    msf, ccm, funda = load_msf(raw_dir), load_ccm(raw_dir), load_funda(raw_dir)
    for name, over, rebuild, kw in ROBUSTNESS:
        if rebuild:
            log(f"  {name} (rebuilding panel)")
            cfg_v = deep_update(cfg, over)
            p = build_panel(msf, ccm, funda, cfg_v, log=lambda *a: None)
            p = p[p["month"] + 1 < holdout][PANEL_COLS].reset_index(drop=True)
            evaluate(name, cfg_v, p, kw)
            del p
    return pd.DataFrame(rows)


def log_config_trials(res: Path, cfg: dict, rob: pd.DataFrame) -> None:
    """Spec 5.6: configuration choices count as trials. One row per robustness variant."""
    path = res / "trials.csv"
    trials = pd.read_csv(path, dtype={"config_hash": str}) if path.exists() else pd.DataFrame()
    if len(trials) and "kind" not in trials:
        trials["kind"] = "grid"
    if len(trials):
        trials = trials[trials["kind"] != "robustness"]
    r = rob[rob["strategy"] == "replica_eq"].drop_duplicates("variant")
    new = pd.DataFrame({"kind": "robustness", "config_hash": r["config_hash"],
                        "variant": r["variant"], "sharpe_net": r["sharpe"]})
    pd.concat([trials, new], ignore_index=True).to_csv(path, index=False)
