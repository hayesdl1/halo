"""M6: evaluation statistics, the end-to-end report, and no false positives on null data."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from halo.config import load_config
from halo.discovery import run_discovery
from halo.eval_steps import run_evaluation, tuned_score
from halo.evaluate import (
    add_controls,
    double_sorts,
    factor_frame,
    factor_table,
    fama_macbeth,
    nw_t,
    perf_stats,
    reality_check,
    regress,
    stationary_bootstrap_index,
)
from halo.fixtures import fake_fred, make_fixture, to_fake_wrds
from halo.panel import run_build
from halo.portfolio import long_short
from halo.pull_steps import step_compustat, step_crsp
from halo.report import build_report
from halo.schema import load_schema_map
from halo.tuning import run_tuning

OVERRIDES = {"tuning": {"walkforward_years": [2008, 2014]}, "holdout": {"start": "2015-01"}}


def pipeline(tmp: Path, planted):
    """Pulls, panel, tuning: exactly what the user's machine runs, on a fixture."""
    f = make_fixture(seed=1, planted=planted)
    common = f.msf["permno"].isin(f.truth["permno"])
    last = f.msf["mthcaldt"].max()
    f.msf.loc[f.msf.loc[(f.msf["mthcaldt"] == last) & common, "mthcap"].idxmax(),
              "mthcap"] = 2.5e9
    db = to_fake_wrds(f, "standard")
    run_discovery(db, tmp / "d.md", tmp / "m.yaml", log=lambda *a: None)
    smap, cfg = load_schema_map(tmp / "m.yaml"), load_config(overrides=OVERRIDES)
    step_crsp(db, smap, cfg, tmp / "raw", "2015-12-31", log=lambda *a: None)
    step_compustat(db, smap, cfg, tmp / "raw", fetch=fake_fred, log=lambda *a: None)
    run_build(tmp / "raw", tmp / "processed", cfg, log=lambda *a: None)
    run_tuning(tmp / "processed" / "panel.parquet", tmp / "results", cfg, log=lambda *a: None)
    return cfg


@pytest.fixture(scope="module")
def planted_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("m6p")
    cfg = pipeline(tmp, {"clr": 0.03})
    run_evaluation(tmp / "raw", tmp / "processed" / "panel.parquet", tmp / "results", cfg,
                   n_boot=50, n_random=100, log=lambda *a: None)
    return tmp, cfg


@pytest.fixture(scope="module")
def null_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("m6n")
    cfg = pipeline(tmp, None)
    run_evaluation(tmp / "raw", tmp / "processed" / "panel.parquet", tmp / "results", cfg,
                   n_boot=50, n_random=100, robustness=False, log=lambda *a: None)
    return tmp, cfg


def test_all_outputs_written(planted_run):
    tmp, _ = planted_run
    res = tmp / "results"
    for name in ["perf_summary", "quintile_returns", "factor_regressions", "spanning",
                 "fama_macbeth", "double_sorts", "subperiods", "random_weights", "comparisons",
                 "reality_check", "composition", "data_checks", "settings", "robustness",
                 "trials"]:
        assert (res / f"{name}.csv").exists(), name
    for chart in ["quintile_growth", "replica_long_short", "tuned_vs_replica", "weights_path",
                  "random_vs_tuned", "annual_long_short"]:
        assert (res / "charts" / f"{chart}.png").exists(), chart
    trials = pd.read_csv(res / "trials.csv")
    assert (trials["kind"] == "robustness").sum() == 16
    assert (trials["kind"] == "grid").sum() == 3003 * 7


def test_report_reproduces_from_csvs(planted_run):
    tmp, _ = planted_run
    res = tmp / "results"
    text = (res / "REPORT.md").read_text()
    assert build_report(res) == text
    assert "\u2014" not in text
    perf = pd.read_csv(res / "perf_summary.csv")
    sr = perf.query("strategy == 'tuned_l1' and returns == 'net'")["sharpe"].iloc[0]
    assert "| tuned_l1 | 84 |" in text and f"{sr:.2f}" in text


def test_no_holdout_months_anywhere(planted_run):
    tmp, cfg = planted_run
    holdout = pd.Period(cfg["holdout"]["start"], "M")
    q = pd.read_csv(tmp / "results" / "quintile_returns.csv")
    assert pd.PeriodIndex(q["hold_month"], freq="M").max() < holdout
    for w in pd.read_csv(tmp / "results" / "perf_summary.csv")["window"].unique():
        assert pd.Period(w.split(" to ")[1], "M") < holdout


def test_planted_signal_is_significant(planted_run):
    tmp, _ = planted_run
    fr = pd.read_csv(tmp / "results" / "factor_regressions.csv")
    row = fr.query("strategy == 'replica_eq (full)' and model == 'FF5+UMD' and returns == 'net'")
    assert row["alpha_t"].iloc[0] > 3


def test_null_fixture_has_no_false_positive(null_run):
    """Spec M6 acceptance: with no planted signal, no headline statistic reaches |t| > 3."""
    tmp, _ = null_run
    res = tmp / "results"
    fr = pd.read_csv(res / "factor_regressions.csv")
    assert (fr["alpha_t"].abs() < 3).all(), fr[["strategy", "model", "alpha_t"]]
    fm = pd.read_csv(res / "fama_macbeth.csv")
    assert (fm.loc[fm["variable"] == "score", "t_nw"].abs() < 3).all()
    ds = pd.read_csv(res / "double_sorts.csv")
    assert (ds.loc[ds["tercile"] == "average", "t_nw"].abs() < 3).all()
    perf = pd.read_csv(res / "perf_summary.csv")
    main = perf[perf["strategy"].isin(["tuned_l1", "tuned_l0.5", "replica_eq"])]
    assert (main["t_nw"].abs() < 3).all()


def test_stale_tuning_results_are_refused(planted_run, tmp_path):
    tmp, cfg = planted_run
    res = tmp_path / "results"
    res.mkdir()
    oos = pd.read_csv(tmp / "results" / "oos_returns.csv")
    oos["replica_eq_net"] += 0.001
    oos.to_csv(res / "oos_returns.csv", index=False)
    (res / "weights_path.csv").write_text((tmp / "results" / "weights_path.csv").read_text())
    with pytest.raises(RuntimeError, match="Rerun scripts/04"):
        run_evaluation(tmp / "raw", tmp / "processed" / "panel.parquet", res, cfg,
                       robustness=False, log=lambda *a: None)


def test_tuned_score_uses_each_years_weights(planted_run):
    tmp, _ = planted_run
    p = pd.read_parquet(tmp / "processed" / "panel.parquet")
    wp = pd.DataFrame({"year": [2009], **{f"w_{m}": [1.0 if m == "fas" else 0.0]
                                          for m in ["tai", "fai", "fas", "clr", "cxi", "cxl"]}})
    s = tuned_score(p, wp)
    rows = ((p["month"] + 1).dt.year == 2009) & p["eligible"] & (p["n_metrics_available"] >= 4)
    assert np.allclose(s[rows], p.loc[rows, "pct_fas"], equal_nan=True)
    assert s[(p["month"] + 1).dt.year != 2009].isna().all()


def test_perf_stats_by_hand():
    r = pd.Series([0.1, -0.5, 0.2, 0.0])
    s = perf_stats(r)
    assert s["max_drawdown"] == pytest.approx(-0.5)
    assert s["geo_ann"] == pytest.approx((1.1 * 0.5 * 1.2) ** 3 - 1)
    assert s["hit_rate"] == pytest.approx(0.5)
    assert s["worst_month"] == pytest.approx(-0.5)


def test_nw_t_close_to_plain_t_for_iid():
    x = pd.Series(np.random.default_rng(0).normal(0.1, 1, 5000))
    plain = x.mean() / (x.std() / np.sqrt(len(x)))
    assert nw_t(x, 6) == pytest.approx(plain, rel=0.1)


def test_regress_recovers_beta():
    rng = np.random.default_rng(1)
    idx = pd.period_range("2000-01", periods=240, freq="M")
    f = pd.DataFrame({"mktrf": rng.normal(0, 0.04, 240)}, index=idx)
    y = pd.Series(0.002 + 0.7 * f["mktrf"] + rng.normal(0, 0.005, 240), index=idx)
    r = regress(y, f, ["mktrf"])
    assert r["b_mktrf"] == pytest.approx(0.7, abs=0.03)
    assert r["alpha_ann"] == pytest.approx(0.024, abs=0.01)


def test_reality_check_noise_vs_signal():
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 0.03, (240, 200))
    assert reality_check(noise, n_boot=200, seed=0)["p_one_sided"] > 0.05
    strong = noise.copy()
    strong[:, 0] += 0.02
    assert reality_check(strong, n_boot=200, seed=0)["p_one_sided"] < 0.01


def test_stationary_bootstrap_index():
    idx = stationary_bootstrap_index(100, 6, np.random.default_rng(0))
    assert idx.shape == (100,) and idx.min() >= 0 and idx.max() < 100
    runs = (np.diff(idx) == 1).mean()
    assert 0.7 < runs < 0.95            # mostly consecutive, about 1 - 1/6


def test_book_equity_and_leverage_by_hand():
    p = pd.DataFrame({"seq": [100.0, np.nan], "ceq": [90.0, 50.0], "pstk": [5.0, 10.0],
                      "pstkrv": [np.nan, np.nan], "pstkl": [8.0, np.nan], "txditc": [4.0, np.nan],
                      "mthcap": [200_000.0, 100_000.0], "revt": [80.0, 40], "cogs": [40.0, 20],
                      "xsga": [10.0, np.nan], "xint": [2.0, 1], "at": [300.0, 100],
                      "at_lag": [250.0, np.nan], "dltt": [60.0, np.nan], "dlc": [15.0, np.nan],
                      "mom_12_1": [0.1, 0.2]})
    c = add_controls(p)
    # BE = 100 + 4 - 8 = 96; ME = 200 ($m)
    assert c["book_to_market"].iloc[0] == pytest.approx(96 / 200)
    # BE = (50 + 10) + 0 - 10 = 50; ME = 100
    assert c["book_to_market"].iloc[1] == pytest.approx(0.5)
    assert c["profitability"].iloc[0] == pytest.approx((80 - 40 - 10 - 2) / 96)
    assert c["asset_growth"].iloc[0] == pytest.approx(0.2)
    assert c["leverage"].iloc[0] == pytest.approx(0.25)
    assert np.isnan(c["leverage"].iloc[1])


def test_factor_table_and_sorts_on_panel(planted_run):
    tmp, cfg = planted_run
    p = pd.read_parquet(tmp / "processed" / "panel.parquet")
    ls = long_short(p, cfg).set_index("hold_month")
    ff = pd.read_parquet(tmp / "raw" / "ff_factors.parquet")
    rates = pd.read_parquet(tmp / "raw" / "fred_rates.parquet")
    t = factor_table({"x": ls["ls_net"]}, factor_frame(ff, rates))
    assert set(t["model"]) == {"CAPM", "FF3", "Carhart", "FF5+UMD", "FF5+UMD+rates"}
    assert t["b_d_term"].notna().any()
    fm = fama_macbeth(p, "1996-02", "2014-12")
    assert set(fm["spec"]) == {"score only", "score + controls"}
    ds = double_sorts(p, "leverage", "1996-02", "2014-12")
    assert list(ds["tercile"].astype(str)) == ["1", "2", "3", "average"]
