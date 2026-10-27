"""M7: the holdout runs once, freezes weights without seeing holdout data, and reports."""

import numpy as np
import pandas as pd
import pytest

from halo import METRICS
from halo.eval_steps import PANEL_COLS, run_evaluation
from halo.fixtures import run_pipeline
from halo.holdout import LOCK, HoldoutLocked, freeze_weights, run_holdout

OVERRIDES = {"tuning": {"walkforward_years": [2008, 2012]},
             "holdout": {"start": "2013-01", "goldman_regime_start": "2015-01"}}


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("m7")
    cfg = run_pipeline(tmp, {"clr": 0.03}, OVERRIDES)
    run_evaluation(tmp / "raw", tmp / "processed" / "panel.parquet", tmp / "results", cfg,
                   n_boot=20, n_random=50, robustness=False, log=lambda *a: None)
    report = run_holdout(tmp / "raw", tmp / "processed" / "panel.parquet", tmp / "results",
                         cfg, log=lambda *a: None)
    return tmp, cfg, report.read_text()


def test_outputs_and_lock(run):
    tmp, _, text = run
    res = tmp / "results"
    for name in ["holdout_weights", "holdout_perf", "holdout_factor_regressions",
                 "holdout_goldman", "holdout_meta"]:
        assert (res / f"{name}.csv").exists(), name
    for chart in ["holdout_quintiles", "holdout_long_short", "holdout_goldman_window"]:
        assert (res / "charts" / f"{chart}.png").exists(), chart
    assert (res / LOCK).exists()
    assert "## Holdout result: 2013-01 to 2015-12 (M7)" in text
    assert text.index("## Holdout result") < text.index("## Data checks")
    assert "WARNING" not in text


def test_second_run_refused_and_force_recorded(run, tmp_path):
    tmp, cfg, _ = run
    with pytest.raises(HoldoutLocked):
        run_holdout(tmp / "raw", tmp / "processed" / "panel.parquet", tmp / "results", cfg,
                    log=lambda *a: None)
    # Force on a copy so the module fixture stays clean.
    res = tmp_path / "results"
    res.mkdir()
    for f in (tmp / "results").glob("*.csv"):
        (res / f.name).write_bytes(f.read_bytes())
    (res / LOCK).write_text((tmp / "results" / LOCK).read_text())
    text = run_holdout(tmp / "raw", tmp / "processed" / "panel.parquet", res, cfg, force=True,
                       log=lambda *a: None).read_text()
    assert "WARNING: the holdout was evaluated more than once" in text
    assert len((res / LOCK).read_text().strip().splitlines()) == 2


def test_frozen_weights_ignore_holdout_returns(run):
    tmp, cfg, _ = run
    p = pd.read_parquet(tmp / "processed" / "panel.parquet", columns=PANEL_COLS)
    holdout = pd.Period(cfg["holdout"]["start"], "M")
    a = freeze_weights(p[p["month"] + 1 < holdout], cfg)
    # Scramble every holdout return and pass the full panel: the guard must ignore them.
    q = p.copy()
    late = q["month"] + 1 >= holdout
    q.loc[late, "ret_next"] = np.random.default_rng(0).normal(0, 0.5, int(late.sum()))
    b = freeze_weights(q, cfg)
    assert np.array_equal(a["weights"], b["weights"])
    assert a["train_end"] == "2012-12"
    w = pd.read_csv(tmp / "results" / "holdout_weights.csv")
    got = w.loc[w["lambda"] == 1.0, [f"w_{m}" for m in METRICS]].to_numpy()[0]
    assert np.allclose(got, a["weights"])
    assert np.allclose(w.loc[w["lambda"] == 0.0, [f"w_{m}" for m in METRICS]], 1 / 6)


def test_windows_and_goldman_check(run):
    tmp, _, _ = run
    res = tmp / "results"
    hp = pd.read_csv(res / "holdout_perf.csv")
    main = hp[hp["window"].str.startswith("holdout")]
    assert (main.loc[main["strategy"] == "replica_eq", "months"] == 36).all()
    regime = hp[hp["window"].str.startswith("Goldman")]
    assert (regime.loc[regime["strategy"] == "replica_eq", "months"] == 12).all()
    # tuned_l0 is the equal-weight replica, computed by the engine.
    a = main.query("strategy == 'tuned_l0' and returns == 'net'")["sharpe"].iloc[0]
    b = main.query("strategy == 'replica_eq' and returns == 'net'")["sharpe"].iloc[0]
    assert a == pytest.approx(b, abs=1e-9)
    g = pd.read_csv(res / "holdout_goldman.csv").set_index("strategy")
    q = pd.read_csv(res / "quintile_returns.csv")      # pre-holdout only, so recompute:
    assert pd.PeriodIndex(q["hold_month"], freq="M").max() < pd.Period("2013-01", "M")
    r = g.loc["replica_eq"]
    assert r["months"] == 12
    assert (1 + r["top_vs_bottom_relative"]) == pytest.approx(
        (1 + r["top_cumulative"]) / (1 + r["bottom_cumulative"]))
    assert r["goldman_reported_europe"] == pytest.approx(0.35)
