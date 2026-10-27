"""M5: the tuning engine, the walk-forward, the holdout guard, and planted-signal recovery."""

import numpy as np
import pandas as pd
import pytest

from halo import METRICS
from halo.config import load_config
from halo.fixtures import make_fixture, panel_inputs
from halo.panel import build_panel
from halo.portfolio import long_short
from halo.score import weighted_score
from halo.tuning import Engine, neighbours, run_tuning, simplex_grid, walk_forward

WF = {"tuning": {"walkforward_years": [2008, 2014]}}


@pytest.fixture(scope="module")
def planted_panel():
    f = make_fixture(seed=1, planted={"clr": 0.03})
    return build_panel(*panel_inputs(f), load_config(), log=lambda *a: None)


def test_grid_and_neighbours():
    g = simplex_grid(0.1)
    assert g.shape == (3003, 6)
    assert np.allclose(g.sum(axis=1), 1) and (g >= 0).all()
    nb = neighbours(g, 0.2)
    corner = int(np.flatnonzero(g[:, 0] == 1)[0])
    assert len(nb[corner]) == 6                 # itself plus moving 0.1 to each of 5 others
    assert max(len(n) for n in nb) == 31        # itself plus all 6 x 5 moves


@pytest.mark.parametrize("basis", ["both_sides", "spec"])
def test_engine_matches_reference(planted_panel, basis):
    cfg = load_config(overrides={"portfolio": {"cost_basis": basis}})
    eng = Engine.from_panel(planted_panel, cfg, holdout_guard=False)
    pct = planted_panel[[f"pct_{m}" for m in METRICS]].set_axis(list(METRICS), axis=1)
    for weights, col in [(np.full(6, 1 / 6), "score"), (np.eye(6)[METRICS.index("clr")],
                                                         "clr_only")]:
        # The reference gets the same score rule: weighted mean of ranks, 4 of 6 required.
        p = planted_panel.assign(clr_only=weighted_score(pct, weights, 4)[0])
        ref = long_short(p, cfg, score_col=col)
        r = eng.run(weights[None, :])
        got = pd.DataFrame({k: v[:, 0] for k, v in r.items()}, index=eng.months)
        j = ref.join(got.add_prefix("e_"), how="inner")
        assert len(j) == len(ref)
        for a, b in [("q5", "top"), ("q1", "bottom"), ("ls_net", "net"),
                     ("to_long", "to_long"), ("to_short", "to_short")]:
            assert np.allclose(j[a], j[f"e_{b}"], atol=1e-12), (a, col)


def test_time_varying_weights_equal_fixed_when_constant(planted_panel):
    cfg = load_config()
    eng = Engine.from_panel(planted_panel, cfg, holdout_guard=False)
    w = np.array([[0.5, 0, 0, 0.5, 0, 0], [0, 0, 1, 0, 0, 0]])
    fixed = eng.run(w)
    path = eng.run(np.repeat(w[None], len(eng.months), axis=0))
    assert np.allclose(fixed["net"], path["net"], equal_nan=True)


def test_holdout_is_never_used(planted_panel):
    cfg = load_config(overrides={"holdout": {"start": "2009-01"},
                                 "tuning": {"walkforward_years": [2008, 2008]}})
    eng = Engine.from_panel(planted_panel, cfg)
    assert eng.hold_months.max() == pd.Period("2008-12", "M")
    leaky = Engine.from_panel(planted_panel, cfg, holdout_guard=False)
    with pytest.raises(AssertionError, match="holdout"):
        walk_forward(leaky, cfg)


def test_planted_signal_recovered_and_replica_worse(planted_panel):
    cfg = load_config(overrides=WF)
    wf = walk_forward(Engine.from_panel(planted_panel, cfg), cfg, config_hash="t")
    assert wf.weights_path["w_clr"].mean() >= 0.7
    sharpe = {c[:-4]: wf.oos[c].mean() / wf.oos[c].std() for c in wf.oos if c.endswith("_net")}
    assert sharpe["tuned_l1"] > sharpe["replica_eq"]
    assert len(wf.trials) == 3003 * 7
    assert (wf.oos.index.year >= 2008).all() and (wf.oos.index.year <= 2014).all()
    # lambda = 0 is exactly Goldman's equal weight.
    assert np.allclose(wf.oos["tuned_l0_net"], wf.oos["replica_eq_net"], equal_nan=True)


def test_run_tuning_writes_results_and_logs_trials(planted_panel, tmp_path):
    cfg = load_config(overrides=WF)
    panel_path = tmp_path / "panel.parquet"
    planted_panel.to_parquet(panel_path)
    report = run_tuning(panel_path, tmp_path / "results", cfg, log=lambda *a: None)
    for name in ["weights_path", "oos_returns", "oos_summary", "trials"]:
        assert (tmp_path / "results" / f"{name}.csv").exists()
    assert "Chosen weights by year" in report
    # A second configuration appends; rerunning the same one replaces its own rows.
    run_tuning(panel_path, tmp_path / "results", cfg, log=lambda *a: None)
    cfg2 = load_config(overrides={**WF, "portfolio": {"cost_bps": 25}})
    run_tuning(panel_path, tmp_path / "results", cfg2, log=lambda *a: None)
    trials = pd.read_csv(tmp_path / "results" / "trials.csv")
    assert trials["config_hash"].nunique() == 2
    assert len(trials) == 2 * 3003 * 7
