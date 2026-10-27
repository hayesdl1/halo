"""M3: panel construction, metrics, ranks, score, and the no-look-ahead guarantee."""

import numpy as np
import pandas as pd
import pytest

from halo import METRICS
from halo.checks import FAIL, PASS
from halo.config import load_config
from halo.crsp import load_msf
from halo.discovery import run_discovery
from halo.fixtures import fake_fred, make_fixture, to_fake_wrds
from halo.industries import ff12, sector
from halo.metrics import compute_metrics
from halo.panel import PANEL_FILE, build_panel, run_build
from halo.pull_steps import step_compustat, step_crsp
from halo.schema import load_schema_map
from halo.score import pct_ranks, rank_inputs, sector_groups, weighted_score
from halo.validate import check_panel

END = "2015-12-31"


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    """Run the real M2 pull steps on the fixture, as the user's machine would."""
    tmp = tmp_path_factory.mktemp("m3")
    f = make_fixture(seed=3)
    common = f.msf["permno"].isin(f.truth["permno"])
    last = f.msf["mthcaldt"].max()
    idx = f.msf.loc[(f.msf["mthcaldt"] == last) & common, "mthcap"].idxmax()
    f.msf.loc[idx, "mthcap"] = 2.5e9
    db = to_fake_wrds(f, "standard")
    run_discovery(db, tmp / "d.md", tmp / "m.yaml", log=lambda *a: None)
    smap, cfg = load_schema_map(tmp / "m.yaml"), load_config()
    step_crsp(db, smap, cfg, tmp / "raw", END, log=lambda *a: None)
    step_compustat(db, smap, cfg, tmp / "raw", fetch=fake_fred, log=lambda *a: None)
    return tmp


@pytest.fixture(scope="module")
def inputs(raw):
    from halo.compustat import load_funda
    from halo.link import load_ccm
    return load_msf(raw / "raw"), load_ccm(raw / "raw"), load_funda(raw / "raw")


@pytest.fixture(scope="module")
def panel(inputs):
    return build_panel(*inputs, load_config(), log=lambda *a: None)


def test_run_build_passes_and_writes(raw):
    report, checks = run_build(raw / "raw", raw / "processed", load_config(), log=lambda *a: None)
    fails = [c.line() for c in checks if c.status == FAIL]
    assert not fails, fails
    assert {c.id: c.status for c in checks}["P9"] == PASS
    back = pd.read_parquet(raw / "processed" / PANEL_FILE)
    assert str(back["month"].dtype) == "period[M]"
    assert "Exhibit 6" in report


def test_panel_columns_from_spec(panel):
    need = ["permno", "gvkey", "month", "mthret", "mthprc", "mthcap", "siccd", "primaryexch",
            "eligible", *METRICS, *[f"pct_{m}" for m in METRICS], "n_metrics_available",
            "score", "fundamentals_datadate", "fundamentals_age_months", "ret_next"]
    assert set(need) <= set(panel.columns)
    assert not panel.duplicated(["permno", "month"]).any()


def test_eligibility_rules(panel):
    cfg = load_config()
    e = panel[panel["eligible"]]
    assert (e["mthprc"] >= 5).all()
    assert not e["sic"].between(6000, 6999).any()
    assert e["primaryexch"].isin(["N", "A", "Q"]).all()
    assert e["gvkey"].notna().all() and e["fundamentals_datadate"].notna().all()
    assert (e["size_rank"] <= cfg["universe"]["n_stocks"]).all()
    # Ranks exist only on eligible rows.
    assert panel.loc[~panel["eligible"], [f"pct_{m}" for m in METRICS]].isna().all().all()


def test_ranks_and_score(panel):
    e = panel[panel["eligible"]]
    for m in METRICS:
        r = e[f"pct_{m}"].dropna()
        assert r.between(0, 1).all() and (r > 0).all()
        # The top rank is 1, or just below 1 when several stocks tie at the top.
        top = e.groupby("month")[f"pct_{m}"].max().dropna()
        assert (top <= 1).all() and (top > 0.9).all()
    s = e[e["scored"]]
    manual = s[[f"pct_{m}" for m in METRICS]].mean(axis=1)
    assert np.allclose(manual, s["score"])
    assert (s["n_metrics_available"] >= 4).all()
    assert e.loc[e["n_metrics_available"] < 4, "score"].isna().all()


def test_ret_next_is_next_months_return(panel):
    p = panel.sort_values(["permno", "month"])
    nxt = p.groupby("permno")["mthret"].shift(-1)
    same = p["has_next_row"]
    assert np.allclose(p.loc[same, "ret_next"].fillna(-9), nxt[same].fillna(-9))
    assert p.loc[~same, "ret_next"].isna().all()


@pytest.mark.parametrize("as_of", ["1999-06", "2004-12", "2011-03"])
def test_no_look_ahead(inputs, panel, as_of):
    """Rebuilding with only data knowable at ``as_of`` reproduces the full panel up to then."""
    cut = build_panel(*inputs, load_config(), as_of=as_of, log=lambda *a: None)
    m = pd.Period(as_of, "M")
    cols = ["eligible", "gvkey", "fundamentals_datadate", "score", "score_sn",
            *[f"pct_{x}" for x in METRICS], *METRICS]
    a = panel[panel["month"] <= m].set_index(["permno", "month"])[cols].sort_index()
    b = cut.set_index(["permno", "month"])[cols].sort_index()
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
    # The last month's next-month return is in the future, so it must be unknown.
    assert cut.loc[cut["month"] == m, "ret_next"].isna().all()


def test_exhibit6_fails_when_metrics_are_inverted(panel):
    flipped = panel.copy()
    flipped["score"] = 1 - flipped["score"]
    status = {c.id: c.status for c in check_panel(flipped, load_config())}
    assert status["P9"] == FAIL


def test_metric_rules_by_hand():
    f = pd.DataFrame({
        "at": [100, 100, 0, 100, 100], "che": [10, np.nan, 5, 10, 10],
        "intan": [20, 20, 0, np.nan, 20], "sale": [50, 50, 50, 0, 50],
        "ppent": [40, 40, 40, 40, 40], "ppegt": [60, 60, 60, 60, 60],
        "emp": [2, 2, 2, 0, np.nan], "capx": [5, 5, 5, 5, np.nan],
        "ebitda": [10, -1, np.nan, 10, 10], "oibdp": [10, -1, 8, 10, 10],
    })
    m = compute_metrics(f)
    assert m.loc[0, "tai"] == pytest.approx((100 - 10 - 20) / 50)
    assert m.loc[0, "fai"] == pytest.approx(40 / 50)
    assert m.loc[0, "fas"] == pytest.approx(40 / 100)
    assert m.loc[0, "clr"] == pytest.approx(70 / 2)
    assert m.loc[0, "cxi"] == pytest.approx(5 / 50)
    assert m.loc[0, "cxl"] == pytest.approx(5 / 10)
    assert np.isnan(m.loc[1, "tai"])                      # che missing
    assert np.isnan(m.loc[1, "cxl"]) and m.loc[1, "ebitda_nonpos"]
    assert np.isnan(m.loc[2, "tai"]) and np.isnan(m.loc[2, "fas"])   # at <= 0
    assert m.loc[2, "cxl"] == pytest.approx(5 / 8)        # ebitda falls back to oibdp
    assert m.loc[3, "intan_filled"]
    assert np.isnan(m.loc[3, "tai"]) and np.isnan(m.loc[3, "cxi"])   # sale <= 0
    assert np.isnan(m.loc[3, "clr"])                      # emp <= 0
    assert np.isnan(m.loc[4, "clr"]) and np.isnan(m.loc[4, "cxi"])   # emp, capx missing
    assert compute_metrics(f, ppe="ppegt").loc[0, "fai"] == pytest.approx(60 / 50)


def test_ranks_with_ties_and_max_rank():
    month = pd.Series([1, 1, 1, 1, 2, 2])
    vals = pd.DataFrame({"x": [3.0, 1.0, 1.0, np.nan, 5.0, 2.0]})
    r = pct_ranks(vals, month)["x"]
    assert r.tolist()[:3] == pytest.approx([1.0, 0.5, 0.5])   # ties share the average rank
    assert np.isnan(r.iloc[3])
    assert r.tolist()[4:] == pytest.approx([1.0, 0.5])
    metrics = pd.DataFrame({m: [1.0, 2.0, 3.0] for m in METRICS})
    metrics["cxl"] = [0.5, np.nan, 0.2]
    metrics["ebitda_nonpos"] = [False, True, False]
    ranked = pct_ranks(rank_inputs(metrics, "max_rank"), pd.Series([1, 1, 1]))
    assert ranked["cxl"].tolist() == pytest.approx([2 / 3, 1.0, 1 / 3])
    assert np.isnan(pct_ranks(rank_inputs(metrics, "nan"), pd.Series([1, 1, 1]))["cxl"][1])


def test_weighted_score_min_metrics():
    pct = pd.DataFrame([[0.2, 0.4, 0.6, 0.8, np.nan, np.nan],
                        [0.2, 0.4, 0.6, np.nan, np.nan, np.nan]], columns=list(METRICS))
    s, n = weighted_score(pct, [1, 1, 1, 1, 1, 1], 4)
    assert s.iloc[0] == pytest.approx(0.5) and np.isnan(s.iloc[1])
    assert n.tolist() == [4, 3]
    s, _ = weighted_score(pct, [1, 0, 0, 0, 0, 0], 4)
    assert s.iloc[0] == pytest.approx(0.2)


def test_sector_groups_merge_small_groups():
    sic = pd.Series([4911] * 12 + [4922, 4931, 3711])
    month = pd.Series([1] * 15)
    g = sector_groups(sic, month, 10)
    assert (g.iloc[:12] == "2d_49").all()
    assert g.iloc[14] == "1d_3"


def test_industry_mapping():
    sic = pd.Series([4911, 4950, 7372, 7380, 3674, 6021, 1311, 0, np.nan])
    assert ff12(sic).tolist() == ["Utils", "Other", "BusEq", "Other", "BusEq", "Money",
                                  "Enrgy", "Unknown", "Unknown"]
    assert sector(sic).tolist()[2] == "Software"
