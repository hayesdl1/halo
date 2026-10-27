"""M4: baskets, returns, turnover with drift, costs, and the shifted-fundamentals test."""

import numpy as np
import pandas as pd
import pytest

from halo.config import load_config
from halo.fixtures import make_fixture, panel_inputs
from halo.panel import build_panel
from halo.portfolio import (
    annualised_sharpe,
    assign_baskets,
    basket_returns,
    basket_turnover,
    holdings,
    long_short,
    missing_share,
)
from halo.validate import check_panel


def toy_panel():
    """Two months, four stocks. Hand-computed results are in the tests below."""
    return pd.DataFrame({
        "month": pd.PeriodIndex(["2000-01"] * 4 + ["2000-02"] * 4, freq="M"),
        "permno": [1, 2, 3, 4] * 2,
        "score": [0.1, 0.2, 0.8, 0.9, 0.1, 0.85, 0.8, 0.2],
        "mthcap": [100.0, 300.0, 100.0, 100.0] * 2,
        "ret_next": [0.0, 0.1, 0.5, 0.0, 0.02, 0.0, np.nan, 0.04],
        "has_next_row": [True] * 8,
        "eligible": [True] * 8,
    })


CFG = {"portfolio": {"basket": "half", "weighting": "equal", "cost_bps": 10,
                     "cost_basis": "spec"}}


def test_baskets_equal_counts_and_order_free():
    df = pd.DataFrame({"month": pd.Period("2000-01", "M"), "permno": range(10),
                       "s": [5, 1, 3, 3, 3, 9, 7, 2, 8, 6]})
    b = assign_baskets(df, "s", "quintile")
    assert b.value_counts().tolist() == [2] * 5
    # Ties (three 3s) break by permno: 2 and 3 share basket 2, 4 goes to basket 3.
    assert b[[2, 3, 4]].tolist() == [2, 2, 3]
    shuffled = df.sample(frac=1, random_state=0)
    pd.testing.assert_series_equal(assign_baskets(shuffled, "s", "quintile").sort_index(), b)
    assert assign_baskets(df, "s", "tercile").max() == 3
    assert set(assign_baskets(df, "s", "half")) == {1, 2}


def test_value_weights():
    h = holdings(toy_panel(), CFG, weighting="value")
    jan_low = h[(h["month"] == pd.Period("2000-01", "M")) & (h["basket"] == 1)]
    assert jan_low.set_index("permno")["w"].to_dict() == pytest.approx({1: 0.25, 2: 0.75})


def test_returns_drop_missing_and_rescale():
    br = basket_returns(holdings(toy_panel(), CFG))
    feb = pd.Period("2000-02", "M")
    assert br.loc[(pd.Period("2000-01", "M"), 1), "ret"] == pytest.approx(0.05)
    assert br.loc[(pd.Period("2000-01", "M"), 2), "ret"] == pytest.approx(0.25)
    # Basket 2 in February holds permnos 2 and 3; 3 has no return, so 2 carries it alone.
    assert br.loc[(feb, 2), "ret"] == pytest.approx(0.0)
    assert br.loc[(feb, 2), "n_missing"] == 1
    assert br.loc[(feb, 1), "ret"] == pytest.approx(0.03)


def test_turnover_with_drift_by_hand():
    to = basket_turnover(holdings(toy_panel(), CFG))
    jan, feb = pd.Period("2000-01", "M"), pd.Period("2000-02", "M")
    assert to[(jan, 1)] == pytest.approx(0.5)         # built from nothing
    # Basket 1, January: permnos 1, 2 at 0.5 each, returns 0 and 10%, drift to 0.5/1.05
    # and 0.55/1.05. February target: permnos 1 and 4 at 0.5 each.
    d1, d2 = 0.5 / 1.05, 0.55 / 1.05
    assert to[(feb, 1)] == pytest.approx(0.5 * (abs(0.5 - d1) + d2 + 0.5))
    # Basket 2, January: permnos 3, 4, returns 50% and 0, drift to 0.6 and 0.4.
    # February target: permnos 2 and 3.
    assert to[(feb, 2)] == pytest.approx(0.5 * (0.5 + abs(0.5 - 0.6) + 0.4))


def test_long_short_costs():
    ls = long_short(toy_panel(), CFG)
    feb = pd.Period("2000-02", "M")
    to = ls.loc[feb, "to_long"] + ls.loc[feb, "to_short"]
    assert ls.loc[feb, "cost"] == pytest.approx(to * 10 / 1e4)
    assert ls.loc[feb, "ls_net"] == pytest.approx(ls.loc[feb, "ls_gross"] - ls.loc[feb, "cost"])
    assert ls.loc[feb, "hold_month"] == pd.Period("2000-03", "M")
    both = long_short(toy_panel(), CFG, cost_basis="both_sides")
    assert both.loc[feb, "cost"] == pytest.approx(2 * ls.loc[feb, "cost"])
    zero = long_short(toy_panel(), CFG, cost_bps=0)
    assert (zero["ls_net"] == zero["ls_gross"]).all()
    jan = pd.Period("2000-01", "M")
    assert ls.loc[jan, "univ"] == pytest.approx(np.mean([0.0, 0.1, 0.5, 0.0]))
    assert ls.loc[jan, "top_minus_univ"] == pytest.approx(0.25 - 0.15)
    assert missing_share(ls) == pytest.approx(1 / 8)


@pytest.fixture(scope="module")
def planted():
    """A fixture where only clr predicts returns and ratios move a lot year to year."""
    f = make_fixture(seed=1, planted={"clr": 0.03}, time_sd_scale=10)
    return panel_inputs(f)


@pytest.mark.parametrize("months", [12, 24])
def test_shifting_fundamentals_later_does_not_help(planted, months):
    """Spec M4: using fundamentals later than allowed must not improve results. If it did,
    the correct build would be leaking something, or the signal would be fake."""
    cfg = load_config()
    msf, ccm, funda = planted
    right = build_panel(msf, ccm, funda, cfg, log=lambda *a: None)
    late = funda.assign(datadate=funda["datadate"] + pd.DateOffset(months=months))
    shifted = build_panel(msf, ccm, late, cfg, log=lambda *a: None)
    s_right = annualised_sharpe(long_short(right, cfg, "pct_clr")["ls_net"])
    s_late = annualised_sharpe(long_short(shifted, cfg, "pct_clr")["ls_net"])
    assert s_right > 1.0, "the planted signal should be found with correct timing"
    assert s_late < s_right


def test_fixture_panel_turnover_and_missing(planted):
    cfg = load_config()
    p = build_panel(*planted, cfg, log=lambda *a: None)
    ls = long_short(p, cfg)
    assert ls["to_long"].iloc[0] == pytest.approx(0.5)
    later = ls.iloc[1:]
    # Annual fundamentals and monthly rebalancing: modest turnover after the first month.
    assert later[["to_long", "to_short"]].mean().max() < 0.25
    assert missing_share(ls) < 0.005
    p10 = {c.id: c for c in check_panel(p, cfg)}["P10"]
    assert p10.status == "INFO"
