"""The synthetic fixture must look like the real WRDS pulls (spec sections 3 and 9)."""

import numpy as np
import pandas as pd
import pytest

from halo import METRICS
from halo.fixtures import _metric_values, make_fixture

MSF_COLS = ["permno", "mthcaldt", "mthret", "mthretx", "mthprc", "mthcap", "mthprcflg"]
SECINFO_COLS = ["permno", "secinfostartdt", "secinfoenddt", "ticker", "siccd", "sharetype",
                "securitytype", "securitysubtype", "usincflg", "issuertype", "primaryexch",
                "conditionaltype", "tradingstatusflg"]
CCM_COLS = ["gvkey", "lpermno", "linktype", "linkprim", "linkdt", "linkenddt"]
FUNDA_COLS = ["gvkey", "datadate", "fyear", "at", "che", "intan", "sale", "ppent", "ppegt",
              "emp", "capx", "ebitda", "oibdp", "dp", "xrd", "sich"]


def test_shape_and_columns(fixture_default):
    f = fixture_default
    assert list(f.msf.columns) == MSF_COLS
    assert list(f.secinfo.columns) == SECINFO_COLS
    assert list(f.ccm.columns) == CCM_COLS
    assert set(FUNDA_COLS + ["indfmt", "datafmt", "popsrc", "consol"]) == set(f.funda.columns)
    assert len(f.truth) == 200
    months = f.msf["mthcaldt"].dt.to_period("M")
    assert months.min() == pd.Period("1996-01", "M")
    assert months.nunique() == 240


def test_dtypes(fixture_default):
    f = fixture_default
    assert f.msf["permno"].dtype == np.int64
    assert f.msf["mthcaldt"].dtype == "datetime64[ns]"
    for col in ["mthret", "mthprc", "mthcap"]:
        assert f.msf[col].dtype == np.float64
    assert isinstance(f.msf["mthprcflg"].dtype, pd.CategoricalDtype)
    assert f.funda["gvkey"].map(lambda g: isinstance(g, str) and len(g) == 6).all()
    assert f.ccm["gvkey"].map(lambda g: isinstance(g, str) and len(g) == 6).all()
    assert f.funda["datadate"].dtype == "datetime64[ns]"


def test_ciz_conventions(fixture_default):
    msf = fixture_default.msf
    month = msf["mthcaldt"].dt.to_period("M")
    assert not pd.DataFrame({"p": msf["permno"], "m": month}).duplicated().any()
    assert (msf["mthprc"] > 0).all(), "CIZ prices are always positive"
    assert (msf["mthret"].dropna() > -1).all()
    # mthcaldt is a weekday at or before calendar month end.
    assert (msf["mthcaldt"].dt.day_of_week < 5).all()
    assert (msf["mthcaldt"] <= month.dt.to_timestamp(how="end")).all()
    # No legacy missing-return codes.
    assert not msf["mthret"].isin([-66, -77, -88, -99]).any()


def test_delisting_return_inside_last_month(fixture_default):
    f = fixture_default
    delisted = f.truth.loc[f.truth["delisted"], "permno"]
    assert len(delisted) > 10
    last = f.msf[f.msf["permno"].isin(delisted)].groupby("permno").tail(1)
    assert last["mthret"].notna().all()


def test_missing_returns_rare(fixture_default):
    msf = fixture_default.msf
    not_first = msf.groupby("permno").cumcount() > 0
    assert msf.loc[not_first, "mthret"].isna().mean() < 0.005


def test_universe_filters_have_work_to_do(fixture_default):
    si = fixture_default.secinfo
    common = (
        (si["sharetype"] == "NS") & (si["securitytype"] == "EQTY")
        & (si["securitysubtype"] == "COM") & (si["usincflg"] == "Y")
        & si["issuertype"].isin(["ACOR", "CORP"])
    )
    assert (~common).sum() == fixture_default.config.n_noncommon
    assert si.loc[common, "permno"].nunique() == 200
    # Some firms change SIC mid-sample.
    assert si.groupby("permno")["siccd"].nunique().max() > 1
    # Financials exist so the exclusion matters.
    assert si["siccd"].between(6000, 6999).any()


def test_ccm_has_noise_to_filter(fixture_default):
    ccm = fixture_default.ccm
    assert ccm["linkenddt"].isna().any(), "active links have NULL linkenddt"
    assert (~ccm["linktype"].isin(["LC", "LU"])).any()
    assert (~ccm["linkprim"].isin(["P", "C"])).any()
    good = ccm[ccm["linktype"].isin(["LC", "LU"]) & ccm["linkprim"].isin(["P", "C"])]
    assert good["lpermno"].nunique() == 200


def test_compustat_has_edge_cases(fixture_default):
    fu = fixture_default.funda
    std = fu[fu["indfmt"] == "INDL"]
    assert (fu["indfmt"] != "INDL").any()
    assert not std.duplicated(["gvkey", "datadate"]).any()
    for col in ["intan", "capx", "emp", "ebitda", "che"]:
        assert std[col].isna().any(), col
    assert (std["ebitda"] <= 0).any()
    assert (std["sale"] <= 0).any()
    assert std["datadate"].dt.month.nunique() > 1, "fiscal year ends vary"
    assert std["datadate"].min() < pd.Timestamp("1996-01-01")


def test_factors_in_decimals(fixture_default):
    fac = fixture_default.factors
    assert fac["rf"].between(0, 0.02).all()
    assert fac["mktrf"].abs().max() < 0.5
    assert len(fac) == 240


def test_deterministic():
    kw = dict(seed=3, n_stocks=30, n_years=5)
    a, b = make_fixture(**kw), make_fixture(**kw)
    for name, df in a.tables().items():
        pd.testing.assert_frame_equal(df, b.tables()[name], obj=name)


def test_sector_ordering_matches_goldman_exhibit6(fixture_default):
    f = fixture_default
    fu = f.funda[(f.funda["indfmt"] == "INDL") & (f.funda["datadate"].dt.year == 2014)]
    m = _metric_values(fu).assign(gvkey=fu["gvkey"].values)
    m = m.merge(f.truth[["gvkey", "sector", "financial"]])
    m = m[~m["financial"]]
    m["score"] = m[list(METRICS)].rank(pct=True).mean(axis=1)
    order = m.groupby("sector")["score"].median().sort_values().index.tolist()
    assert "utilities" in order[-3:]
    assert {"software", "data_processing"} <= set(order[:3])


def _long_short_t(f, metric, lag=6):
    """Q5 minus Q1 on a lagged metric rank, built independently of the fixture internals."""
    fu = f.funda[f.funda["indfmt"] == "INDL"].copy()
    fu[metric] = _metric_values(fu)[metric]
    fu["avail"] = (fu["datadate"].dt.to_period("M") + lag).dt.to_timestamp()
    fu = fu.merge(f.truth[["gvkey", "permno"]])[["permno", "avail", metric]].dropna()
    msf = f.msf[f.msf["permno"].isin(f.truth["permno"])].copy()
    msf["month"] = msf["mthcaldt"].dt.to_period("M").dt.to_timestamp()
    msf = msf.sort_values("month")
    panel = pd.merge_asof(msf, fu.sort_values("avail"), left_on="month", right_on="avail",
                          by="permno", direction="backward")
    panel = panel.sort_values(["permno", "month"])
    panel["next_ret"] = panel.groupby("permno")["mthret"].shift(-1)
    panel = panel.dropna(subset=[metric, "next_ret"])
    panel["q"] = panel.groupby("month")[metric].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
    ls = panel.groupby(["month", "q"])["next_ret"].mean().unstack()
    spread = ls[4] - ls[0]
    return spread.mean() / spread.std() * np.sqrt(len(spread))


def test_planted_signal_is_detectable():
    f = make_fixture(seed=1, planted={"clr": 0.03})
    assert _long_short_t(f, "clr") > 3


@pytest.mark.parametrize("seed", [11, 12])
def test_null_fixture_has_no_signal(seed):
    f = make_fixture(seed=seed)
    assert abs(_long_short_t(f, "clr")) < 3


def test_write_roundtrip(tmp_path):
    f = make_fixture(seed=4, n_stocks=20, n_years=3)
    f.write(tmp_path)
    back = pd.read_parquet(tmp_path / "msf.parquet")
    pd.testing.assert_frame_equal(back, f.msf, check_categorical=False)
