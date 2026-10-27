"""M2 pull steps against SqliteWRDS, in both schema layouts (spec sections 3 and 9)."""

import numpy as np
import pandas as pd
import pytest

from halo.checks import FAIL, PASS
from halo.compustat import attach_fundamentals
from halo.config import load_config
from halo.crsp import crsp_screen, load_msf
from halo.discovery import run_discovery
from halo.factors import parse_fred_csv, pull_ff
from halo.fixtures import fake_fred, make_fixture, to_fake_wrds
from halo.link import link_gvkey
from halo.pull import ph, resolve_columns, sql_list, year_chunks
from halo.pull_steps import step_compustat, step_crsp
from halo.schema import load_schema_map
from halo.wrds_io import SqliteWRDS

END = "2003-12-31"


@pytest.fixture(scope="module")
def fx():
    f = make_fixture(seed=7, n_stocks=80, n_years=8)
    last = f.msf["mthcaldt"].max()
    common = f.msf["permno"].isin(f.truth["permno"])
    idx = f.msf.loc[(f.msf["mthcaldt"] == last) & common, "mthcap"].idxmax()
    f.msf.loc[idx, "mthcap"] = 2.5e9      # a mega-cap, so the units check has a target
    return f


def setup(fx, tmp_path, layout="standard", db=None):
    db = db or to_fake_wrds(fx, layout)
    run_discovery(db, tmp_path / "d.md", tmp_path / "m.yaml", log=lambda *a: None)
    return db, load_schema_map(tmp_path / "m.yaml"), load_config()


def statuses(checks):
    return {c.id: c.status for c in checks}


@pytest.mark.parametrize("layout", ["standard", "variant"])
def test_both_steps_pass(fx, tmp_path, layout):
    db, smap, cfg = setup(fx, tmp_path, layout)
    raw = tmp_path / "raw"
    _, c1 = step_crsp(db, smap, cfg, raw, END, log=lambda *a: None)
    _, c2 = step_compustat(db, smap, cfg, raw, fetch=fake_fred, log=lambda *a: None)
    fails = [c for c in c1 + c2 if c.status == FAIL]
    assert not fails, [c.line() for c in fails]
    for name in ["crsp_secinfo", "ccm_link", "comp_funda", "ff_factors", "fred_rates"]:
        assert (raw / f"{name}.parquet").exists()
    assert len(list(raw.glob("crsp_msf_*.parquet"))) == 4   # 1994-96, 1997-99, 2000-02, 2003


def test_msf_dtypes_and_scope(fx, tmp_path):
    db, smap, cfg = setup(fx, tmp_path)
    step_crsp(db, smap, cfg, tmp_path / "raw", END, log=lambda *a: None)
    msf = load_msf(tmp_path / "raw")
    assert msf["permno"].dtype == np.int64
    assert msf["mthcaldt"].dtype == "datetime64[ns]"
    assert msf["mthret"].dtype == np.float64
    assert isinstance(msf["mthretflg"].dtype, pd.CategoricalDtype)
    assert str(msf["month"].dtype) == "period[M]"
    # Common-stock filter applied in SQL: ADRs, ETFs and units are gone.
    assert set(msf["permno"]) <= set(fx.truth["permno"])
    assert msf["mthcaldt"].max() <= pd.Timestamp(END)


def test_layouts_pull_the_same_stock_months(fx, tmp_path):
    out = []
    for layout in ["standard", "variant"]:
        db, smap, cfg = setup(fx, tmp_path / layout, layout)
        step_crsp(db, smap, cfg, tmp_path / layout / "raw", END, log=lambda *a: None)
        m = load_msf(tmp_path / layout / "raw")
        out.append(set(zip(m["permno"], m["month"], strict=True)))
    assert out[0] == out[1]


def test_exchange_filter_not_applied_in_sql(fx, tmp_path):
    """A stock suspended in its last month must keep that month's return (spec 3.3 note)."""
    f = make_fixture(seed=7, n_stocks=80, n_years=8)
    f.msf = fx.msf.copy()
    db = to_fake_wrds(f, "standard")
    raw_msf = db.raw_sql("SELECT * FROM crsp.msf_v2")
    de = raw_msf["mthretflg"] == "DE"
    raw_msf.loc[de, "tradingstatusflg"] = "X"
    db.add_table("crsp", "msf_v2", raw_msf)
    db, smap, cfg = setup(fx, tmp_path, db=db)
    step_crsp(db, smap, cfg, tmp_path / "raw", END, log=lambda *a: None)
    msf = load_msf(tmp_path / "raw")
    last = msf[msf["mthretflg"] == "DE"]
    assert len(last) > 0 and last["mthret"].notna().all()
    assert not crsp_screen(msf, cfg).loc[last.index, "crsp_ok"].any()


def test_distribution_duplicates_collapse_but_conflicts_fail(fx, tmp_path):
    db = to_fake_wrds(fx, "standard")
    m = db.raw_sql("SELECT * FROM crsp.msf_v2").assign(disexdt=None)
    dup = m.iloc[:50].assign(disexdt="2001-01-15")
    db.add_table("crsp", "msf_v2", pd.concat([m, dup], ignore_index=True))
    db, smap, cfg = setup(fx, tmp_path, db=db)
    _, checks = step_crsp(db, smap, cfg, tmp_path / "raw", END, log=lambda *a: None)
    assert statuses(checks)["CRSP1"] == PASS

    bad = m.iloc[:5].assign(mthret=m["mthret"].iloc[:5] + 0.01, disexdt="2001-01-15")
    db.add_table("crsp", "msf_v2", pd.concat([m, bad], ignore_index=True))
    _, checks = step_crsp(db, smap, cfg, tmp_path / "raw2", END, log=lambda *a: None)
    assert statuses(checks)["CRSP1"] == FAIL


def test_percent_returns_fail(fx, tmp_path):
    db = to_fake_wrds(fx, "standard")
    m = db.raw_sql("SELECT * FROM crsp.msf_v2")
    m["mthret"] = m["mthret"] * 100
    db.add_table("crsp", "msf_v2", m)
    db, smap, cfg = setup(fx, tmp_path, db=db)
    _, checks = step_crsp(db, smap, cfg, tmp_path / "raw", END, log=lambda *a: None)
    assert statuses(checks)["CRSP4"] == FAIL


def test_missing_links_fail_coverage(fx, tmp_path):
    db = to_fake_wrds(fx, "standard")
    ccm = db.raw_sql("SELECT * FROM crsp.ccmxpf_lnkhist")
    db.add_table("crsp", "ccmxpf_lnkhist", ccm.iloc[::3])
    db, smap, cfg = setup(fx, tmp_path, db=db)
    raw = tmp_path / "raw"
    step_crsp(db, smap, cfg, raw, END, log=lambda *a: None)
    _, checks = step_compustat(db, smap, cfg, raw, fetch=fake_fred, log=lambda *a: None)
    assert statuses(checks)["X1"] == FAIL


def test_control_items_found_live_when_missing_from_map(fx, tmp_path):
    db, smap, cfg = setup(fx, tmp_path)
    for c in ["dltt", "dlc"]:
        del smap["tables"]["funda"]["columns"][c]     # as in a map made before M2
    logs = []
    m = resolve_columns(db, smap, "funda", ["gvkey", "dltt", "dlc", "nosuch"], ["gvkey"],
                        log=logs.append)
    assert m == {"gvkey": "gvkey", "dltt": "dltt", "dlc": "dlc"}
    assert any("found on WRDS by name" in s for s in logs)
    assert any("nosuch" in s for s in logs)
    with pytest.raises(ValueError, match="required"):
        resolve_columns(db, smap, "funda", ["nosuch"], ["nosuch"], log=logs.append)


def test_lpermno_is_cast_to_int(fx, tmp_path):
    db = to_fake_wrds(fx, "standard")
    ccm = db.raw_sql("SELECT * FROM crsp.ccmxpf_lnkhist")
    ccm["lpermno"] = ccm["lpermno"].astype(float)
    ccm.loc[0, "lpermno"] = None                 # links to non-CRSP securities
    db.add_table("crsp", "ccmxpf_lnkhist", ccm)
    db, smap, cfg = setup(fx, tmp_path, db=db)
    step_crsp(db, smap, cfg, tmp_path / "raw", END, log=lambda *a: None)
    out = pd.read_parquet(tmp_path / "raw" / "ccm_link.parquet")
    assert out["permno"].dtype == np.int64
    assert len(out) == len(ccm[ccm["linktype"].isin(["LC", "LU", "LS"])
                               & ccm["linkprim"].isin(["P", "C"])]) - 1


def test_link_tie_break_and_dates():
    ccm = pd.DataFrame({
        "gvkey": ["000001", "000002", "000003", "000004"],
        "permno": [1, 1, 2, 3],
        "linktype": ["LC", "LU", "LC", "LC"],
        "linkprim": ["C", "P", "P", "P"],
        "linkdt": pd.to_datetime(["2000-01-01", "1999-01-01", "2000-01-01", "2000-01-01"]),
        "linkenddt": pd.to_datetime([None, None, "2000-06-15", None]),
    })
    keys = pd.DataFrame({"permno": [1, 2, 2, 3],
                         "month": pd.PeriodIndex(["2001-01", "2000-05", "2000-06", "1999-12"],
                                                 freq="M")})
    out, n_multi = link_gvkey(keys, ccm, load_config())
    assert out["gvkey"].tolist()[:1] == ["000002"]        # P beats C
    assert out["gvkey"].iloc[1] == "000003"               # within the link window
    assert pd.isna(out["gvkey"].iloc[2])                  # link ended mid-June
    assert pd.isna(out["gvkey"].iloc[3])                  # before linkdt
    assert n_multi == 1


def test_attach_fundamentals_lag_and_expiry():
    f = pd.DataFrame({"gvkey": ["000001"] * 3,
                      "datadate": pd.to_datetime(["2000-12-31", "2001-12-31", "2003-12-31"]),
                      "at": [1.0, 2.0, 3.0]})
    k = pd.DataFrame({"gvkey": ["000001"] * 6,
                      "month": pd.PeriodIndex(["2001-05", "2001-06", "2002-06", "2003-06",
                                               "2003-07", "2004-06"], freq="M")})
    out = attach_fundamentals(k, f, 6, 18)
    assert out["at"].tolist()[1:3] == [1.0, 2.0]
    assert np.isnan(out["at"].iloc[0])                    # not yet 6 months old
    assert out["at"].iloc[3] == 2.0 and out["fundamentals_age_months"].iloc[3] == 18
    assert np.isnan(out["at"].iloc[4])                    # 19 months old: expired
    assert out["at"].iloc[5] == 3.0


def test_ff_percent_units_converted(fx, tmp_path):
    db, smap, cfg = setup(fx, tmp_path)
    smap["decisions"]["factor_units"]["ff_monthly"] = "percent"
    ff = pull_ff(db, smap, tmp_path, log=lambda *a: None)
    assert ff["mktrf"].abs().mean() < 0.01


def test_fred_parsing():
    old = "DATE,GS10\n1996-01-01,5.65\n1996-02-01,.\n"
    df = parse_fred_csv(old, "gs10")
    assert df["gs10"].iloc[0] == pytest.approx(0.0565)
    assert np.isnan(df["gs10"].iloc[1])
    assert str(df["month"].iloc[0]) == "1996-01"


def test_helpers():
    assert ph(SqliteWRDS(), "start") == ":start"
    assert ph(object(), "start") == "%(start)s"
    assert sql_list(["NS", "N/A"]) == "'NS', 'N/A'"
    with pytest.raises(ValueError):
        sql_list(["x'; DROP TABLE t; --"])
    assert year_chunks("1994-01-01", "2001-06-30", 3) == [
        ("1994-01-01", "1996-12-31"), ("1997-01-01", "1999-12-31"), ("2000-01-01", "2001-06-30")]


def test_fred_failure_is_reported_not_raised(fx, tmp_path):
    db, smap, cfg = setup(fx, tmp_path)
    raw = tmp_path / "raw"
    step_crsp(db, smap, cfg, raw, END, log=lambda *a: None)

    def blocked(url):
        raise OSError("Tunnel connection failed: 403 Forbidden")

    report, checks = step_compustat(db, smap, cfg, raw, fetch=blocked, log=lambda *a: None)
    assert statuses(checks)["RATE0"] == FAIL
    assert "Compustat funda" in report and "X1" in statuses(checks)
