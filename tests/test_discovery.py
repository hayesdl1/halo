"""Schema discovery (M1) against a SQLite stand-in for WRDS, in both schema layouts."""

import pandas as pd
import pytest
import yaml

from halo.discovery import CONFIRMED, ERROR, REFUTED, UNKNOWN, run_discovery
from halo.fixtures import make_fixture, to_fake_wrds
from halo.schema import load_schema_map, select_list


@pytest.fixture(scope="module")
def small():
    f = make_fixture(seed=5, n_stocks=60, n_years=6)
    # Give the largest firm on the last date a realistic mega-cap size ($2.5 trillion),
    # so the mthcap units check has something to find.
    last = f.msf["mthcaldt"].max()
    idx = f.msf.loc[f.msf["mthcaldt"] == last, "mthcap"].idxmax()
    f.msf.loc[idx, "mthcap"] = 2.5e9
    return f


def run(db, tmp_path, **kw):
    d = run_discovery(db, tmp_path / "doc.md", tmp_path / "map.yaml", log=lambda *a: None, **kw)
    return d, {c.id: c for c in d.checks}


@pytest.mark.parametrize("layout,msf_table,start_col", [
    ("standard", "crsp.msf_v2", "secinfostartdt"),
    ("variant", "crsp_a_stock62.wrds_msf62v2_query", "securitybegdt"),
])
def test_both_layouts_pass(small, tmp_path, layout, msf_table, start_col):
    d, checks = run(to_fake_wrds(small, layout), tmp_path)
    assert d.failures == []
    assert d.tables["msf"].fqn == msf_table
    assert d.tables["secinfo"].columns["secinfostartdt"] == start_col
    for cid in ["V1", "V2", "V3", "V7", "V8", "R1", "R2", "R3", "R5", "R6", "R8", "F1", "F2",
                "F3", "L1", "L2", "L3", "C1", "V10"]:
        assert checks[cid].status == CONFIRMED, (cid, checks[cid].evidence)


def test_outputs_written_and_readable(small, tmp_path):
    db = to_fake_wrds(small, "variant")
    run(db, tmp_path)
    m = load_schema_map(tmp_path / "map.yaml")
    assert m["tables"]["ccm"]["library"] == "crsp_a_ccm"
    assert m["tables"]["dsf"] is None
    assert m["decisions"]["filter_source"] == "secinfo"
    assert m["decisions"]["factor_units"]["ff_monthly"] == "decimal"
    doc = (tmp_path / "doc.md").read_text()
    assert "**PASSED.**" in doc
    assert "\u2014" not in doc, "no em dashes in generated docs"
    # The generated map drives real queries and yields canonical names.
    e = m["tables"]["secinfo"]
    out = db.raw_sql(f"SELECT {select_list(m, 'secinfo', ['permno', 'secinfostartdt'])} "
                     f"FROM {e['library']}.{e['table']}")
    assert list(out.columns) == ["permno", "secinfostartdt"]
    with pytest.raises(KeyError):
        select_list(m, "msf", ["mthshrout"])  # absent in the fixture


def test_prefer_library(small, tmp_path):
    db = to_fake_wrds(small, "standard")
    variant = to_fake_wrds(small, "variant")
    for lib in variant.list_libraries():
        for t in variant.list_tables(lib):
            if lib.startswith("crsp"):
                db.add_table(lib, t, variant.raw_sql(f"SELECT * FROM {lib}.{t}"))
    d, _ = run(db, tmp_path)
    assert d.tables["msf"].fqn == "crsp.msf_v2"
    assert "crsp_a_stock62.wrds_msf62v2_query" in d.tables["msf"].alternatives
    d, _ = run(db, tmp_path, prefer_library="crsp_a_stock62")
    assert d.tables["msf"].fqn == "crsp_a_stock62.wrds_msf62v2_query"


def test_missing_link_table_fails_loudly(small, tmp_path):
    db = to_fake_wrds(small, "standard")
    db._libs["crsp"].remove("ccmxpf_lnkhist")
    d, checks = run(db, tmp_path)
    assert checks["V7"].status == REFUTED
    assert any("ccm" in f for f in d.failures)
    assert "**FAILED.**" in (tmp_path / "doc.md").read_text()


def test_table_without_required_columns_is_skipped(small, tmp_path):
    db = to_fake_wrds(small, "variant")
    db.add_table("crsp_a_stock62", "msf_v2_summary", pd.DataFrame({"permno": [1], "n": [2]}))
    d, _ = run(db, tmp_path)
    assert d.tables["msf"].fqn == "crsp_a_stock62.wrds_msf62v2_query"


def _mutated(small, fn, layout="standard"):
    f = make_fixture(seed=5, n_stocks=60, n_years=6)
    f.msf = small.msf.copy()
    fn(f)
    return to_fake_wrds(f, layout)


def test_negative_prices_and_legacy_codes_refuted(small, tmp_path):
    def break_it(f):
        f.msf.loc[f.msf.index[:5], "mthprc"] *= -1
        f.msf.loc[f.msf.index[10:13], "mthret"] = -99.0
    d, checks = run(_mutated(small, break_it), tmp_path)
    assert checks["R2"].status == REFUTED
    assert checks["R5"].status == REFUTED
    assert any(f.startswith("R2") for f in d.failures)
    assert any(f.startswith("R5") for f in d.failures)


def test_cap_in_dollars_refuted(small, tmp_path):
    def dollars(f):
        f.msf["mthcap"] = f.msf["mthcap"] * 1000
    _, checks = run(_mutated(small, dollars), tmp_path)
    assert checks["R3"].status == REFUTED


def test_legacy_columns_refuted(small, tmp_path):
    def legacy(f):
        f.msf["shrcd"] = 10
        f.msf["dlret"] = None
    d, checks = run(_mutated(small, legacy), tmp_path)
    assert checks["R6"].status == REFUTED
    assert checks["R1"].status == REFUTED
    assert any(f.startswith("R6") for f in d.failures)


def test_legacy_freeze(small, tmp_path):
    db = to_fake_wrds(small, "standard")
    legacy = small.msf[["permno", "mthcaldt"]].rename(columns={"mthcaldt": "date"})
    legacy = legacy[legacy["date"] < small.msf["mthcaldt"].max() - pd.DateOffset(months=6)]
    db.add_table("crsp", "msf", legacy)
    _, checks = run(db, tmp_path)
    # Fixture data ends in 2001, before the 2024 freeze, and v2 runs later than legacy.
    assert checks["R7"].status == CONFIRMED
    db2 = to_fake_wrds(small, "standard")
    _, checks2 = run(db2, tmp_path)
    assert checks2["R7"].status == UNKNOWN


def test_percent_factors_refuted(small, tmp_path):
    db = to_fake_wrds(small, "standard")
    fac = small.factors.copy()
    for c in ["mktrf", "smb", "hml", "rf", "umd"]:
        fac[c] = fac[c] * 100
    db.add_table("ff", "factors_monthly", fac[["date", "mktrf", "smb", "hml", "rf", "umd"]])
    d, checks = run(db, tmp_path)
    assert checks["V10"].status == REFUTED
    m = yaml.safe_load((tmp_path / "map.yaml").read_text())
    assert m["decisions"]["factor_units"]["ff_monthly"] == "percent"


def test_query_error_becomes_error_row(small, tmp_path):
    db = to_fake_wrds(small, "standard")
    real = db.raw_sql

    def flaky(sql, **kw):
        if "LENGTH(" in sql:
            raise RuntimeError("permission denied for relation ccmxpf_lnkhist")
        return real(sql, **kw)

    db.raw_sql = flaky
    d, checks = run(db, tmp_path)
    assert checks["L"].status == ERROR
    assert any("permission denied" in f for f in d.failures)
    assert (tmp_path / "doc.md").exists()


def test_schema_map_missing_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="00_discover_schema"):
        load_schema_map(tmp_path / "nope.yaml")


def _deny(db, blocked):
    """Mimic WRDS views whose source schema this account cannot read."""
    real = db.raw_sql

    def guarded(sql, **kw):
        if any(f"{t} " in sql + " " for t in blocked):
            raise RuntimeError("(psycopg2.errors.InsufficientPrivilege) permission denied "
                               "for schema crsp_a_stock\n[SQL: ...]")
        return real(sql, **kw)

    db.raw_sql = guarded
    return db


def _both_layouts(small):
    db = to_fake_wrds(small, "standard")
    variant = to_fake_wrds(small, "variant")
    for lib in variant.list_libraries():
        if lib.startswith("crsp"):
            for t in variant.list_tables(lib):
                db.add_table(lib, t, variant.raw_sql(f"SELECT * FROM {lib}.{t}"))
    return db


def test_unreadable_tables_fall_back_to_readable_ones(small, tmp_path):
    db = _deny(_both_layouts(small), ["crsp.msf_v2", "crsp.stksecurityinfohist"])
    d, checks = run(db, tmp_path)
    assert d.failures == [], d.failures
    assert d.tables["msf"].fqn == "crsp_a_stock62.wrds_msf62v2_query"
    assert d.tables["secinfo"].fqn == "crsp_a_stock62.stksecurityinfohist62"
    assert checks["A1"].status == "INFO"
    assert "crsp.msf_v2" in checks["A1"].evidence
    assert "[SQL" not in checks["A1"].evidence, "only the first line of the error is kept"
    m = load_schema_map(tmp_path / "map.yaml")
    assert m["tables"]["msf"]["library"] == "crsp_a_stock62"


def test_all_candidates_unreadable_is_a_failure(small, tmp_path):
    db = _deny(to_fake_wrds(small, "standard"), ["crsp.msf_v2"])
    d, checks = run(db, tmp_path)
    assert "msf" in d.unresolved
    assert "permission denied on: ['crsp.msf_v2']" in d.unresolved["msf"]
    assert any("needed table msf" in f for f in d.failures)


def test_unreadable_legacy_table_gives_unknown(small, tmp_path):
    db = to_fake_wrds(small, "standard")
    db.add_table("crsp", "msf", small.msf[["permno", "mthcaldt"]].rename(
        columns={"mthcaldt": "date"}))
    _, checks = run(_deny(db, ["crsp.msf"]), tmp_path)
    assert checks["R7"].status == UNKNOWN


def test_negative_rf_before_sample_start_is_fine(small, tmp_path):
    db = to_fake_wrds(small, "standard")
    early = small.factors.iloc[:12].copy()
    early["date"] = pd.date_range("1931-01-01", periods=12, freq="MS")
    early["rf"] = -0.0006
    fac = pd.concat([early, small.factors], ignore_index=True)
    db.add_table("ff", "factors_monthly", fac[["date", "mktrf", "smb", "hml", "rf", "umd"]])
    _, checks = run(db, tmp_path)
    assert checks["V10"].status == CONFIRMED
    _, checks = run(db, tmp_path, sample_start="1926-01-01")
    assert checks["V10"].status == REFUTED


def test_secinfo_whole_life_columns_flagged(small, tmp_path):
    # Mimic a security info table whose only date columns span the security's whole life:
    # every row of a permno with a SIC change then shares one window, so windows overlap.
    si = small.secinfo.copy()
    # A mid-sample SIC change: same permno, second row from halfway through its life.
    for _, row in si[si["secinfoenddt"].dt.year > 2090].head(3).iterrows():
        mid = row["secinfostartdt"] + pd.DateOffset(years=2)
        si.loc[row.name, "secinfoenddt"] = mid
        si = pd.concat([si, row.to_frame().T.assign(
            secinfostartdt=mid + pd.Timedelta(days=1), siccd=row["siccd"] + 1)],
            ignore_index=True)
    si = si.astype(small.secinfo.dtypes.to_dict())
    db0 = to_fake_wrds(small, "variant")
    db0.add_table("crsp_a_stock62", "stksecurityinfohist62", si.rename(columns={
        "secinfostartdt": "securitybegdt", "secinfoenddt": "securityenddt"}))
    _, checks = run(db0, tmp_path)
    assert checks["S1"].status == CONFIRMED, "true validity windows do not overlap"
    life = si.groupby("permno").agg(b=("secinfostartdt", "min"), e=("secinfoenddt", "max"))
    si = si.join(life, on="permno").drop(columns=["secinfostartdt", "secinfoenddt"])
    si = si.rename(columns={"b": "securitybegdt", "e": "securityenddt"})
    db = to_fake_wrds(small, "variant")
    db.add_table("crsp_a_stock62", "stksecurityinfohist62", si)
    _, checks = run(db, tmp_path)
    assert checks["S1"].status == REFUTED
    assert "whole life" in checks["S1"].evidence
    _, checks = run(to_fake_wrds(small, "standard"), tmp_path)
    assert checks["S1"].status == CONFIRMED
