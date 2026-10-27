"""CRSP monthly stock file and security info: pull, load, screen, check.

Pull scope (spec 3.3, decided in M2): the five common-stock filters run in SQL. The
three exchange and trading-status fields are kept as columns and applied at
eligibility time, so a stock held into a month in which it is delisted or suspended
keeps that month's return.

M1 found that the monthly view on this account repeats a stock-month when it has
several distributions. The pull selects only the columns it needs with DISTINCT and
the checks then require exactly one row per (permno, month).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from halo.checks import FAIL, FLAG, INFO, PASS, Check, check
from halo.pull import ph, resolve_columns, select_sql, sql_list, year_chunks
from halo.schema import CIZ_FILTER_COLUMNS, qualified, table_entry

COMMON_STOCK_FILTERS = ("sharetype", "securitytype", "securitysubtype", "usincflg", "issuertype")
EXCHANGE_FILTERS = ("primaryexch", "conditionaltype", "tradingstatusflg")
MSF_PRICE_COLUMNS = ["permno", "mthcaldt", "mthret", "mthretx", "mthprc", "mthcap",
                     "mthprcflg", "mthretflg"]
MSF_REQUIRED = ["permno", "mthcaldt", "mthret", "mthprc", "mthcap"]
FLAG_COLUMNS = ["mthprcflg", "mthretflg", *CIZ_FILTER_COLUMNS]
MSF_GLOB = "crsp_msf_*.parquet"


def msf_query(db, smap: dict, cfg: dict, log=print) -> tuple[str, dict[str, str]]:
    """SQL for one date chunk of the monthly file, with :start and :end parameters."""
    source = smap["decisions"].get("filter_source") or "msf"
    msf_map = resolve_columns(db, smap, "msf", MSF_PRICE_COLUMNS, MSF_REQUIRED, log)
    m_fqn = qualified(table_entry(smap, "msf"))
    if source == "msf":
        info_map = resolve_columns(db, smap, "msf", ["siccd", *CIZ_FILTER_COLUMNS], [], log)
        info_alias, join = "m", ""
    else:
        info_map = resolve_columns(db, smap, "secinfo", ["siccd", *CIZ_FILTER_COLUMNS], [], log)
        si = resolve_columns(db, smap, "secinfo", ["permno", "secinfostartdt", "secinfoenddt"],
                             ["permno", "secinfostartdt", "secinfoenddt"], log)
        s_fqn = qualified(table_entry(smap, "secinfo"))
        join = (f' JOIN {s_fqn} s ON s."{si["permno"]}" = m."{msf_map["permno"]}" '
                f'AND m."{msf_map["mthcaldt"]}" BETWEEN s."{si["secinfostartdt"]}" '
                f'AND s."{si["secinfoenddt"]}"')
        info_alias = "s"
    filters = cfg["universe"]["common_stock_filter"]
    conds, skipped = [], []
    for col in COMMON_STOCK_FILTERS:
        if col in info_map:
            conds.append(f'{info_alias}."{info_map[col]}" IN ({sql_list(filters[col])})')
        else:
            skipped.append(col)
    if skipped:
        log(f"  WARNING: common-stock filter columns not available, not applied: {skipped}")
    date = f'm."{msf_map["mthcaldt"]}"'
    where = [f"{date} >= {ph(db, 'start')}", f"{date} <= {ph(db, 'end')}", *conds]
    sql = (f"SELECT DISTINCT {select_sql(msf_map, 'm')}, {select_sql(info_map, info_alias)} "
           f"FROM {m_fqn} m{join} WHERE {' AND '.join(where)}")
    return sql, {**msf_map, **info_map}


def _msf_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["permno"] = df["permno"].astype("int64")
    df["mthcaldt"] = pd.to_datetime(df["mthcaldt"]).astype("datetime64[ns]")
    for c in ["mthret", "mthretx", "mthprc", "mthcap"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    if "siccd" in df:
        df["siccd"] = pd.to_numeric(df["siccd"], errors="coerce").astype("Int64")
    for c in FLAG_COLUMNS:
        if c in df:
            df[c] = df[c].astype("string").astype("category")
    return df


def pull_msf(db, smap: dict, cfg: dict, out_dir: Path, end: str, log=print) -> int:
    """Write one parquet file per date chunk and return the total row count."""
    sql, _ = msf_query(db, smap, cfg, log)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(MSF_GLOB):
        old.unlink()
    total = 0
    chunks = year_chunks(cfg["sample"]["crsp_start"], end, cfg["sample"]["crsp_chunk_years"])
    for lo, hi in chunks:
        df = db.raw_sql(sql, date_cols=["mthcaldt"], params={"start": lo, "end": hi})
        df = _msf_dtypes(df)
        path = out_dir / f"crsp_msf_{lo[:4]}_{hi[:4]}.parquet"
        df.to_parquet(path, index=False)
        log(f"  {lo} to {hi}: {len(df):,} rows -> {path.name}")
        total += len(df)
    return total


def load_msf(raw_dir: Path) -> pd.DataFrame:
    files = sorted(Path(raw_dir).glob(MSF_GLOB))
    if not files:
        raise FileNotFoundError(f"no {MSF_GLOB} in {raw_dir}; run scripts/01_pull_crsp.py")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    for c in FLAG_COLUMNS:           # concat drops categories that differ between chunks
        if c in df:
            df[c] = df[c].astype("category")
    df["month"] = df["mthcaldt"].dt.to_period("M")
    return df


def pull_secinfo(db, smap: dict, out_dir: Path, log=print) -> pd.DataFrame:
    wanted = ["permno", "secinfostartdt", "secinfoenddt", "ticker", "siccd", *CIZ_FILTER_COLUMNS]
    mapping = resolve_columns(db, smap, "secinfo", wanted,
                              ["permno", "secinfostartdt", "secinfoenddt"], log)
    df = db.raw_sql(f"SELECT {select_sql(mapping)} FROM {qualified(table_entry(smap, 'secinfo'))}",
                    date_cols=["secinfostartdt", "secinfoenddt"])
    df["permno"] = df["permno"].astype("int64")
    for c in ["secinfostartdt", "secinfoenddt"]:
        df[c] = pd.to_datetime(df[c]).astype("datetime64[ns]")
    if "siccd" in df:
        df["siccd"] = pd.to_numeric(df["siccd"], errors="coerce").astype("Int64")
    for c in CIZ_FILTER_COLUMNS:
        if c in df:
            df[c] = df[c].astype("string").astype("category")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_parquet(Path(out_dir) / "crsp_secinfo.parquet", index=False)
    df.attrs["mapping"] = mapping
    return df


def crsp_screen(msf: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """CRSP-side eligibility at each month-end (spec 3.8, before link and fundamentals).

    Adds ``crsp_ok`` (exchange filter, price, financials, non-missing cap) and
    ``size_rank`` (1 = largest among crsp_ok stocks that month), plus ``in_top_n``.
    """
    u = cfg["universe"]
    ok = msf["mthcap"].notna() & (msf["mthprc"] >= u["min_price"])
    exch = cfg["universe"]["exchange_filter"]
    if "primaryexch" in msf:
        ok &= msf["primaryexch"].isin(u["exchanges"])
    for col in ("conditionaltype", "tradingstatusflg"):
        if col in msf:
            ok &= msf[col].isin(exch[col])
    if u["exclude_financials"] and "siccd" in msf:
        lo, hi = u["financial_sic_range"]
        ok &= ~msf["siccd"].between(lo, hi).fillna(False).astype(bool)
    out = msf.assign(crsp_ok=ok.astype(bool))
    cap = out["mthcap"].where(out["crsp_ok"])
    out["size_rank"] = cap.groupby(out["month"]).rank(ascending=False, method="first")
    out["in_top_n"] = out["size_rank"] <= u["n_stocks"]
    return out


def check_msf(msf: pd.DataFrame, cfg: dict) -> list[Check]:
    out: list[Check] = []
    n = len(msf)
    dup = int(msf.duplicated(["permno", "month"]).sum())
    out.append(check("CRSP1", "One row per (permno, month)", dup == 0,
                     f"{dup:,} duplicate rows of {n:,}. If non-zero, the monthly view repeats "
                     "stock-months with conflicting values even after DISTINCT"))
    nonpos = int((msf["mthprc"] <= 0).sum())
    out.append(check("CRSP2", "Prices positive where present", nonpos == 0,
                     f"{nonpos:,} rows with mthprc <= 0; {msf['mthprc'].isna().sum():,} NULL"))
    r = msf["mthret"]
    below = int((r < -1).sum())
    out.append(check("CRSP3", "Monthly returns are >= -100%", below == 0,
                     f"{below:,} below -1; {int((r == -1).sum()):,} exactly -1 (total loss)"))
    share = float((r > 3).mean())
    status = FAIL if share > 0.01 else (FLAG if share > 0.0001 else PASS)
    out.append(Check("CRSP4", "Share of returns above 300% is under 0.01% (FAIL above 1%, "
                     "which would mean percent units)", status,
                     f"{int((r > 3).sum()):,} rows, {share:.4%}. Returns are never winsorized"))
    out.append(Check("CRSP5", "NULL returns by return flag (spec 3.4 diagnostics)", INFO,
                     _null_by_flag(msf)))
    last = msf.sort_values("mthcaldt").groupby("permno").tail(1)
    if "mthretflg" in msf:
        de = last[last["mthretflg"] == "DE"]
        share_de = de["mthret"].notna().mean() if len(de) else float("nan")
        out.append(check("CRSP6", "Delisting months (mthretflg DE) carry a return",
                         len(de) == 0 or share_de >= 0.9,
                         f"{len(de):,} permnos end with a DE row; {share_de:.1%} have a "
                         "non-null mthret", FLAG))
    last_month = msf["month"].max()
    top = msf.loc[msf["month"] == last_month, "mthcap"].max() * 1000
    out.append(check("CRSP7", "mthcap is in thousands (largest firm in the trillions)",
                     1e12 <= top < 1e14, f"largest on {last_month}: ${top / 1e12:,.2f} trillion"))
    months = msf.groupby(msf["month"].dt.year)["month"].nunique()
    gaps = months[(months < 12) & (months.index < last_month.year)
                  & (months.index > msf["month"].min().year)]
    out.append(check("CRSP8", "Every year has 12 months of data", gaps.empty,
                     f"range {msf['month'].min()} to {last_month}; short years: "
                     f"{gaps.to_dict() or 'none'}"))
    out.append(_stocks_per_year(crsp_screen(msf, cfg), cfg))
    return out


def _null_by_flag(msf: pd.DataFrame) -> str:
    if "mthretflg" not in msf:
        return f"{msf['mthret'].isna().sum():,} NULL returns; no mthretflg column"
    t = msf.groupby("mthretflg", observed=True)["mthret"].agg(
        rows="size", null=lambda s: int(s.isna().sum()))
    return "; ".join(f"{k}: {int(v['rows']):,} rows, {int(v['null']):,} NULL"
                     for k, v in t.iterrows())


def _stocks_per_year(screened: pd.DataFrame, cfg: dict) -> Check:
    dec = screened[screened["month"].dt.month == 12]
    start = pd.Period(cfg["sample"]["signal_start"], "M").year
    dec = dec[dec["month"].dt.year >= start - 1]
    counts = dec[dec["crsp_ok"]].groupby(dec["month"].dt.year)["permno"].nunique()
    drops = counts.pct_change()
    bad = drops[drops < -0.30]
    series = ", ".join(f"{y}: {c:,}" for y, c in counts.items())
    return check("CRSP9", "Stocks passing the CRSP screen each December are stable (no year "
                 "drops more than 30%)", bad.empty,
                 f"{series}. Drops over 30%: {bad.round(3).to_dict() or 'none'}")


def coverage_years(df: pd.DataFrame, flag: str, cfg: dict) -> pd.Series:
    """Share of top-N stocks each December with ``flag`` true, from the signal start year."""
    start = pd.Period(cfg["sample"]["signal_start"], "M").year
    dec = df[(df["month"].dt.month == 12) & df["in_top_n"] & (df["month"].dt.year >= start)]
    return dec.groupby(dec["month"].dt.year)[flag].mean()

