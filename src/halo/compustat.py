"""Compustat annual fundamentals (spec 3.6): pull, load, as-of attachment, checks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from halo.checks import INFO, Check, check
from halo.pull import ph, resolve_columns, select_sql, sql_list
from halo.schema import qualified, table_entry

FUNDA_FILE = "comp_funda.parquet"
BASE_ITEMS = ["at", "che", "intan", "sale", "ppent", "ppegt", "emp", "capx", "ebitda", "oibdp",
              "dp", "xrd", "sich"]
METRIC_INPUTS = ["at", "che", "intan", "sale", "ppent", "emp", "capx", "ebitda"]
REQUIRED = ["gvkey", "datadate", "at", "sale", "ppent", "capx", "emp", "indfmt", "datafmt",
            "popsrc", "consol"]
STD_FILTER = {"indfmt": "INDL", "datafmt": "STD", "popsrc": "D", "consol": "C"}


def pull_funda(db, smap: dict, cfg: dict, out_dir: Path, log=print) -> pd.DataFrame:
    items = BASE_ITEMS + list(cfg["fundamentals"]["control_items"])
    wanted = ["gvkey", "datadate", "fyear", *items, *STD_FILTER]
    m = resolve_columns(db, smap, "funda", wanted, REQUIRED, log)
    conds = [f'"{m[k]}" IN ({sql_list([v])})' for k, v in STD_FILTER.items()]
    conds.append(f'"{m["datadate"]}" >= {ph(db, "start")}')
    cols = {c: a for c, a in m.items() if c not in STD_FILTER}
    sql = (f"SELECT {select_sql(cols)} FROM {qualified(table_entry(smap, 'funda'))} "
           f"WHERE {' AND '.join(conds)}")
    df = db.raw_sql(sql, date_cols=["datadate"],
                    params={"start": cfg["sample"]["compustat_min_datadate"]})
    df["gvkey"] = df["gvkey"].astype(str).str.strip().str.zfill(6)
    df["datadate"] = pd.to_datetime(df["datadate"]).astype("datetime64[ns]")
    if "fyear" in df:
        df["fyear"] = pd.to_numeric(df["fyear"], errors="coerce").astype("Int64")
    for c in items:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_parquet(Path(out_dir) / FUNDA_FILE, index=False)
    df.attrs["mapping"] = cols
    return df


def load_funda(raw_dir: Path) -> pd.DataFrame:
    path = Path(raw_dir) / FUNDA_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/02_pull_compustat.py")
    return pd.read_parquet(path)


def _month_index(p: pd.Series) -> pd.Series:
    return p.dt.year * 12 + p.dt.month - 1


def attach_fundamentals(keys: pd.DataFrame, funda: pd.DataFrame, lag_months: int,
                        expiry_months: int) -> pd.DataFrame:
    """Latest fundamentals usable at each (gvkey, month) in ``keys``. No look-ahead.

    A record with ``datadate`` D becomes usable at month-end D + lag and is dropped once
    the month is more than ``expiry_months`` after D. Adds ``fundamentals_datadate``,
    ``fundamentals_age_months`` and every column of ``funda``. Rows without a usable
    record get NaN.
    """
    f = funda.copy()
    f["_avail"] = _month_index(f["datadate"].dt.to_period("M")) + lag_months
    # Two records can become usable in the same month after a fiscal year-end change.
    f = f.sort_values(["gvkey", "_avail", "datadate"]).drop_duplicates(["gvkey", "_avail"],
                                                                       keep="last")
    k = keys.copy()
    k["_m"] = _month_index(k["month"])
    k["_row"] = range(len(k))
    linked = k[k["gvkey"].notna()].sort_values("_m")
    merged = pd.merge_asof(linked, f.sort_values("_avail"), left_on="_m", right_on="_avail",
                           by="gvkey", direction="backward")
    age = merged["_m"] - _month_index(merged["datadate"].dt.to_period("M"))
    expired = age > expiry_months
    fcols = [c for c in f.columns if c not in ("gvkey", "_avail")]
    merged.loc[expired, fcols] = pd.NA
    merged["fundamentals_age_months"] = age.where(~expired)
    merged = merged.rename(columns={"datadate": "fundamentals_datadate"})
    out = k.merge(merged.drop(columns=[c for c in k.columns if c not in ("_row",)]),
                  on="_row", how="left")
    return out.drop(columns=["_m", "_row", "_avail"]).reset_index(drop=True)


def check_funda(funda: pd.DataFrame) -> list[Check]:
    dup = int(funda.duplicated(["gvkey", "datadate"]).sum())
    out = [
        check("COMP1", "One record per (gvkey, datadate) after the standard filters", dup == 0,
              f"{dup:,} duplicates of {len(funda):,}"),
        check("COMP2", "gvkey is a 6-character string",
              bool((funda["gvkey"].str.len() == 6).all()),
              f"lengths {funda['gvkey'].str.len().min()} to {funda['gvkey'].str.len().max()}"),
    ]
    yr = funda.groupby(funda["datadate"].dt.year)
    share = yr[[c for c in METRIC_INPUTS if c in funda]].agg(lambda s: s.notna().mean())
    lines = [f"{y}: " + ", ".join(f"{c} {v:.0%}" for c, v in row.items())
             for y, row in share.iterrows()]
    out.append(Check("COMP3", "Non-missing share of each metric input by datadate year "
                     "(all Compustat firms, before the universe screen)", INFO,
                     " | ".join(lines)))
    return out
