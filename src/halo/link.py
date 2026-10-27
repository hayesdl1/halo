"""CRSP/Compustat link (spec 3.5): pull, load, point-in-time permno to gvkey."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from halo.checks import INFO, Check, check
from halo.pull import resolve_columns, select_sql, sql_list
from halo.schema import qualified, table_entry

CCM_FILE = "ccm_link.parquet"
CCM_COLUMNS = ["gvkey", "lpermno", "linktype", "linkprim", "linkdt", "linkenddt"]


def pull_ccm(db, smap: dict, cfg: dict, out_dir: Path, log=print) -> pd.DataFrame:
    m = resolve_columns(db, smap, "ccm", CCM_COLUMNS, CCM_COLUMNS, log)
    sql = (f"SELECT {select_sql(m)} FROM {qualified(table_entry(smap, 'ccm'))} "
           f'WHERE "{m["linktype"]}" IN ({sql_list(cfg["link"]["pull_linktypes"])}) '
           f'AND "{m["linkprim"]}" IN ({sql_list(cfg["link"]["linkprim"])}) '
           f'AND "{m["lpermno"]}" IS NOT NULL')
    df = db.raw_sql(sql, date_cols=["linkdt", "linkenddt"])
    lp = pd.to_numeric(df["lpermno"])
    if not np.allclose(lp, lp.round()):
        raise ValueError("lpermno has non-integer values; cannot cast to permno")
    df = df.drop(columns="lpermno").assign(permno=lp.round().astype("int64"))
    df["gvkey"] = df["gvkey"].astype(str).str.strip().str.zfill(6)
    for c in ["linkdt", "linkenddt"]:
        df[c] = pd.to_datetime(df[c]).astype("datetime64[ns]")
    for c in ["linktype", "linkprim"]:
        df[c] = df[c].astype("string").astype("category")
    df = df[["gvkey", "permno", "linktype", "linkprim", "linkdt", "linkenddt"]]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_parquet(Path(out_dir) / CCM_FILE, index=False)
    df.attrs["mapping"] = m
    return df


def load_ccm(raw_dir: Path) -> pd.DataFrame:
    path = Path(raw_dir) / CCM_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/01_pull_crsp.py")
    return pd.read_parquet(path)


def link_gvkey(keys: pd.DataFrame, ccm: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, int]:
    """gvkey valid for each (permno, month) in ``keys``, as of that month-end.

    A link is valid when linkdt <= month-end and (linkenddt is NULL or month-end <=
    linkenddt). If more than one gvkey is valid, keep the primary (P) link, then the
    most recent linkdt. Returns the keys with a ``gvkey`` column (NaN where unlinked)
    and the number of (permno, month) pairs that needed the tie-break.
    """
    links = ccm[ccm["linktype"].isin(cfg["link"]["linktypes"])
                & ccm["linkprim"].isin(cfg["link"]["linkprim"])]
    k = keys[["permno", "month"]].drop_duplicates()
    asof = k["month"].dt.to_timestamp(how="end").dt.normalize()
    k = k.assign(_asof=asof.values)
    m = k.merge(links[["permno", "gvkey", "linkprim", "linkdt", "linkenddt"]], on="permno")
    valid = (m["linkdt"] <= m["_asof"]) & (m["linkenddt"].isna() | (m["_asof"] <= m["linkenddt"]))
    m = m[valid]
    m = m.assign(_p=(m["linkprim"] == "P").astype(int))
    m = m.sort_values(["permno", "month", "_p", "linkdt"], ascending=[True, True, False, False])
    n_multi = int((m.groupby(["permno", "month"])["gvkey"].nunique() > 1).sum())
    best = m.drop_duplicates(["permno", "month"])[["permno", "month", "gvkey"]]
    out = keys.merge(best, on=["permno", "month"], how="left")
    return out, n_multi


def check_ccm(ccm: pd.DataFrame) -> list[Check]:
    lens = ccm["gvkey"].str.len()
    out = [
        check("CCM1", "gvkey is a 6-character string", bool((lens == 6).all()),
              f"lengths {lens.min()} to {lens.max()}"),
        check("CCM2", "No duplicate link rows", not ccm.duplicated().any(),
              f"{int(ccm.duplicated().sum()):,} duplicates of {len(ccm):,}"),
        check("CCM3", "linkdt <= linkenddt where both exist",
              bool((ccm["linkenddt"].isna() | (ccm["linkdt"] <= ccm["linkenddt"])).all()),
              f"{int((ccm['linkdt'] > ccm['linkenddt']).sum()):,} reversed"),
    ]
    vc = ccm.groupby(["linktype", "linkprim"], observed=True).size()
    out.append(Check("CCM4", "Links pulled by type and primacy", INFO,
                     ", ".join(f"{t}/{p}: {n:,}" for (t, p), n in vc.items())
                     + f"; {int(ccm['linkenddt'].isna().sum()):,} active (NULL linkenddt)"))
    return out
