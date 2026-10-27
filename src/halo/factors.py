"""Fama-French factors from WRDS (spec 3.7) and Treasury/corporate yields from FRED."""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import pandas as pd

from halo.checks import INFO, Check, check
from halo.pull import resolve_columns, select_sql
from halo.schema import qualified, table_entry

FACTORS_FILE = "ff_factors.parquet"
RATES_FILE = "fred_rates.parquet"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


def pull_ff(db, smap: dict, out_dir: Path, log=print) -> pd.DataFrame:
    """Monthly MKT-RF, SMB, HML, RMW, CMA, UMD, RF in decimals, keyed by ``month``."""
    units = smap["decisions"].get("factor_units", {})
    frames = []
    for concept, wanted, required in [
        ("ff_monthly", ["date", "mktrf", "smb", "hml", "rf", "umd"], ["date", "mktrf", "rf"]),
        ("ff5_monthly", ["date", "rmw", "cma"], ["date", "rmw", "cma"]),
    ]:
        m = resolve_columns(db, smap, concept, wanted, required, log)
        df = db.raw_sql(f"SELECT {select_sql(m)} FROM {qualified(table_entry(smap, concept))}",
                        date_cols=["date"])
        num = [c for c in df.columns if c != "date"]
        df[num] = df[num].apply(pd.to_numeric, errors="coerce").astype("float64")
        if units.get(concept) == "percent":
            df[num] = df[num] / 100
        df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
        frames.append(df.drop(columns="date"))
    out = frames[0].merge(frames[1], on="month", how="left").sort_values("month")
    out = out[["month", "mktrf", "smb", "hml", "rmw", "cma", "umd", "rf"]
              if "umd" in out else ["month", "mktrf", "smb", "hml", "rmw", "cma", "rf"]]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out.to_parquet(Path(out_dir) / FACTORS_FILE, index=False)
    return out.reset_index(drop=True)


def _fetch_url(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (halo-strategy)"})
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 (fixed https FRED URL)
        return r.read().decode()


def parse_fred_csv(text: str, name: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text))
    date_col = df.columns[0]           # "observation_date" now, "DATE" in older files
    val = pd.to_numeric(df[df.columns[1]], errors="coerce")   # "." marks missing
    return pd.DataFrame({"month": pd.to_datetime(df[date_col]).dt.to_period("M"),
                         name: val / 100})


def pull_fred(cfg: dict, out_dir: Path, fetch=_fetch_url, log=print) -> pd.DataFrame:
    """Monthly yields in decimals, term and default spreads, and their monthly changes."""
    out = None
    for name, series in cfg["rates"]["fred_series"].items():
        df = parse_fred_csv(fetch(FRED_URL.format(series=series)), name)
        log(f"  FRED {series}: {df['month'].min()} to {df['month'].max()}")
        out = df if out is None else out.merge(df, on="month", how="outer")
    out = out.sort_values("month").reset_index(drop=True)
    out["term"] = out["gs10"] - out["tb3ms"]
    out["default"] = out["baa"] - out["aaa"]
    out["d_term"] = out["term"].diff()
    out["d_def"] = out["default"].diff()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out.to_parquet(Path(out_dir) / RATES_FILE, index=False)
    return out


def load_factors(raw_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(Path(raw_dir) / FACTORS_FILE)


def load_rates(raw_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(Path(raw_dir) / RATES_FILE)


def check_factors(ff: pd.DataFrame, cfg: dict) -> list[Check]:
    # Factors are needed for every holding month, from the first return month on.
    start = pd.Period(cfg["sample"]["return_start"], "M")
    s = ff[ff["month"] >= start]
    months = pd.period_range(start, s["month"].max(), freq="M")
    missing = [str(x) for x in sorted(set(months) - set(s["month"]))]
    return [
        check("FF1", "rf is between 0 and 0.02 per month from the first return month",
              bool(s["rf"].between(0, 0.02).all()),
              f"rf {s['rf'].min():.4f} to {s['rf'].max():.4f}"),
        check("FF2", "Factors are in decimals (mean |mktrf| under 0.2)",
              float(s["mktrf"].abs().mean()) < 0.2, f"mean |mktrf| {s['mktrf'].abs().mean():.4f}"),
        check("FF3", "No missing months or factor values from the first return month",
              not missing and not s.drop(columns="month").isna().any().any(),
              f"{s['month'].min()} to {s['month'].max()}; missing months: {missing[:5] or 'none'}"
              f"; NULLs: {s.drop(columns='month').isna().sum().to_dict()}"),
    ]


def check_rates(rates: pd.DataFrame, cfg: dict) -> list[Check]:
    start = pd.Period(cfg["sample"]["signal_start"], "M")
    s = rates[rates["month"] >= start]
    # The newest month or two may be published for some series only. Check up to the last
    # month where all four exist; later code only uses months with all four.
    complete = s[["gs10", "tb3ms", "baa", "aaa"]].notna().all(axis=1)
    trailing = 0
    if complete.any():
        last = complete[complete].index[-1]
        trailing = int((s.index > last).sum())
        s = s.loc[:last]
    levels = s[["gs10", "tb3ms", "baa", "aaa"]]
    return [
        check("RATE1", "Yields are decimals between -0.01 and 0.25",
              bool(((levels > -0.01) & (levels < 0.25)).all().all()),
              f"min {levels.min().round(4).to_dict()}, max {levels.max().round(4).to_dict()}"),
        check("RATE2", "No missing yields from the signal start", not levels.isna().any().any(),
              f"NULLs: {levels.isna().sum().to_dict()}; range {s['month'].min()} to "
              f"{s['month'].max()}; {trailing} newer incomplete months ignored"),
        Check("RATE3", "Spread ranges", INFO,
              f"term {s['term'].min():.4f} to {s['term'].max():.4f}; default "
              f"{s['default'].min():.4f} to {s['default'].max():.4f}"),
    ]
