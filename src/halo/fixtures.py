"""Synthetic WRDS-shaped data for tests and CI. Never touches WRDS.

`make_fixture` builds the five raw tables the pull scripts will produce, in the
standard WRDS CIZ column names from HALO_STRATEGY_SPEC.md section 3:

- ``msf``: CRSP monthly stock file (``permno, mthcaldt, mthret, mthretx, mthprc,
  mthcap, mthprcflg``). ``mthcaldt`` is the last business day of the month, prices are
  positive, ``mthcap`` is in thousands of dollars, missing returns are NaN, and the
  delisting return is already folded into the last month's ``mthret``.
- ``secinfo``: security info history (``permno, secinfostartdt, secinfoenddt, ticker,
  siccd`` and the eight CIZ filter columns). Some firms change SIC mid-sample, and a
  few non-common securities (ADRs, ETFs, units) are included so universe filters
  have something to remove.
- ``ccm``: CCM link history (``gvkey, lpermno, linktype, linkprim, linkdt,
  linkenddt``). ``linkenddt`` is NaT for active links. Includes non-standard links
  (``NR``, ``LX``, ``J``) and a few linking gaps.
- ``funda``: Compustat annual (``gvkey, datadate, fyear``, the dollar items in
  millions, ``emp`` in thousands, and the four standard filter columns). Fiscal year
  ends vary by firm. Some inputs are missing and some EBITDA is negative.
- ``factors``: Fama-French five factors plus momentum and rf, in decimals, dated at
  calendar month-end.

Column names are the standard CIZ names. Milestone 1 maps the user's schema onto
these names through config/schema_map.yaml, so code downstream of the pull scripts
only ever sees these names.

Returns are generated as market beta times the market factor, plus a sector shock,
plus idiosyncratic noise, plus an optional planted premium. The planted premium for
metric ``k`` is ``strength * (rank_k - 0.5)`` where ``rank_k`` is the cross-sectional
percentile rank of the firm's metric ``k``, computed from the fundamentals the firm
had published at least ``lag_months`` earlier. With ``planted=None`` (the null
fixture) no characteristic predicts returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from halo import METRICS

# Sector archetypes. Ratios are medians; firms draw around them with lognormal noise.
# turnover = sale/at, ppe = ppent/at, cash = che/at, intan = intan/at,
# capx = capx/sale, margin = ebitda/sale, ape = at ($m) per thousand employees,
# rd = xrd/sale, dep = dp/ppegt.
SECTORS: dict[str, dict] = {
    "utilities": dict(sic=[4911, 4931, 4941], turnover=0.25, ppe=0.70, cash=0.02, intan=0.05,
                      capx=0.30, margin=0.35, ape=2500.0, rd=0.0, dep=0.035, beta=0.5),
    "pipelines": dict(sic=[4922, 4923], turnover=0.35, ppe=0.65, cash=0.03, intan=0.08,
                      capx=0.25, margin=0.30, ape=3000.0, rd=0.0, dep=0.04, beta=0.8),
    "railroads": dict(sic=[4011, 4013], turnover=0.35, ppe=0.75, cash=0.03, intan=0.02,
                      capx=0.18, margin=0.40, ape=1200.0, rd=0.0, dep=0.035, beta=0.9),
    "oil_gas": dict(sic=[1311, 2911], turnover=0.45, ppe=0.60, cash=0.05, intan=0.02,
                    capx=0.25, margin=0.30, ape=4000.0, rd=0.002, dep=0.07, beta=1.1),
    "manufacturing": dict(sic=[3312, 3711, 3714, 3560], turnover=0.90, ppe=0.30, cash=0.08,
                          intan=0.15, capx=0.05, margin=0.13, ape=350.0, rd=0.03, dep=0.08,
                          beta=1.1),
    "retail": dict(sic=[5311, 5411, 5812], turnover=1.80, ppe=0.35, cash=0.08, intan=0.10,
                   capx=0.04, margin=0.08, ape=40.0, rd=0.0, dep=0.09, beta=0.9),
    "pharma": dict(sic=[2834, 2836], turnover=0.55, ppe=0.12, cash=0.20, intan=0.35,
                   capx=0.04, margin=0.25, ape=900.0, rd=0.15, dep=0.09, beta=0.8),
    "software": dict(sic=[7372, 7373], turnover=0.60, ppe=0.04, cash=0.30, intan=0.40,
                     capx=0.03, margin=0.25, ape=500.0, rd=0.20, dep=0.25, beta=1.3),
    "data_processing": dict(sic=[7374, 7370], turnover=0.80, ppe=0.06, cash=0.20, intan=0.40,
                            capx=0.04, margin=0.22, ape=150.0, rd=0.10, dep=0.20, beta=1.2),
    "services": dict(sic=[8742, 7380, 8711], turnover=1.30, ppe=0.08, cash=0.12, intan=0.35,
                     capx=0.02, margin=0.12, ape=60.0, rd=0.01, dep=0.15, beta=1.0),
    "banks": dict(sic=[6021, 6022], turnover=0.07, ppe=0.01, cash=0.10, intan=0.03,
                  capx=0.03, margin=0.35, ape=8000.0, rd=0.0, dep=0.10, beta=1.1),
    "insurance": dict(sic=[6311, 6331], turnover=0.25, ppe=0.01, cash=0.05, intan=0.05,
                      capx=0.01, margin=0.12, ape=2500.0, rd=0.0, dep=0.10, beta=0.9),
}

FINANCIAL_SECTORS = ("banks", "insurance")


@dataclass
class FixtureConfig:
    n_stocks: int = 200
    start: str = "1996-01"           # first CRSP month
    n_years: int = 20
    fundamentals_start_year: int = 1994
    seed: int = 0
    n_noncommon: int = 10            # ADRs, ETFs, units: must be removed by universe filters
    planted: dict[str, float] | None = None   # e.g. {"clr": 0.02} for 2% a month top vs bottom
    lag_months: int = 6              # lag the planted premium uses to read fundamentals
    idio_vol: float = 0.08           # monthly idiosyncratic volatility
    sector_vol: float = 0.03
    p_missing_return: float = 0.001  # NULL mthret in a non-delisting month
    p_delist: float = 0.25           # share of firms that delist before the end
    p_late_ipo: float = 0.25         # share of firms that list after the start
    missing_rates: dict[str, float] = field(default_factory=lambda: {
        "intan": 0.05, "che": 0.01, "capx": 0.02, "emp": 0.03, "ebitda": 0.03, "xrd": 0.30,
    })


@dataclass
class Fixture:
    msf: pd.DataFrame
    secinfo: pd.DataFrame
    ccm: pd.DataFrame
    funda: pd.DataFrame
    factors: pd.DataFrame
    truth: pd.DataFrame              # one row per firm: sector, permno, gvkey, listing dates
    config: FixtureConfig

    def tables(self) -> dict[str, pd.DataFrame]:
        return {"msf": self.msf, "secinfo": self.secinfo, "ccm": self.ccm,
                "funda": self.funda, "factors": self.factors}

    def write(self, directory) -> None:
        """Write each table as parquet, mirroring what the pull scripts write to data/raw."""
        from pathlib import Path

        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for name, df in self.tables().items():
            df.to_parquet(d / f"{name}.parquet", index=False)


def _sector_assignment(rng: np.random.Generator, n: int) -> np.ndarray:
    names = list(SECTORS)
    # Roughly match a US large-cap mix: more manufacturing, services, retail, software.
    weights = np.array([6, 3, 2, 6, 18, 10, 8, 10, 6, 11, 12, 8], dtype=float)
    return rng.choice(names, size=n, p=weights / weights.sum())


def _month_ends(start: str, n_months: int) -> pd.PeriodIndex:
    return pd.period_range(start=start, periods=n_months, freq="M")


def _last_business_day(periods: pd.PeriodIndex) -> pd.DatetimeIndex:
    """Stand-in for CRSP mthcaldt: last weekday of the month (holidays ignored)."""
    return periods.to_timestamp(how="start") + pd.offsets.BMonthEnd(0)


def _metric_values(f: pd.DataFrame) -> pd.DataFrame:
    """Plain metric arithmetic used only to plant the return premium in the fixture.

    Deliberately independent of halo.metrics so tests compare two implementations.
    """
    tang = f["at"] - f["che"] - f["intan"].fillna(0.0)
    ebitda = f["ebitda"].fillna(f["oibdp"])
    sale = f["sale"].where(f["sale"] > 0)
    return pd.DataFrame({
        "tai": tang / sale,
        "fai": f["ppent"] / sale,
        "fas": f["ppent"] / f["at"].where(f["at"] > 0),
        "clr": tang / f["emp"].where(f["emp"] > 0),
        "cxi": f["capx"] / sale,
        "cxl": f["capx"] / ebitda.where(ebitda > 0),
    })


def make_fixture(cfg: FixtureConfig | None = None, **kwargs) -> Fixture:
    """Build a reproducible synthetic dataset. Keyword arguments override FixtureConfig."""
    cfg = cfg or FixtureConfig(**kwargs)
    if cfg.planted:
        unknown = set(cfg.planted) - set(METRICS)
        assert not unknown, f"planted metrics must be in {METRICS}, got {sorted(unknown)}"
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_stocks
    n_months = cfg.n_years * 12
    months = _month_ends(cfg.start, n_months)
    first_m, last_m = months[0], months[-1]

    # ---- firms --------------------------------------------------------------------
    sector = _sector_assignment(rng, n)
    permno = 10001 + np.arange(n) * 7
    gvkey = np.array([f"{1001 + i * 3:06d}" for i in range(n)])
    fye_month = rng.choice([12, 6, 9, 3], size=n, p=[0.65, 0.15, 0.1, 0.1])

    # Listing window, as month offsets from the sample start.
    list_start = np.where(rng.random(n) < cfg.p_late_ipo,
                          rng.integers(1, max(2, n_months - 36), size=n), 0)
    delists = rng.random(n) < cfg.p_delist
    life = rng.integers(min(24, n_months // 2), n_months, size=n)
    list_end = np.where(delists, np.minimum(n_months - 1, list_start + life), n_months - 1)
    delists &= list_end < n_months - 1

    # Firm-level log deviations from the sector medians. Independent across ratios so the
    # six metrics are correlated through sector but not identical within sector.
    ratio_keys = ["turnover", "ppe", "cash", "intan", "capx", "margin", "ape", "rd", "dep"]
    firm_dev = {k: rng.normal(0, 0.35, size=n) for k in ratio_keys}
    beta = np.array([SECTORS[s]["beta"] for s in sector]) * np.exp(rng.normal(0, 0.2, size=n))
    size0 = np.exp(rng.normal(np.log(3000), 1.4, size=n))   # total assets at start, $m

    # ---- Compustat annual ---------------------------------------------------------
    fyears = np.arange(cfg.fundamentals_start_year, int(str(last_m)[:4]) + 1)
    rows = []
    for i in range(n):
        s = SECTORS[sector[i]]
        at = size0[i]
        start_fy = cfg.fundamentals_start_year + max(0, list_start[i] // 12 - 2)
        end_period = months[list_end[i]]
        for fy in fyears:
            if fy < start_fy:
                continue
            # Compustat convention: fiscal years ending Jan-May belong to the prior fyear.
            dd_year = fy + 1 if fye_month[i] <= 5 else fy
            datadate = pd.Period(year=dd_year, month=fye_month[i], freq="M").to_timestamp(how="end")
            datadate = datadate.normalize()
            if pd.Period(datadate, "M") > end_period:
                break
            at *= np.exp(rng.normal(0.05, 0.12))
            def draw(k, sd=0.10, i=i, s=s):
                return s[k] * np.exp(firm_dev[k][i] + rng.normal(0, sd))
            ppe_share = min(draw("ppe"), 0.9)
            cash_share = min(draw("cash"), 0.6)
            intan_share = min(draw("intan"), 0.9 - ppe_share)
            cash_share = min(cash_share, max(0.0, 0.95 - ppe_share - intan_share))
            sale = at * draw("turnover")
            margin = s["margin"] * np.exp(firm_dev["margin"][i]) + rng.normal(0, 0.06)
            ebitda = sale * margin
            ppent = at * ppe_share
            ppegt = ppent * np.exp(rng.normal(np.log(1.7), 0.1))
            rows.append(dict(
                gvkey=gvkey[i], datadate=datadate, fyear=fy,
                at=at, che=at * cash_share, intan=at * intan_share, sale=sale,
                ppent=ppent, ppegt=ppegt, emp=at / draw("ape", 0.05),
                capx=sale * draw("capx", 0.2), ebitda=ebitda,
                oibdp=ebitda * np.exp(rng.normal(0, 0.01)), dp=ppegt * draw("dep", 0.05),
                xrd=sale * draw("rd") if s["rd"] > 0 else np.nan,
                sich=float(SECTORS[sector[i]]["sic"][0]),
            ))
    funda = pd.DataFrame(rows)
    for col, p in cfg.missing_rates.items():
        funda.loc[rng.random(len(funda)) < p, col] = np.nan
    # A handful of non-positive sales and a few firms with no sich in early years.
    funda.loc[rng.random(len(funda)) < 0.003, "sale"] = 0.0
    funda.loc[(funda["fyear"] < 1998) & (rng.random(len(funda)) < 0.3), "sich"] = np.nan
    funda["indfmt"], funda["datafmt"], funda["popsrc"], funda["consol"] = "INDL", "STD", "D", "C"
    # Duplicate a few records in the FS format so the INDL filter matters.
    fs = funda.sample(frac=0.02, random_state=cfg.seed).assign(indfmt="FS")
    funda = pd.concat([funda, fs], ignore_index=True)
    funda = funda.sort_values(["gvkey", "datadate", "indfmt"], ignore_index=True)

    # ---- factors ------------------------------------------------------------------
    fac = pd.DataFrame({
        "mktrf": rng.normal(0.007, 0.045, n_months),
        "smb": rng.normal(0.001, 0.03, n_months),
        "hml": rng.normal(0.002, 0.03, n_months),
        "rmw": rng.normal(0.003, 0.02, n_months),
        "cma": rng.normal(0.002, 0.02, n_months),
        "umd": rng.normal(0.005, 0.045, n_months),
        "rf": np.clip(rng.normal(0.002, 0.0008, n_months), 0.0, 0.006),
    })
    fac.insert(0, "date", months.to_timestamp(how="end").normalize())

    # ---- planted premium ----------------------------------------------------------
    premium = np.zeros((n_months, n))
    if cfg.planted:
        std = funda[funda["indfmt"] == "INDL"]
        mv = pd.concat([std[["gvkey", "datadate"]], _metric_values(std)], axis=1)
        mv["avail"] = (mv["datadate"].dt.to_period("M") + cfg.lag_months)
        gv_index = {g: j for j, g in enumerate(gvkey)}
        for metric, strength in cfg.planted.items():
            grid = np.full((n_months, n), np.nan)
            for _, r in mv.dropna(subset=[metric]).iterrows():
                a = (r["avail"] - first_m).n
                if a < n_months:
                    grid[max(a, 0):, gv_index[r["gvkey"]]] = r[metric]
            # grid[t] holds the value known at month-end t; it predicts the return in t+1.
            known = pd.DataFrame(grid).rank(axis=1, pct=True).to_numpy()
            prem = strength * (np.nan_to_num(known, nan=0.5) - 0.5)
            premium[1:] += prem[:-1]

    # ---- CRSP monthly -------------------------------------------------------------
    sector_names = list(SECTORS)
    sector_shock = rng.normal(0, cfg.sector_vol, (n_months, len(sector_names)))
    sec_idx = np.array([sector_names.index(s) for s in sector])
    rf = fac["rf"].to_numpy()[:, None]
    ret = (rf + beta[None, :] * fac["mktrf"].to_numpy()[:, None]
           + sector_shock[:, sec_idx] + rng.normal(0, cfg.idio_vol, (n_months, n)) + premium)
    ret = np.maximum(ret, -0.95)
    for i in np.where(delists)[0]:
        ret[list_end[i], i] = rng.choice([-0.30, -0.55, 0.15])   # delisting return folded in

    # Shares outstanding (thousands) sized so mthcap tracks book assets loosely.
    p0 = np.exp(rng.normal(np.log(30), 0.8, size=n))
    shrout0 = size0 * 1000 * np.exp(rng.normal(0.2, 0.5, size=n)) / p0
    mcal = _last_business_day(months)
    recs = []
    for i in range(n):
        price = p0[i]
        shrout = shrout0[i]
        for t in range(list_start[i], list_end[i] + 1):
            r = ret[t, i]
            if t > list_start[i]:
                price *= 1 + r - 0.002   # mthretx excludes a small dividend yield
            shrout *= 1 + max(0.0, rng.normal(0.001, 0.003))
            recs.append((permno[i], mcal[t], r, r - 0.002, price, price * shrout))
    msf = pd.DataFrame(recs, columns=["permno", "mthcaldt", "mthret", "mthretx", "mthprc",
                                      "mthcap"])
    # First listed month has no prior price, so its return is NULL, as in CRSP.
    first_rows = msf.groupby("permno").head(1).index
    msf.loc[first_rows, ["mthret", "mthretx"]] = np.nan
    last_rows = set(msf.groupby("permno").tail(1).index)
    miss = (rng.random(len(msf)) < cfg.p_missing_return) & ~msf.index.isin(list(last_rows))
    msf.loc[miss, ["mthret", "mthretx"]] = np.nan
    msf["mthprcflg"] = pd.Categorical(
        rng.choice(["TR", "BA"], size=len(msf), p=[0.97, 0.03]), categories=["TR", "BA", "DA"]
    )

    # ---- non-common securities ----------------------------------------------------
    nc_permno = 90001 + np.arange(cfg.n_noncommon) * 7
    nc_kind = np.resize(["adr", "etf", "unit", "foreign"], cfg.n_noncommon)
    nc_recs = []
    for j, p in enumerate(nc_permno):
        price, shrout = 40.0, 50_000.0 * (j + 1) * 20
        for t in range(n_months):
            r = rng.normal(0.008, 0.06)
            price *= 1 + r
            nc_recs.append((p, mcal[t], r, r - 0.001, price, price * shrout))
    nc = pd.DataFrame(nc_recs, columns=msf.columns[:-1])
    nc["mthprcflg"] = pd.Categorical(["TR"] * len(nc), categories=["TR", "BA", "DA"])
    msf = pd.concat([msf, nc], ignore_index=True)

    # ---- security info history ----------------------------------------------------
    common = dict(sharetype="NS", securitytype="EQTY", securitysubtype="COM", usincflg="Y",
                  issuertype="CORP", conditionaltype="RW", tradingstatusflg="A")
    exch = rng.choice(["N", "Q", "A"], size=n, p=[0.55, 0.40, 0.05])
    far_future = pd.Timestamp("2099-12-31")
    sec_rows = []
    for i in range(n):
        beg = mcal[list_start[i]].replace(day=1)
        end = mcal[list_end[i]] if delists[i] else far_future
        sic = SECTORS[sector[i]]["sic"]
        base = dict(permno=permno[i], ticker=f"T{i:04d}", primaryexch=exch[i], **common)
        if len(sic) > 1 and rng.random() < 0.15 and list_end[i] - list_start[i] > 60:
            # Point-in-time SIC change inside the same sector.
            mid = mcal[list_start[i] + (list_end[i] - list_start[i]) // 2]
            sec_rows.append(dict(base, secinfostartdt=beg, secinfoenddt=mid, siccd=sic[0]))
            sec_rows.append(dict(base, secinfostartdt=mid + pd.Timedelta(days=1),
                                 secinfoenddt=end, siccd=sic[1]))
        else:
            sec_rows.append(dict(base, secinfostartdt=beg, secinfoenddt=end,
                                 siccd=int(rng.choice(sic))))
    for j, p in enumerate(nc_permno):
        row = dict(permno=p, ticker=f"X{j:03d}", primaryexch="N", siccd=3711,
                   secinfostartdt=mcal[0].replace(day=1), secinfoenddt=far_future, **common)
        kind = nc_kind[j]
        if kind == "adr":
            row.update(sharetype="AD", usincflg="N")
        elif kind == "etf":
            row.update(securitytype="FUND", securitysubtype="ETF", issuertype="FUND", siccd=6726)
        elif kind == "unit":
            row.update(securitysubtype="UNT")
        else:
            row.update(usincflg="N")
        sec_rows.append(row)
    secinfo = pd.DataFrame(sec_rows)
    for col in ["sharetype", "securitytype", "securitysubtype", "usincflg", "issuertype",
                "primaryexch", "conditionaltype", "tradingstatusflg"]:
        secinfo[col] = secinfo[col].astype("category")
    secinfo["siccd"] = secinfo["siccd"].astype("int64")
    secinfo = secinfo[["permno", "secinfostartdt", "secinfoenddt", "ticker", "siccd",
                       "sharetype", "securitytype", "securitysubtype", "usincflg",
                       "issuertype", "primaryexch", "conditionaltype", "tradingstatusflg"]]

    # ---- CCM link -----------------------------------------------------------------
    link_rows = []
    for i in range(n):
        beg = mcal[list_start[i]].replace(day=1) - pd.offsets.MonthBegin(1)
        end = mcal[list_end[i]] if delists[i] else pd.NaT
        u = rng.random()
        if u < 0.05 and list_end[i] - list_start[i] > 48:
            # Gap: link lapses for a year, then resumes as LU.
            g0 = mcal[list_start[i] + 12]
            g1 = mcal[list_start[i] + 24].replace(day=1)
            link_rows.append((gvkey[i], permno[i], "LC", "P", beg, g0))
            link_rows.append((gvkey[i], permno[i], "LU", "P", g1, end))
        elif u < 0.10:
            link_rows.append((gvkey[i], permno[i], "LU", "C", beg, end))
        else:
            link_rows.append((gvkey[i], permno[i], "LC", "P", beg, end))
        if rng.random() < 0.04:
            # Secondary link that the linkprim filter must drop.
            link_rows.append((gvkey[i], permno[i], "LC", "J", beg, end))
    # Links that the linktype filter must drop: a placeholder gvkey with no CRSP match.
    for j in range(3):
        link_rows.append((f"{900001 + j:06d}", permno[j], "NR", "P", mcal[0], pd.NaT))
        link_rows.append((f"{900101 + j:06d}", permno[j], "LX", "P", mcal[0], pd.NaT))
    ccm = pd.DataFrame(link_rows, columns=["gvkey", "lpermno", "linktype", "linkprim", "linkdt",
                                           "linkenddt"])
    ccm["linkdt"] = pd.to_datetime(ccm["linkdt"])
    ccm["linkenddt"] = pd.to_datetime(ccm["linkenddt"])
    for col in ["linktype", "linkprim"]:
        ccm[col] = ccm[col].astype("category")

    # ---- dtypes, spec section 9 ---------------------------------------------------
    msf["permno"] = msf["permno"].astype("int64")
    msf["mthcaldt"] = msf["mthcaldt"].astype("datetime64[ns]")
    for col in ["mthret", "mthretx", "mthprc", "mthcap"]:
        msf[col] = msf[col].astype("float64")
    msf = msf.sort_values(["permno", "mthcaldt"], ignore_index=True)
    secinfo["permno"] = secinfo["permno"].astype("int64")
    for col in ["secinfostartdt", "secinfoenddt"]:
        secinfo[col] = secinfo[col].astype("datetime64[ns]")
    ccm["lpermno"] = ccm["lpermno"].astype("int64")
    ccm["gvkey"] = ccm["gvkey"].astype(str)
    for col in ["linkdt", "linkenddt"]:
        ccm[col] = ccm[col].astype("datetime64[ns]")
    funda["gvkey"] = funda["gvkey"].astype(str)
    funda["datadate"] = funda["datadate"].astype("datetime64[ns]")
    funda["fyear"] = funda["fyear"].astype("int64")
    fac["date"] = fac["date"].astype("datetime64[ns]")

    truth = pd.DataFrame({
        "permno": permno, "gvkey": gvkey, "sector": sector, "fye_month": fye_month,
        "first_month": months[list_start], "last_month": months[list_end], "delisted": delists,
        "financial": np.isin(sector, FINANCIAL_SECTORS), "beta": beta,
    })
    return Fixture(msf=msf, secinfo=secinfo, ccm=ccm, funda=funda, factors=fac, truth=truth,
                   config=cfg)


def to_fake_wrds(fixture: Fixture, layout: str = "standard"):
    """Load a fixture into a SqliteWRDS laid out like a real WRDS account.

    ``standard`` mimics the WRDS ``crsp`` schema: ``crsp.msf_v2`` carrying the CIZ
    filter columns and point-in-time ``siccd``, ``crsp.stksecurityinfohist`` and
    ``crsp.ccmxpf_lnkhist``. ``variant`` mimics the user's ``crsp_a_stock62``
    schema: the stock file holds only price and return columns, and security info
    uses ``securitybegdt`` / ``securityenddt``. Both get ``comp.funda`` and the
    ``ff`` factor tables.
    """
    from halo.wrds_io import SqliteWRDS

    db = SqliteWRDS()
    msf, secinfo = fixture.msf, fixture.secinfo
    if layout == "standard":
        m = msf.merge(secinfo.drop(columns=["ticker"]), on="permno")
        m = m[(m["mthcaldt"] >= m["secinfostartdt"]) & (m["mthcaldt"] <= m["secinfoenddt"])]
        m = m.drop(columns=["secinfostartdt", "secinfoenddt"])
        db.add_table("crsp", "msf_v2", m)
        db.add_table("crsp", "stksecurityinfohist", secinfo)
        db.add_table("crsp", "ccmxpf_lnkhist", fixture.ccm)
    elif layout == "variant":
        db.add_table("crsp_a_stock62", "wrds_msf62v2_query", msf)
        db.add_table("crsp_a_stock62", "stksecurityinfohist62", secinfo.rename(
            columns={"secinfostartdt": "securitybegdt", "secinfoenddt": "securityenddt"}))
        db.add_table("crsp_a_ccm", "ccmxpf_lnkhist", fixture.ccm)
    else:
        raise ValueError(f"unknown layout {layout!r}")
    db.add_table("comp", "funda", fixture.funda)
    fac = fixture.factors
    db.add_table("ff", "factors_monthly", fac[["date", "mktrf", "smb", "hml", "rf", "umd"]])
    db.add_table("ff", "fivefactors_monthly", fac)
    return db
