"""The two M2 pull steps as functions, so tests can run them against SqliteWRDS.

Each returns the printed report and the list of checks. Scripts exit non-zero when any
check FAILs.
"""

from __future__ import annotations

from pathlib import Path

from halo.checks import FAIL, Check, render
from halo.compustat import check_funda, pull_funda
from halo.crsp import check_msf, load_msf, pull_msf, pull_secinfo
from halo.factors import check_factors, check_rates, pull_ff, pull_fred
from halo.link import check_ccm, load_ccm, pull_ccm
from halo.pull import summary
from halo.schema import qualified, table_entry
from halo.validate import cross_checks


def step_crsp(db, smap: dict, cfg: dict, raw_dir: Path, end: str,
              log=print) -> tuple[str, list[Check]]:
    log("Pulling CRSP monthly stock file in chunks:")
    pull_msf(db, smap, cfg, raw_dir, end, log)
    msf = load_msf(raw_dir)
    log("Pulling security info:")
    si = pull_secinfo(db, smap, raw_dir, log)
    log("Pulling CCM link:")
    ccm = pull_ccm(db, smap, cfg, raw_dir, log)
    checks = check_msf(msf, cfg) + check_ccm(ccm)
    report = "\n\n".join([
        summary("CRSP monthly", msf.drop(columns="month"), "permno", "mthcaldt",
                qualified(table_entry(smap, "msf")), msf_mapping(smap)),
        summary("Security info", si, "permno", "secinfostartdt",
                qualified(table_entry(smap, "secinfo")), si.attrs.get("mapping", {})),
        summary("CCM link", ccm, "gvkey", "linkdt", qualified(table_entry(smap, "ccm")),
                ccm.attrs.get("mapping", {})),
        render("Checks: CRSP and link", checks),
    ])
    return report, checks


def msf_mapping(smap: dict) -> dict:
    return {c: a for c, a in table_entry(smap, "msf")["columns"].items() if a}


def step_compustat(db, smap: dict, cfg: dict, raw_dir: Path, fetch=None,
                   skip_fred: bool = False, log=print) -> tuple[str, list[Check]]:
    log("Pulling Compustat annual fundamentals:")
    funda = pull_funda(db, smap, cfg, raw_dir, log)
    log("Pulling Fama-French factors:")
    ff = pull_ff(db, smap, raw_dir, log)
    parts = [
        summary("Compustat funda", funda, "gvkey", "datadate",
                qualified(table_entry(smap, "funda")), funda.attrs.get("mapping", {})),
        summary("Factors", ff, "month", "month", qualified(table_entry(smap, "ff_monthly"))
                + " + " + qualified(table_entry(smap, "ff5_monthly")), {}),
    ]
    checks = check_funda(funda) + check_factors(ff, cfg)
    if not skip_fred:
        log("Downloading FRED rates:")
        kw = {"fetch": fetch} if fetch else {}
        try:
            rates = pull_fred(cfg, raw_dir, log=log, **kw)
        except Exception as exc:
            checks.append(Check("RATE0", "FRED download", FAIL,
                                f"{type(exc).__name__}: {exc}. Retry later, or rerun with "
                                "--skip-fred and download the four series separately"))
        else:
            parts.append(summary("FRED rates", rates, "month", "month",
                                 "fred.stlouisfed.org", {}))
            checks += check_rates(rates, cfg)
    log("Cross-checking against the CRSP pull:")
    msf, ccm = load_msf(raw_dir), load_ccm(raw_dir)
    checks += cross_checks(msf, ccm, funda, cfg)
    parts.append(render("Checks: Compustat, factors, rates, cross-table", checks))
    return "\n\n".join(parts), checks
