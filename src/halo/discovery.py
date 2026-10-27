"""Milestone 1: discover the real WRDS schema and test every [VERIFY] claim in spec section 3.

``run_discovery(db)`` works on any object with the ``wrds.Connection`` interface,
so the same code runs against WRDS on the user's machine and against
``SqliteWRDS`` in tests. It only reads metadata and aggregates (names, types,
counts, min and max dates, value frequencies of flag columns). It never writes
rows of licensed data to disk.

Outputs:
- ``config/schema_map.yaml``: the chosen table for each concept and the mapping
  from canonical column names to the names on WRDS. All later code reads this.
- ``docs/schema_discovery.md``: the evidence, including a table that marks each
  [VERIFY] item CONFIRMED, REFUTED, UNKNOWN or ERROR.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from halo.schema import (
    CIZ_FILTER_COLUMNS,
    CONCEPTS,
    LIBRARY_PREFIXES,
    Concept,
    check_identifier,
)

CONFIRMED, REFUTED, UNKNOWN, ERROR, INFO = "CONFIRMED", "REFUTED", "UNKNOWN", "ERROR", "INFO"

# Expected values of the spec 3.3 filters.
EXPECTED_FILTER_VALUES = {
    "sharetype": ["NS"], "securitytype": ["EQTY"], "securitysubtype": ["COM"],
    "usincflg": ["Y"], "issuertype": ["ACOR", "CORP"], "primaryexch": ["N", "A", "Q"],
    "conditionaltype": ["RW"], "tradingstatusflg": ["A"],
}
FUNDA_FILTER = ("\"indfmt\" = 'INDL' AND \"datafmt\" = 'STD' AND \"popsrc\" = 'D' "
                "AND \"consol\" = 'C'")


@dataclass
class Resolved:
    concept: Concept
    library: str
    table: str
    columns: dict[str, str | None]          # canonical -> actual (None if absent)
    all_columns: pd.DataFrame               # name, type as reported by WRDS
    alternatives: list[str] = field(default_factory=list)
    row_count: int | None = None
    date_ranges: dict[str, tuple] = field(default_factory=dict)
    null_shares: dict[str, float] = field(default_factory=dict)

    @property
    def fqn(self) -> str:
        return f"{self.library}.{self.table}"

    def col(self, canonical: str) -> str:
        actual = self.columns.get(canonical)
        if not actual:
            raise KeyError(f"{self.fqn} has no column for {canonical}")
        return f'"{check_identifier(actual)}"'


@dataclass
class Check:
    id: str
    claim: str
    status: str
    evidence: str
    critical: bool = False                  # a REFUTED critical check fails the run


class Discovery:
    def __init__(self, db, prefer_library: str | None = None, log=print,
                 sample_start: str = "1996-01-01"):
        self.db = db
        self.prefer = prefer_library
        self.log = log
        self.sample_start = pd.Timestamp(sample_start).strftime("%Y-%m-%d")
        self.no_access: dict[str, str] = {}      # lib.table -> first line of the error
        self.libraries: list[str] = []
        self.tables: dict[str, Resolved] = {}
        self.unresolved: dict[str, str] = {}
        self.checks: list[Check] = []
        self.distributions: dict[str, pd.DataFrame] = {}
        self.decisions: dict = {}
        self._table_cache: dict[str, list[str]] = {}

    # ---- helpers -----------------------------------------------------------------

    def sql(self, query: str, date_cols=None) -> pd.DataFrame:
        return self.db.raw_sql(query, date_cols=date_cols)

    def list_tables(self, lib: str) -> list[str]:
        if lib not in self._table_cache:
            try:
                self._table_cache[lib] = [t.lower() for t in (self.db.list_tables(lib) or [])]
            except Exception as exc:  # no permission on this library
                self.log(f"  cannot list tables in {lib}: {exc}")
                self._table_cache[lib] = []
        return self._table_cache[lib]

    def add(self, *args, **kwargs) -> Check:
        c = Check(*args, **kwargs)
        self.checks.append(c)
        return c

    def guarded(self, check_id: str, claim: str, fn, critical: bool = False) -> None:
        """Run one check; an exception becomes an ERROR row instead of stopping discovery."""
        try:
            fn()
        except Exception as exc:
            self.add(check_id, claim, ERROR, f"{type(exc).__name__}: {_first_line(exc)}",
                     critical)

    # ---- step 1: libraries and tables ---------------------------------------------

    def find_libraries(self) -> None:
        libs = [lib.lower() for lib in self.db.list_libraries()]
        self.libraries = sorted(lib for lib in libs if lib.startswith(LIBRARY_PREFIXES))
        self.log(f"Libraries matching crsp*, comp*, ff*: {len(self.libraries)}")

    def resolve(self, concept: Concept) -> None:
        candidates = list(concept.candidates)
        if self.prefer:
            candidates.sort(key=lambda c: not re.fullmatch(c[0], self.prefer))
        matches: list[Resolved] = []
        seen: set[str] = set()
        for lib_re, tab_re in candidates:
            for lib in self.libraries:
                if not re.fullmatch(lib_re, lib):
                    continue
                for tab in self.list_tables(lib):
                    if not re.fullmatch(tab_re, tab) or f"{lib}.{tab}" in seen:
                        continue
                    seen.add(f"{lib}.{tab}")
                    desc = self.db.describe_table(lib, tab)
                    names = [str(n).lower() for n in desc["name"]]
                    mapping = {canon: next((a for a in alts if a in names), None)
                               for canon, alts in concept.columns.items()}
                    if all(mapping[r] for r in concept.required):
                        if not self.can_query(lib, tab):
                            continue
                        all_cols = pd.DataFrame({"name": names,
                                                 "type": [str(t) for t in desc["type"]]})
                        matches.append(Resolved(concept, lib, tab, mapping, all_cols))
                    else:
                        missing = [r for r in concept.required if not mapping[r]]
                        self.log(f"  {lib}.{tab} matched {concept.name} by name but lacks "
                                 f"{missing}; skipped")
        if not matches:
            blocked = [t for t in seen if t in self.no_access]
            self.unresolved[concept.name] = (
                f"no candidate table has the required columns and can be queried; "
                f"permission denied on: {blocked or 'none'}")
            self.log(f"  {concept.name}: NOT FOUND")
            return
        chosen = matches[0]
        chosen.alternatives = [m.fqn for m in matches[1:]]
        self.tables[concept.name] = chosen
        self.log(f"  {concept.name}: {chosen.fqn}"
                 + (f" (also found: {', '.join(chosen.alternatives)})"
                    if chosen.alternatives else ""))

    def can_query(self, lib: str, tab: str) -> bool:
        """Describing a view needs no rights on its source; reading it does. Probe once."""
        fqn = f"{check_identifier(lib)}.{check_identifier(tab)}"
        if fqn in self.no_access:
            return False
        try:
            self.sql(f"SELECT 1 AS ok FROM {fqn} LIMIT 1")
            return True
        except Exception as exc:
            self.no_access[fqn] = _first_line(exc)
            self.log(f"  {fqn}: no permission to read ({self.no_access[fqn]}); trying the next")
            return False

    # ---- step 2: profile each table ------------------------------------------------

    def profile(self, r: Resolved) -> None:
        fqn = r.fqn
        if r.concept.heavy:
            r.row_count = int(self.db.get_row_count(r.library, r.table))
            return
        present = [c for c, a in r.columns.items() if a]
        parts = ", ".join(f'SUM(CASE WHEN {r.col(c)} IS NULL THEN 1 ELSE 0 END) AS "{c}"'
                          for c in present)
        out = self.sql(f"SELECT COUNT(*) AS n_rows, {parts} FROM {fqn}").iloc[0]
        r.row_count = int(out["n_rows"])
        r.null_shares = {c: float(out[c] or 0) / max(r.row_count, 1) for c in present}
        for c in r.concept.date_columns:
            if r.columns.get(c):
                d = self.sql(f"SELECT MIN({r.col(c)}) AS lo, MAX({r.col(c)}) AS hi FROM {fqn}",
                             date_cols=["lo", "hi"]).iloc[0]
                r.date_ranges[c] = (_date(d["lo"]), _date(d["hi"]))

    def value_counts(self, r: Resolved, canonical: str, where: str = "",
                     limit: int = 25) -> pd.DataFrame:
        col = r.col(canonical)
        df = self.sql(f"SELECT {col} AS value, COUNT(*) AS n FROM {r.fqn} {where} "
                      f"GROUP BY {col} ORDER BY n DESC LIMIT {int(limit)}")
        df["value"] = df["value"].astype(object).where(df["value"].notna(), "NULL")
        self.distributions[f"{r.fqn}.{r.columns[canonical]}"] = df
        return df

    # ---- step 3: the [VERIFY] checks ----------------------------------------------

    def check_names(self) -> None:
        t = self.tables
        msf = t.get("msf")
        self.add("V1", "Monthly stock table exists (crsp.msf_v2, or a variant such as "
                 "crsp_a_stock62.wrds_msf62v2_query)",
                 CONFIRMED if msf else REFUTED,
                 f"using {msf.fqn}" if msf else "no monthly CIZ table found", critical=True)
        if msf:
            for cid, cols in [("V2", ("mthcaldt", "mthret")), ("V3", ("mthprc", "mthcap"))]:
                self.add(cid, f"Monthly columns are named {' and '.join(cols)}", CONFIRMED,
                         ", ".join(f"{c} -> {msf.columns[c]}" for c in cols), critical=True)
            flg = msf.columns.get("mthretflg")
            similar = [c for c in msf.all_columns["name"]
                       if re.search(r"flg|del|dlst", c) and c != flg]
            self.add("V4", "A return flag column mthretflg exists",
                     CONFIRMED if flg else REFUTED,
                     ("present" if flg else "absent; delisting months will be identified as "
                      "the last row per permno")
                     + f". Other flag or delisting columns: {similar or 'none'}")
            if flg:
                self.guarded("V4b", "mthretflg values", lambda: self.add(
                    "V4b", "mthretflg values", INFO,
                    _fmt_counts(self.value_counts(msf, "mthretflg"))))
        si = t.get("secinfo")
        if si:
            self.add("V5", "Security info validity window columns",
                     INFO, f"{si.fqn}: start = {si.columns['secinfostartdt']}, "
                     f"end = {si.columns['secinfoenddt']}")
            has_sic = si.columns.get("siccd") or (msf and msf.columns.get("siccd"))
            where = []
            if si.columns.get("siccd"):
                where.append(f"{si.fqn}.siccd")
            if msf and msf.columns.get("siccd"):
                where.append(f"{msf.fqn}.siccd")
            self.add("V6", "Point-in-time SIC (siccd) is available in CRSP",
                     CONFIRMED if has_sic else REFUTED,
                     ", ".join(where) if has_sic else "not found; fall back to Compustat sich")
        ccm = t.get("ccm")
        self.add("V7", "CCM link table location", CONFIRMED if ccm else REFUTED,
                 ccm.fqn if ccm else "ccmxpf_lnkhist not found in any crsp* library",
                 critical=True)
        fa = t.get("funda")
        if fa:
            missing = [c for c, a in fa.columns.items() if not a]
            self.add("V8", "Compustat funda has every column in spec 3.6",
                     CONFIRMED if not missing else REFUTED,
                     f"{fa.fqn}; missing: {missing}" if missing else f"{fa.fqn}; all present")
        for name, cols in [("ff_monthly", ("mktrf", "smb", "hml", "rf", "umd")),
                           ("ff5_monthly", ("mktrf", "smb", "hml", "rmw", "cma", "rf"))]:
            r = t.get(name)
            missing = [c for c in cols if r and not r.columns.get(c)]
            self.add("V9" if name == "ff_monthly" else "V9b",
                     f"Factor table for {name} with {', '.join(cols)}",
                     CONFIRMED if r and not missing else REFUTED,
                     f"{r.fqn}; missing: {missing}" if r else "not found")

    def check_delisting(self) -> None:
        msf = self.tables["msf"]
        p, d, ret = msf.col("permno"), msf.col("mthcaldt"), msf.col("mthret")
        forbidden = [c for c in msf.concept.forbidden if c in set(msf.all_columns["name"])]
        df = self.sql(
            f"SELECT m.{p} AS permno, m.{d} AS d, m.{ret} AS r, l.n AS n "
            f"FROM {msf.fqn} m JOIN (SELECT {p} AS p, MAX({d}) AS last, COUNT(*) AS n "
            f"FROM {msf.fqn} GROUP BY {p}) l ON m.{p} = l.p AND m.{d} = l.last",
            date_cols=["d"])
        cutoff = df["d"].max() - pd.Timedelta(days=365)
        ended = df[(df["d"] < cutoff) & (df["n"] >= 2)]
        share = ended["r"].notna().mean() if len(ended) else float("nan")
        ok = len(ended) > 0 and share >= 0.9 and not forbidden
        self.add("R1", "Delisting returns are inside mthret: the last month of stocks that "
                 "stopped trading has a non-null return, and no dlret/dlretx columns exist",
                 CONFIRMED if ok else (UNKNOWN if not len(ended) else REFUTED),
                 f"{len(ended)} permnos ended more than a year before the last date; "
                 f"{share:.1%} have a non-null last mthret; legacy delisting columns present: "
                 f"{forbidden or 'none'}", critical=bool(forbidden))

    def check_prices_caps_returns(self) -> None:
        msf = self.tables["msf"]
        prc, cap = msf.col("mthprc"), msf.col("mthcap")
        ret, d = msf.col("mthret"), msf.col("mthcaldt")
        s = self.sql(
            f"SELECT SUM(CASE WHEN {prc} <= 0 THEN 1 ELSE 0 END) AS nonpos, "
            f"SUM(CASE WHEN {prc} IS NULL THEN 1 ELSE 0 END) AS prc_null, "
            f"SUM(CASE WHEN {ret} IN (-66, -77, -88, -99) THEN 1 ELSE 0 END) AS codes, "
            f"SUM(CASE WHEN {ret} <= -1 THEN 1 ELSE 0 END) AS le_minus1, "
            f"SUM(CASE WHEN {ret} > 3 THEN 1 ELSE 0 END) AS gt_300, "
            f"SUM(CASE WHEN {ret} IS NULL THEN 1 ELSE 0 END) AS ret_null, "
            f"COUNT(*) AS n FROM {msf.fqn}").iloc[0]
        s = s.fillna(0).astype(int)
        self.add("R2", "Prices are always positive (no negative bid-ask flag prices)",
                 CONFIRMED if s["nonpos"] == 0 else REFUTED,
                 f"{s['nonpos']} rows with mthprc <= 0, {s['prc_null']} NULL prices, "
                 f"of {s['n']}", critical=True)
        self.add("R5", "Missing returns are NULL, not -66/-77/-88/-99 codes",
                 CONFIRMED if s["codes"] == 0 else REFUTED,
                 f"{s['codes']} legacy codes, {s['ret_null']} NULL returns "
                 f"({s['ret_null'] / max(s['n'], 1):.2%}), {s['le_minus1']} returns <= -100%, "
                 f"{s['gt_300']} returns > 300% ({s['gt_300'] / max(s['n'], 1):.4%})",
                 critical=True)
        top = self.sql(f"SELECT MAX({cap}) AS cap FROM {msf.fqn} "
                       f"WHERE {d} = (SELECT MAX({d}) FROM {msf.fqn})").iloc[0]["cap"]
        dollars = float(top) * 1000
        ok = 1e12 <= dollars < 1e14
        self.add("R3", "mthcap is in thousands of dollars (largest firm in the trillions "
                 "after multiplying by 1000)", CONFIRMED if ok else REFUTED,
                 f"largest mthcap on the last date = {float(top):,.0f}, "
                 f"x1000 = ${dollars / 1e12:,.2f} trillion", critical=True)
        if msf.columns.get("mthprcflg"):
            self.guarded("R2b", "mthprcflg values", lambda: self.add(
                "R2b", "mthprcflg values (expect TR, BA, DA among them)", INFO,
                _fmt_counts(self.value_counts(msf, "mthprcflg"))))
        self.add("R4", "mthret compounds daily returns with dividends reinvested",
                 UNKNOWN, "documented by CRSP; not testable from aggregates. Accepted as is.")

    def check_legacy_columns(self) -> None:
        msf = self.tables["msf"]
        found = [c for c in ("shrcd", "exchcd") if c in set(msf.all_columns["name"])]
        self.add("R6", "Legacy shrcd and exchcd do not exist in the CIZ table",
                 CONFIRMED if not found else REFUTED,
                 f"found in {msf.fqn}: {found}" if found else f"absent from {msf.fqn}",
                 critical=bool(found))

    def check_legacy_freeze(self) -> None:
        legacy, msf = self.tables.get("legacy_msf"), self.tables["msf"]
        v2_hi = msf.date_ranges.get("mthcaldt", (None, None))[1]
        if not legacy:
            self.add("R7", "Legacy tables frozen after 2024-12-31", UNKNOWN,
                     f"no legacy msf found; {msf.fqn} runs to {v2_hi}")
            return
        hi = legacy.date_ranges.get("date", (None, None))[1]
        if hi is None or v2_hi is None:
            self.add("R7", "Legacy tables frozen after 2024-12-31", UNKNOWN,
                     f"end dates unavailable: {legacy.fqn} {hi}, {msf.fqn} {v2_hi}")
            return
        frozen = hi is not None and hi <= dt.date(2024, 12, 31)
        newer = v2_hi is not None and hi is not None and v2_hi > hi
        self.add("R7", "Legacy tables frozen after 2024-12-31, v2 tables continue",
                 CONFIRMED if frozen and newer else REFUTED,
                 f"{legacy.fqn} ends {hi}; {msf.fqn} ends {v2_hi}")

    def check_month_dates(self) -> None:
        msf = self.tables["msf"]
        dates = self.sql(f"SELECT DISTINCT {msf.col('mthcaldt')} AS d FROM {msf.fqn}",
                         date_cols=["d"])["d"].sort_values()
        per_month = dates.dt.to_period("M").value_counts()
        weekend = int((dates.dt.day_of_week >= 5).sum())
        month_end = dates.dt.to_period("M").dt.to_timestamp(how="end").dt.normalize()
        early = int((dates < month_end).sum())
        ok = weekend == 0 and early > 0 and per_month.max() == 1
        self.add("R8", "mthcaldt is the last trading day of the month (one date per month, "
                 "never a weekend, sometimes before the calendar month end)",
                 CONFIRMED if ok else REFUTED,
                 f"{len(dates)} distinct dates, {weekend} on weekends, {early} before calendar "
                 f"month end, max dates in one month = {per_month.max()}")

    def check_filters(self) -> None:
        msf, si = self.tables.get("msf"), self.tables.get("secinfo")
        where_found = {}
        for name, r in [("msf", msf), ("secinfo", si)]:
            if r:
                where_found[name] = [c for c in CIZ_FILTER_COLUMNS if r.columns.get(c)]
        source = next((n for n in ("msf", "secinfo")
                       if len(where_found.get(n, [])) == len(CIZ_FILTER_COLUMNS)), None)
        if source is None and where_found:
            source = max(where_found, key=lambda n: len(where_found[n]))
        missing = [c for c in CIZ_FILTER_COLUMNS if c not in where_found.get(source, [])]
        self.decisions["filter_source"] = source
        self.decisions["filter_columns_missing"] = missing
        self.add("F1", "All eight spec 3.3 filter columns exist (five common-stock, three "
                 "exchange)", CONFIRMED if not missing else REFUTED,
                 f"taking filters from {source}; present in msf: {where_found.get('msf')}, "
                 f"secinfo: {where_found.get('secinfo')}; missing: {missing or 'none'}")
        r = self.tables.get(source) if source else None
        if not r:
            return
        none_seen, some_unseen = {}, {}
        for c in CIZ_FILTER_COLUMNS:
            if not r.columns.get(c):
                continue
            vc = self.value_counts(r, c)
            absent = [v for v in EXPECTED_FILTER_VALUES[c] if v not in set(vc["value"])]
            if len(absent) == len(EXPECTED_FILTER_VALUES[c]):
                none_seen[c] = absent
            elif absent:
                some_unseen[c] = absent
        self.add("F2", "Each filter column has at least one of its expected values "
                 "(NS, EQTY, COM, Y, ACOR/CORP, N/A/Q, RW, A)",
                 CONFIRMED if not none_seen else REFUTED,
                 (f"no expected value at all in: {none_seen}. " if none_seen else "")
                 + (f"expected but unused values: {some_unseen}. " if some_unseen else "")
                 + "Full frequencies below.", critical=bool(none_seen))
        if r.columns.get("sharetype"):
            vc = self.distributions[f"{r.fqn}.{r.columns['sharetype']}"]
            self.add("F3", "sharetype = 'COM' returns nothing (COM belongs to securitysubtype)",
                     CONFIRMED if "COM" not in set(vc["value"]) else REFUTED,
                     f"sharetype values: {', '.join(map(str, vc['value']))}")
        conds = [f"{r.col(c)} IN ({', '.join(repr(v) for v in EXPECTED_FILTER_VALUES[c])})"
                 for c in CIZ_FILTER_COLUMNS if r.columns.get(c)]
        if conds:
            p = r.col("permno")
            n = self.sql(f"SELECT COUNT(*) AS n, COUNT(DISTINCT {p}) AS k FROM {r.fqn} "
                         f"WHERE {' AND '.join(conds)}").iloc[0]
            self.add("F4", "Rows passing every available filter", INFO,
                     f"{int(n['n']):,} rows, {int(n['k']):,} distinct permnos in {r.fqn}")

    def check_secinfo_windows(self) -> None:
        si = self.tables["secinfo"]
        p, a, b = si.col("permno"), si.col("secinfostartdt"), si.col("secinfoenddt")
        overlap = self.sql(f"SELECT COUNT(*) AS n FROM {si.fqn} x JOIN {si.fqn} y "
                           f"ON x.{p} = y.{p} AND x.{a} < y.{a} AND x.{b} >= y.{a}"
                           ).iloc[0]["n"]
        same_start = self.sql(f"SELECT COUNT(*) AS n FROM (SELECT {p}, {a} FROM {si.fqn} "
                              f"GROUP BY {p}, {a} HAVING COUNT(*) > 1) AS dup").iloc[0]["n"]
        n = int(overlap or 0) + int(same_start or 0)
        names = f"{si.columns['secinfostartdt']} / {si.columns['secinfoenddt']}"
        whole_life = si.columns["secinfostartdt"] != "secinfostartdt"
        self.add("S1", "Security info rows have non-overlapping validity windows per permno, "
                 "so a date-range join gives one row per (permno, date)",
                 CONFIRMED if n == 0 else REFUTED,
                 f"{int(overlap or 0):,} overlapping pairs and {int(same_start or 0):,} "
                 f"(permno, start) duplicates using {names}"
                 + (". These columns may be the security's whole life rather than the "
                    "validity window of each row" if whole_life else ""))

    def check_ccm(self) -> None:
        ccm = self.tables["ccm"]
        lt = self.value_counts(ccm, "linktype")
        self.value_counts(ccm, "linkprim")
        has_ls = "LS" in set(lt["value"])
        self.decisions["link_types_available"] = [str(v) for v in lt["value"]]
        self.add("L1", "Link types LC and LU exist; LS availability",
                 CONFIRMED if {"LC", "LU"} <= set(lt["value"]) else REFUTED,
                 f"linktypes: {_fmt_counts(lt)}; LS {'available' if has_ls else 'not present'}")
        g = ccm.col("gvkey")
        s = self.sql(f"SELECT MIN(LENGTH({g})) AS lo, MAX(LENGTH({g})) AS hi, "
                     f"SUM(CASE WHEN {ccm.col('linkenddt')} IS NULL THEN 1 ELSE 0 END) "
                     f"AS open_links FROM {ccm.fqn}").iloc[0]
        gtype = ccm.all_columns.set_index("name").loc[ccm.columns["gvkey"], "type"]
        is_text = any(k in gtype.upper() for k in ("CHAR", "TEXT", "STRING"))
        self.add("L2", "gvkey is a 6-character string",
                 CONFIRMED if is_text and s["lo"] == 6 and s["hi"] == 6 else REFUTED,
                 f"type {gtype}, length {s['lo']} to {s['hi']}", critical=not is_text)
        hi = ccm.date_ranges.get("linkenddt", (None, None))[1]
        self.add("L3", "Active links have NULL linkenddt",
                 CONFIRMED if s["open_links"] and s["open_links"] > 0 else REFUTED,
                 f"{int(s['open_links'] or 0):,} NULL linkenddt; max non-null linkenddt {hi}. "
                 "If REFUTED, active links probably use a far-future date instead")

    def check_funda(self) -> None:
        fa = self.tables["funda"]
        for c in ("indfmt", "datafmt", "popsrc", "consol"):
            self.value_counts(fa, c)
        present = [c for c in ("at", "che", "intan", "sale", "ppent", "ppegt", "emp", "capx",
                               "ebitda", "oibdp", "dp", "xrd", "sich") if fa.columns.get(c)]
        base = f"FROM {fa.fqn} WHERE {FUNDA_FILTER} AND {fa.col('datadate')} >= '1994-01-01'"
        parts = ", ".join(f'SUM(CASE WHEN {fa.col(c)} IS NULL THEN 1 ELSE 0 END) AS "{c}"'
                          for c in present)
        out = self.sql(f"SELECT COUNT(*) AS n, COUNT(DISTINCT {fa.col('gvkey')}) AS k, "
                       f"{parts} {base}").iloc[0]
        n = int(out["n"])
        shares = {c: 1 - float(out[c] or 0) / max(n, 1) for c in present}
        self.decisions["funda_nonnull_since_1994"] = {c: round(v, 4) for c, v in shares.items()}
        self.add("C1", "Standard filter (INDL, STD, D, C) from 1994 returns data",
                 CONFIRMED if n > 0 else REFUTED,
                 f"{n:,} rows, {int(out['k']):,} gvkeys; non-null share: "
                 + ", ".join(f"{c} {v:.1%}" for c, v in shares.items()), critical=n == 0)
        if fa.columns.get("ebitda") and fa.columns.get("oibdp"):
            e = self.sql(f"SELECT SUM(CASE WHEN {fa.col('ebitda')} IS NULL AND "
                         f"{fa.col('oibdp')} IS NOT NULL THEN 1 ELSE 0 END) AS fill, "
                         f"SUM(CASE WHEN {fa.col('ebitda')} IS NOT NULL AND {fa.col('oibdp')} "
                         f"IS NOT NULL AND {fa.col('ebitda')} <> {fa.col('oibdp')} THEN 1 ELSE 0 "
                         f"END) AS differ {base}").iloc[0]
            self.add("C2", "oibdp is useful as an ebitda fallback", INFO,
                     f"{int(e['fill'] or 0):,} rows have NULL ebitda but non-null oibdp; "
                     f"{int(e['differ'] or 0):,} rows have both and they differ")

    def check_factors(self) -> None:
        for name in ("ff_monthly", "ff5_monthly"):
            r = self.tables.get(name)
            if not r:
                continue
            s = self.sql(f"SELECT AVG(ABS({r.col('mktrf')})) AS m, MIN({r.col('rf')}) AS rf_lo, "
                         f"MAX({r.col('rf')}) AS rf_hi FROM {r.fqn} "
                         f"WHERE {r.col('date')} >= '{self.sample_start}'").iloc[0]
            lo, hi = r.date_ranges.get("date", (None, None))
            first_of_month = lo is not None and lo.day == 1 and hi.day == 1
            self.decisions.setdefault("factor_date_convention", {})[name] = (
                "first_of_month" if first_of_month else "month_end_or_other")
            decimal = float(s["m"]) < 0.2
            self.decisions.setdefault("factor_units", {})[name] = (
                "decimal" if decimal else "percent")
            rf_ok = (0 <= float(s["rf_lo"]) and float(s["rf_hi"]) <= 0.02) if decimal else None
            self.add("V10" if name == "ff_monthly" else "V10b",
                     f"{r.fqn} is in decimals and monthly rf is between 0 and 0.02 "
                     f"from {self.sample_start}",
                     CONFIRMED if decimal and rf_ok else REFUTED,
                     f"mean |mktrf| = {float(s['m']):.4f} ({'decimal' if decimal else 'percent'}); "
                     f"rf from {float(s['rf_lo']):.4f} to {float(s['rf_hi']):.4f}; "
                     f"date range {_range(r.date_ranges.get('date'))} "
                     f"({'first of month' if first_of_month else 'not first of month'}); umd "
                     f"{'present' if r.columns.get('umd') else 'absent'}")

    # ---- run -----------------------------------------------------------------------

    def run(self) -> None:
        self.find_libraries()
        self.log("Resolving tables:")
        for concept in CONCEPTS.values():
            self.resolve(concept)
        if self.no_access:
            self.add("A1", "Tables skipped because this account cannot read them", INFO,
                     "; ".join(f"{t} ({e})" for t, e in sorted(self.no_access.items())))
        self.log("Profiling tables:")
        for r in self.tables.values():
            self.log(f"  {r.fqn}")
            try:
                self.profile(r)
            except Exception as exc:
                self.add(f"P-{r.concept.name}", f"profile {r.fqn}", ERROR, repr(exc))
        self.log("Running checks:")
        self.check_names()
        steps = [
            ("msf", "R1", "delisting returns inside mthret", self.check_delisting),
            ("msf", "R2/R3/R5", "prices, caps, returns", self.check_prices_caps_returns),
            ("msf", "R6", "legacy columns absent", self.check_legacy_columns),
            ("msf", "R7", "legacy freeze", self.check_legacy_freeze),
            ("msf", "R8", "month-end dates", self.check_month_dates),
            ("msf", "F", "universe filters", self.check_filters),
            ("secinfo", "S1", "security info windows", self.check_secinfo_windows),
            ("ccm", "L", "CCM link", self.check_ccm),
            ("funda", "C", "Compustat", self.check_funda),
            ("ff_monthly", "V10", "factors", self.check_factors),
        ]
        for needs, cid, claim, fn in steps:
            if needs in self.tables:
                self.log(f"  {claim}")
                self.guarded(cid, claim, fn, critical=True)

    @property
    def failures(self) -> list[str]:
        out = [f"needed table {name} not found: {why}" for name, why in self.unresolved.items()
               if CONCEPTS[name].needed]
        out += [f"{c.id} {c.status}: {c.claim} ({c.evidence})" for c in self.checks
                if c.status == ERROR or (c.critical and c.status == REFUTED)]
        return out


# ---- output --------------------------------------------------------------------------

def _date(x):
    if x is None or pd.isna(x):
        return None
    return pd.Timestamp(x).date()


def _first_line(exc: Exception) -> str:
    text = str(exc).strip()
    return text.splitlines()[0] if text else type(exc).__name__


def _range(v) -> str:
    return "n/a" if not v else f"{v[0]} to {v[1]}"


def _fmt_counts(df: pd.DataFrame) -> str:
    return ", ".join(f"{v} ({int(n):,})" for v, n in zip(df["value"], df["n"], strict=True))


def build_schema_map(d: Discovery) -> dict:
    tables = {}
    for name, concept in CONCEPTS.items():
        r = d.tables.get(name)
        tables[name] = None if r is None else {
            "library": r.library, "table": r.table, "description": concept.description,
            "columns": dict(r.columns), "also_found": r.alternatives,
        }
    return {
        "generated_by": "scripts/00_discover_schema.py",
        "generated_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": "Generated file. Rerun discovery instead of editing by hand. Canonical names "
                "on the left, names on WRDS on the right, null if absent.",
        "tables": tables,
        "decisions": _plain(d.decisions),
    }


def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_plain(v) for v in x]
    if hasattr(x, "item"):
        return x.item()
    return x


def build_markdown(d: Discovery) -> str:
    now = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# WRDS schema discovery",
        "",
        f"Generated by `scripts/00_discover_schema.py` on {now}. Do not edit by hand; rerun "
        "the script. Contains names, types and aggregates only, no rows of data.",
        "",
        "## Result",
        "",
    ]
    fails = d.failures
    lines += (["**FAILED.** Fix these before M2:", ""] + [f"- {f}" for f in fails]
              if fails else ["**PASSED.** Every needed table was found and no critical check "
                             "was refuted."])
    lines += ["", "## Libraries matching crsp*, comp*, ff*", "",
              ", ".join(f"`{lib}`" for lib in d.libraries) or "(none)", "",
              "## Tables used", "",
              "| Concept | Table | Rows | Date range | Also found |", "|---|---|---|---|---|"]
    for name, concept in CONCEPTS.items():
        r = d.tables.get(name)
        if r is None:
            lines.append(f"| {name} | NOT FOUND{'' if concept.needed else ' (optional)'} | | | |")
            continue
        ranges = "; ".join(f"{c}: {_range(v)}" for c, v in r.date_ranges.items())
        rows = f"{r.row_count:,}" + (" (estimate)" if concept.heavy else "") \
            if r.row_count is not None else ""
        lines.append(f"| {name} | `{r.fqn}` | {rows} | {ranges} | "
                     f"{', '.join(r.alternatives) or ''} |")
    lines += ["", "## [VERIFY] checks", "",
              "| ID | Claim | Status | Evidence |", "|---|---|---|---|"]
    for c in d.checks:
        mark = f"**{c.status}**" if c.status in (REFUTED, ERROR) else c.status
        lines.append(f"| {c.id} | {c.claim} | {mark} | {_md(c.evidence)} |")
    lines += ["", "## Column mapping (canonical name on the left)", ""]
    for name, r in d.tables.items():
        types = r.all_columns.set_index("name")["type"].to_dict()
        lines += [f"### {name}: `{r.fqn}`", "",
                  "| Canonical | On WRDS | Type | NULL share |", "|---|---|---|---|"]
        for canon, actual in r.columns.items():
            ns = r.null_shares.get(canon)
            lines.append(f"| {canon} | {actual or '**absent**'} | {types.get(actual, '')} | "
                         f"{'' if ns is None else f'{ns:.2%}'} |")
        lines.append("")
        n_cols = len(r.all_columns)
        if n_cols <= 120:
            lines += [f"All {n_cols} columns: " + ", ".join(
                f"`{a}` {t}" for a, t in zip(r.all_columns["name"], r.all_columns["type"],
                                             strict=True)), ""]
        else:
            lines += [f"{n_cols} columns in total (too many to list).", ""]
    if d.distributions:
        lines += ["## Value frequencies of flag columns (top 25)", ""]
        for key, df in d.distributions.items():
            lines.append(f"- `{key}`: {_fmt_counts(df)}")
        lines.append("")
    if d.decisions:
        lines += ["## Decisions recorded in schema_map.yaml", "", "```yaml",
                  yaml.safe_dump(_plain(d.decisions), sort_keys=False).rstrip(), "```", ""]
    return "\n".join(lines) + "\n"


def _md(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def run_discovery(db, doc_path: str | Path, map_path: str | Path,
                  prefer_library: str | None = None, log=print,
                  sample_start: str = "1996-01-01") -> Discovery:
    d = Discovery(db, prefer_library=prefer_library, log=log, sample_start=sample_start)
    d.run()
    doc_path, map_path = Path(doc_path), Path(map_path)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(build_markdown(d))
    with open(map_path, "w") as f:
        f.write("# Generated by scripts/00_discover_schema.py. Do not edit by hand.\n")
        yaml.safe_dump(build_schema_map(d), f, sort_keys=False)
    return d
