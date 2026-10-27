"""Helpers shared by the WRDS pull scripts (spec section 9)."""

from __future__ import annotations

import datetime as dt
import re

import pandas as pd

from halo.schema import check_identifier, table_entry

LITERAL = re.compile(r"^[A-Za-z0-9/ _.-]+$")


def ph(db, name: str) -> str:
    """Placeholder for a bound parameter in this connection's style."""
    check_identifier(name)
    return f":{name}" if getattr(db, "paramstyle", "pyformat") == "named" else f"%({name})s"


def sql_list(values) -> str:
    """Quoted IN-list of short config literals such as filter codes. Never user data."""
    out = []
    for v in values:
        if not LITERAL.match(str(v)):
            raise ValueError(f"unsafe literal in config: {v!r}")
        out.append("'" + str(v) + "'")
    return ", ".join(out)


def resolve_columns(db, smap: dict, concept: str, wanted: list[str], required: list[str],
                    log=print) -> dict[str, str]:
    """Map canonical names to WRDS names for one table.

    Names come from the schema map. A canonical name the map does not know (because it
    was added after discovery ran) is accepted only if a column of that exact name
    exists on WRDS, checked live with describe_table. Required names that cannot be
    found raise; optional ones are dropped with a log line.
    """
    entry = table_entry(smap, concept)
    known = entry["columns"]
    live = None
    out, missing = {}, []
    for canonical in wanted:
        if canonical in known:
            if known[canonical]:
                out[canonical] = known[canonical]
                continue
        else:
            if live is None:
                desc = db.describe_table(entry["library"], entry["table"])
                live = {str(n).lower() for n in desc["name"]}
            if canonical in live:
                out[canonical] = canonical
                log(f"  {concept}.{canonical}: not in schema map, found on WRDS by name")
                continue
        missing.append(canonical)
    req_missing = [c for c in missing if c in required]
    if req_missing:
        raise ValueError(f"{concept}: required columns missing on WRDS: {req_missing}. "
                         "Rerun scripts/00_discover_schema.py and check the schema map.")
    for c in missing:
        log(f"  {concept}.{c}: not available; continuing without it")
    return out


def select_sql(mapping: dict[str, str], alias: str | None = None) -> str:
    prefix = f"{check_identifier(alias)}." if alias else ""
    return ", ".join(f'{prefix}"{check_identifier(a)}" AS "{check_identifier(c)}"'
                     for c, a in mapping.items())


def year_chunks(start: str, end: str, years: int) -> list[tuple[str, str]]:
    """Inclusive [start, end] date ranges covering whole calendar-year blocks."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    out = []
    y = s.year
    while y <= e.year:
        lo = max(s, pd.Timestamp(year=y, month=1, day=1))
        hi = min(e, pd.Timestamp(year=y + years - 1, month=12, day=31))
        out.append((lo.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d")))
        y += years
    return out


def today() -> str:
    return dt.date.today().strftime("%Y-%m-%d")


def summary(name: str, df: pd.DataFrame, id_col: str, date_col: str | None,
            source: str, mapping: dict[str, str]) -> str:
    """The spec 9 print block: rows, date range, distinct ids, NULL shares, names used."""
    lines = [f"== {name}: {source} =="]
    lines.append(f"rows: {len(df):,}   distinct {id_col}: {df[id_col].nunique():,}")
    if date_col and len(df):
        lo, hi = df[date_col].min(), df[date_col].max()
        if isinstance(lo, pd.Timestamp):
            lo, hi = lo.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d")
        lines.append(f"date range: {lo} to {hi}")
    nulls = df.isna().mean()
    lines.append("NULL share: " + ", ".join(f"{c} {v:.2%}" for c, v in nulls.items()))
    renamed = {c: a for c, a in mapping.items() if c != a}
    lines.append(f"schema map entries used: {source} columns "
                 + (f"(renamed: {renamed})" if renamed else "(all canonical names)"))
    return "\n".join(lines)
