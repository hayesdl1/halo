"""WRDS table and column names: what to look for, and how to read the resolved map.

Downstream code always uses the canonical names below, which are the standard WRDS
CIZ and Compustat names that src/halo/fixtures.py also uses. The real names on the
user's WRDS account are discovered by scripts/00_discover_schema.py and written to
config/schema_map.yaml. Pull scripts translate with ``select_list``, so nothing
after a pull ever sees a non-canonical name.

Table candidates are tried in order. Each is a ``(library_regex, table_regex)``
pair matched in full against ``db.list_libraries()`` and ``db.list_tables(lib)``.
The first candidate whose table has every required column wins.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from halo import REPO_ROOT

DEFAULT_SCHEMA_MAP_PATH = REPO_ROOT / "config" / "schema_map.yaml"
IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")
LIBRARY_PREFIXES = ("crsp", "comp", "ff")

# Filter columns from spec 3.3. They may sit in the stock table, the security info
# table, or both, depending on the schema variant.
CIZ_FILTER_COLUMNS = ("sharetype", "securitytype", "securitysubtype", "usincflg", "issuertype",
                      "primaryexch", "conditionaltype", "tradingstatusflg")


@dataclass(frozen=True)
class Concept:
    name: str
    description: str
    candidates: tuple[tuple[str, str], ...]
    # canonical column -> accepted names on WRDS, in order of preference
    columns: dict[str, tuple[str, ...]]
    required: tuple[str, ...]
    date_columns: tuple[str, ...] = ()
    needed: bool = True              # discovery fails if a needed concept is not found
    # Legacy column names that must NOT exist in a CIZ table (spec 3.2 rule 1 and 6).
    forbidden: tuple[str, ...] = ()
    heavy: bool = False              # skip full-table aggregates (daily file)
    extra: dict = field(default_factory=dict)


def _same(*names: str) -> dict[str, tuple[str, ...]]:
    return {n: (n,) for n in names}


CONCEPTS: dict[str, Concept] = {c.name: c for c in [
    Concept(
        name="msf",
        description="CRSP monthly stock file, CIZ format",
        candidates=(
            ("crsp", "msf_v2"),
            ("crsp_a_stock62", "wrds_msf62v2_query"),
            ("crsp_a_stock62", r".*msf.*v2.*"),
            ("crsp_a_stock", "msf_v2"),
            (r"crsp\w*", "msf_v2"),
            (r"crsp\w*", r".*msf.*v2.*"),
        ),
        columns={
            **_same("permno", "mthcaldt", "mthret", "mthretx", "mthprc", "mthcap",
                    "mthprcflg", "mthretflg", "mthprcdt", "mthshrout", "siccd"),
            **_same(*CIZ_FILTER_COLUMNS),
        },
        required=("permno", "mthcaldt", "mthret", "mthprc", "mthcap"),
        date_columns=("mthcaldt",),
        forbidden=("dlret", "dlretx", "shrcd", "exchcd"),
    ),
    Concept(
        name="dsf",
        description="CRSP daily stock file, CIZ format (reported only; not used in v1)",
        candidates=(
            ("crsp", "dsf_v2"),
            ("crsp_a_stock62", "wrds_dsf62v2_query"),
            (r"crsp\w*", r".*dsf.*v2.*"),
        ),
        columns=_same("permno", "dlycaldt", "dlyret", "dlyprc", "dlycap", "dlyprcflg"),
        required=("permno", "dlycaldt", "dlyret"),
        needed=False,
        heavy=True,
    ),
    Concept(
        name="secinfo",
        description="CRSP security info history (point-in-time classifications)",
        candidates=(
            ("crsp", "stksecurityinfohist"),
            ("crsp_a_stock62", "stksecurityinfohist62"),
            (r"crsp\w*", r"stksecurityinfohist\w*"),
            ("crsp", "stocknames_v2"),
            (r"crsp\w*", r"stocknames_v2\w*"),
        ),
        columns={
            "permno": ("permno",),
            "secinfostartdt": ("secinfostartdt", "securitybegdt", "namedt"),
            "secinfoenddt": ("secinfoenddt", "securityenddt", "nameenddt"),
            "ticker": ("ticker",),
            "siccd": ("siccd",),
            **_same(*CIZ_FILTER_COLUMNS),
        },
        required=("permno", "secinfostartdt", "secinfoenddt"),
        date_columns=("secinfostartdt", "secinfoenddt"),
    ),
    Concept(
        name="ccm",
        description="CRSP/Compustat merged link history",
        candidates=(
            ("crsp", "ccmxpf_lnkhist"),
            ("crsp_a_ccm", "ccmxpf_lnkhist"),
            (r"crsp\w*", "ccmxpf_lnkhist"),
            (r"crsp\w*", "ccmxpf_linktable"),
        ),
        columns={
            "gvkey": ("gvkey",),
            "lpermno": ("lpermno", "permno"),
            "linktype": ("linktype",),
            "linkprim": ("linkprim",),
            "linkdt": ("linkdt",),
            "linkenddt": ("linkenddt",),
        },
        required=("gvkey", "lpermno", "linktype", "linkprim", "linkdt", "linkenddt"),
        date_columns=("linkdt", "linkenddt"),
    ),
    Concept(
        name="funda",
        description="Compustat North America fundamentals annual",
        candidates=(("comp", "funda"), ("comp_na_daily_all", "funda"), (r"comp\w*", "funda")),
        columns=_same("gvkey", "datadate", "fyear", "at", "che", "intan", "sale", "ppent",
                      "ppegt", "emp", "capx", "ebitda", "oibdp", "dp", "xrd", "sich",
                      "indfmt", "datafmt", "popsrc", "consol",
                      # section 7 controls (added in M2)
                      "dltt", "dlc", "ceq", "seq", "pstk", "pstkrv", "pstkl", "txditc",
                      "revt", "cogs", "xsga", "xint"),
        required=("gvkey", "datadate", "at", "sale", "ppent", "capx", "emp", "indfmt",
                  "datafmt", "popsrc", "consol"),
        date_columns=("datadate",),
    ),
    Concept(
        name="ff_monthly",
        description="Fama-French 3 factors plus momentum, monthly",
        candidates=(("ff", "factors_monthly"), (r"ff\w*", "factors_monthly")),
        columns={"date": ("date", "dateff"), **_same("mktrf", "smb", "hml", "rf", "umd")},
        required=("date", "mktrf", "smb", "hml", "rf"),
        date_columns=("date",),
    ),
    Concept(
        name="ff5_monthly",
        description="Fama-French 5 factors, monthly",
        candidates=(("ff", "fivefactors_monthly"), (r"ff\w*", "fivefactors_monthly")),
        columns={"date": ("date", "dateff"),
                 **_same("mktrf", "smb", "hml", "rmw", "cma", "rf", "umd")},
        required=("date", "mktrf", "rmw", "cma", "rf"),
        date_columns=("date",),
        needed=False,
    ),
    Concept(
        name="legacy_msf",
        description="Legacy CRSP monthly file (only to check the freeze date; never used)",
        candidates=(("crsp", "msf"), ("crsp_a_stock", "msf"), ("crsp_a_stock62", r"msf\d*")),
        columns={"date": ("date",), "permno": ("permno",)},
        required=("date",),
        date_columns=("date",),
        needed=False,
    ),
]}


def check_identifier(name: str) -> str:
    """Refuse anything that is not a plain lowercase SQL identifier."""
    if not IDENTIFIER.match(name):
        raise ValueError(f"unsafe SQL identifier: {name!r}")
    return name


def qualified(entry: dict) -> str:
    return f"{check_identifier(entry['library'])}.{check_identifier(entry['table'])}"


def load_schema_map(path: str | Path | None = None) -> dict:
    path = Path(path or DEFAULT_SCHEMA_MAP_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/00_discover_schema.py first (see RUNBOOK.md)."
        )
    with open(path) as f:
        return yaml.safe_load(f)


def table_entry(schema_map: dict, concept: str) -> dict:
    entry = schema_map["tables"].get(concept)
    if not entry or not entry.get("table"):
        raise KeyError(f"schema map has no table for {concept!r}; rerun discovery")
    return entry


def select_list(schema_map: dict, concept: str, columns: list[str] | None = None,
                alias: str | None = None) -> str:
    """SQL select list mapping actual WRDS names to canonical names.

    Columns missing on WRDS raise, unless they are optional and not requested.
    """
    entry = table_entry(schema_map, concept)
    mapping = entry["columns"]
    wanted = columns or [c for c, actual in mapping.items() if actual]
    prefix = f"{check_identifier(alias)}." if alias else ""
    parts = []
    for canonical in wanted:
        actual = mapping.get(canonical)
        if not actual:
            raise KeyError(f"{concept}.{canonical} does not exist on WRDS per the schema map")
        parts.append(f'{prefix}"{check_identifier(actual)}" AS "{check_identifier(canonical)}"')
    return ", ".join(parts)
