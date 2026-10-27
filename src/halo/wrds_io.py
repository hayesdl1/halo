"""WRDS connection, plus a SQLite stand-in with the same interface for tests.

Only the user's machine ever calls ``connect``. Tests and CI use ``SqliteWRDS``,
which implements the subset of ``wrds.Connection`` this project relies on:
``list_libraries``, ``list_tables``, ``describe_table``, ``get_row_count``,
``raw_sql`` and ``close``. Libraries are SQLite attached databases, so
``library.table`` SQL works the same way it does on WRDS Postgres.

SQL written for both backends must stay portable: plain SELECT, JOIN, GROUP BY,
CASE, MIN/MAX/SUM/COUNT, LENGTH, double-quoted identifiers, ISO date literals.
"""

from __future__ import annotations

import os
import sqlite3
import sys

import pandas as pd

from halo.schema import check_identifier


def connect():
    """Open a WRDS connection. The password comes from ~/.pgpass, never the repo."""
    try:
        import wrds
    except ImportError:
        sys.exit("The wrds package is not installed. Run: pip install -e '.[wrds]'")
    username = os.environ.get("WRDS_USERNAME")
    if not username:
        sys.exit("Set WRDS_USERNAME first, for example: export WRDS_USERNAME=jdoe")
    return wrds.Connection(wrds_username=username)


class SqliteWRDS:
    """In-memory stand-in for wrds.Connection, loaded from DataFrames."""

    paramstyle = "named"             # sqlite3 binds :name; wrds (psycopg2) binds %(name)s

    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
        self._libs: dict[str, list[str]] = {}

    def add_table(self, library: str, table: str, df: pd.DataFrame) -> None:
        check_identifier(library)
        check_identifier(table)
        if library not in self._libs:
            self.conn.execute(f"ATTACH DATABASE ':memory:' AS {library}")
            self._libs[library] = []
        out = df.copy()
        for col in out.columns:
            s = out[col]
            if pd.api.types.is_datetime64_any_dtype(s):
                out[col] = s.dt.strftime("%Y-%m-%d").astype(object).where(s.notna(), None)
            elif isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(s):
                out[col] = s.astype(object).where(s.notna(), None)
        out.to_sql("_staging", self.conn, index=False, if_exists="replace")
        self.conn.execute(f"DROP TABLE IF EXISTS {library}.{table}")
        self.conn.execute(f"CREATE TABLE {library}.{table} AS SELECT * FROM main._staging")
        self.conn.execute("DROP TABLE main._staging")
        self._libs[library].append(table)

    def list_libraries(self) -> list[str]:
        return list(self._libs)

    def list_tables(self, library: str) -> list[str]:
        return list(self._libs.get(library, []))

    def describe_table(self, library: str, table: str) -> pd.DataFrame:
        info = pd.read_sql_query(
            f"PRAGMA {check_identifier(library)}.table_info({check_identifier(table)})", self.conn
        )
        return pd.DataFrame({"name": info["name"], "nullable": info["notnull"] == 0,
                             "type": info["type"].replace("", "TEXT"), "comment": None})

    def get_row_count(self, library: str, table: str) -> int:
        sql = f"SELECT COUNT(*) FROM {check_identifier(library)}.{check_identifier(table)}"
        return int(self.conn.execute(sql).fetchone()[0])

    def raw_sql(self, sql: str, date_cols=None, params=None, **_) -> pd.DataFrame:
        return pd.read_sql_query(sql, self.conn, params=params, parse_dates=date_cols)

    def close(self) -> None:
        self.conn.close()

    def __del__(self):
        try:
            self.conn.close()
        except Exception:
            pass
