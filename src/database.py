"""Read-only SQLite access and validation for BIPilot.

This file replaces src/database.py. It does not call Gemini and is safe to
reuse for normal queries, cached history and full-CSV exports.
"""

from contextlib import closing
from pathlib import Path
import re
import sqlite3

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "clean_database" / "bipilot.db"


class SQLSafetyError(ValueError):
    """The generated statement is not an authorized read-only query."""


class SQLValidationError(ValueError):
    """SQLite could not compile the generated SELECT statement."""


_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
}
if hasattr(sqlite3, "SQLITE_RECURSIVE"):
    _ALLOWED_ACTIONS.add(sqlite3.SQLITE_RECURSIVE)


def _read_only_authorizer(action, arg1, arg2, database_name, trigger_name):
    return (
        sqlite3.SQLITE_OK
        if action in _ALLOWED_ACTIONS
        else sqlite3.SQLITE_DENY
    )


def get_connection():
    """Open the existing database in read-only mode (do not create it)."""
    if not DB_PATH.is_file():
        raise FileNotFoundError(f"BIPilot database not found: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH.resolve().as_uri() + "?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")  # Internal setup, before authorizer.
    conn.set_authorizer(_read_only_authorizer)
    return conn


def _validate_query_on_connection(query: str, conn: sqlite3.Connection):
    if not isinstance(query, str) or not re.match(
        r"^\s*(?:SELECT|WITH)\b", query, flags=re.IGNORECASE
    ):
        raise SQLSafetyError("Only one read-only SELECT query is allowed.")

    try:
        # Compiles the complete query without fetching its results. SQLite
        # rejects missing SELECT, unknown columns/tables and extra statements.
        conn.execute("EXPLAIN QUERY PLAN " + query).fetchall()
    except sqlite3.Error as exc:
        error = str(exc)
        if "not authorized" in error.lower() or "readonly" in error.lower():
            raise SQLSafetyError(error) from exc
        raise SQLValidationError(error) from exc


def validate_query(query: str) -> None:
    """Validate SQL without running the user's result-producing SELECT."""
    with closing(get_connection()) as conn:
        _validate_query_on_connection(query, conn)


def execute_query(query: str) -> pd.DataFrame:
    """Validate then execute a single, read-only SQLite SELECT."""
    with closing(get_connection()) as conn:
        _validate_query_on_connection(query, conn)
        return pd.read_sql_query(query, conn)
