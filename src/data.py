"""Read the SQL-cleaned Telco and Orange analysis tables.

Data ingestion, typing, null normalization, deduplication, and constraints belong
in the SQL scripts.  This module intentionally performs no database cleaning.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import pandas as pd


TELCO_REQUIRED_COLUMNS = {"customer_id", "churn_value"}
ORANGE_REQUIRED_COLUMNS = {"campaign_row_id", "t", "y"}
_SQL_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class AnalysisData:
    """The two independent datasets used by the project."""

    telco: pd.DataFrame
    orange: pd.DataFrame


def _database_url(database_url: str | None) -> str:
    url = database_url or os.getenv("DATABASE_URL")
    if not url:
        raise ValueError(
            "Pass database_url or set DATABASE_URL, for example "
            "postgresql+psycopg://user:password@localhost/telco_customer_db"
        )
    return url


def _validate_identifier(value: str, label: str) -> None:
    if not _SQL_IDENTIFIER.fullmatch(value):
        raise ValueError(f"Invalid SQL {label}: {value!r}")


def _validate_columns(
    frame: pd.DataFrame, required: set[str], source: str
) -> pd.DataFrame:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"{source} contains no rows")
    return frame


def load_sql_table(
    database_url: str | None,
    table: str,
    *,
    schema: str = "public",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load a previously cleaned PostgreSQL table without transforming it."""

    _validate_identifier(table, "table name")
    _validate_identifier(schema, "schema name")
    if columns:
        for column in columns:
            _validate_identifier(column, "column name")

    try:
        from sqlalchemy import create_engine
    except ImportError as exc:
        raise ImportError(
            "Database loading requires sqlalchemy and a PostgreSQL driver; "
            "install sqlalchemy and psycopg[binary]."
        ) from exc

    engine = create_engine(_database_url(database_url))
    try:
        return pd.read_sql_table(
            table_name=table,
            con=engine,
            schema=schema,
            columns=columns,
        )
    finally:
        engine.dispose()


def load_telco_from_postgres(
    database_url: str | None = None,
    *,
    table: str = "customer_retention",
    schema: str = "public",
) -> pd.DataFrame:
    """Load the SQL-cleaned customer table used for churn modeling."""

    frame = load_sql_table(database_url, table, schema=schema)
    return _validate_columns(frame, TELCO_REQUIRED_COLUMNS, f"{schema}.{table}")


def load_orange_from_postgres(
    database_url: str | None = None,
    *,
    table: str = "orange_campaign",
    schema: str = "public",
) -> pd.DataFrame:
    """Load the SQL-cleaned randomized campaign table used for A/B testing."""

    frame = load_sql_table(database_url, table, schema=schema)
    return _validate_columns(frame, ORANGE_REQUIRED_COLUMNS, f"{schema}.{table}")


def load_analysis_data(
    database_url: str | None = None,
    *,
    telco_table: str = "customer_retention",
    orange_table: str = "orange_campaign",
    schema: str = "public",
) -> AnalysisData:
    """Load both SQL-cleaned datasets through one notebook-friendly call."""

    return AnalysisData(
        telco=load_telco_from_postgres(
            database_url, table=telco_table, schema=schema
        ),
        orange=load_orange_from_postgres(
            database_url, table=orange_table, schema=schema
        ),
    )
