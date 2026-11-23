"""Execute the cohort SQL file against a pandas frame with DuckDB."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from readmission.config import SQL_PATH


def load_named_queries(path: Path | None = None) -> list[tuple[str, str]]:
    """Parse ``-- name:`` blocks into (name, sql) pairs."""
    text = (path or SQL_PATH).read_text(encoding="utf-8")
    queries: list[tuple[str, str]] = []
    name: str | None = None
    buffer: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("-- name:"):
            if name and "".join(buffer).strip():
                queries.append((name, "\n".join(buffer).strip().rstrip(";")))
            name = stripped.split(":", 1)[1].strip()
            buffer = []
            continue
        if stripped.startswith("--"):
            continue
        buffer.append(line)
    if name and "".join(buffer).strip():
        queries.append((name, "\n".join(buffer).strip().rstrip(";")))
    if not queries:
        raise ValueError(f"No named queries found in {path or SQL_PATH}.")
    return queries


def run_cohort_queries(
    cohort: pd.DataFrame,
    path: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Register ``encounters`` and run every named query."""
    connection = duckdb.connect()
    try:
        connection.register("encounters", cohort)
        results = {}
        for name, sql in load_named_queries(path):
            results[name] = connection.execute(sql).fetchdf()
        return results
    finally:
        connection.close()
