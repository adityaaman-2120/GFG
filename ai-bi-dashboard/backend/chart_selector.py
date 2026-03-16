from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd


def choose_chart_type(dataframe: pd.DataFrame) -> str:
    """Inspect a DataFrame and return 'line', 'bar', or 'pie'.

    Rules (evaluated in priority order):
    1. Any column whose name contains 'date' or whose values parse as ISO dates → line
    2. Exactly one categorical column + one or more numeric columns → bar
    3. One categorical column with ≤ 8 distinct values + one numeric column → pie
    Fallback: bar
    """
    if dataframe.empty:
        return "bar"

    categorical_cols = [
        col for col in dataframe.columns
        if pd.api.types.is_object_dtype(dataframe[col])
        or pd.api.types.is_string_dtype(dataframe[col])
        and not pd.api.types.is_numeric_dtype(dataframe[col])
    ]
    numeric_cols = [
        col for col in dataframe.columns
        if pd.api.types.is_numeric_dtype(dataframe[col])
    ]

    # Rule 1 — date column present → line
    for col in dataframe.columns:
        if "date" in col.lower():
            return "line"
        if pd.api.types.is_object_dtype(dataframe[col]) or pd.api.types.is_string_dtype(dataframe[col]):
            sample = dataframe[col].dropna().head(5)
            if len(sample) > 0 and all(_is_iso_date(v) for v in sample):
                return "line"

    # Rule 3 — small category distribution → pie
    # (checked before generic bar so a small-cardinality categorical wins pie)
    if len(categorical_cols) == 1 and len(numeric_cols) >= 1:
        n_unique = dataframe[categorical_cols[0]].nunique()
        if n_unique <= 8:
            return "pie"

    # Rule 2 — categorical vs numeric → bar
    if categorical_cols and numeric_cols:
        return "bar"

    return "bar"


def _is_iso_date(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        date.fromisoformat(value)
        return True
    except ValueError:
        return False


def select_chart_type(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "table"

    first_row = rows[0]
    columns = list(first_row.keys())
    if len(columns) == 1:
        return "metric"

    numeric_columns = [column for column in columns if _is_number(first_row.get(column))]

    # Multi-row name/value pairs (e.g. summary UNION queries) → table
    if len(columns) == 2 and len(rows) > 1:
        x_value = first_row[columns[0]]
        if _looks_like_date(x_value):
            return "line"
        if not _is_number(x_value) and len(numeric_columns) == 1:
            return "bar"
        if not _is_number(x_value) and not _is_number(first_row[columns[1]]):
            return "table"
        if len(numeric_columns) == 2:
            return "scatter"

    if len(rows) <= 10 and numeric_columns:
        return "bar"

    return "table"


def build_insight(question: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f"No results matched the question: {question.strip()}"

    first_row = rows[0]
    columns = list(first_row.keys())
    if len(rows) == 1 and len(columns) == 1:
        column = columns[0].replace("_", " ")
        return f"{column.title()} is {first_row[columns[0]]}."

    if len(columns) >= 2:
        label_column = columns[0]
        value_column = next(
            (column for column in columns[1:] if _is_number(first_row.get(column))),
            None,
        )
        if value_column:
            top_row = max(rows, key=lambda row: float(row[value_column]))
            label = top_row[label_column]
            value = round(float(top_row[value_column]), 2)
            metric = value_column.replace("_", " ")
            return f"Top result is {label} with {value} {metric}."

    return f"Returned {len(rows)} rows for the requested question."


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _looks_like_date(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        date.fromisoformat(value)
        return True
    except ValueError:
        return False
