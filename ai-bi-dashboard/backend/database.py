from __future__ import annotations

import html
import io
import plistlib
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_CANDIDATES = [
    BASE_DIR / "data" / "amazon_sales.csv",
    BASE_DIR / "data" / "Amazon Sales.csv",
]
DB_PATH = BASE_DIR / "database" / "sales.db"
TABLE_NAME = "sales"
ACTIVE_TABLE = TABLE_NAME
ACTIVE_TABLE_COLUMNS: dict[str, str] = {
    "order_id": "INTEGER",
    "order_date": "TEXT",
    "product_id": "INTEGER",
    "product_category": "TEXT",
    "price": "REAL",
    "discount_percent": "REAL",
    "quantity_sold": "INTEGER",
    "customer_region": "TEXT",
    "payment_method": "TEXT",
    "rating": "REAL",
    "review_count": "INTEGER",
    "discounted_price": "REAL",
    "total_revenue": "REAL",
}
ACTIVE_TABLE_LOCK = threading.Lock()
EXPECTED_COLUMNS = [
    "order_id",
    "order_date",
    "product_id",
    "product_category",
    "price",
    "discount_percent",
    "quantity_sold",
    "customer_region",
    "payment_method",
    "rating",
    "review_count",
    "discounted_price",
    "total_revenue",
]

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    order_id INTEGER PRIMARY KEY,
    order_date TEXT NOT NULL,
    product_id INTEGER NOT NULL,
    product_category TEXT NOT NULL,
    price REAL NOT NULL,
    discount_percent REAL NOT NULL,
    quantity_sold INTEGER NOT NULL,
    customer_region TEXT NOT NULL,
    payment_method TEXT NOT NULL,
    rating REAL NOT NULL,
    review_count INTEGER NOT NULL,
    discounted_price REAL NOT NULL,
    total_revenue REAL NOT NULL
)
"""


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def ensure_database() -> None:
    with get_connection() as connection:
        connection.execute(CREATE_TABLE_SQL)
        row_count = connection.execute(
            f"SELECT COUNT(*) AS count FROM {TABLE_NAME}"
        ).fetchone()["count"]
        if row_count:
            with ACTIVE_TABLE_LOCK:
                global ACTIVE_TABLE, ACTIVE_TABLE_COLUMNS
                ACTIVE_TABLE = TABLE_NAME
                ACTIVE_TABLE_COLUMNS = _fetch_table_columns(connection, TABLE_NAME)
            return
        sync_sales_table_from_csv(connection)
        with ACTIVE_TABLE_LOCK:
            ACTIVE_TABLE = TABLE_NAME
            ACTIVE_TABLE_COLUMNS = _fetch_table_columns(connection, TABLE_NAME)


def execute_query(sql_query: str) -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(sql_query).fetchall()
    return [dict(row) for row in rows]


def execute_sql(query: str) -> pd.DataFrame:
    """Run a SELECT query against sales.db and return the results as a DataFrame."""
    with sqlite3.connect(DB_PATH) as connection:
        return pd.read_sql_query(query, connection)


def table_schema_prompt() -> str:
    with ACTIVE_TABLE_LOCK:
        active_table = ACTIVE_TABLE
        columns = ACTIVE_TABLE_COLUMNS.copy()

    column_lines = "\n".join(f"- {name} {dtype}" for name, dtype in columns.items())
    return f"""
Table name: {active_table}
Columns:
{column_lines}
Rules:
- Only generate a single SQLite SELECT query.
- Use the {active_table} table only.
- Prefer aggregates and LIMIT 20 for grouped results unless the user asks for a full export.
- When the question asks for ranking, sort descending on the relevant metric.
""".strip()


def get_active_schema() -> tuple[str, dict[str, str]]:
    with ACTIVE_TABLE_LOCK:
        return ACTIVE_TABLE, ACTIVE_TABLE_COLUMNS.copy()


def load_uploaded_csv(csv_bytes: bytes, filename: str = "uploaded.csv") -> dict[str, Any]:
    if not csv_bytes:
        raise ValueError("Uploaded CSV file is empty.")

    if csv_bytes.startswith(b"bplist00"):
        csv_text = extract_text_from_webarchive(csv_bytes)
        dataframe = pd.read_csv(io.StringIO(csv_text), sep=None, engine="python")
    else:
        dataframe = _read_csv_with_fallback(csv_bytes)

    if dataframe.empty:
        raise ValueError("Uploaded CSV has no rows.")

    dataframe = _sanitize_dataframe_columns(dataframe)
    dataframe = _normalize_uploaded_dataframe(dataframe)

    table_suffix = int(time.time())
    table_name = f"uploaded_{table_suffix}"

    with get_connection() as connection:
        dataframe.to_sql(table_name, connection, if_exists="replace", index=False)
        columns = _fetch_table_columns(connection, table_name)

    with ACTIVE_TABLE_LOCK:
        global ACTIVE_TABLE, ACTIVE_TABLE_COLUMNS
        ACTIVE_TABLE = table_name
        ACTIVE_TABLE_COLUMNS = columns

    return {
        "table_name": table_name,
        "row_count": int(len(dataframe.index)),
        "columns": list(columns.keys()),
        "source_file": filename,
    }


def _read_csv_with_fallback(csv_bytes: bytes) -> pd.DataFrame:
    encodings = (
        "utf-8-sig",
        "utf-8",
        "utf-16",
        "utf-16-le",
        "utf-16-be",
        "cp1252",
        "cp1251",
        "latin-1",
    )
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            text = csv_bytes.decode(encoding)
            # sep=None lets pandas infer delimiter (comma/semicolon/tab) for uploaded files.
            return pd.read_csv(io.StringIO(text), sep=None, engine="python")
        except (UnicodeDecodeError, LookupError) as error:
            last_error = error
        except pd.errors.ParserError as error:
            last_error = error

    # Final attempt with replacement to avoid hard decode failures.
    try:
        fallback_text = csv_bytes.decode("latin-1", errors="replace")
        dataframe = pd.read_csv(io.StringIO(fallback_text), sep=None, engine="python")
        if dataframe.empty:
            raise ValueError("Uploaded CSV has no readable rows.")
        return dataframe
    except Exception as error:  # noqa: BLE001
        last_error = error

    raise ValueError(
        "Could not decode or parse the CSV file. Please export it as CSV UTF-8 and try again."
    ) from last_error


def sync_sales_table_from_csv(connection: sqlite3.Connection) -> None:
    dataset_path = resolve_dataset_path()
    dataframe = load_sales_dataframe(dataset_path)
    dataframe.to_sql(
        TABLE_NAME,
        connection,
        if_exists="append",
        index=False,
        dtype={
            "order_id": "INTEGER",
            "order_date": "TEXT",
            "product_id": "INTEGER",
            "product_category": "TEXT",
            "price": "REAL",
            "discount_percent": "REAL",
            "quantity_sold": "INTEGER",
            "customer_region": "TEXT",
            "payment_method": "TEXT",
            "rating": "REAL",
            "review_count": "INTEGER",
            "discounted_price": "REAL",
            "total_revenue": "REAL",
        },
    )


def resolve_dataset_path() -> Path:
    for dataset_path in DATASET_CANDIDATES:
        if dataset_path.exists():
            return dataset_path
    candidate_list = ", ".join(str(path) for path in DATASET_CANDIDATES)
    raise FileNotFoundError(f"Dataset not found. Checked: {candidate_list}")


def load_sales_dataframe(dataset_path: Path) -> pd.DataFrame:
    raw_bytes = dataset_path.read_bytes()
    if raw_bytes.startswith(b"bplist00"):
        csv_text = extract_text_from_webarchive(raw_bytes)
        dataframe = pd.read_csv(io.StringIO(csv_text))
    else:
        dataframe = _read_csv_with_fallback(raw_bytes)

    missing_columns = [column for column in EXPECTED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    dataframe = dataframe[EXPECTED_COLUMNS].copy()
    dataframe["order_id"] = pd.to_numeric(dataframe["order_id"], errors="raise").astype("int64")
    dataframe["order_date"] = pd.to_datetime(
        dataframe["order_date"],
        errors="raise",
    ).dt.strftime("%Y-%m-%d")
    dataframe["product_id"] = pd.to_numeric(dataframe["product_id"], errors="raise").astype("int64")
    dataframe["product_category"] = dataframe["product_category"].astype("string").str.strip()
    dataframe["price"] = pd.to_numeric(dataframe["price"], errors="raise").astype("float64")
    dataframe["discount_percent"] = pd.to_numeric(
        dataframe["discount_percent"], errors="raise"
    ).astype("float64")
    dataframe["quantity_sold"] = pd.to_numeric(
        dataframe["quantity_sold"], errors="raise"
    ).astype("int64")
    dataframe["customer_region"] = dataframe["customer_region"].astype("string").str.strip()
    dataframe["payment_method"] = dataframe["payment_method"].astype("string").str.strip()
    dataframe["rating"] = pd.to_numeric(dataframe["rating"], errors="raise").astype("float64")
    dataframe["review_count"] = pd.to_numeric(
        dataframe["review_count"], errors="raise"
    ).astype("int64")
    dataframe["discounted_price"] = pd.to_numeric(
        dataframe["discounted_price"], errors="raise"
    ).astype("float64")
    dataframe["total_revenue"] = pd.to_numeric(
        dataframe["total_revenue"], errors="raise"
    ).astype("float64")
    return dataframe


def read_dataset_text(dataset_path: Path) -> str:
    raw_bytes = dataset_path.read_bytes()
    if raw_bytes.startswith(b"bplist00"):
        return extract_text_from_webarchive(raw_bytes)
    return _decode_text_with_fallback(raw_bytes)


def _decode_text_with_fallback(raw_bytes: bytes) -> str:
    encodings = (
        "utf-8-sig",
        "utf-8",
        "utf-16",
        "utf-16-le",
        "utf-16-be",
        "cp1252",
        "cp1251",
        "latin-1",
    )
    for encoding in encodings:
        try:
            return raw_bytes.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw_bytes.decode("latin-1", errors="replace")


def extract_text_from_webarchive(raw_bytes: bytes) -> str:
    archive = plistlib.loads(raw_bytes)
    html_document = archive["WebMainResource"]["WebResourceData"].decode(
        "utf-8", errors="replace"
    )
    match = re.search(r"<pre[^>]*>(?P<payload>.*)</pre>", html_document, re.DOTALL)
    if not match:
        raise ValueError("Could not locate embedded CSV text in dataset file.")
    return html.unescape(match.group("payload")).strip()


def _sanitize_dataframe_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    normalized: list[str] = []
    used: set[str] = set()

    for idx, column in enumerate(dataframe.columns):
        name = str(column).strip().lower()
        name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
        if not name:
            name = f"column_{idx + 1}"
        if name[0].isdigit():
            name = f"col_{name}"

        candidate = name
        suffix = 2
        while candidate in used:
            candidate = f"{name}_{suffix}"
            suffix += 1

        used.add(candidate)
        normalized.append(candidate)

    renamed = dataframe.copy()
    renamed.columns = normalized
    return renamed


def _normalize_uploaded_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    normalized = dataframe.copy()

    for column in normalized.columns:
        series = normalized[column]

        if pd.api.types.is_numeric_dtype(series):
            continue

        as_text = series.astype("string").str.strip()
        lowered = as_text.str.lower()
        as_text = as_text.mask(lowered.isin({"", "nan", "none", "null", "na", "n/a"}))

        non_null_count = int(as_text.notna().sum())
        if non_null_count == 0:
            normalized[column] = as_text
            continue

        numeric_candidate = (
            as_text
            .str.replace(r"[\$,₹€£]", "", regex=True)
            .str.replace(",", "", regex=False)
            .str.replace(r"%$", "", regex=True)
            .str.replace(r"^\((.+)\)$", r"-\1", regex=True)
        )
        numeric_values = pd.to_numeric(numeric_candidate, errors="coerce")
        numeric_ratio = float(numeric_values.notna().sum()) / float(non_null_count)
        if numeric_ratio >= 0.85:
            normalized[column] = numeric_values
            continue

        if any(token in column for token in ("date", "time", "day", "month", "year")):
            datetime_values = pd.to_datetime(as_text, errors="coerce")
            datetime_ratio = float(datetime_values.notna().sum()) / float(non_null_count)
            if datetime_ratio >= 0.85:
                normalized[column] = datetime_values.dt.strftime("%Y-%m-%d")
                continue

        normalized[column] = as_text

    return normalized


def _fetch_table_columns(connection: sqlite3.Connection, table_name: str) -> dict[str, str]:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    if not rows:
        raise ValueError(f"Table {table_name} has no columns.")

    return {
        row["name"]: (row["type"] or "TEXT")
        for row in rows
    }
