from __future__ import annotations

import json
import os
import re
import warnings
from typing import Any, Final

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import google.generativeai as genai
except ImportError:  # pragma: no cover
    genai = None

ALLOWED_TABLE: Final[str] = "sales"
FORBIDDEN_SQL_PATTERNS: Final[tuple[str, ...]] = (
    " insert ",
    " update ",
    " delete ",
    " drop ",
    " alter ",
    " create ",
    " attach ",
    " detach ",
    " pragma ",
)
ALLOWED_COLUMNS: Final[set[str]] = {
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
}

FOLLOW_UP_PREFIXES: Final[tuple[str, ...]] = (
    "now",
    "then",
    "also",
    "and",
    "filter",
    "only",
)

REGION_VALUES: Final[dict[str, str]] = {
    "asia": "Asia",
    "europe": "Europe",
    "africa": "Africa",
    "oceania": "Oceania",
    "north america": "North America",
    "south america": "South America",
    "middle east": "Middle East",
}


def modify_sql(
    previous_sql: str,
    follow_up_question: str,
    table_name: str = "sales",
    allowed_columns: set[str] | None = None,
) -> str:
    """Modify a previously generated SQL query based on a follow-up question.

    Uses Gemini when available and falls back to deterministic follow-up rules.
    Every returned query is validated for table/column safety.
    """
    compact_previous = " ".join(previous_sql.strip().split()).rstrip(";")
    if not compact_previous:
        raise ValueError("Missing previous SQL query.")

    question = follow_up_question.strip()
    if not question:
        raise ValueError("Follow-up question cannot be empty.")

    effective_allowed = allowed_columns or set(ALLOWED_COLUMNS)
    schema_columns = "\n".join(f"- {column}" for column in sorted(effective_allowed))

    if genai and os.getenv("GOOGLE_API_KEY"):
        model_sql = _modify_with_gemini(
            previous_sql=compact_previous,
            follow_up_question=question,
            table_name=table_name,
            schema_columns=schema_columns,
        )
        if model_sql:
            try:
                return validate_sql_query(
                    model_sql,
                    table_name=table_name,
                    allowed_columns=effective_allowed,
                )
            except ValueError:
                pass

    return apply_follow_up_to_sql(
        question,
        compact_previous,
        table_name=table_name,
        allowed_columns=effective_allowed,
    )


def generate_sql_query(
    question: str,
    schema_prompt: str,
    table_name: str = "sales",
    allowed_columns: dict[str, str] | None = None,
) -> str:
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    if table_name == "sales" and _is_ambiguous_or_unsupported(question):
        raise ValueError("Question cannot be answered with available schema.")

    effective_columns = set((allowed_columns or {}).keys()) or set(ALLOWED_COLUMNS)

    if genai and os.getenv("GOOGLE_API_KEY"):
        sql_query = _generate_with_gemini(question, schema_prompt)
        if sql_query:
            try:
                return validate_sql_query(
                    sql_query,
                    table_name=table_name,
                    allowed_columns=effective_columns,
                )
            except ValueError:
                # Fall back to deterministic rules for unsupported/invalid model SQL.
                pass

    if table_name == "sales":
        rules_sql = _generate_with_rules(question)
    else:
        rules_sql = _generate_with_generic_rules(question, table_name, allowed_columns or {})

    if rules_sql == "NOT_POSSIBLE":
        raise ValueError("Question cannot be answered with available schema.")
    return validate_sql_query(
        rules_sql,
        table_name=table_name,
        allowed_columns=effective_columns,
    )


def is_follow_up_question(question: str) -> bool:
    normalized = question.strip().lower()
    if not normalized:
        return False

    if any(normalized.startswith(prefix + " ") for prefix in FOLLOW_UP_PREFIXES):
        return True

    followup_markers = (
        "same",
        "previous",
        "above",
        "that",
        "those",
        "filter",
    )
    return any(marker in normalized for marker in followup_markers)


def apply_follow_up_to_sql(
    question: str,
    previous_sql: str,
    table_name: str = "sales",
    allowed_columns: set[str] | None = None,
) -> str:
    compact_query = " ".join(previous_sql.strip().split()).rstrip(";")
    if not compact_query:
        raise ValueError("Missing previous SQL query.")

    effective_allowed = allowed_columns or set(ALLOWED_COLUMNS)

    region_value = _extract_region_value(question)
    payment_method = _extract_payment_method(question)
    new_limit = _extract_limit(question)
    generic_conditions = _extract_follow_up_conditions(question, effective_allowed)

    if not region_value and not payment_method and new_limit is None and not generic_conditions:
        raise ValueError("Unsupported follow-up request.")

    updated_query = compact_query
    applied_change = False

    if region_value:
        if "customer_region" not in effective_allowed:
            raise ValueError("Unsupported follow-up request.")
        safe_region = region_value.replace("'", "''")
        condition = f"customer_region = '{safe_region}'"
        updated_query = _append_condition(updated_query, condition)
        applied_change = True

    if payment_method:
        if "payment_method" not in effective_allowed:
            raise ValueError("Unsupported follow-up request.")
        safe_method = payment_method.replace("'", "''")
        condition = f"payment_method = '{safe_method}'"
        updated_query = _append_condition(updated_query, condition)
        applied_change = True

    for condition in generic_conditions:
        updated_query = _append_condition(updated_query, condition)
        applied_change = True

    if new_limit is not None:
        updated_query = _replace_or_append_limit(updated_query, new_limit)
        applied_change = True

    if not applied_change:
        raise ValueError("Unsupported follow-up request.")

    return validate_sql_query(
        updated_query,
        table_name=table_name,
        allowed_columns=effective_allowed,
    )


def _extract_follow_up_conditions(question: str, allowed_columns: set[str]) -> list[str]:
    conditions: list[str] = []
    for column in sorted(allowed_columns):
        variants = (column.lower(), column.lower().replace("_", " "))
        pattern_prefix = "(?:" + "|".join(re.escape(variant) for variant in variants) + ")"

        number_pattern = re.compile(
            pattern_prefix + r"\s*(?:=|is|equals|equal to)\s*(-?\d+(?:\.\d+)?)",
            re.IGNORECASE,
        )
        text_pattern = re.compile(
            pattern_prefix + r"\s*(?:=|is|equals|equal to|as|like)\s*['\"]?([a-z0-9_ .\-]+)['\"]?",
            re.IGNORECASE,
        )

        number_match = number_pattern.search(question)
        if number_match:
            conditions.append(f"{column} = {number_match.group(1)}")
            continue

        text_match = text_pattern.search(question)
        if text_match:
            raw_value = text_match.group(1).strip()
            if raw_value:
                safe_value = raw_value.replace("'", "''")
                conditions.append(f"{column} = '{safe_value}'")

    return conditions


def validate_sql_query(
    sql_query: str,
    table_name: str = "sales",
    allowed_columns: set[str] | None = None,
) -> str:
    compact_query = " ".join(sql_query.strip().split())
    compact_query = compact_query.rstrip(";")
    lowered_query = f" {compact_query.lower()} "
    allowed_table = table_name.lower()
    effective_allowed = allowed_columns or set(ALLOWED_COLUMNS)

    if not compact_query:
        raise ValueError("Generated SQL query is empty.")
    if ";" in compact_query:
        raise ValueError("Multiple SQL statements are not allowed.")
    if not (lowered_query.strip().startswith("select") or lowered_query.strip().startswith("with")):
        raise ValueError("Only SELECT queries are allowed.")
    if f" {allowed_table} " not in lowered_query:
        raise ValueError(f"Query must target the {table_name} table.")
    if any(pattern in lowered_query for pattern in FORBIDDEN_SQL_PATTERNS):
        raise ValueError("Generated SQL contains forbidden statements.")
    if not _uses_only_allowed_columns(compact_query, effective_allowed):
        raise ValueError("Query uses unsupported columns.")

    return compact_query


def _is_ambiguous_or_unsupported(question: str) -> bool:
    lowered = question.lower()
    padded = f" {lowered} "

    # The rules engine does not support arbitrary SQL-like condition parsing.
    if " where " in padded or " having " in padded:
        return True

    known_intents = (
        "revenue",
        "sales",
        "order",
        "orders",
        "quantity",
        "units",
        "category",
        "region",
        "payment",
        "rating",
        "review",
        "discount",
        "product",
        "date",
        "daily",
        "monthly",
        "trend",
        "summary",
        "overview",
        "count",
        "top",
        "average",
        "avg",
        "scatter",
        "correlation",
    )

    # Reject questions that do not map to any known sales-domain intent.
    if not any(token in lowered for token in known_intents):
        return True

    # Reject explicit mentions of fields that don't exist in the schema.
    unsupported_hints = (
        "country",
        "city",
        "state",
        "customer age",
        "gender",
        "profit",
        "margin",
        "shipping",
        "warehouse",
        "supplier",
        "inventory",
        "stock",
    )
    return any(token in lowered for token in unsupported_hints)


def _extract_region_value(question: str) -> str | None:
    lowered = question.lower()
    for token, value in REGION_VALUES.items():
        if token in lowered:
            return value
    return None


def _extract_payment_method(question: str) -> str | None:
    lowered = question.lower()
    method_map = {
        "credit card": "Credit Card",
        "debit card": "Debit Card",
        "paypal": "PayPal",
        "cash": "Cash",
        "upi": "UPI",
        "bank transfer": "Bank Transfer",
    }
    for token, value in method_map.items():
        if token in lowered:
            return value
    return None


def _extract_limit(question: str) -> int | None:
    lowered = question.lower()
    match = re.search(r"(?:top|limit)\s+(\d{1,3})\b", lowered)
    if not match:
        return None
    limit = int(match.group(1))
    return max(1, min(limit, 500))


def _append_condition(sql_query: str, condition: str) -> str:
    lowered = sql_query.lower()
    clause_positions = [
        index
        for index in (
            lowered.find(" group by "),
            lowered.find(" order by "),
            lowered.find(" limit "),
            lowered.find(" having "),
        )
        if index != -1
    ]

    where_index = lowered.find(" where ")
    if where_index != -1:
        end_index = min([index for index in clause_positions if index > where_index], default=len(sql_query))
        return f"{sql_query[:end_index]} AND {condition}{sql_query[end_index:]}"

    insert_at = min(clause_positions, default=len(sql_query))
    prefix = sql_query[:insert_at].rstrip()
    suffix = sql_query[insert_at:]
    spacer = " " if suffix and not suffix.startswith(" ") else ""
    return f"{prefix} WHERE {condition}{spacer}{suffix}"


def _replace_or_append_limit(sql_query: str, limit: int) -> str:
    lowered = sql_query.lower()
    limit_match = re.search(r"\blimit\s+\d+\b", lowered)
    if limit_match:
        start, end = limit_match.span()
        return f"{sql_query[:start]}LIMIT {limit}{sql_query[end:]}"
    return f"{sql_query} LIMIT {limit}"


def gemini_nl_to_sql(question: str) -> str:
    """Convert a natural language question to SQLite SQL using Gemini.

    Returns:
    - A validated SQLite SELECT query targeting only the `sales` table and allowed columns
    - "NOT_POSSIBLE" when the question cannot be answered under the schema/rules
    """
    question = question.strip()
    if not question:
        return "NOT_POSSIBLE"

    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not genai or not api_key or api_key == "your_google_api_key_here":
        return "NOT_POSSIBLE"

    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    schema_columns = "\n".join(f"- {column}" for column in sorted(ALLOWED_COLUMNS))
    prompt = f"""
You convert natural language questions into SQL for SQLite.

Dataset schema:
Table: sales
Columns:
{schema_columns}

Rules:
- Only generate SQLite compatible SQL.
- Only use table: sales.
- Only use the listed columns.
- Output one SQL statement only.
- If query cannot be answered from this schema, return exactly: NOT_POSSIBLE

Return strict JSON only:
{{"sql_query": "SELECT ..."}}
or
{{"sql_query": "NOT_POSSIBLE"}}

Question: {question}
""".strip()

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        text = (getattr(response, "text", "") or "").strip()
        if not text:
            return "NOT_POSSIBLE"

        cleaned = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            payload = json.loads(cleaned)
            sql_query = str(payload.get("sql_query", "")).strip()
        except json.JSONDecodeError:
            sql_query = cleaned

        if sql_query.upper() == "NOT_POSSIBLE":
            return "NOT_POSSIBLE"

        validated_query = validate_sql_query(sql_query)
        if not _uses_only_allowed_columns(validated_query, set(ALLOWED_COLUMNS)):
            return "NOT_POSSIBLE"

        return validated_query

    except Exception:  # noqa: BLE001
        return "NOT_POSSIBLE"


def _uses_only_allowed_columns(sql_query: str, allowed_columns: set[str]) -> bool:
    lowered = sql_query.lower()
    normalized = re.sub(r"'[^']*'", "''", lowered)

    # Validate any explicit table.column references.
    qualified_identifiers = re.findall(r"\b[a-z_][a-z0-9_]*\.([a-z_][a-z0-9_]*)\b", normalized)
    if any(identifier not in allowed_columns for identifier in qualified_identifiers):
        return False

    table_identifiers = set(
        re.findall(r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)\b", normalized)
    )
    alias_identifiers = set(
        re.findall(r"\bas\s+([a-z_][a-z0-9_]*)\b", normalized)
    )

    # If identifiers that look like columns are present but none match allowed columns,
    # the query likely references unsupported fields.
    identifiers = set(re.findall(r"\b[a-z_][a-z0-9_]*\b", normalized))
    known_tokens = {
        "select",
        "from",
        "where",
        "group",
        "by",
        "order",
        "limit",
        "having",
        "as",
        "and",
        "or",
        "not",
        "in",
        "is",
        "null",
        "like",
        "between",
        "case",
        "when",
        "then",
        "else",
        "end",
        "asc",
        "desc",
        "with",
        "distinct",
        "round",
        "sum",
        "avg",
        "min",
        "max",
        "count",
        "substr",
        "cast",
        "integer",
        "real",
        "text",
        "sales",
    }
    known_tokens.update(table_identifiers)
    known_tokens.update(alias_identifiers)
    unknown_tokens = identifiers - allowed_columns - known_tokens

    # Allow short aliases and common metric aliases.
    allowed_aliases = {
        "x",
        "y",
        "z",
        "m",
        "n",
        "value",
        "metric",
        "month",
        "total",
        "cnt",
        "avg_revenue",
        "total_revenue",
        "total_orders",
        "average_rating",
        "avg_discount",
        "quantity_sold",
    }
    unknown_tokens = {token for token in unknown_tokens if token not in allowed_aliases}
    return not unknown_tokens


def generate_insight(question: str, rows: list[dict[str, Any]]) -> str:
    """Return a one-sentence business insight for the given query results.

    Tries Gemini first; falls back to a rule-based summary when the API key is
    absent or the call fails.
    """
    if not rows:
        return f"No data found for: {question.strip()}"

    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if genai and api_key and api_key != "your_google_api_key_here":
        insight = _gemini_insight(question, rows, api_key)
        if insight:
            return insight

    return _rules_insight(question, rows)


def _gemini_insight(question: str, rows: list[dict[str, Any]], api_key: str) -> str | None:
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    # Send at most 20 rows to keep the prompt short.
    sample = rows[:20]
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        prompt = f"""You are a business analyst. Given the question and data below, write ONE concise sentence of insight (max 30 words). Mention specific numbers or percentages where helpful. Do not start with "I" or "The data shows".

Question: {question}

Data (JSON):
{json.dumps(sample, indent=2)}

Respond with the insight sentence only — no preamble, no markdown.""".strip()

        response = model.generate_content(prompt)
        text = (getattr(response, "text", "") or "").strip()
        # Reject empty or multi-sentence responses
        if text and len(text) < 300:
            return text
    except Exception:  # noqa: BLE001
        pass
    return None


def _rules_insight(question: str, rows: list[dict[str, Any]]) -> str:
    """Simple rule-based fallback insight."""
    columns = list(rows[0].keys())

    # Single metric (e.g. COUNT(*))
    if len(rows) == 1 and len(columns) == 1:
        col = columns[0].replace("_", " ")
        return f"{col.title()} is {rows[0][columns[0]]:,}." if isinstance(rows[0][columns[0]], int | float) else f"{col.title()} is {rows[0][columns[0]]}."

    # Find the first numeric value column
    label_col = columns[0]
    value_col = next(
        (c for c in columns[1:] if isinstance(rows[0].get(c), int | float)),
        None,
    )
    if not value_col:
        return f"Returned {len(rows)} rows."

    total = sum(float(r[value_col]) for r in rows if r.get(value_col) is not None)
    top = max(rows, key=lambda r: float(r.get(value_col) or 0))
    top_label = top[label_col]
    top_value = float(top[value_col])
    pct = round(top_value / total * 100, 1) if total else 0
    metric = value_col.replace("_", " ")
    return (
        f"{top_label} leads with {top_value:,.2f} {metric}"
        + (f", contributing {pct}% of total." if pct else ".")
    )


def _generate_with_gemini(question: str, schema_prompt: str) -> str | None:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key or api_key == "your_google_api_key_here":
        return None

    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        prompt = f"""
You generate safe SQLite SELECT queries for a business intelligence dashboard.

{schema_prompt}

Return JSON only with this shape:
{{"sql_query": "SELECT ..."}}

User question: {question}
""".strip()
        response = model.generate_content(prompt)
        text = getattr(response, "text", "") or ""
        if not text:
            return None

        cleaned_text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            payload = json.loads(cleaned_text)
        except json.JSONDecodeError:
            match = re.search(r"SELECT .*", text, re.IGNORECASE | re.DOTALL)
            return match.group(0).strip() if match else None
        return payload.get("sql_query")

    except Exception:  # noqa: BLE001 — any API or network failure falls back to rules
        return None


def _modify_with_gemini(
    previous_sql: str,
    follow_up_question: str,
    table_name: str,
    schema_columns: str,
) -> str | None:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not genai or not api_key or api_key == "your_google_api_key_here":
        return None

    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    prompt = f"""
You revise SQLite SQL queries for a BI dashboard.

Schema constraints:
- Table: {table_name}
- Allowed columns:
{schema_columns}

Rules:
- Modify the previous SQL to satisfy the follow-up question.
- Keep intent from previous SQL unless follow-up explicitly changes it.
- Return exactly one SQLite SELECT query.
- Use only the allowed table and columns.
- Do not invent columns.
- Do not return markdown.
- If impossible under constraints, return exactly: NOT_POSSIBLE

Previous SQL:
{previous_sql}

Follow-up question:
{follow_up_question}

Return strict JSON only:
{{"sql_query": "SELECT ..."}}
or
{{"sql_query": "NOT_POSSIBLE"}}
""".strip()

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        text = (getattr(response, "text", "") or "").strip()
        if not text:
            return None

        cleaned_text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            payload = json.loads(cleaned_text)
            sql_query = (payload.get("sql_query") or "").strip()
        except json.JSONDecodeError:
            sql_query = text

        if not sql_query or sql_query.upper() == "NOT_POSSIBLE":
            return None

        # Accept first SELECT statement if model includes extra text.
        match = re.search(r"\bselect\b.*", sql_query, re.IGNORECASE | re.DOTALL)
        return match.group(0).strip() if match else None
    except Exception:  # noqa: BLE001
        return None


def _generate_with_rules(question: str) -> str:
    lower_question = question.lower()

    # ── category breakdowns ──────────────────────────────────────────────────
    if "by category" in lower_question or "per category" in lower_question:
        if "rating" in lower_question:
            return (
                "SELECT product_category, ROUND(AVG(rating), 2) AS average_rating "
                "FROM sales GROUP BY product_category ORDER BY average_rating DESC LIMIT 10"
            )
        if "quantity" in lower_question or "units" in lower_question or "sold" in lower_question:
            return (
                "SELECT product_category, SUM(quantity_sold) AS quantity_sold "
                "FROM sales GROUP BY product_category ORDER BY quantity_sold DESC LIMIT 10"
            )
        if "discount" in lower_question:
            return (
                "SELECT product_category, ROUND(AVG(discount_percent), 2) AS avg_discount "
                "FROM sales GROUP BY product_category ORDER BY avg_discount DESC LIMIT 10"
            )
        return (
            "SELECT product_category, ROUND(SUM(total_revenue), 2) AS total_revenue "
            "FROM sales GROUP BY product_category ORDER BY total_revenue DESC LIMIT 10"
        )

    # ── region breakdowns ────────────────────────────────────────────────────
    if "by region" in lower_question or "per region" in lower_question or "show revenue by region" in lower_question:
        if "quantity" in lower_question or "units" in lower_question:
            return (
                "SELECT customer_region, SUM(quantity_sold) AS quantity_sold "
                "FROM sales GROUP BY customer_region ORDER BY quantity_sold DESC LIMIT 10"
            )
        if "rating" in lower_question:
            return (
                "SELECT customer_region, ROUND(AVG(rating), 2) AS average_rating "
                "FROM sales GROUP BY customer_region ORDER BY average_rating DESC LIMIT 10"
            )
        return (
            "SELECT customer_region, ROUND(SUM(total_revenue), 2) AS total_revenue "
            "FROM sales GROUP BY customer_region ORDER BY total_revenue DESC LIMIT 10"
        )

    # ── payment method breakdowns ────────────────────────────────────────────
    if "payment" in lower_question:
        if "count" in lower_question or "orders" in lower_question or "number" in lower_question:
            return (
                "SELECT payment_method, COUNT(*) AS total_orders "
                "FROM sales GROUP BY payment_method ORDER BY total_orders DESC LIMIT 10"
            )
        return (
            "SELECT payment_method, ROUND(SUM(total_revenue), 2) AS total_revenue "
            "FROM sales GROUP BY payment_method ORDER BY total_revenue DESC LIMIT 10"
        )

    # ── top / best products ──────────────────────────────────────────────────
    if "top" in lower_question and "product" in lower_question:
        if "rating" in lower_question:
            return (
                "SELECT product_id, product_category, ROUND(AVG(rating), 2) AS average_rating "
                "FROM sales GROUP BY product_id ORDER BY average_rating DESC LIMIT 10"
            )
        if "quantity" in lower_question or "sold" in lower_question:
            return (
                "SELECT product_id, product_category, SUM(quantity_sold) AS quantity_sold "
                "FROM sales GROUP BY product_id ORDER BY quantity_sold DESC LIMIT 10"
            )
        return (
            "SELECT product_id, product_category, ROUND(SUM(total_revenue), 2) AS total_revenue "
            "FROM sales GROUP BY product_id ORDER BY total_revenue DESC LIMIT 10"
        )

    if "best product" in lower_question or "best selling" in lower_question:
        return (
            "SELECT product_id, product_category, ROUND(SUM(total_revenue), 2) AS total_revenue "
            "FROM sales GROUP BY product_id ORDER BY total_revenue DESC LIMIT 10"
        )

    # ── discount metrics ─────────────────────────────────────────────────────
    if "discount" in lower_question:
        if "by category" in lower_question or "per category" in lower_question:
            return (
                "SELECT product_category, ROUND(AVG(discount_percent), 2) AS avg_discount "
                "FROM sales GROUP BY product_category ORDER BY avg_discount DESC LIMIT 10"
            )
        if "highest" in lower_question:
            return (
                "SELECT product_category, ROUND(MAX(discount_percent), 2) AS max_discount "
                "FROM sales GROUP BY product_category ORDER BY max_discount DESC LIMIT 10"
            )
        return "SELECT ROUND(AVG(discount_percent), 2) AS average_discount_percent FROM sales"

    # ── rating metrics ───────────────────────────────────────────────────────
    if "average rating" in lower_question or "avg rating" in lower_question:
        return "SELECT ROUND(AVG(rating), 2) AS average_rating FROM sales"

    if "rating" in lower_question and ("by category" in lower_question or "per category" in lower_question):
        return (
            "SELECT product_category, ROUND(AVG(rating), 2) AS average_rating "
            "FROM sales GROUP BY product_category ORDER BY average_rating DESC LIMIT 10"
        )

    # ── scatter / correlation ── (must come before generic revenue check) ────
    if "scatter" in lower_question or "correlation" in lower_question:
        if "discount" in lower_question and "rating" in lower_question:
            return (
                "SELECT discount_percent, ROUND(AVG(rating), 2) AS average_rating "
                "FROM sales GROUP BY discount_percent ORDER BY discount_percent ASC LIMIT 50"
            )
        if "quantity" in lower_question or "units" in lower_question:
            return (
                "SELECT quantity_sold, ROUND(AVG(total_revenue), 2) AS avg_revenue "
                "FROM sales GROUP BY quantity_sold ORDER BY quantity_sold ASC LIMIT 50"
            )
        return (
            "SELECT ROUND(price, 0) AS price, ROUND(SUM(total_revenue), 2) AS total_revenue "
            "FROM sales GROUP BY ROUND(price, 0) ORDER BY price ASC LIMIT 50"
        )

    # ── revenue over time ────────────────────────────────────────────────────
    if ("total revenue" in lower_question or "revenue" in lower_question) and (
        "daily" in lower_question
        or "by date" in lower_question
        or "over time" in lower_question
        or "trend" in lower_question
        or "monthly" in lower_question
        or "time" in lower_question
    ):
        if "monthly" in lower_question:
            return (
                "SELECT substr(order_date, 1, 7) AS month, ROUND(SUM(total_revenue), 2) AS total_revenue "
                "FROM sales GROUP BY month ORDER BY month ASC LIMIT 24"
            )
        return (
            "SELECT order_date, ROUND(SUM(total_revenue), 2) AS total_revenue "
            "FROM sales GROUP BY order_date ORDER BY order_date ASC LIMIT 30"
        )

    # ── aggregate revenue ────────────────────────────────────────────────────
    if "total revenue" in lower_question or (
        "revenue" in lower_question and "breakdown" not in lower_question
    ):
        return "SELECT ROUND(SUM(total_revenue), 2) AS total_revenue FROM sales"

    # ── order counts ─────────────────────────────────────────────────────────
    if (
        "total orders" in lower_question
        or "number of orders" in lower_question
        or "how many orders" in lower_question
        or "count" in lower_question
    ):
        return "SELECT COUNT(*) AS total_orders FROM sales"

    # ── quantity sold ────────────────────────────────────────────────────────
    if "quantity" in lower_question or "units sold" in lower_question:
        return "SELECT SUM(quantity_sold) AS total_units_sold FROM sales"


    # ── summary / overview ───────────────────────────────────────────────────
    if "summary" in lower_question or "overview" in lower_question or "dashboard" in lower_question:
        return (
            "SELECT 'Total Orders' AS metric, CAST(COUNT(*) AS TEXT) AS value FROM sales "
            "UNION ALL SELECT 'Total Revenue', CAST(ROUND(SUM(total_revenue), 2) AS TEXT) FROM sales "
            "UNION ALL SELECT 'Avg Rating', CAST(ROUND(AVG(rating), 2) AS TEXT) FROM sales "
            "UNION ALL SELECT 'Avg Discount %', CAST(ROUND(AVG(discount_percent), 2) AS TEXT) FROM sales"
        )

    # ── default fallback ─────────────────────────────────────────────────────
    return "NOT_POSSIBLE"


def _generate_with_generic_rules(
    question: str,
    table_name: str,
    allowed_columns: dict[str, str],
) -> str:
    lowered = question.lower()
    padded = f" {lowered} "
    explicit_limit = _extract_limit(question)
    columns = list(allowed_columns.keys())
    numeric_columns = [
        name for name, dtype in allowed_columns.items()
        if any(token in (dtype or "").upper() for token in ("INT", "REAL", "NUM", "DEC", "FLOAT", "DOUBLE"))
    ]
    non_numeric_columns = [name for name in columns if name not in numeric_columns]
    date_columns = [
        name
        for name in columns
        if "date" in name or "time" in name or "month" in name or "year" in name
    ]
    inferred_filters = _extract_generic_filters(question, allowed_columns)

    if not columns:
        return "NOT_POSSIBLE"

    if "count" in lowered or "how many" in lowered:
        base_sql = f"SELECT COUNT(*) AS total_rows FROM {table_name}"
        if inferred_filters:
            base_sql = _append_where_clause(base_sql, inferred_filters)
        return base_sql

    metric_column = _best_matching_column(lowered, numeric_columns)
    group_column = _best_matching_column(lowered, columns)

    # Semantic column preference for business-style questions.
    if not metric_column:
        if any(token in lowered for token in ("revenue", "sales", "amount", "value", "income", "turnover")):
            metric_column = _pick_by_preferred_keywords(
                numeric_columns,
                (
                    "total_revenue",
                    "revenue",
                    "sales",
                    "amount",
                    "value",
                    "income",
                    "turnover",
                    "price",
                    "total",
                ),
            )
        elif any(token in lowered for token in ("quantity", "unit", "volume")):
            metric_column = _pick_by_keywords(numeric_columns, ("quantity", "unit", "volume", "count", "qty"))
        elif any(token in lowered for token in ("rating", "score", "review")):
            metric_column = _pick_by_keywords(numeric_columns, ("rating", "score", "review", "stars"))
        elif any(token in lowered for token in ("discount", "margin", "profit")):
            metric_column = _pick_by_keywords(numeric_columns, ("discount", "margin", "profit"))

    if not metric_column and numeric_columns:
        metric_column = numeric_columns[0]

    if not group_column:
        if "region" in lowered:
            group_column = _pick_by_keywords(non_numeric_columns, ("region", "country", "state", "zone", "market"))
        elif "category" in lowered:
            group_column = _pick_by_keywords(non_numeric_columns, ("category", "segment", "type", "class"))
        elif "product" in lowered or "item" in lowered:
            group_column = _pick_by_keywords(non_numeric_columns, ("product", "item", "sku", "name"))
        elif "payment" in lowered or "channel" in lowered or "method" in lowered:
            group_column = _pick_by_keywords(non_numeric_columns, ("payment", "channel", "method", "mode"))
        elif date_columns and any(token in lowered for token in ("date", "day", "month", "year", "time", "trend", "over time")):
            group_column = date_columns[0]

    # Trend/time style questions.
    if metric_column and date_columns and any(token in lowered for token in ("trend", "over time", "daily", "monthly", "time")):
        date_col = group_column if group_column in date_columns else date_columns[0]
        sql = (
            f"SELECT {date_col}, ROUND(SUM({metric_column}), 2) AS total_{metric_column} "
            f"FROM {table_name} GROUP BY {date_col} ORDER BY {date_col} ASC"
        )
        sql = _append_where_clause(sql, inferred_filters)
        if explicit_limit is not None:
            sql = f"{sql} LIMIT {explicit_limit}"
        return sql

    if metric_column and ("average" in lowered or "avg" in lowered):
        if group_column and group_column != metric_column and (" by " in padded or " per " in padded):
            sql = (
                f"SELECT {group_column}, ROUND(AVG({metric_column}), 2) AS average_{metric_column} "
                f"FROM {table_name} GROUP BY {group_column} ORDER BY average_{metric_column} DESC"
            )
            sql = _append_where_clause(sql, inferred_filters)
            if explicit_limit is not None:
                sql = f"{sql} LIMIT {explicit_limit}"
            return sql
        sql = f"SELECT ROUND(AVG({metric_column}), 2) AS average_{metric_column} FROM {table_name}"
        sql = _append_where_clause(sql, inferred_filters)
        return sql

    if metric_column and ("sum" in lowered or "total" in lowered):
        if group_column and group_column != metric_column and (" by " in padded or " per " in padded):
            sql = (
                f"SELECT {group_column}, ROUND(SUM({metric_column}), 2) AS total_{metric_column} "
                f"FROM {table_name} GROUP BY {group_column} ORDER BY total_{metric_column} DESC"
            )
            sql = _append_where_clause(sql, inferred_filters)
            if explicit_limit is not None:
                sql = f"{sql} LIMIT {explicit_limit}"
            return sql
        sql = f"SELECT ROUND(SUM({metric_column}), 2) AS total_{metric_column} FROM {table_name}"
        sql = _append_where_clause(sql, inferred_filters)
        return sql

    if group_column and metric_column and (" by " in padded or " per " in padded):
        sql = (
            f"SELECT {group_column}, ROUND(SUM({metric_column}), 2) AS total_{metric_column} "
            f"FROM {table_name} GROUP BY {group_column} ORDER BY total_{metric_column} DESC"
        )
        sql = _append_where_clause(sql, inferred_filters)
        if explicit_limit is not None:
            sql = f"{sql} LIMIT {explicit_limit}"
        return sql

    if metric_column and "top" in lowered and group_column and group_column != metric_column:
        top_limit = explicit_limit if explicit_limit is not None else 10
        sql = (
            f"SELECT {group_column}, ROUND(SUM({metric_column}), 2) AS total_{metric_column} "
            f"FROM {table_name} GROUP BY {group_column} ORDER BY total_{metric_column} DESC LIMIT {top_limit}"
        )
        sql = _append_where_clause(sql, inferred_filters)
        return sql

    if any(token in lowered for token in ("show", "list", "preview", "sample", "data")):
        preview_limit = explicit_limit if explicit_limit is not None else 50
        sql = f"SELECT * FROM {table_name}"
        sql = _append_where_clause(sql, inferred_filters)
        return f"{sql} LIMIT {preview_limit}"

    sql = f"SELECT * FROM {table_name}"
    sql = _append_where_clause(sql, inferred_filters)
    if explicit_limit is not None:
        sql = f"{sql} LIMIT {explicit_limit}"
    return sql


def _extract_generic_filters(question: str, allowed_columns: dict[str, str]) -> list[str]:
    lowered = question.lower()
    has_filter_intent = any(token in lowered for token in ("filter", "where", "only", "with", "equals", "is"))
    if not has_filter_intent:
        return []

    conditions: list[str] = []
    for column, dtype in allowed_columns.items():
        column_variants = {column.lower(), column.lower().replace("_", " ")}
        pattern_prefix = "(?:" + "|".join(re.escape(variant) for variant in column_variants) + ")"

        string_pattern = re.compile(
            pattern_prefix + r"\s*(?:=|is|equals|equal to|as|like)\s*['\"]?([a-z0-9_ .\-]+)['\"]?",
            re.IGNORECASE,
        )
        number_pattern = re.compile(
            pattern_prefix + r"\s*(?:=|is|equals|equal to)\s*(-?\d+(?:\.\d+)?)",
            re.IGNORECASE,
        )

        if any(token in (dtype or "").upper() for token in ("INT", "REAL", "NUM", "DEC", "FLOAT", "DOUBLE")):
            match = number_pattern.search(question)
            if match:
                conditions.append(f"{column} = {match.group(1)}")
                continue

        match = string_pattern.search(question)
        if match:
            raw_value = match.group(1).strip()
            if raw_value:
                safe_value = raw_value.replace("'", "''")
                conditions.append(f"{column} = '{safe_value}'")

    return conditions


def _append_where_clause(sql_query: str, conditions: list[str]) -> str:
    if not conditions:
        return sql_query

    lowered = sql_query.lower()
    clause_positions = [
        index
        for index in (
            lowered.find(" group by "),
            lowered.find(" order by "),
            lowered.find(" limit "),
            lowered.find(" having "),
        )
        if index != -1
    ]

    condition_sql = " AND ".join(conditions)
    where_index = lowered.find(" where ")
    if where_index != -1:
        end_index = min([index for index in clause_positions if index > where_index], default=len(sql_query))
        return f"{sql_query[:end_index]} AND {condition_sql}{sql_query[end_index:]}"

    insert_at = min(clause_positions, default=len(sql_query))
    prefix = sql_query[:insert_at].rstrip()
    suffix = sql_query[insert_at:]
    spacer = " " if suffix and not suffix.startswith(" ") else ""
    return f"{prefix} WHERE {condition_sql}{spacer}{suffix}"


def _best_matching_column(question_lower: str, candidates: list[str]) -> str | None:
    padded = f" {question_lower} "
    for candidate in candidates:
        spaced = candidate.replace("_", " ")
        if f" {candidate} " in padded or f" {spaced} " in padded:
            return candidate
    return None


def _pick_by_keywords(candidates: list[str], keywords: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if any(keyword in candidate for keyword in keywords):
            return candidate
    return None


def _pick_by_preferred_keywords(candidates: list[str], preferred_keywords: tuple[str, ...]) -> str | None:
    for keyword in preferred_keywords:
        for candidate in candidates:
            if keyword in candidate:
                return candidate
    return None
