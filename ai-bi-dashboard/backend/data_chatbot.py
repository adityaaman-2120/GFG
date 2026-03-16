from __future__ import annotations

import difflib
import io
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import Any

import pandas as pd

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover
    genai = None

logger = logging.getLogger(__name__)

FOLLOW_UP_PREFIXES = ("now", "then", "also", "and", "only", "change", "update", "switch")


@dataclass
class DatasetState:
    dataframe: pd.DataFrame | None = None
    metadata: dict[str, Any] | None = None
    source_file: str = ""


@dataclass
class SessionState:
    last_question: str | None = None
    last_plan: dict[str, Any] | None = None


_DATASET_LOCK = threading.Lock()
_DATASET_STATE = DatasetState()
_SESSION_LOCK = threading.Lock()
_SESSION_MEMORY: dict[str, SessionState] = {}


def load_dataset(csv_bytes: bytes, filename: str = "uploaded.csv") -> dict[str, Any]:
    dataframe = _read_csv_with_fallback(csv_bytes)
    if dataframe.empty:
        raise ValueError("Uploaded CSV has no rows.")

    dataframe = _sanitize_dataframe_columns(dataframe)
    dataframe = _normalize_dataframe_types(dataframe)
    metadata = analyze_dataset(dataframe)

    with _DATASET_LOCK:
        _DATASET_STATE.dataframe = dataframe
        _DATASET_STATE.metadata = metadata
        _DATASET_STATE.source_file = filename

    with _SESSION_LOCK:
        _SESSION_MEMORY.clear()

    logger.info(
        "Dataset loaded | file=%s | shape=%s | columns=%s",
        filename,
        dataframe.shape,
        metadata["columns"],
    )
    return {
        "source_file": filename,
        "row_count": int(dataframe.shape[0]),
        "columns": metadata["columns"],
        "metadata": metadata,
    }


def analyze_dataset(dataframe: pd.DataFrame) -> dict[str, Any]:
    numeric_columns = [
        column for column in dataframe.columns if pd.api.types.is_numeric_dtype(dataframe[column])
    ]
    datetime_columns = [
        column for column in dataframe.columns
        if _looks_like_datetime_column(column, dataframe[column])
    ]
    categorical_columns = [
        column
        for column in dataframe.columns
        if column not in numeric_columns and column not in datetime_columns
    ]

    possible_metrics = numeric_columns.copy()

    metadata = {
        "columns": list(dataframe.columns),
        "shape": {"rows": int(dataframe.shape[0]), "columns": int(dataframe.shape[1])},
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "possible_metrics": possible_metrics,
        "dtypes": {column: str(dtype) for column, dtype in dataframe.dtypes.items()},
    }
    return metadata


def parse_user_query(
    question: str,
    metadata: dict[str, Any],
    previous_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    lowered = question.strip().lower()
    columns = metadata["columns"]
    numeric_columns = metadata["numeric_columns"]
    categorical_columns = metadata["categorical_columns"]
    datetime_columns = metadata["datetime_columns"]

    gemini_plan = _parse_with_gemini(question, metadata)
    if gemini_plan:
        plan = _normalize_query_plan(gemini_plan, metadata)
    else:
        plan = {
            "intent": "table",
            "filters": [],
            "group_by": None,
            "aggregation": None,
            "metric": None,
            "x": None,
            "y": None,
            "sort_by": None,
            "sort_order": "desc",
            "limit": None,
            "visualization": None,
            "distribution": False,
            "comparison": False,
            "correlation": False,
        }

        mentioned_columns = _find_mentioned_columns(question, columns)
        missing_columns = _find_missing_columns(question, columns)
        missing_columns.extend(_extract_unmatched_column_references(question, columns))

        if any(token in lowered for token in ("correlation", "relate", "relationship", "scatter")):
            plan["intent"] = "correlation"
            plan["correlation"] = True
            selected_numeric = [column for column in mentioned_columns if column in numeric_columns]
            if len(selected_numeric) >= 2:
                plan["x"] = selected_numeric[0]
                plan["y"] = selected_numeric[1]
            elif len(numeric_columns) >= 2:
                plan["x"] = numeric_columns[0]
                plan["y"] = numeric_columns[1]

        elif any(token in lowered for token in ("distribution", "histogram", "spread")):
            plan["intent"] = "distribution"
            plan["distribution"] = True
            metric = _choose_metric(mentioned_columns, numeric_columns, lowered)
            plan["metric"] = metric

        else:
            aggregation = _detect_aggregation(lowered)
            group_column = _detect_group_by(question, columns, categorical_columns, datetime_columns)
            metric = _choose_metric(mentioned_columns, numeric_columns, lowered)

            if _is_ranking_query(lowered):
                if group_column in numeric_columns and not metric:
                    metric = group_column
                if not metric and group_column in numeric_columns:
                    metric = group_column
                if (not group_column or group_column in numeric_columns) and categorical_columns:
                    group_column = _pick_preferred_group_column(question, categorical_columns) or categorical_columns[0]
                if not aggregation:
                    aggregation = "sum"

            if aggregation or group_column:
                plan["intent"] = "aggregation"
            if aggregation:
                plan["aggregation"] = aggregation
            if group_column:
                plan["group_by"] = group_column
            if metric:
                plan["metric"] = metric

            if any(token in lowered for token in ("compare", "comparison", "across categories", "versus")):
                plan["comparison"] = True
                plan["intent"] = "aggregation"
                if not plan["group_by"] and categorical_columns:
                    plan["group_by"] = categorical_columns[0]
                if not plan["metric"] and numeric_columns:
                    plan["metric"] = numeric_columns[0]
                if not plan["aggregation"]:
                    plan["aggregation"] = "sum"

        plan["filters"] = _extract_filters(question, metadata)
        sort_by, sort_order = _extract_sort(lowered, columns, plan.get("metric"), plan.get("group_by"))
        plan["sort_by"] = sort_by
        plan["sort_order"] = sort_order
        plan["limit"] = _extract_limit(lowered)

        if plan["intent"] == "table" and mentioned_columns:
            # Return only referenced columns for table-style asks.
            plan["selected_columns"] = mentioned_columns[:8]
        else:
            plan["selected_columns"] = []

        plan["missing_columns"] = list(dict.fromkeys(missing_columns))

    plan["query_text"] = question

    plan = _merge_follow_up_plan(question, plan, previous_plan, metadata)
    plan = _normalize_query_plan(plan, metadata)

    logger.info("Parsed query | question=%s | plan=%s", question, plan)
    return plan


def run_dataframe_query(dataframe: pd.DataFrame, parsed_query: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    working_df = dataframe.copy()
    operation_steps: list[str] = ["df"]

    for filter_spec in parsed_query.get("filters", []):
        column = filter_spec["column"]
        operator = filter_spec["operator"]
        value = filter_spec["value"]

        if operator == "eq":
            working_df = working_df[working_df[column].astype("string").str.lower() == str(value).lower()]
            operation_steps.append(f"[{column} == {value!r}]")
        elif operator == "contains":
            working_df = working_df[
                working_df[column].astype("string").str.contains(str(value), case=False, na=False)
            ]
            operation_steps.append(f"[{column}.contains({value!r})]")
        elif operator == "gt":
            working_df = working_df[pd.to_numeric(working_df[column], errors="coerce") > float(value)]
            operation_steps.append(f"[{column} > {value}]")
        elif operator == "gte":
            working_df = working_df[pd.to_numeric(working_df[column], errors="coerce") >= float(value)]
            operation_steps.append(f"[{column} >= {value}]")
        elif operator == "lt":
            working_df = working_df[pd.to_numeric(working_df[column], errors="coerce") < float(value)]
            operation_steps.append(f"[{column} < {value}]")
        elif operator == "lte":
            working_df = working_df[pd.to_numeric(working_df[column], errors="coerce") <= float(value)]
            operation_steps.append(f"[{column} <= {value}]")

    intent = parsed_query.get("intent", "table")
    result_df = working_df.copy()

    if intent == "correlation":
        x_col = parsed_query.get("x")
        y_col = parsed_query.get("y")
        if x_col and y_col:
            result_df = result_df[[x_col, y_col]].dropna().reset_index(drop=True)
            operation_steps.append(f"[[{x_col}, {y_col}]].dropna()")

    elif intent == "distribution":
        metric = parsed_query.get("metric")
        if metric:
            metric_series = pd.to_numeric(result_df[metric], errors="coerce").dropna()
            bins = min(12, max(5, int(metric_series.nunique() ** 0.5) if not metric_series.empty else 6))
            counts, edges = pd.cut(metric_series, bins=bins, include_lowest=True).value_counts(sort=False), None
            distribution_df = counts.reset_index()
            distribution_df.columns = [metric, "count"]
            result_df = distribution_df
            operation_steps.append(f"{metric}.histogram_bins({bins})")

    elif intent == "aggregation":
        group_by = parsed_query.get("group_by")
        metric = parsed_query.get("metric")
        aggregation = parsed_query.get("aggregation") or "sum"

        if group_by and metric:
            grouped = result_df.groupby(group_by, dropna=False)[metric]
            if aggregation == "mean":
                result_df = grouped.mean().reset_index(name=f"avg_{metric}")
            elif aggregation == "count":
                result_df = grouped.count().reset_index(name=f"count_{metric}")
            elif aggregation == "max":
                result_df = grouped.max().reset_index(name=f"max_{metric}")
            elif aggregation == "min":
                result_df = grouped.min().reset_index(name=f"min_{metric}")
            else:
                result_df = grouped.sum().reset_index(name=f"total_{metric}")
            operation_steps.append(f"groupby({group_by})[{metric}].{aggregation}()")
        elif metric:
            if aggregation == "mean":
                value = pd.to_numeric(result_df[metric], errors="coerce").mean()
                result_df = pd.DataFrame([{f"avg_{metric}": float(value) if pd.notna(value) else None}])
            elif aggregation == "count":
                result_df = pd.DataFrame([{f"count_{metric}": int(result_df[metric].count())}])
            elif aggregation == "max":
                value = pd.to_numeric(result_df[metric], errors="coerce").max()
                result_df = pd.DataFrame([{f"max_{metric}": float(value) if pd.notna(value) else None}])
            elif aggregation == "min":
                value = pd.to_numeric(result_df[metric], errors="coerce").min()
                result_df = pd.DataFrame([{f"min_{metric}": float(value) if pd.notna(value) else None}])
            else:
                value = pd.to_numeric(result_df[metric], errors="coerce").sum()
                result_df = pd.DataFrame([{f"total_{metric}": float(value) if pd.notna(value) else None}])
            operation_steps.append(f"{metric}.{aggregation}()")

    selected_columns = parsed_query.get("selected_columns") or []
    if intent == "table" and selected_columns:
        keep = [column for column in selected_columns if column in result_df.columns]
        if keep:
            result_df = result_df[keep]
            operation_steps.append(f"[{keep}]")

    sort_by = parsed_query.get("sort_by")
    if sort_by and sort_by in result_df.columns:
        ascending = parsed_query.get("sort_order", "desc") == "asc"
        result_df = result_df.sort_values(sort_by, ascending=ascending, na_position="last")
        operation_steps.append(f"sort_values({sort_by}, asc={ascending})")

    limit = parsed_query.get("limit")
    if isinstance(limit, int) and limit > 0:
        result_df = result_df.head(limit)
        operation_steps.append(f"head({limit})")

    result_df = result_df.reset_index(drop=True)
    operation_trace = " -> ".join(operation_steps)

    logger.info(
        "Dataframe query executed | shape=%s | operation=%s | preview=%s",
        result_df.shape,
        operation_trace,
        result_df.head(3).to_dict(orient="records"),
    )
    return result_df, operation_trace


def generate_visualization(result_df: pd.DataFrame, parsed_query: dict[str, Any], metadata: dict[str, Any]) -> str:
    query_text = str(parsed_query.get("query_text", "")).lower()
    if result_df.empty:
        chart_type = "table"
    elif parsed_query.get("intent") == "correlation":
        chart_type = "scatter"
    elif parsed_query.get("intent") == "distribution":
        chart_type = "bar"
    elif result_df.shape[0] == 1 and result_df.shape[1] == 1:
        chart_type = "metric"
    else:
        numeric_columns = [
            column for column in result_df.columns if pd.api.types.is_numeric_dtype(result_df[column])
        ]
        datetime_columns = [
            column for column in result_df.columns
            if _looks_like_datetime_column(column, result_df[column])
        ]
        categorical_columns = [
            column for column in result_df.columns
            if column not in numeric_columns and column not in datetime_columns
        ]

        if datetime_columns and numeric_columns:
            chart_type = "line"
        elif len(numeric_columns) >= 2:
            chart_type = "scatter"
        elif len(categorical_columns) >= 1 and len(numeric_columns) >= 1:
            if result_df.shape[0] <= 8 and _is_parts_of_whole_query(query_text):
                chart_type = "pie"
            else:
                chart_type = "bar"
        else:
            chart_type = "table"

    logger.info("Visualization selected | type=%s", chart_type)
    return chart_type


def chatbot_response(question: str, session_id: str) -> dict[str, Any]:
    with _DATASET_LOCK:
        dataframe = _DATASET_STATE.dataframe.copy() if _DATASET_STATE.dataframe is not None else None
        metadata = dict(_DATASET_STATE.metadata or {})

    if dataframe is None or not metadata:
        raise ValueError("No dataset loaded. Upload a CSV file first.")

    with _SESSION_LOCK:
        session_state = _SESSION_MEMORY.setdefault(session_id, SessionState())
        previous_plan = dict(session_state.last_plan) if session_state.last_plan else None

    parsed_query = parse_user_query(question, metadata, previous_plan=previous_plan)

    graceful_message = _assess_query_clarity(question, parsed_query, metadata)
    if graceful_message:
        return {
            "data": [],
            "chart_type": "table",
            "operation": "query_needs_clarification",
            "insight": graceful_message,
            "message": graceful_message,
            "parsed_query": parsed_query,
        }

    result_df, operation = run_dataframe_query(dataframe, parsed_query)
    chart_type = generate_visualization(result_df, parsed_query, metadata)

    if result_df.empty:
        insight = "No rows matched the current query and filters."
        response = {
            "data": [],
            "chart_type": chart_type,
            "operation": operation,
            "insight": insight,
            "message": insight,
            "parsed_query": parsed_query,
        }
    else:
        response = {
            "data": result_df.head(5000).to_dict(orient="records"),
            "chart_type": chart_type,
            "operation": operation,
            "insight": _build_insight(question, result_df, parsed_query),
            "message": None,
            "parsed_query": parsed_query,
        }

    with _SESSION_LOCK:
        session_state.last_question = question
        session_state.last_plan = parsed_query

    return response


def get_dataset_metadata() -> dict[str, Any] | None:
    with _DATASET_LOCK:
        if not _DATASET_STATE.metadata:
            return None
        return dict(_DATASET_STATE.metadata)


def _parse_with_gemini(question: str, metadata: dict[str, Any]) -> dict[str, Any] | None:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not genai or not api_key or api_key == "your_google_api_key_here":
        return None

    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    prompt = f"""
You are a data query parser for a pandas dataframe chatbot.
Return JSON only.

Dataset columns: {metadata['columns']}
Numeric columns: {metadata['numeric_columns']}
Categorical columns: {metadata['categorical_columns']}
Datetime columns: {metadata['datetime_columns']}

User question: {question}

Return JSON with this schema:
{{
  "intent": "table|aggregation|correlation|distribution",
  "filters": [{{"column": "", "operator": "eq|contains|gt|gte|lt|lte", "value": ""}}],
  "group_by": "column or null",
  "aggregation": "sum|mean|count|max|min|null",
  "metric": "column or null",
  "x": "column or null",
  "y": "column or null",
  "sort_by": "column or null",
  "sort_order": "asc|desc",
  "limit": 0,
  "visualization": "bar|line|scatter|histogram|table|metric|null"
}}

Use only the listed columns. If unsure, return nulls instead of inventing.
""".strip()

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        text = (getattr(response, "text", "") or "").strip()
        if not text:
            return None
        cleaned = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:  # noqa: BLE001
        return None
    return None


def _normalize_query_plan(plan: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(plan)
    columns = metadata["columns"]

    normalized.setdefault("filters", [])
    normalized.setdefault("sort_order", "desc")
    normalized.setdefault("intent", "table")
    normalized.setdefault("selected_columns", [])
    normalized.setdefault("missing_columns", [])
    normalized.setdefault("query_text", "")

    for key in ("group_by", "metric", "x", "y", "sort_by"):
        value = normalized.get(key)
        if isinstance(value, str):
            matched = _match_column_name(value, columns)
            normalized[key] = matched
            if value and not matched:
                normalized["missing_columns"].append(value)
        else:
            normalized[key] = None

    normalized_filters: list[dict[str, Any]] = []
    for filter_item in normalized.get("filters", []):
        if not isinstance(filter_item, dict):
            continue
        column_raw = str(filter_item.get("column", "")).strip()
        column = _match_column_name(column_raw, columns)
        if not column:
            if column_raw:
                normalized["missing_columns"].append(column_raw)
            continue
        operator = str(filter_item.get("operator", "eq")).lower()
        if operator not in {"eq", "contains", "gt", "gte", "lt", "lte"}:
            operator = "eq"
        normalized_filters.append(
            {
                "column": column,
                "operator": operator,
                "value": filter_item.get("value"),
            }
        )
    normalized["filters"] = normalized_filters

    limit = normalized.get("limit")
    if isinstance(limit, (int, float)) and int(limit) > 0:
        normalized["limit"] = min(int(limit), 5000)
    else:
        normalized["limit"] = None

    normalized["missing_columns"] = list(dict.fromkeys(normalized.get("missing_columns", [])))
    return normalized


def _merge_follow_up_plan(
    question: str,
    plan: dict[str, Any],
    previous_plan: dict[str, Any] | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    lowered = question.strip().lower()
    is_follow_up = any(lowered.startswith(prefix + " ") for prefix in FOLLOW_UP_PREFIXES)
    if not previous_plan or not is_follow_up:
        return plan

    merged = dict(previous_plan)
    merged_filters = list(previous_plan.get("filters", []))

    # Merge new filters and modifiers while preserving previous intent/axes.
    if plan.get("filters"):
        merged_filters.extend(plan["filters"])
    merged["filters"] = merged_filters

    for key in ("group_by", "metric", "x", "y", "sort_by", "sort_order", "limit", "intent", "aggregation"):
        if plan.get(key) not in (None, "", []):
            merged[key] = plan[key]

    if plan.get("selected_columns"):
        merged["selected_columns"] = plan["selected_columns"]

    merged["missing_columns"] = list(dict.fromkeys(plan.get("missing_columns", [])))
    merged["query_text"] = question
    return _normalize_query_plan(merged, metadata)


def _find_mentioned_columns(question: str, columns: list[str]) -> list[str]:
    lowered = question.lower()
    matched: list[str] = []
    for column in columns:
        variants = (column.lower(), column.lower().replace("_", " "))
        if any(re.search(rf"\b{re.escape(variant)}\b", lowered) for variant in variants):
            matched.append(column)
    return matched


def _find_missing_columns(question: str, columns: list[str]) -> list[str]:
    missing: list[str] = []
    phrase_patterns = (
        r"between\s+([a-zA-Z0-9_ ]+)\s+and\s+([a-zA-Z0-9_ ]+)",
        r"where\s+([a-zA-Z0-9_ ]+)\s*(?:=|is|equals)",
    )
    known = {column.lower() for column in columns}
    known_spaced = {column.lower().replace("_", " ") for column in columns}
    ignored_tokens = {
        "category",
        "categories",
        "values",
        "value",
        "data",
        "top",
        "bottom",
        "highest",
        "lowest",
        "compare",
        "comparison",
        "distribution",
        "correlation",
    }

    for pattern in phrase_patterns:
        for match in re.findall(pattern, question, flags=re.IGNORECASE):
            values = list(match) if isinstance(match, tuple) else [match]
            for value in values:
                token = _clean_column_phrase(value)
                if not token:
                    continue
                if token.isdigit():
                    continue
                if token in known or token in known_spaced:
                    continue
                if token in ignored_tokens:
                    continue
                if token not in missing:
                    missing.append(token)

    # Lightweight check for explicit "column ..." references.
    for phrase in re.findall(r"column\s+([a-zA-Z0-9_ ]+)", question, flags=re.IGNORECASE):
        token = _clean_column_phrase(phrase)
        if not token or token in known or token in known_spaced or token in ignored_tokens:
            continue
        if token not in missing:
            missing.append(token)

    return missing


def _extract_unmatched_column_references(question: str, columns: list[str]) -> list[str]:
    unmatched: list[str] = []
    patterns = (
        r"\b(?:by|per|across)\s+([a-zA-Z0-9_ ]+)",
        r"\bcorrelation\s+between\s+([a-zA-Z0-9_ ]+)\s+and\s+([a-zA-Z0-9_ ]+)",
        r"\bplot\s+([a-zA-Z0-9_ ]+)\s+by\s+([a-zA-Z0-9_ ]+)",
        r"\bcompare\s+([a-zA-Z0-9_ ]+)\s+and\s+([a-zA-Z0-9_ ]+)",
    )

    ignored = {
        "category",
        "categories",
        "value",
        "values",
        "data",
        "everything",
        "all",
        "overall",
    }

    for pattern in patterns:
        for match in re.findall(pattern, question, flags=re.IGNORECASE):
            values = list(match) if isinstance(match, tuple) else [match]
            for value in values:
                token = _clean_column_phrase(value)
                if not token or token in ignored or token.isdigit():
                    continue
                if _match_column_name(token, columns):
                    continue
                if token not in unmatched:
                    unmatched.append(token)
    return unmatched


def _assess_query_clarity(question: str, parsed_query: dict[str, Any], metadata: dict[str, Any]) -> str | None:
    lowered = question.strip().lower()
    missing_columns = parsed_query.get("missing_columns", [])
    if missing_columns:
        suggestions = []
        for missing in missing_columns:
            alternatives = difflib.get_close_matches(str(missing), metadata["columns"], n=3, cutoff=0.35)
            if alternatives:
                suggestions.append(f"{missing} -> {', '.join(alternatives)}")
        message = "Some requested columns were not found in this dataset."
        if suggestions:
            message += " Similar columns: " + " | ".join(suggestions)
        return message

    if _is_highly_complex_query(lowered):
        return (
            "This prompt mixes multiple tasks at once. Please ask one at a time, such as: "
            "'show trend of <metric> over <time column>' or 'show correlation between <num1> and <num2>'."
        )

    if _is_vague_query(question, lowered, parsed_query, metadata):
        sample_numeric = ", ".join(metadata["numeric_columns"][:3]) or "a numeric column"
        sample_group = ", ".join(metadata["categorical_columns"][:3]) or "a category column"
        return (
            "Please be more specific. Try queries like: "
            f"'show average {sample_numeric.split(', ')[0]} by {sample_group.split(', ')[0]}' "
            "or 'show correlation between <num1> and <num2>'."
        )

    return None


def _is_highly_complex_query(lowered_question: str) -> bool:
    clusters = [
        ("trend", "over time", "daily", "monthly", "line"),
        ("correlation", "scatter", "relationship"),
        ("distribution", "histogram", "spread"),
        ("compare", "comparison", "versus"),
        ("filter", "where", "only"),
    ]
    matched_clusters = sum(1 for cluster in clusters if any(token in lowered_question for token in cluster))
    return matched_clusters >= 3


def _is_vague_query(
    original_question: str,
    lowered_question: str,
    parsed_query: dict[str, Any],
    metadata: dict[str, Any],
) -> bool:
    vague_markers = (
        "something",
        "anything",
        "insight",
        "analyze",
        "analysis",
        "what about",
        "help me",
        "useful",
        "performance",
    )
    has_vague_marker = any(marker in lowered_question for marker in vague_markers)
    if not has_vague_marker:
        return False

    has_column_reference = bool(_find_mentioned_columns(original_question, metadata["columns"]))
    explicit_action_markers = (
        " by ",
        " where ",
        " filter ",
        " correlation ",
        " distribution ",
        " histogram ",
        " compare ",
        " top ",
        " sort ",
        "plot ",
        " chart ",
    )
    padded = f" {lowered_question} "
    has_explicit_action = any(marker in padded for marker in explicit_action_markers)

    # If user gave neither concrete columns nor concrete action, request clarification.
    return not has_column_reference and not has_explicit_action


def _detect_aggregation(question_lower: str) -> str | None:
    if any(token in question_lower for token in ("average", "avg", "mean")):
        return "mean"
    if any(token in question_lower for token in ("count", "how many", "number of")):
        return "count"
    if any(token in question_lower for token in ("maximum", "max", "highest")):
        return "max"
    if any(token in question_lower for token in ("minimum", "min", "lowest")):
        return "min"
    if any(token in question_lower for token in ("sum", "total")):
        return "sum"
    return None


def _detect_group_by(
    question: str,
    columns: list[str],
    categorical_columns: list[str],
    datetime_columns: list[str],
) -> str | None:
    by_match = re.search(r"\b(?:by|per|across)\s+([a-zA-Z0-9_ ]+)", question, flags=re.IGNORECASE)
    if by_match:
        raw_phrase = by_match.group(1).strip()
        candidate = _match_column_name(raw_phrase, columns)
        if not candidate:
            cleaned_phrase = _clean_column_phrase(raw_phrase)
            candidate = _match_column_name(cleaned_phrase, columns)
        if candidate:
            return candidate

    mentioned = _find_mentioned_columns(question, columns)
    for column in mentioned:
        if column in categorical_columns or column in datetime_columns:
            return column

    return None


def _choose_metric(mentioned_columns: list[str], numeric_columns: list[str], lowered_question: str) -> str | None:
    for column in mentioned_columns:
        if column in numeric_columns:
            return column

    keyword_hints = (
        "revenue",
        "price",
        "amount",
        "cost",
        "usage",
        "memory",
        "cpu",
        "count",
        "rate",
        "score",
    )
    for column in numeric_columns:
        if any(keyword in column.lower() for keyword in keyword_hints) and any(keyword in lowered_question for keyword in keyword_hints):
            return column

    return numeric_columns[0] if numeric_columns else None


def _extract_filters(question: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = []
    columns = metadata["columns"]
    numeric_columns = set(metadata["numeric_columns"])

    for column in columns:
        variants = (column.lower(), column.lower().replace("_", " "))
        pattern_prefix = "(?:" + "|".join(re.escape(variant) for variant in variants) + ")"

        numeric_patterns = [
            (re.compile(pattern_prefix + r"\s*(?:>=|greater than or equal to)\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE), "gte"),
            (re.compile(pattern_prefix + r"\s*(?:<=|less than or equal to)\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE), "lte"),
            (re.compile(pattern_prefix + r"\s*(?:>|greater than)\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE), "gt"),
            (re.compile(pattern_prefix + r"\s*(?:<|less than)\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE), "lt"),
            (re.compile(pattern_prefix + r"\s*(?:=|is|equals|equal to)\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE), "eq"),
        ]

        string_patterns = [
            (re.compile(pattern_prefix + r"\s*(?:=|is|equals|equal to|as)\s*['\"]?([a-z0-9_ .\-]+)['\"]?", re.IGNORECASE), "eq"),
            (re.compile(pattern_prefix + r"\s*(?:contains|like)\s*['\"]?([a-z0-9_ .\-]+)['\"]?", re.IGNORECASE), "contains"),
        ]

        if column in numeric_columns:
            for pattern, operator in numeric_patterns:
                match = pattern.search(question)
                if match:
                    filters.append({"column": column, "operator": operator, "value": float(match.group(1))})
                    break
        else:
            for pattern, operator in string_patterns:
                match = pattern.search(question)
                if match:
                    filters.append({"column": column, "operator": operator, "value": match.group(1).strip()})
                    break

    return filters


def _extract_sort(
    lowered: str,
    columns: list[str],
    metric: str | None,
    group_by: str | None,
) -> tuple[str | None, str]:
    sort_order = "desc"
    if any(token in lowered for token in ("ascending", "low to high", "smallest")):
        sort_order = "asc"

    sort_match = re.search(r"sort(?:ed)?\s+by\s+([a-zA-Z0-9_ ]+)", lowered)
    if sort_match:
        sort_by = _match_column_name(sort_match.group(1), columns)
        if sort_by:
            return sort_by, sort_order

    if any(token in lowered for token in ("top", "highest", "largest", "bottom", "lowest")):
        if any(token in lowered for token in ("bottom", "lowest", "smallest")):
            sort_order = "asc"
        return metric or group_by, sort_order

    return None, sort_order


def _extract_limit(lowered: str) -> int | None:
    match = re.search(r"\b(?:top|limit|first|last)\s+(\d{1,4})\b", lowered)
    if not match:
        return None
    limit = int(match.group(1))
    return max(1, min(limit, 5000))


def _match_column_name(raw_value: str, columns: list[str]) -> str | None:
    direct_candidate = raw_value.strip().lower().replace(" ", "_")
    if direct_candidate:
        for column in columns:
            if column.lower() == direct_candidate:
                return column
            if column.lower().replace("_", " ") == raw_value.strip().lower():
                return column

    candidate = _clean_column_phrase(raw_value).replace(" ", "_")
    if not candidate:
        return None

    for column in columns:
        if column.lower() == candidate:
            return column
        if column.lower().replace("_", " ") == raw_value.strip().lower():
            return column

    matches = difflib.get_close_matches(candidate, [column.lower() for column in columns], n=1, cutoff=0.75)
    if matches:
        match = matches[0]
        for column in columns:
            if column.lower() == match:
                return column
    return None


def _clean_column_phrase(raw_value: str) -> str:
    cleaned = raw_value.strip().lower()
    cleaned = re.sub(r"\b(top|bottom|highest|lowest|first|last|limit)\s+\d+\b", "", cleaned)
    cleaned = re.sub(r"\b(categories|values|value)\b", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _is_ranking_query(lowered_question: str) -> bool:
    return any(token in lowered_question for token in ("top", "highest", "largest", "bottom", "lowest"))


def _pick_preferred_group_column(question: str, categorical_columns: list[str]) -> str | None:
    lowered = question.lower()
    for column in categorical_columns:
        spaced = column.replace("_", " ").lower()
        if spaced in lowered or column.lower() in lowered:
            return column
    return None


def _is_parts_of_whole_query(lowered_question: str) -> bool:
    return any(
        token in lowered_question
        for token in (
            "share",
            "proportion",
            "percentage",
            "percent",
            "part of whole",
            "pie",
            "composition",
            "breakdown",
        )
    )


def _build_insight(question: str, result_df: pd.DataFrame, parsed_query: dict[str, Any]) -> str:
    if result_df.empty:
        return "No data found for this query."

    if parsed_query.get("intent") == "correlation" and result_df.shape[1] >= 2:
        x_col, y_col = result_df.columns[0], result_df.columns[1]
        corr = result_df[x_col].corr(result_df[y_col])
        if pd.notna(corr):
            return f"Correlation between {x_col} and {y_col} is {corr:.3f}."

    if result_df.shape[0] == 1 and result_df.shape[1] == 1:
        key = result_df.columns[0]
        value = result_df.iloc[0, 0]
        return f"{key.replace('_', ' ').title()} is {value}."

    numeric_columns = [
        column for column in result_df.columns if pd.api.types.is_numeric_dtype(result_df[column])
    ]
    if len(result_df.columns) >= 2 and numeric_columns:
        label_col = result_df.columns[0]
        value_col = numeric_columns[0]
        top_row = result_df.sort_values(value_col, ascending=False).iloc[0]
        return f"Top {label_col.replace('_', ' ')} is {top_row[label_col]} with {top_row[value_col]:,.2f}."

    return f"Returned {len(result_df)} rows for: {question.strip()}"


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
            return pd.read_csv(io.StringIO(text), sep=None, engine="python")
        except Exception as error:  # noqa: BLE001
            last_error = error

    raise ValueError(
        "Could not decode or parse the CSV file. Please export it as CSV UTF-8 and try again."
    ) from last_error


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


def _normalize_dataframe_types(dataframe: pd.DataFrame) -> pd.DataFrame:
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

        datetime_values = pd.to_datetime(as_text, errors="coerce")
        datetime_ratio = float(datetime_values.notna().sum()) / float(non_null_count)
        if datetime_ratio >= 0.85 and _looks_like_datetime_column(column, series):
            normalized[column] = datetime_values.dt.strftime("%Y-%m-%d")
            continue

        normalized[column] = as_text

    return normalized


def _looks_like_datetime_column(column_name: str, series: pd.Series) -> bool:
    lowered_name = column_name.lower()
    if any(token in lowered_name for token in ("date", "time", "timestamp", "month", "year", "day")):
        return True

    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    sample = series.dropna().astype("string").head(20)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce")
    return float(parsed.notna().sum()) / float(len(sample)) >= 0.8
