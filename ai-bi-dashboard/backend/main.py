from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load .env file if python-dotenv is available (optional but helpful locally)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from .chart_selector import select_chart_type
    from .database import ensure_database, execute_query, get_active_schema, load_uploaded_csv, table_schema_prompt
    from .gemini import generate_insight, generate_sql_query, is_follow_up_question, modify_sql
except ImportError:  # pragma: no cover
    from chart_selector import select_chart_type
    from database import ensure_database, execute_query, get_active_schema, load_uploaded_csv, table_schema_prompt
    from gemini import generate_insight, generate_sql_query, is_follow_up_question, modify_sql


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Natural language BI question")
    session_id: str = Field(default="default", min_length=1, max_length=128)


class QueryResponse(BaseModel):
    data: list[dict[str, Any]]
    chart_type: str
    sql_query: str
    insight: str
    message: str | None = None


class UploadCsvResponse(BaseModel):
    table_name: str
    row_count: int
    columns: list[str]
    source_file: str
    message: str


FALLBACK_MESSAGE = "Sorry, this question cannot be answered using available data."
MEMORY_LOCK = threading.Lock()
MAX_SESSION_MEMORY = 1000


@dataclass
class ConversationState:
    last_question: str | None = None
    last_sql_query: str | None = None
    updated_at: float = 0.0


CONVERSATION_MEMORY: dict[str, ConversationState] = {}


def _sanitize_session_id(raw_session_id: str) -> str:
    safe = "".join(character for character in raw_session_id.strip() if character.isalnum() or character in ("-", "_"))
    return safe[:128] or "default"


def _get_or_create_session_state(session_id: str) -> ConversationState:
    with MEMORY_LOCK:
        state = CONVERSATION_MEMORY.get(session_id)
        if state is None:
            if len(CONVERSATION_MEMORY) >= MAX_SESSION_MEMORY:
                oldest = min(CONVERSATION_MEMORY, key=lambda key: CONVERSATION_MEMORY[key].updated_at)
                CONVERSATION_MEMORY.pop(oldest, None)
            state = ConversationState(updated_at=time.time())
            CONVERSATION_MEMORY[session_id] = state
        return state


def _fallback_response(sql_query: str = "") -> QueryResponse:
    return QueryResponse(
        data=[],
        chart_type="table",
        sql_query=sql_query,
        insight=FALLBACK_MESSAGE,
        message=FALLBACK_MESSAGE,
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_database()
    yield


app = FastAPI(title="AI BI Dashboard API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload-csv", response_model=UploadCsvResponse)
async def upload_csv(file: UploadFile = File(...)) -> UploadCsvResponse:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        file_bytes = await file.read()
        upload_info = load_uploaded_csv(file_bytes, file.filename)
    except ValueError as error:
        detail = str(error)
        if "codec can't decode" in detail.lower() or "could not decode" in detail.lower():
            detail = "Could not decode or parse the CSV file. Please export it as CSV UTF-8 and try again."
        raise HTTPException(status_code=400, detail=detail) from error
    except Exception as error:  # pragma: no cover
        detail = str(error)
        if "codec can't decode" in detail.lower() or "could not decode" in detail.lower():
            detail = "Could not decode or parse the CSV file. Please export it as CSV UTF-8 and try again."
            raise HTTPException(status_code=400, detail=detail) from error
        raise

    with MEMORY_LOCK:
        CONVERSATION_MEMORY.clear()

    return UploadCsvResponse(
        table_name=upload_info["table_name"],
        row_count=upload_info["row_count"],
        columns=upload_info["columns"],
        source_file=upload_info["source_file"],
        message="CSV uploaded successfully. You can query it now.",
    )


@app.post("/query", response_model=QueryResponse)
def query_dashboard(request: QueryRequest) -> QueryResponse:
    try:
        active_table, active_columns = get_active_schema()
        session_id = _sanitize_session_id(request.session_id)
        session_state = _get_or_create_session_state(session_id)

        with MEMORY_LOCK:
            previous_sql = session_state.last_sql_query

        if is_follow_up_question(request.question):
            if not previous_sql:
                return _fallback_response()
            sql_query = modify_sql(
                previous_sql=previous_sql,
                follow_up_question=request.question,
                table_name=active_table,
                allowed_columns=set(active_columns.keys()),
            )
        else:
            sql_query = generate_sql_query(
                request.question,
                table_schema_prompt(),
                table_name=active_table,
                allowed_columns=active_columns,
            )

        result_rows = execute_query(sql_query)
    except ValueError:
        return _fallback_response()
    except Exception as error:  # pragma: no cover
        logger.exception("Unhandled error during /query: %s", error)
        return _fallback_response()

    if not result_rows:
        return _fallback_response(sql_query)

    try:
        response = QueryResponse(
            data=result_rows,
            chart_type=select_chart_type(result_rows),
            sql_query=sql_query,
            insight=generate_insight(request.question, result_rows),
            message=None,
        )
        with MEMORY_LOCK:
            session_state.last_question = request.question
            session_state.last_sql_query = sql_query
            session_state.updated_at = time.time()
        return response
    except Exception as error:  # pragma: no cover
        logger.exception("Failed while preparing response payload: %s", error)
        return _fallback_response(sql_query)
