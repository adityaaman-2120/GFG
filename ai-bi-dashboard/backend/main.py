from __future__ import annotations

import logging
from contextlib import asynccontextmanager
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
    from .data_chatbot import chatbot_response, get_dataset_metadata, load_dataset
    from .database import ensure_database, resolve_dataset_path
except ImportError:  # pragma: no cover
    from data_chatbot import chatbot_response, get_dataset_metadata, load_dataset
    from database import ensure_database, resolve_dataset_path


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
    metadata: dict[str, Any]


FALLBACK_MESSAGE = "Sorry, this question cannot be answered using available data."


def _sanitize_session_id(raw_session_id: str) -> str:
    safe = "".join(character for character in raw_session_id.strip() if character.isalnum() or character in ("-", "_"))
    return safe[:128] or "default"


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
    try:
        dataset_path = resolve_dataset_path()
        load_dataset(dataset_path.read_bytes(), dataset_path.name)
        logger.info("Default dataset initialized from %s", dataset_path)
    except Exception as error:  # pragma: no cover
        logger.warning("Could not preload default dataset: %s", error)
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
        upload_info = load_dataset(file_bytes, file.filename)
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

    return UploadCsvResponse(
        table_name="in_memory_dataset",
        row_count=upload_info["row_count"],
        columns=upload_info["columns"],
        source_file=upload_info["source_file"],
        message="CSV uploaded successfully. You can query it now.",
        metadata=upload_info["metadata"],
    )


@app.post("/query", response_model=QueryResponse)
def query_dashboard(request: QueryRequest) -> QueryResponse:
    try:
        session_id = _sanitize_session_id(request.session_id)
        logger.info("Incoming /query | session_id=%s | question=%s", session_id, request.question)
        payload = chatbot_response(request.question, session_id)
    except ValueError as error:
        logger.warning("Query validation failed | question=%s | error=%s", request.question, error)
        return _fallback_response()
    except Exception as error:  # pragma: no cover
        logger.exception("Unhandled error during /query: %s", error)
        return _fallback_response()

    result_rows = payload.get("data", [])
    operation = str(payload.get("operation", ""))
    if not result_rows:
        return _fallback_response(operation)

    try:
        return QueryResponse(
            data=result_rows,
            chart_type=str(payload.get("chart_type", "table")),
            sql_query=operation,
            insight=str(payload.get("insight", FALLBACK_MESSAGE)),
            message=payload.get("message"),
        )
    except Exception as error:  # pragma: no cover
        logger.exception("Failed while preparing response payload: %s", error)
        return _fallback_response(operation)


@app.get("/dataset-metadata")
def dataset_metadata() -> dict[str, Any]:
    metadata = get_dataset_metadata()
    if not metadata:
        raise HTTPException(status_code=404, detail="No dataset metadata available. Upload a CSV first.")
    return metadata
