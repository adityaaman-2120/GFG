// Points to the FastAPI backend.  Override in .env.local without editing code.
const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const REQUEST_TIMEOUT_MS = 8000;

function getApiCandidates(): string[] {
  const candidates = [API_BASE];
  if (API_BASE.includes("localhost")) {
    candidates.push(API_BASE.replace("localhost", "127.0.0.1"));
  }
  if (API_BASE.includes("127.0.0.1")) {
    candidates.push(API_BASE.replace("127.0.0.1", "localhost"));
  }
  return [...new Set(candidates)];
}

export interface QueryResponse {
  data: Record<string, unknown>[];
  chart_type: "bar" | "line" | "scatter" | "metric" | "table" | string;
  sql_query: string;
  insight: string;
  message?: string | null;
}

export interface UploadCsvResponse {
  table_name: string;
  row_count: number;
  columns: string[];
  source_file: string;
  message: string;
}

async function fetchWithTimeout(
  input: string,
  init: RequestInit,
  timeoutMs: number = REQUEST_TIMEOUT_MS
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(input, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timeoutId);
  }
}

export async function askDashboard(question: string, sessionId: string): Promise<QueryResponse> {
  let response: Response | null = null;
  let reachedBackend = false;
  let timeoutReached = false;

  for (const baseUrl of getApiCandidates()) {
    try {
      response = await fetchWithTimeout(`${baseUrl}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, session_id: sessionId }),
      });
      reachedBackend = true;
      break;
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        timeoutReached = true;
      }
      // Try next loopback candidate.
    }
  }

  if (!reachedBackend || !response) {
    if (timeoutReached) {
      throw new Error("Backend request timed out. Please ensure FastAPI is running and retry.");
    }
    throw new Error(
      `Cannot reach backend at ${getApiCandidates().join(" or ")}. Make sure FastAPI is running.`
    );
  }

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(
      (error as { detail?: string }).detail ?? `HTTP ${response.status}`
    );
  }

  return response.json() as Promise<QueryResponse>;
}

export async function uploadCsv(file: File): Promise<UploadCsvResponse> {
  let response: Response | null = null;
  let reachedBackend = false;
  let timeoutReached = false;

  for (const baseUrl of getApiCandidates()) {
    try {
      const form = new FormData();
      form.append("file", file);

      response = await fetchWithTimeout(`${baseUrl}/upload-csv`, {
        method: "POST",
        body: form,
      });
      reachedBackend = true;
      break;
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        timeoutReached = true;
      }
      // Try next loopback candidate.
    }
  }

  if (!reachedBackend || !response) {
    if (timeoutReached) {
      throw new Error("CSV upload timed out. Please check backend status and file size, then retry.");
    }
    throw new Error(
      `Cannot reach backend at ${getApiCandidates().join(" or ")}. Make sure FastAPI is running.`
    );
  }

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(
      (error as { detail?: string }).detail ?? `HTTP ${response.status}`
    );
  }

  return response.json() as Promise<UploadCsvResponse>;
}
