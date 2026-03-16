import { useEffect, useMemo, useRef, useState } from "react";
import { Loader2, Sparkles, Database, ChevronDown, Search, Zap, BarChart3, Upload, MessageSquareText, SendHorizontal } from "lucide-react";
import { askDashboard, QueryResponse, uploadCsv } from "../lib/api";
import DashboardChart from "../components/DashboardChart";

const MAX_UPLOAD_MB = 25;

const SUGGESTIONS = [
  "Revenue by category",
  "Revenue by region",
  "Top 10 products",
  "Daily revenue trend",
  "Payment methods",
  "Avg rating by category",
  "Average discount",
  "Total orders count",
  "Revenue vs quantity",
];

const CHART_META: Record<string, { icon: string; label: string }> = {
  bar:     { icon: "📊", label: "Bar Chart" },
  line:    { icon: "📈", label: "Line Chart" },
  pie:     { icon: "🥧", label: "Pie Chart" },
  scatter: { icon: "✦",  label: "Scatter Plot" },
  metric:  { icon: "🔢", label: "Single Metric" },
  table:   { icon: "📋", label: "Data Table" },
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  sql?: string;
};

function generateSessionId() {
  return `sess_${Math.random().toString(36).slice(2)}_${Date.now().toString(36)}`;
}

export default function Home() {
  const [question, setQuestion] = useState("");
  const [result, setResult]     = useState<QueryResponse | null>(null);
  const [loading, setLoading]   = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError]       = useState<string | null>(null);
  const [sqlOpen, setSqlOpen]   = useState(false);
  const [datasetLabel, setDatasetLabel] = useState("Amazon Sales · 50 k rows");
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [sessionId, setSessionId] = useState("default");
  const chatScrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const initialAssistantMessage = useMemo<ChatMessage>(
    () => ({
      id: "welcome",
      role: "assistant",
      text: "I can keep context in this chat. Ask a query, then use follow-ups like: Now filter this for Asia.",
    }),
    []
  );

  useEffect(() => {
    if (typeof window === "undefined") return;
    const key = "ai-bi-session-id";
    const existing = window.localStorage.getItem(key);
    if (existing) {
      setSessionId(existing);
      return;
    }
    const created = generateSessionId();
    window.localStorage.setItem(key, created);
    setSessionId(created);
  }, []);

  useEffect(() => {
    if (!chatScrollRef.current) return;
    chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
  }, [chatMessages, loading]);

  async function handleSubmit(q?: string) {
    const query = (q ?? question).trim();
    if (!query) return;

    const userMessage: ChatMessage = {
      id: `u_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      role: "user",
      text: query,
    };

    setChatMessages((previous) => [...previous, userMessage]);
    setLoading(true);
    setError(null);
    setResult(null);
    setSqlOpen(false);
    try {
      const response = await askDashboard(query, sessionId);
      setResult(response);
      setQuestion(query);

      const assistantText = response.message?.trim()
        ? response.message
        : response.insight;

      setChatMessages((previous) => [
        ...previous,
        {
          id: `a_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
          role: "assistant",
          text: assistantText,
          sql: response.sql_query,
        },
      ]);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unexpected error";
      setError(message);
      setChatMessages((previous) => [
        ...previous,
        {
          id: `a_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
          role: "assistant",
          text: message,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  async function handleUpload(file: File) {
    if (!file.name.toLowerCase().endsWith(".csv")) {
      setError("Please upload a .csv file.");
      return;
    }

    const maxBytes = MAX_UPLOAD_MB * 1024 * 1024;
    if (file.size > maxBytes) {
      setError(`CSV is too large. Please upload a file up to ${MAX_UPLOAD_MB} MB.`);
      return;
    }

    setUploading(true);
    setError(null);
    try {
      const info = await uploadCsv(file);
      setDatasetLabel(`${info.source_file} · ${info.row_count.toLocaleString()} rows`);
      setResult(null);
      setQuestion("");
      setSqlOpen(false);
      setChatMessages((previous) => [
        ...previous,
        {
          id: `a_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
          role: "assistant",
          text: `Uploaded ${info.source_file} (${info.row_count.toLocaleString()} rows). Continue chatting to analyze this dataset.`,
        },
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to upload CSV.");
    } finally {
      setUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  }

  return (
    <div className="min-h-screen" style={{ background: "var(--bg)" }}>

      {/* ── Sticky nav ─────────────────────────────────────────────── */}
      <nav className="sticky top-0 z-20 border-b border-[var(--border)] bg-[var(--surface)]/80 backdrop-blur-md px-6 py-3 flex items-center gap-3">
        <BarChart3 className="w-5 h-5 text-indigo-400" />
        <span className="font-bold text-white tracking-tight text-sm">AI BI Dashboard</span>
        <span className="ml-auto flex items-center gap-1.5 text-xs text-emerald-400 font-medium">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse inline-block" />
          {datasetLabel}
        </span>
      </nav>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-8 lg:py-10">
        <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_360px] gap-6 xl:gap-8 items-start">

        <section>

        {/* ── Hero ───────────────────────────────────────────────────── */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-extrabold mb-2 gradient-text">
            Ask your data anything
          </h1>
          <p className="text-slate-400 text-sm">
            Natural language → SQL → Chart · Powered by Gemini AI
          </p>
        </div>

        {/* ── Upload row ──────────────────────────────────────────────── */}
        <div className="mb-4 flex flex-wrap items-center gap-2">
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,text/csv"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) {
                void handleUpload(file);
              }
            }}
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading || loading}
            className="px-4 py-2 rounded-xl text-sm font-medium flex items-center gap-2
              border border-[var(--border)] text-slate-200 hover:text-white
              hover:border-cyan-500/60 hover:bg-cyan-500/10 transition-colors
              disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {uploading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
            {uploading ? "Uploading CSV..." : "Upload CSV"}
          </button>
          <span className="text-xs text-slate-400">After upload, queries run on the new dataset immediately.</span>
          <span className="text-xs text-slate-500">Max file size: {MAX_UPLOAD_MB} MB</span>
        </div>

        {/* ── Search box ─────────────────────────────────────────────── */}
        <div className="flex gap-2 mb-4">
          <div className="relative flex-1">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500 pointer-events-none" />
            <input
              ref={inputRef}
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
              placeholder="e.g. Show revenue by category"
              className="w-full pl-10 pr-4 py-3.5 rounded-xl text-sm outline-none
                bg-[var(--surface)] border border-[var(--border)] text-[var(--text)]
                placeholder:text-slate-500 focus:border-indigo-500
                focus:ring-2 focus:ring-indigo-500/30 transition-all"
            />
          </div>
          <button
            onClick={() => handleSubmit()}
            disabled={loading || uploading || !question.trim()}
            className="px-5 py-3 rounded-xl text-sm font-semibold flex items-center gap-2
              bg-indigo-600 hover:bg-indigo-500 active:scale-95 text-white transition-all
              disabled:opacity-40 disabled:cursor-not-allowed shadow-lg shadow-indigo-500/20"
          >
            {loading
              ? <Loader2 className="w-4 h-4 animate-spin" />
              : <Zap className="w-4 h-4" />}
            {loading ? "Thinking…" : "Analyze"}
          </button>
        </div>

        {/* ── Suggestion chips ───────────────────────────────────────── */}
        <div className="flex flex-wrap gap-2 mb-10">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              onClick={() => { setQuestion(s); handleSubmit(s); }}
              disabled={loading}
              className="text-xs px-3 py-1.5 rounded-full border transition-all
                border-[var(--border)] text-slate-400
                hover:border-indigo-500/60 hover:text-indigo-300 hover:bg-indigo-500/10
                disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {s}
            </button>
          ))}
        </div>

        {/* ── Loading spinner ─────────────────────────────────────────── */}
        {loading && (
          <div className="flex flex-col items-center justify-center py-24 gap-4">
            <div className="spinner" />
            <p className="text-slate-500 text-sm animate-pulse">Generating insights…</p>
          </div>
        )}

        {/* ── Error ──────────────────────────────────────────────────── */}
        {!loading && error && (
          <div className="rounded-xl border border-red-800/50 bg-red-950/30 px-5 py-4 text-sm text-red-300 flex gap-3 items-start">
            <span className="text-base mt-0.5">⚠️</span>
            <span>{error}</span>
          </div>
        )}

        {/* ── Result ─────────────────────────────────────────────────── */}
        {result && !loading && (
          <div className="space-y-4 animate-fadein">

            {/* Insight card */}
            <div className="rounded-2xl border border-indigo-500/30 bg-gradient-to-r from-indigo-500/10 to-purple-500/10 px-5 py-4 flex gap-3 items-start">
              <Sparkles className="w-4 h-4 text-indigo-400 mt-0.5 shrink-0" />
              <p className="text-sm text-indigo-100 leading-relaxed">{result.insight}</p>
            </div>

            {/* Chart card */}
            <div className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] overflow-hidden">
              {/* Chart header */}
              <div className="flex items-center justify-between px-5 py-3.5 border-b border-[var(--border)] bg-slate-800/40">
                <span className="text-sm font-medium text-slate-200 truncate mr-4">
                  {question}
                </span>
                {CHART_META[result.chart_type] && (
                  <span className="shrink-0 text-xs px-2.5 py-1 rounded-full
                    bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 font-medium">
                    {CHART_META[result.chart_type].icon}{" "}
                    {CHART_META[result.chart_type].label}
                  </span>
                )}
              </div>
              {/* Chart body */}
              <div className="p-5 pt-6">
                <DashboardChart data={result.data} chartType={result.chart_type} />
              </div>
            </div>

            {/* SQL drawer */}
            <div className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] overflow-hidden">
              <button
                onClick={() => setSqlOpen((v) => !v)}
                className="w-full flex items-center justify-between px-5 py-3.5 text-sm
                  text-slate-400 hover:text-slate-200 transition-colors"
              >
                <span className="flex items-center gap-2">
                  <Database className="w-3.5 h-3.5" />
                  Generated SQL
                </span>
                <ChevronDown className={`w-4 h-4 transition-transform duration-200 ${sqlOpen ? "rotate-180" : ""}`} />
              </button>
              {sqlOpen && (
                <pre className="px-5 pb-5 pt-1 text-xs text-emerald-300 font-mono
                  whitespace-pre-wrap leading-relaxed border-t border-[var(--border)]">
                  {result.sql_query}
                </pre>
              )}
            </div>

          </div>
        )}
        </section>

        <aside className="lg:sticky lg:top-20 h-[75vh] min-h-[520px] rounded-2xl border border-[var(--border)] bg-[var(--surface)] overflow-hidden flex flex-col">
          <div className="px-4 py-3 border-b border-[var(--border)] bg-slate-800/30 flex items-center gap-2">
            <MessageSquareText className="w-4 h-4 text-cyan-300" />
            <h2 className="text-sm font-semibold text-slate-100">BI chat</h2>
            <span className="ml-auto text-[10px] uppercase tracking-wider text-slate-400">Session memory on</span>
          </div>

          <div ref={chatScrollRef} className="flex-1 overflow-y-auto p-3 space-y-3">
            {[initialAssistantMessage, ...chatMessages].map((message) => (
              <div
                key={message.id}
                className={`rounded-xl px-3 py-2.5 text-sm leading-relaxed border ${
                  message.role === "user"
                    ? "ml-8 bg-indigo-500/15 border-indigo-500/30 text-indigo-100"
                    : "mr-8 bg-slate-900/50 border-slate-700 text-slate-200"
                }`}
              >
                <p>{message.text}</p>
                {message.sql && (
                  <p className="mt-2 text-[11px] text-emerald-300 font-mono whitespace-pre-wrap break-all">
                    SQL: {message.sql}
                  </p>
                )}
              </div>
            ))}

            {loading && (
              <div className="mr-8 rounded-xl px-3 py-2.5 text-sm border bg-slate-900/50 border-slate-700 text-slate-300 flex items-center gap-2">
                <Loader2 className="w-4 h-4 animate-spin" />
                Thinking over your data...
              </div>
            )}
          </div>

          <div className="border-t border-[var(--border)] p-3 bg-slate-950/30">
            <div className="flex gap-2">
              <input
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
                placeholder="Type follow-up: add filter for Asia"
                className="flex-1 rounded-xl border border-[var(--border)] bg-slate-900/40 px-3 py-2 text-sm text-[var(--text)] outline-none focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/20"
              />
              <button
                onClick={() => handleSubmit()}
                disabled={loading || uploading || !question.trim()}
                className="px-3 py-2 rounded-xl text-sm font-semibold flex items-center gap-1.5 bg-cyan-600 hover:bg-cyan-500 text-white transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <SendHorizontal className="w-4 h-4" />
                Send
              </button>
            </div>
            <p className="mt-2 text-[11px] text-slate-400">
              Follow-ups in this panel keep context with session ID: {sessionId}
            </p>
          </div>
        </aside>

        </div>
      </main>
    </div>
  );
}
