import { useMemo, useState } from "react";
import { Bar, Line, Pie, Scatter } from "react-chartjs-2";
import {
  type ChartData,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  ArcElement,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";

import {
  ChevronLeft,
  ChevronRight,
  ZoomIn,
  ZoomOut,
} from "lucide-react";

import { Chart as ChartJS } from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  ArcElement,
  Tooltip,
  Legend,
  Filler
);

const PALETTE = [
  "#6366f1", "#22d3ee", "#a78bfa", "#34d399",
  "#fb923c", "#f472b6", "#facc15", "#60a5fa",
  "#4ade80", "#f87171", "#e879f9", "#2dd4bf",
];

const darkTooltip = {
  backgroundColor: "#1e293b",
  borderColor: "#334155",
  borderWidth: 1,
  titleColor: "#f1f5f9",
  bodyColor: "#94a3b8",
  padding: 10,
  cornerRadius: 8,
};

const darkLegend = {
  labels: {
    color: "#94a3b8",
    font: { size: 12 },
    padding: 20,
    usePointStyle: true,
  },
};

interface ChartProps {
  data: Record<string, unknown>[];
  chartType: string;
}

type SortOrder = "none" | "desc" | "asc";

function titleize(value: string) {
  return value.replace(/_/g, " ");
}

function isNumericValue(value: unknown) {
  return typeof value === "number" && Number.isFinite(value);
}

function toNumber(value: unknown) {
  return typeof value === "number" ? value : Number(value ?? 0);
}

function ChartActions({
  zoomLevel,
  canPanLeft,
  canPanRight,
  onZoomIn,
  onZoomOut,
  onPanLeft,
  onPanRight,
  onResetZoom,
}: {
  zoomLevel: number;
  canPanLeft: boolean;
  canPanRight: boolean;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onPanLeft: () => void;
  onPanRight: () => void;
  onResetZoom: () => void;
}) {
  return (
    <div className="mb-4 flex flex-wrap justify-end gap-2">
      <button
        onClick={onZoomOut}
        disabled={zoomLevel <= 1}
        className="text-xs px-2.5 py-1.5 rounded-lg border border-[var(--border)] text-slate-300 hover:text-white hover:border-indigo-500/60 hover:bg-indigo-500/10 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
      >
        <span className="inline-flex items-center gap-1">
          <ZoomOut size={14} />
          Zoom Out
        </span>
      </button>

      <button
        onClick={onZoomIn}
        className="text-xs px-2.5 py-1.5 rounded-lg border border-[var(--border)] text-slate-300 hover:text-white hover:border-indigo-500/60 hover:bg-indigo-500/10 transition-colors"
      >
        <span className="inline-flex items-center gap-1">
          <ZoomIn size={14} />
          Zoom In
        </span>
      </button>

      <button
        onClick={onPanLeft}
        disabled={!canPanLeft}
        className="text-xs px-2.5 py-1.5 rounded-lg border border-[var(--border)] text-slate-300 hover:text-white hover:border-indigo-500/60 hover:bg-indigo-500/10 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
      >
        <span className="inline-flex items-center gap-1">
          <ChevronLeft size={14} />
          Left
        </span>
      </button>

      <button
        onClick={onPanRight}
        disabled={!canPanRight}
        className="text-xs px-2.5 py-1.5 rounded-lg border border-[var(--border)] text-slate-300 hover:text-white hover:border-indigo-500/60 hover:bg-indigo-500/10 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
      >
        <span className="inline-flex items-center gap-1">
          Right
          <ChevronRight size={14} />
        </span>
      </button>

      <button
        onClick={onResetZoom}
        className="text-xs px-2.5 py-1.5 rounded-lg border border-[var(--border)] text-slate-300 hover:text-white hover:border-indigo-500/60 hover:bg-indigo-500/10 transition-colors"
      >
        Reset Zoom
      </button>
    </div>
  );
}

function FilterControls({
  labelKey,
  numericColumns,
  activeMetric,
  setActiveMetric,
  search,
  setSearch,
  topN,
  setTopN,
  sortOrder,
  setSortOrder,
}: {
  labelKey: string;
  numericColumns: string[];
  activeMetric: string;
  setActiveMetric: (value: string) => void;
  search: string;
  setSearch: (value: string) => void;
  topN: number;
  setTopN: (value: number) => void;
  sortOrder: SortOrder;
  setSortOrder: (value: SortOrder) => void;
}) {
  return (
    <div className="mb-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
      <label className="text-xs text-slate-400 flex flex-col gap-1">
        Search {titleize(labelKey)}
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Type to filter..."
          className="rounded-lg border border-[var(--border)] bg-slate-900/40 px-2.5 py-1.5 text-slate-200 outline-none focus:border-indigo-500"
        />
      </label>

      <label className="text-xs text-slate-400 flex flex-col gap-1">
        Metric
        <select
          value={activeMetric}
          onChange={(e) => setActiveMetric(e.target.value)}
          className="rounded-lg border border-[var(--border)] bg-slate-900/40 px-2.5 py-1.5 text-slate-200 outline-none focus:border-indigo-500"
        >
          {numericColumns.map((col) => (
            <option key={col} value={col}>{titleize(col)}</option>
          ))}
        </select>
      </label>

      <label className="text-xs text-slate-400 flex flex-col gap-1">
        Top Rows
        <select
          value={topN}
          onChange={(e) => setTopN(Number(e.target.value))}
          className="rounded-lg border border-[var(--border)] bg-slate-900/40 px-2.5 py-1.5 text-slate-200 outline-none focus:border-indigo-500"
        >
          <option value={5}>Top 5</option>
          <option value={10}>Top 10</option>
          <option value={20}>Top 20</option>
          <option value={50}>Top 50</option>
          <option value={100}>Top 100</option>
        </select>
      </label>

      <label className="text-xs text-slate-400 flex flex-col gap-1">
        Sort
        <select
          value={sortOrder}
          onChange={(e) => setSortOrder(e.target.value as SortOrder)}
          className="rounded-lg border border-[var(--border)] bg-slate-900/40 px-2.5 py-1.5 text-slate-200 outline-none focus:border-indigo-500"
        >
          <option value="none">Original Order</option>
          <option value="desc">High to Low</option>
          <option value="asc">Low to High</option>
        </select>
      </label>
    </div>
  );
}

function getColumns(row: Record<string, unknown>) {
  const keys = Object.keys(row);
  const labelKey = keys[0];
  const valueKey = keys.find((k) => k !== labelKey && typeof row[k] === "number");
  return { labelKey, valueKey };
}

export default function DashboardChart({ data, chartType }: ChartProps) {
  const firstRow = data[0] ?? {};
  const labelKey = Object.keys(firstRow)[0] ?? "label";
  const numericColumns = Object.keys(firstRow).filter((key) =>
    data.some((row) => isNumericValue(row[key]))
  );

  const [activeMetric, setActiveMetric] = useState(numericColumns[0] ?? "");
  const [search, setSearch] = useState("");
  const [topN, setTopN] = useState(20);
  const [sortOrder, setSortOrder] = useState<SortOrder>("none");
  const [zoomLevel, setZoomLevel] = useState(1);
  const [offset, setOffset] = useState(0);

  const commonOptions = useMemo(
    () => ({
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "nearest" as const, intersect: false },
    }),
    []
  );

  if (!data.length) return null;

  const filteredData = useMemo(() => {
    const term = search.trim().toLowerCase();
    let rows = data.filter((row) => {
      if (!term) return true;
      return String(row[labelKey] ?? "").toLowerCase().includes(term);
    });

    if (activeMetric && sortOrder !== "none") {
      rows = [...rows].sort((a, b) => {
        const delta = toNumber(a[activeMetric]) - toNumber(b[activeMetric]);
        return sortOrder === "asc" ? delta : -delta;
      });
    }

    return rows.slice(0, topN);
  }, [data, labelKey, search, sortOrder, activeMetric, topN]);

  const visibleCount = Math.max(3, Math.floor(filteredData.length / zoomLevel));
  const maxOffset = Math.max(0, filteredData.length - visibleCount);
  const safeOffset = Math.min(offset, maxOffset);
  const zoomedData = filteredData.slice(safeOffset, safeOffset + visibleCount);

  if (!zoomedData.length) {
    return (
      <div className="rounded-xl border border-[var(--border)] bg-slate-900/30 px-4 py-10 text-center text-sm text-slate-400">
        No rows match the current filters.
      </div>
    );
  }

  // Single metric
  if (chartType === "metric" || (zoomedData.length === 1 && Object.keys(zoomedData[0]).length === 1)) {
    const [key, value] = Object.entries(zoomedData[0])[0];
    return (
      <div className="flex flex-col items-center justify-center py-16 gap-3">
        <div className="text-6xl font-extrabold bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">
          {typeof value === "number" ? Number(value).toLocaleString() : String(value)}
        </div>
        <div className="text-slate-400 text-sm uppercase tracking-widest font-medium">
          {titleize(key)}
        </div>
      </div>
    );
  }

  const { valueKey } = getColumns(zoomedData[0]);
  if (!valueKey) return <TableView data={zoomedData} />;

  const metricKey = numericColumns.includes(activeMetric) ? activeMetric : valueKey;
  const labels = zoomedData.map((row) => String(row[labelKey] ?? ""));
  const values = zoomedData.map((row) => toNumber(row[metricKey]));

  const sharedScales = {
    x: {
      ticks: { color: "#94a3b8", maxRotation: 40, font: { size: 11 } },
      grid: { color: "#334155" },
    },
    y: {
      ticks: { color: "#94a3b8" },
      grid: { color: "#334155" },
    },
  };

  const resetZoom = () => {
    setZoomLevel(1);
    setOffset(0);
  };

  const zoomIn = () => {
    setZoomLevel((current) => Math.min(12, current + 1));
    setOffset(0);
  };

  const zoomOut = () => {
    setZoomLevel((current) => Math.max(1, current - 1));
    setOffset(0);
  };

  const panLeft = () => {
    const step = Math.max(1, Math.floor(visibleCount / 3));
    setOffset((current) => Math.max(0, current - step));
  };

  const panRight = () => {
    const step = Math.max(1, Math.floor(visibleCount / 3));
    setOffset((current) => Math.min(maxOffset, current + step));
  };

  const controls = (
    <FilterControls
      labelKey={labelKey}
      numericColumns={numericColumns}
      activeMetric={metricKey}
      setActiveMetric={setActiveMetric}
      search={search}
      setSearch={setSearch}
      topN={topN}
      setTopN={setTopN}
      sortOrder={sortOrder}
      setSortOrder={setSortOrder}
    />
  );

  const actions = (
    <ChartActions
      zoomLevel={zoomLevel}
      canPanLeft={safeOffset > 0}
      canPanRight={safeOffset < maxOffset}
      onZoomIn={zoomIn}
      onZoomOut={zoomOut}
      onPanLeft={panLeft}
      onPanRight={panRight}
      onResetZoom={resetZoom}
    />
  );

  // PIE
  if (chartType === "pie") {
    const pieData: ChartData<"pie"> = {
      labels,
      datasets: [{
        data: values,
        backgroundColor: PALETTE,
        borderColor: "#0f172a",
        borderWidth: 2,
        hoverOffset: 8,
      }],
    };
    return (
      <div>
        {controls}
        {actions}
        <div style={{ height: 360, display: "flex", justifyContent: "center" }}>
        <Pie
          data={pieData}
          options={{
            ...commonOptions,
            plugins: {
              tooltip: {
                ...darkTooltip,
                callbacks: {
                  label: (c) => ` ${c.label}: ${Number(c.raw).toLocaleString()}`,
                },
              },
              legend: darkLegend,
            },
          }}
        />
        </div>
      </div>
    );
  }

  // LINE
  if (chartType === "line") {
    const lineData: ChartData<"line"> = {
      labels,
      datasets: [{
        label: titleize(metricKey),
        data: values,
        borderColor: "#6366f1",
        backgroundColor: "rgba(99,102,241,0.12)",
        borderWidth: 2,
        pointRadius: filteredData.length > 60 ? 0 : 3,
        pointHoverRadius: 5,
        tension: 0.4,
        fill: true,
      }],
    };
    return (
      <div>
        {controls}
        {actions}
        <div style={{ height: 340 }}>
        <Line
          data={lineData}
          options={{
            ...commonOptions,
            scales: sharedScales,
            plugins: {
              tooltip: darkTooltip,
              legend: { display: false },
            },
          }}
        />
        </div>
      </div>
    );
  }

  // SCATTER
  if (chartType === "scatter") {
    const xKey = numericColumns[0] ?? Object.keys(filteredData[0])[0];
    const yKey = numericColumns[1] ?? Object.keys(filteredData[0])[1];
    const scatterData: ChartData<"scatter"> = {
      datasets: [{
        label: `${xKey} vs ${yKey}`,
        data: zoomedData.map((row) => ({ x: toNumber(row[xKey]), y: toNumber(row[yKey]) })),
        backgroundColor: "rgba(99,102,241,0.6)",
        pointRadius: 4,
        pointHoverRadius: 6,
      }],
    };
    return (
      <div>
        <div className="mb-4 grid grid-cols-1 sm:grid-cols-3 gap-2.5">
          <label className="text-xs text-slate-400 flex flex-col gap-1">
            Search {titleize(labelKey)}
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Type to filter..."
              className="rounded-lg border border-[var(--border)] bg-slate-900/40 px-2.5 py-1.5 text-slate-200 outline-none focus:border-indigo-500"
            />
          </label>
          <label className="text-xs text-slate-400 flex flex-col gap-1">
            Top Rows
            <select
              value={topN}
              onChange={(e) => setTopN(Number(e.target.value))}
              className="rounded-lg border border-[var(--border)] bg-slate-900/40 px-2.5 py-1.5 text-slate-200 outline-none focus:border-indigo-500"
            >
              <option value={20}>Top 20</option>
              <option value={50}>Top 50</option>
              <option value={100}>Top 100</option>
              <option value={200}>Top 200</option>
            </select>
          </label>
          <label className="text-xs text-slate-400 flex flex-col gap-1">
            Sort by X
            <select
              value={sortOrder}
              onChange={(e) => setSortOrder(e.target.value as SortOrder)}
              className="rounded-lg border border-[var(--border)] bg-slate-900/40 px-2.5 py-1.5 text-slate-200 outline-none focus:border-indigo-500"
            >
              <option value="none">Original Order</option>
              <option value="desc">High to Low</option>
              <option value="asc">Low to High</option>
            </select>
          </label>
        </div>
        {actions}
        <div style={{ height: 340 }}>
        <Scatter
          data={scatterData}
          options={{
            ...commonOptions,
            scales: {
              x: {
                ticks: { color: "#94a3b8" },
                grid: { color: "#334155" },
                title: { display: true, text: titleize(xKey), color: "#94a3b8" },
              },
              y: {
                ticks: { color: "#94a3b8" },
                grid: { color: "#334155" },
                title: { display: true, text: titleize(yKey), color: "#94a3b8" },
              },
            },
            plugins: {
              tooltip: darkTooltip,
              legend: { display: false },
            },
          }}
        />
        </div>
      </div>
    );
  }

  // BAR (default)
  const barData: ChartData<"bar"> = {
    labels,
    datasets: [{
      label: titleize(metricKey),
      data: values,
      backgroundColor: labels.map((_, i) => PALETTE[i % PALETTE.length] + "cc"),
      borderColor: labels.map((_, i) => PALETTE[i % PALETTE.length]),
      borderWidth: 1,
      borderRadius: 6,
    }],
  };
  return (
    <div>
      {controls}
      {actions}
      <div style={{ height: 340 }}>
      <Bar
        data={barData}
        options={{
          ...commonOptions,
          scales: {
            ...sharedScales,
            x: {
              ...sharedScales.x,
              grid: { display: false },
            },
          },
          plugins: {
            tooltip: darkTooltip,
            legend: { display: false },
          },
        }}
      />
      </div>
    </div>
  );
}

function TableView({ data }: { data: Record<string, unknown>[] }) {
  const columns = Object.keys(data[0]);
  return (
    <div className="overflow-x-auto rounded-xl">
      <table className="w-full text-sm text-left text-slate-300">
        <thead className="bg-slate-800/60">
          <tr>
            {columns.map((col) => (
              <th key={col} className="px-4 py-3 text-slate-400 font-medium uppercase tracking-wide text-xs whitespace-nowrap">
                {col.replace(/_/g, " ")}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr key={i} className={`border-t border-slate-800 hover:bg-slate-800/40 transition-colors ${i % 2 !== 0 ? "bg-slate-800/20" : ""}`}>
              {columns.map((col) => (
                <td key={col} className="px-4 py-2.5 whitespace-nowrap">
                  {String(row[col] ?? "")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
