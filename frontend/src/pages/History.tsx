import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  Search,
  FolderOpen,
  ArrowRight,
  Image as ImageIcon,
  ArrowUpDown,
} from "lucide-react";
import type {
  AnalysisResult as AnalysisResultType,
  QualityLabel,
} from "../types/analysis";
import { getAnalyses } from "../services/api";

import ErrorState from "../components/common/ErrorState";
import EmptyState from "../components/common/EmptyState";
import StatusBadge from "../components/common/StatusBadge";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

type SortOption = "newest" | "oldest" | "highest" | "lowest";

function formatDate(iso: string): string {
  const d = new Date(iso);
  const now = new Date();
  const isToday =
    d.getFullYear() === now.getFullYear() &&
    d.getMonth() === now.getMonth() &&
    d.getDate() === now.getDate();

  if (isToday) {
    return (
      "Today, " +
      d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })
    );
  }

  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

function scoreColor(score: number): string {
  if (score >= 80) return "var(--green)";
  if (score >= 60) return "var(--blue, #3b82f6)";
  if (score >= 40) return "var(--yellow)";
  if (score >= 20) return "var(--orange, #f97316)";
  return "var(--red)";
}

function issuesSummary(
  issues: AnalysisResultType["issues"],
): { text: string; isNone: boolean } {
  if (issues.length === 0) return { text: "No major issues", isNone: true };
  if (issues.length <= 2) {
    return {
      text: issues.map((i) => i.type).join(", "),
      isNone: false,
    };
  }
  return {
    text: `${issues.length} Issues — ${issues[0].type}`,
    isNone: false,
  };
}

function sortAnalyses(
  items: AnalysisResultType[],
  sort: SortOption,
): AnalysisResultType[] {
  const copy = [...items];
  switch (sort) {
    case "newest":
      return copy.sort(
        (a, b) =>
          new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
      );
    case "oldest":
      return copy.sort(
        (a, b) =>
          new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
      );
    case "highest":
      return copy.sort((a, b) => b.quality_score - a.quality_score);
    case "lowest":
      return copy.sort((a, b) => a.quality_score - b.quality_score);
  }
}

// ---------------------------------------------------------------------------
// Summary Cards
// ---------------------------------------------------------------------------

function SummaryCards({ analyses }: { analyses: AnalysisResultType[] }) {
  const counts = useMemo(() => {
    const total = analyses.length;
    let excellent = 0;
    let good = 0;
    let fair = 0;
    let poor = 0;
    for (const a of analyses) {
      if (a.quality_label === "Excellent") excellent++;
      else if (a.quality_label === "Good") good++;
      else if (a.quality_label === "Fair") fair++;
      else if (a.quality_label === "Poor" || a.quality_label === "Critical") poor++;
    }
    return { total, excellent, good, fair, poor };
  }, [analyses]);

  const cards = [
    { value: counts.total, label: "Total Analyses", color: "total" },
    { value: counts.excellent + counts.good, label: "Good+", color: "acceptable" },
    { value: counts.fair, label: "Fair", color: "degraded" },
    { value: counts.poor, label: "Poor/Critical", color: "defective" },
  ] as const;

  return (
    <div className="summary-cards">
      {cards.map((c) => (
        <div key={c.label} className="summary-card">
          <span
            className={`summary-card__value summary-card__value--${c.color}`}
          >
            {c.value}
          </span>
          <span className="summary-card__label">{c.label}</span>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Score Bar (inline)
// ---------------------------------------------------------------------------

function ScoreBar({ score }: { score: number }) {
  return (
    <div className="score-bar">
      <div className="score-bar__track">
        <div
          className="score-bar__fill"
          style={{
            width: `${score}%`,
            background: scoreColor(score),
          }}
        />
      </div>
      <span className="score-bar__value" style={{ color: scoreColor(score) }}>
        {score}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Skeleton Rows
// ---------------------------------------------------------------------------

function SkeletonRows() {
  return (
    <>
      {Array.from({ length: 5 }).map((_, i) => (
        <div key={i} className="skeleton-row">
          <div
            className="skeleton-pulse"
            style={{ width: 56, height: 40 }}
          />
          <div
            className="skeleton-pulse"
            style={{ height: 14, width: "60%" }}
          />
          <div
            className="skeleton-pulse"
            style={{ height: 12, width: 90 }}
          />
          <div
            className="skeleton-pulse"
            style={{ height: 20, width: 60, borderRadius: 9999 }}
          />
          <div
            className="skeleton-pulse"
            style={{ height: 12, width: 70 }}
          />
          <div
            className="skeleton-pulse"
            style={{ height: 12, width: 80 }}
          />
          <div
            className="skeleton-pulse"
            style={{ height: 12, width: 80 }}
          />
        </div>
      ))}
    </>
  );
}

// ---------------------------------------------------------------------------
// Desktop Table Row
// ---------------------------------------------------------------------------

function AnalysisRow({ a }: { a: AnalysisResultType }) {
  const iss = issuesSummary(a.issues);

  return (
    <Link
      to={`/history/${a.analysis_id}`}
      className="history-analysis-row"
      tabIndex={0}
    >
      {/* Thumbnail */}
      <div className="history-analysis-row__thumb">
        {a.image_url ? (
          <img src={a.image_url} alt="" />
        ) : (
          <ImageIcon size={16} />
        )}
      </div>

      {/* Filename */}
      <span className="history-analysis-row__filename">{a.filename}</span>

      {/* Score */}
      <ScoreBar score={a.quality_score} />

      {/* Status */}
      <StatusBadge label={a.quality_label} size="sm" />

      {/* Issues */}
      <span
        className={`history-analysis-row__issues ${
          iss.isNone ? "history-analysis-row__issues--none" : ""
        }`}
      >
        {iss.text}
      </span>

      {/* Date */}
      <span className="history-analysis-row__date">
        {formatDate(a.created_at)}
      </span>

      {/* Action */}
      <span className="history-analysis-row__action">
        View Details <ArrowRight size={12} />
      </span>
    </Link>
  );
}

// ---------------------------------------------------------------------------
// Mobile Card
// ---------------------------------------------------------------------------

function MobileCard({ a }: { a: AnalysisResultType }) {
  const iss = issuesSummary(a.issues);

  return (
    <Link
      to={`/history/${a.analysis_id}`}
      className="history-mobile-card"
      tabIndex={0}
    >
      <div className="history-mobile-card__top">
        <div className="history-mobile-card__thumb">
          {a.image_url ? (
            <img src={a.image_url} alt="" />
          ) : (
            <ImageIcon size={14} />
          )}
        </div>
        <div className="history-mobile-card__info">
          <div className="history-mobile-card__filename">{a.filename}</div>
          <div className="history-mobile-card__date">
            {formatDate(a.created_at)}
          </div>
        </div>
      </div>

      <div className="history-mobile-card__metrics">
        <ScoreBar score={a.quality_score} />
        <StatusBadge label={a.quality_label} size="sm" />
      </div>

      <span
        className={`history-mobile-card__issues ${
          iss.isNone ? "history-mobile-card__issues--none" : ""
        }`}
      >
        {iss.text}
      </span>

      <div className="history-mobile-card__footer">
        <span className="history-mobile-card__action">
          View Analysis <ArrowRight size={12} />
        </span>
      </div>
    </Link>
  );
}

// ---------------------------------------------------------------------------
// Main History Page
// ---------------------------------------------------------------------------

export default function History() {
  const [analyses, setAnalyses] = useState<AnalysisResultType[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [labelFilter, setLabelFilter] = useState<QualityLabel | "">("");
  const [sort, setSort] = useState<SortOption>("newest");

  const fetchAnalyses = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getAnalyses();
      setAnalyses(data);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to load history.",
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAnalyses();
  }, []);

  const filtered = useMemo(() => {
    const result = analyses.filter((a) => {
      const matchesSearch =
        !search ||
        a.filename.toLowerCase().includes(search.toLowerCase());
      const matchesLabel = !labelFilter || a.quality_label === labelFilter;
      return matchesSearch && matchesLabel;
    });
    return sortAnalyses(result, sort);
  }, [analyses, search, labelFilter, sort]);

  // -----------------------------------------------------------------------
  // Loading State
  // -----------------------------------------------------------------------
  if (loading) {
    return (
      <div>
        <div className="page-header">
          <div className="page-header__top-row">
            <div>
              <h1 className="page-header__title">Analysis History</h1>
              <p className="page-header__subtitle">
                Review and compare previous visual quality assessments.
              </p>
            </div>
          </div>
        </div>

        <div className="summary-cards">
          {[1, 2, 3, 4].map((n) => (
            <div key={n} className="summary-card">
              <div
                className="skeleton-pulse"
                style={{ height: 32, width: 48, marginBottom: 6 }}
              />
              <div
                className="skeleton-pulse"
                style={{ height: 12, width: 80 }}
              />
            </div>
          ))}
        </div>

        <SkeletonRows />
      </div>
    );
  }

  // -----------------------------------------------------------------------
  // Error State
  // -----------------------------------------------------------------------
  if (error) {
    return (
      <div>
        <div className="page-header">
          <h1 className="page-header__title">Analysis History</h1>
          <p className="page-header__subtitle">
            Review and compare previous visual quality assessments.
          </p>
        </div>
        <ErrorState
          title="Unable to load analysis history"
          message="We couldn't retrieve your previous analyses. Please try again."
          onRetry={fetchAnalyses}
        />
      </div>
    );
  }

  // -----------------------------------------------------------------------
  // Empty State (no analyses at all)
  // -----------------------------------------------------------------------
  if (analyses.length === 0) {
    return (
      <div>
        <div className="page-header">
          <h1 className="page-header__title">Analysis History</h1>
          <p className="page-header__subtitle">
            Review and compare previous visual quality assessments.
          </p>
        </div>
        <EmptyState
          icon={<FolderOpen size={40} />}
          title="No analyses yet"
          message="Upload an image and run an analysis to begin building your visual quality history."
        />
        <div style={{ display: "flex", justifyContent: "center", marginTop: 16 }}>
          <Link to="/" className="btn btn--primary" style={{ textDecoration: "none" }}>
            Go to Dashboard
          </Link>
        </div>
      </div>
    );
  }

  // -----------------------------------------------------------------------
  // Main View
  // -----------------------------------------------------------------------
  return (
    <div>
      {/* ── Page Header ── */}
      <div className="page-header">
        <div className="page-header__top-row">
          <div>
            <h1 className="page-header__title">Analysis History</h1>
            <p className="page-header__subtitle">
              Review and compare previous visual quality assessments.
            </p>
          </div>
          <span className="page-header__badge">
            {analyses.length} {analyses.length === 1 ? "Analysis" : "Analyses"}
          </span>
        </div>
      </div>

      {/* ── Summary Cards ── */}
      <SummaryCards analyses={analyses} />

      {/* ── Controls ── */}
      <div className="history-controls">
        <div style={{ position: "relative", flex: 1, minWidth: 200 }}>
          <Search
            size={16}
            style={{
              position: "absolute",
              left: 12,
              top: "50%",
              transform: "translateY(-50%)",
              color: "var(--text-muted)",
              pointerEvents: "none",
            }}
          />
          <input
            type="text"
            className="history-controls__search"
            placeholder="Search by filename…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{ paddingLeft: 36 }}
            aria-label="Search by filename"
          />
        </div>
        <select
          className="history-controls__select"
          value={labelFilter}
          onChange={(e) =>
            setLabelFilter(e.target.value as QualityLabel | "")
          }
          aria-label="Filter by status"
        >
          <option value="">All Statuses</option>
          <option value="Excellent">Excellent</option>
          <option value="Good">Good</option>
          <option value="Fair">Fair</option>
          <option value="Poor">Poor</option>
          <option value="Critical">Critical</option>
        </select>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <ArrowUpDown
            size={14}
            style={{ color: "var(--text-muted)", flexShrink: 0 }}
          />
          <select
            className="history-controls__select"
            value={sort}
            onChange={(e) => setSort(e.target.value as SortOption)}
            aria-label="Sort order"
          >
            <option value="newest">Newest First</option>
            <option value="oldest">Oldest First</option>
            <option value="highest">Highest Quality Score</option>
            <option value="lowest">Lowest Quality Score</option>
          </select>
        </div>
      </div>

      {/* ── Filtered Empty State ── */}
      {filtered.length === 0 ? (
        <EmptyState
          icon={<FolderOpen size={40} />}
          title="No analyses found"
          message={
            search || labelFilter
              ? "No results match your filters. Try adjusting your search or filter."
              : "No image analyses have been performed yet."
          }
        />
      ) : (
        <>
          {/* ── Desktop Table Header ── */}
          <div className="history-table-header">
            <span></span>
            <span>Filename</span>
            <span>Score</span>
            <span>Status</span>
            <span>Issues</span>
            <span>Analyzed</span>
            <span>Action</span>
          </div>

          {/* ── Desktop Rows ── */}
          <div className="history-list" style={{ display: "contents" }}>
            {/* Desktop: shown via grid rows */}
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {filtered.map((a) => (
                <div key={a.analysis_id}>
                  {/* Desktop row — hidden on mobile */}
                  <div className="history-row-desktop-wrap">
                    <AnalysisRow a={a} />
                  </div>
                  {/* Mobile card — hidden on desktop */}
                  <div className="history-row-mobile-wrap">
                    <MobileCard a={a} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
