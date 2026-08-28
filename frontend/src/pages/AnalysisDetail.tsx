import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import {
  ArrowLeft,
  FileText,
  ClipboardCheck,
  Hash,
  Clock,
  Cpu,
  Timer,
  Image as ImageIcon,
  FolderX,
} from "lucide-react";
import type {
  AnalysisResult as AnalysisResultType,
} from "../types/analysis";
import { getAnalysisById } from "../services/api";
import QualityScore from "../components/analysis/QualityScore";
import IssueCard from "../components/analysis/IssueCard";
import ImageStatistics from "../components/analysis/ImageStatistics";
import FeatureInterpretation from "../components/analysis/FeatureInterpretation";
import AnalyticsReadiness from "../components/analysis/AnalyticsReadiness";
import ContextImpact from "../components/analysis/ContextImpact";
import IssueExplanations from "../components/analysis/IssueExplanations";
import StatusBadge from "../components/common/StatusBadge";
import ErrorState from "../components/common/ErrorState";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatAnalysisId(id: string): string {
  return `VG-${id.slice(-6).toUpperCase()}`;
}

function formatDateTime(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatProcessingTime(ms: number): string {
  return `${(ms / 1000).toFixed(2)} sec`;
}

function inferFormat(filename: string): string {
  const ext = filename.split(".").pop()?.toLowerCase();
  switch (ext) {
    case "jpg":
    case "jpeg":
      return "JPEG";
    case "png":
      return "PNG";
    case "webp":
      return "WebP";
    default:
      return ext?.toUpperCase() ?? "Unknown";
  }
}

// ---------------------------------------------------------------------------
// Skeleton Loading
// ---------------------------------------------------------------------------

function DetailSkeleton() {
  return (
    <div className="detail-skeleton">
      <div className="detail-skeleton__header">
        <div
          className="skeleton-pulse"
          style={{ width: 160, height: 16 }}
        />
      </div>

      <div className="skeleton-pulse" style={{ width: 280, height: 26 }} />

      <div className="detail-skeleton__grid">
        <div className="skeleton-pulse detail-skeleton__image" />
        <div className="skeleton-pulse detail-skeleton__score" />
      </div>

      <div className="skeleton-pulse detail-skeleton__full" />
      <div className="skeleton-pulse detail-skeleton__full" />
      <div className="skeleton-pulse detail-skeleton__full" />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Not Found State
// ---------------------------------------------------------------------------

function NotFoundState() {
  return (
    <div>
      <Link to="/history" className="back-link">
        <ArrowLeft size={15} />
        Back to Analysis History
      </Link>

      <div className="page-header">
        <h1 className="page-header__title">Analysis Details</h1>
        <p className="page-header__subtitle">
          Detailed visual quality assessment and diagnostic results.
        </p>
      </div>

      <div className="state-view">
        <FolderX size={40} className="state-view__icon" style={{ color: "var(--text-muted)" }} />
        <p className="state-view__title">Analysis Not Found</p>
        <p className="state-view__message">
          The requested analysis could not be found or may have been removed.
        </p>
        <Link
          to="/history"
          className="btn btn--ghost"
          style={{ marginTop: 16, textDecoration: "none" }}
        >
          <ArrowLeft size={14} />
          Back to Analysis History
        </Link>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main AnalysisDetail Page
// ---------------------------------------------------------------------------

export default function AnalysisDetail() {
  const { id } = useParams<{ id: string }>();
  const [result, setResult] = useState<AnalysisResultType | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      if (!id) return;
      setLoading(true);
      setError(null);
      try {
        const data = await getAnalysisById(id);
        if (!data) setError("not_found");
        else setResult(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load analysis.");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [id]);

  const refetch = async () => {
    if (!id) return;
    setLoading(true);
    setError(null);
    try {
      const data = await getAnalysisById(id);
      if (!data) setError("not_found");
      else setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load analysis.");
    } finally {
      setLoading(false);
    }
  };

  // -- Loading State --
  if (loading) {
    return (
      <div>
        <div className="page-header">
          <h1 className="page-header__title">Analysis Details</h1>
          <p className="page-header__subtitle">
            Loading analysis details...
          </p>
        </div>
        <DetailSkeleton />
      </div>
    );
  }

  // -- Error State --
  if (error && error !== "not_found") {
    return (
      <div>
        <Link to="/history" className="back-link">
          <ArrowLeft size={15} />
          Back to Analysis History
        </Link>
        <div className="page-header">
          <h1 className="page-header__title">Analysis Details</h1>
          <p className="page-header__subtitle">
            Detailed visual quality assessment and diagnostic results.
          </p>
        </div>
        <ErrorState
          title="Unable to Load Analysis"
          message="We couldn't retrieve the requested analysis. Please try again."
          onRetry={refetch}
        />
      </div>
    );
  }

  // -- Not Found State --
  if (error === "not_found" || (!result && !loading)) {
    return <NotFoundState />;
  }

  if (!result) return null;

  const hasIssues = result.issues.length > 0;

  return (
    <div>
      {/* Back Link */}
      <Link to="/history" className="back-link">
        <ArrowLeft size={15} />
        Back to Analysis History
      </Link>

      {/* Page Header */}
      <div className="page-header">
        <div className="page-header__top-row">
          <div>
            <h1 className="page-header__title">Analysis Details</h1>
            <p className="page-header__subtitle">
              Detailed visual quality assessment and diagnostic results.
            </p>
          </div>
          <span className="page-header__badge">
            Analysis ID: {formatAnalysisId(result.analysis_id)}
          </span>
        </div>
      </div>

      {/* Two-Column: Image Inspection + Quality Assessment */}
      <div className="detail-grid">
        {/* Left: Image Inspection Panel */}
        <div className="card">
          <div className="card__heading">
            <ImageIcon size={15} className="card__heading-icon" />
            <span className="card__heading-text">
              Image Inspection
              <span className="card__heading-sub">— source evidence</span>
            </span>
          </div>

          {/* Image Preview */}
          {result.image_url ? (
            <img
              src={result.image_url}
              alt={result.filename}
              className="detail-image-preview"
            />
          ) : (
            <div className="detail-image-placeholder">
              <ImageIcon size={32} />
            </div>
          )}

          {/* Image Information */}
          <div style={{ marginTop: 16 }}>
            <div className="card__heading" style={{ marginBottom: 12 }}>
              <span className="card__heading-text">Image Information</span>
            </div>
            <div className="image-info-grid">
              <div className="image-info-item">
                <span className="image-info-item__label">Filename</span>
                <span className="image-info-item__value">
                  {result.filename}
                </span>
              </div>
              <div className="image-info-item">
                <span className="image-info-item__label">Format</span>
                <span className="image-info-item__value">
                  {inferFormat(result.filename)}
                </span>
              </div>
              <div className="image-info-item">
                <span className="image-info-item__label">Analyzed</span>
                <span className="image-info-item__value">
                  {formatDateTime(result.created_at)}
                </span>
              </div>
              <div className="image-info-item">
                <span className="image-info-item__label">Resolution</span>
                <span className="image-info-item__value">
                  {result.image_url ? "—" : "Source unavailable"}
                </span>
              </div>
              {result.context && (
                <div className="image-info-item">
                  <span className="image-info-item__label">Context</span>
                  <span className="image-info-item__value">
                    {result.context}
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right: Quality Assessment Panel */}
        <QualityScore
          score={result.quality_score}
          label={result.quality_label}
          analysisConfidence={result.analysis_confidence}
        />
      </div>

      {/* Analytics Readiness (Phase 3) */}
      {result.analytics_readiness_details && (
        <div className="results-full">
          <AnalyticsReadiness
            score={result.analytics_readiness_score}
            status={result.analytics_readiness_status}
            details={result.analytics_readiness_details}
          />
        </div>
      )}

      {/* Detected Issues */}
      <div className="results-full">
        <div className="card">
          <div className="card__heading">
            <span className="card__heading-text">
              Detected Issues
              <span className="card__heading-sub">
                — {hasIssues
                  ? `${result.issues.length} issue${result.issues.length !== 1 ? "s" : ""}`
                  : "none found"}
              </span>
            </span>
            <StatusBadge label={result.quality_label} size="sm" />
          </div>

          {hasIssues ? (
            <div className="detail-issues-grid">
              {result.issues.map((issue, i) => (
                <IssueCard key={`${issue.type}-${i}`} issue={issue} />
              ))}
            </div>
          ) : (
            <div style={{ padding: "20px 0", textAlign: "center" }}>
              <p
                style={{
                  fontSize: 14,
                  color: "var(--green)",
                  fontWeight: 600,
                  marginBottom: 4,
                }}
              >
                ✓ No significant quality issues detected
              </p>
              <p style={{ fontSize: 13, color: "var(--text-muted)" }}>
                The image meets the expected visual quality threshold.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Smart-City Context Impact (Phase 4) */}
      {result.context && (
        <div className="results-full">
          <ContextImpact
            context={result.context}
            impacts={result.context_impacts ?? []}
          />
        </div>
      )}

      {/* Image Quality Metrics */}
      <div className="results-full">
        <ImageStatistics statistics={result.statistics} />
      </div>

      {/* Decision Explanation */}
      <div className="results-full">
        <FeatureInterpretation result={result} />
      </div>

      {/* Issue Explanations (Phase 6) */}
      {result.issue_explanations && result.issue_explanations.length > 0 && (
        <div className="results-full">
          <IssueExplanations explanations={result.issue_explanations} />
        </div>
      )}

      {/* Quality Assessment Summary */}
      <div className="results-full">
        <div className="card">
          <div className="card__heading">
            <ClipboardCheck size={15} className="card__heading-icon" />
            <span className="card__heading-text">
              Quality Assessment Summary
            </span>
          </div>
          <div className="summary-block">{result.summary}</div>
          {result.recommendation && (
            <div
              className={`recommendation-block recommendation-block--${result.quality_label.toLowerCase()}`}
            >
              <span style={{ opacity: 0.7, fontWeight: 500 }}>Recommendation: </span>
              {result.recommendation}
            </div>
          )}
        </div>
      </div>

      {/* Analysis Metadata */}
      <div className="results-full">
        <div className="card">
          <div className="card__heading">
            <FileText size={15} className="card__heading-icon" />
            <span className="card__heading-text">Analysis Metadata</span>
          </div>
          <div className="metadata-grid">
            <div className="metadata-item">
              <span className="metadata-item__label">
                <Hash
                  size={10}
                  style={{
                    display: "inline",
                    verticalAlign: "middle",
                    marginRight: 3,
                  }}
                />
                Analysis ID
              </span>
              <span className="metadata-item__value">
                {formatAnalysisId(result.analysis_id)}
              </span>
            </div>
            <div className="metadata-item">
              <span className="metadata-item__label">
                <Clock
                  size={10}
                  style={{
                    display: "inline",
                    verticalAlign: "middle",
                    marginRight: 3,
                  }}
                />
                Analyzed
              </span>
              <span className="metadata-item__value">
                {formatDateTime(result.created_at)}
              </span>
            </div>
            <div className="metadata-item">
              <span className="metadata-item__label">
                <Timer
                  size={10}
                  style={{
                    display: "inline",
                    verticalAlign: "middle",
                    marginRight: 3,
                  }}
                />
                Processing Time
              </span>
              <span className="metadata-item__value">
                {formatProcessingTime(result.processing_time_ms)}
              </span>
            </div>
            <div className="metadata-item">
              <span className="metadata-item__label">
                <Cpu
                  size={10}
                  style={{
                    display: "inline",
                    verticalAlign: "middle",
                    marginRight: 3,
                  }}
                />
                Model Version
              </span>
              <span className="metadata-item__value">
                {result.model_version ?? "—"}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
