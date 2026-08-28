import {
  FileText,
  ClipboardCheck,
  Hash,
  Clock,
  Cpu,
  Timer,
} from "lucide-react";
import type {
  AnalysisResult as AnalysisResultType,
} from "../../types/analysis";
import QualityScore from "./QualityScore";
import IssueCard from "./IssueCard";
import ImageStatistics from "./ImageStatistics";
import FeatureInterpretation from "./FeatureInterpretation";
import AnalyticsReadiness from "./AnalyticsReadiness";
import ContextImpact from "./ContextImpact";
import IssueExplanations from "./IssueExplanations";

interface AnalysisResultProps {
  result: AnalysisResultType;
}

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

export default function AnalysisResult({ result }: AnalysisResultProps) {
  const hasIssues = result.issues.length > 0;

  return (
    <div>
      {/* ── Overall Quality ── */}
      <div className="results-grid">
        <QualityScore
          score={result.quality_score}
          label={result.quality_label}
          analysisConfidence={result.analysis_confidence}
        />

        {/* ── Detected Issues ── */}
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
          </div>
          {hasIssues ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
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
                }}
              >
                ✓ No significant quality issues detected
              </p>
              <p
                style={{
                  fontSize: 13,
                  color: "var(--text-muted)",
                  marginTop: 4,
                }}
              >
                Image meets all quality standards
              </p>
            </div>
          )}
        </div>
      </div>

      {/* ── Analytics Readiness (Phase 3) ── */}
      {result.analytics_readiness_details && (
        <div className="results-full">
          <AnalyticsReadiness
            score={result.analytics_readiness_score}
            status={result.analytics_readiness_status}
            details={result.analytics_readiness_details}
          />
        </div>
      )}

      {/* ── Smart-City Context Impact (Phase 4) ── */}
      {result.context && (
        <div className="results-full">
          <ContextImpact
            context={result.context}
            impacts={result.context_impacts ?? []}
          />
        </div>
      )}

      {/* ── Quality Metrics ── */}
      <div className="results-full">
        <ImageStatistics statistics={result.statistics} />
      </div>

      {/* ── Why This Decision? ── */}
      <div className="results-full">
        <FeatureInterpretation result={result} />
      </div>

      {/* ── Explainability (Phase 6) ── */}
      {result.issue_explanations && result.issue_explanations.length > 0 && (
        <div className="results-full">
          <IssueExplanations explanations={result.issue_explanations} />
        </div>
      )}

      {/* ── Summary + Recommendation ── */}
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
              <span style={{ opacity: 0.7, fontWeight: 500 }}>
                Recommendation:{" "}
              </span>
              {result.recommendation}
            </div>
          )}
        </div>
      </div>

      {/* ── Analysis Metadata ── */}
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
