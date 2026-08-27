import { FileText, ClipboardCheck, Hash, Clock, Cpu, Timer } from "lucide-react";
import type { AnalysisResult as AnalysisResultType, QualityLabel } from "../../types/analysis";
import QualityScore from "./QualityScore";
import IssueCard from "./IssueCard";
import ImageStatistics from "./ImageStatistics";
import FeatureInterpretation from "./FeatureInterpretation";

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

function recommendationText(label: QualityLabel): string {
  switch (label) {
    case "ACCEPTABLE":
      return "Recommendation: Safe for downstream analysis";
    case "DEGRADED":
      return "Recommendation: Use with caution or consider image enhancement";
    case "DEFECTIVE":
      return "Recommendation: Recapture image or perform manual review";
  }
}

export default function AnalysisResult({ result }: AnalysisResultProps) {
  return (
    <div>
      {/* Two-column workspace: Score + Issues */}
      <div className="results-grid">
        <QualityScore
          score={result.quality_score}
          label={result.quality_label}
          analysisConfidence={result.analysis_confidence}
        />

        <div className="card">
          <div className="card__heading">
            <span className="card__heading-text">
              Detected Issues
              <span className="card__heading-sub">
                — {result.issues.length === 0
                  ? "none found"
                  : `${result.issues.length} issue${result.issues.length !== 1 ? "s" : ""}`}
              </span>
            </span>
          </div>
          {result.issues.length === 0 ? (
            <div style={{ padding: "20px 0", textAlign: "center" }}>
              <p style={{ fontSize: 14, color: "var(--green)", fontWeight: 600 }}>
                ✓ No significant quality issues detected
              </p>
              <p style={{ fontSize: 13, color: "var(--text-muted)", marginTop: 4 }}>
                Image meets all quality standards
              </p>
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              {result.issues.map((issue, i) => (
                <IssueCard key={`${issue.type}-${i}`} issue={issue} />
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Image Quality Metrics */}
      <div className="results-full">
        <ImageStatistics statistics={result.statistics} />
      </div>

      {/* Feature Interpretation */}
      <div className="results-full">
        <FeatureInterpretation result={result} />
      </div>

      {/* Quality Assessment Summary + Recommendation */}
      <div className="results-full">
        <div className="card">
          <div className="card__heading">
            <ClipboardCheck size={15} className="card__heading-icon" />
            <span className="card__heading-text">Quality Assessment Summary</span>
          </div>
          <div className="summary-block">{result.summary}</div>
          <div
            className={`recommendation-block recommendation-block--${result.quality_label.toLowerCase()}`}
          >
            {recommendationText(result.quality_label)}
          </div>
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
                <Hash size={10} style={{ display: "inline", verticalAlign: "middle", marginRight: 3 }} />
                Analysis ID
              </span>
              <span className="metadata-item__value">
                {formatAnalysisId(result.analysis_id)}
              </span>
            </div>
            <div className="metadata-item">
              <span className="metadata-item__label">
                <Clock size={10} style={{ display: "inline", verticalAlign: "middle", marginRight: 3 }} />
                Analyzed
              </span>
              <span className="metadata-item__value">
                {formatDateTime(result.created_at)}
              </span>
            </div>
            <div className="metadata-item">
              <span className="metadata-item__label">
                <Timer size={10} style={{ display: "inline", verticalAlign: "middle", marginRight: 3 }} />
                Processing Time
              </span>
              <span className="metadata-item__value">
                {formatProcessingTime(result.processing_time_ms)}
              </span>
            </div>
            <div className="metadata-item">
              <span className="metadata-item__label">
                <Cpu size={10} style={{ display: "inline", verticalAlign: "middle", marginRight: 3 }} />
                Model Version
              </span>
              <span className="metadata-item__value">v1.0.0</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
