import { AlertTriangle, AlertCircle, Info, XCircle } from "lucide-react";
import type { Issue, Severity } from "../../types/analysis";
import ConfidenceBar from "./ConfidenceBar";

interface IssueCardProps {
  issue: Issue;
}

function severityIcon(severity: Severity) {
  switch (severity) {
    case "critical":
      return <XCircle size={18} />;
    case "high":
      return <AlertTriangle size={18} />;
    case "moderate":
      return <AlertCircle size={18} />;
    case "low":
      return <Info size={18} />;
  }
}

function severityClass(severity: Severity): string {
  return `issue-card__icon issue-card__icon--${severity}`;
}

function severityBadgeClass(severity: Severity): string {
  switch (severity) {
    case "critical":
      return "badge badge--critical";
    case "high":
      return "badge badge--high";
    case "moderate":
      return "badge badge--medium";
    case "low":
      return "badge badge--low";
  }
}

/** Human-readable label for an issue type. */
function issueLabel(type: string): string {
  return type
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export default function IssueCard({ issue }: IssueCardProps) {
  return (
    <div className="issue-card">
      <div className="issue-card__top">
        <div className={severityClass(issue.severity)}>
          {severityIcon(issue.severity)}
        </div>
        <div className="issue-card__info">
          <div className="issue-card__header">
            <span className="issue-card__type">{issueLabel(issue.type)}</span>
            <span className={severityBadgeClass(issue.severity)}>
              {issue.severity}
            </span>
          </div>
        </div>
      </div>

      {/* Impact */}
      {issue.impact && (
        <p className="issue-card__explanation">{issue.impact}</p>
      )}

      {/* Evidence: metric / value / threshold */}
      <div className="issue-card__evidence">
        <div className="issue-card__evidence-item">
          <span className="issue-card__evidence-label">Metric</span>
          <span className="issue-card__evidence-value">{issue.metric}</span>
        </div>
        <div className="issue-card__evidence-item">
          <span className="issue-card__evidence-label">Value</span>
          <span className="issue-card__evidence-value">
            {issue.value.toFixed(2)}
          </span>
        </div>
        <div className="issue-card__evidence-item">
          <span className="issue-card__evidence-label">Threshold</span>
          <span className="issue-card__evidence-value">
            {issue.threshold.toFixed(2)}
          </span>
        </div>
      </div>

      {/* Description */}
      {issue.description && (
        <p className="issue-card__description">{issue.description}</p>
      )}

      {/* Confidence */}
      <div className="issue-card__confidence-row">
        <span className="issue-card__confidence-label">Confidence</span>
        <ConfidenceBar value={Math.round(issue.confidence * 100)} />
      </div>
    </div>
  );
}
