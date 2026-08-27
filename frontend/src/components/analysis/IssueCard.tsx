import { AlertTriangle, AlertCircle, Info } from "lucide-react";
import type { Issue, Severity } from "../../types/analysis";
import StatusBadge from "../common/StatusBadge";
import ConfidenceBar from "./ConfidenceBar";

interface IssueCardProps {
  issue: Issue;
}

function severityIcon(severity: Severity) {
  switch (severity) {
    case "HIGH":
      return <AlertTriangle size={18} />;
    case "MEDIUM":
      return <AlertCircle size={18} />;
    case "LOW":
      return <Info size={18} />;
  }
}

function severityClass(severity: Severity): string {
  return `issue-card__icon issue-card__icon--${severity.toLowerCase()}`;
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
            <span className="issue-card__type">{issue.type}</span>
            <StatusBadge label={issue.severity} size="sm" />
          </div>
        </div>
      </div>
      <p className="issue-card__explanation">{issue.explanation}</p>
      <div className="issue-card__confidence-row">
        <span className="issue-card__confidence-label">Confidence</span>
        <ConfidenceBar value={issue.confidence} />
      </div>
    </div>
  );
}
