import type { QualityLabel } from "../../types/analysis";
import StatusBadge from "../common/StatusBadge";
import ConfidenceBar from "./ConfidenceBar";

interface QualityScoreProps {
  score: number;
  label: QualityLabel;
  analysisConfidence: number;
}

function labelColor(label: QualityLabel): string {
  switch (label) {
    case "Excellent":
      return "var(--green)";
    case "Good":
      return "var(--blue, #3b82f6)";
    case "Fair":
      return "var(--yellow)";
    case "Poor":
      return "var(--orange, #f97316)";
    case "Critical":
      return "var(--red)";
  }
}

function labelMessage(label: QualityLabel): string {
  switch (label) {
    case "Excellent":
      return "Image quality is suitable for automated analysis.";
    case "Good":
      return "Image quality is generally reliable, with minor degradation.";
    case "Fair":
      return "Image quality is usable but some degradation may affect downstream analysis.";
    case "Poor":
      return "Image quality is significantly degraded. Review or recapture is recommended.";
    case "Critical":
      return "Image quality is critically degraded and unreliable for automated analysis.";
  }
}

const RADIUS = 78;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

export default function QualityScore({
  score,
  label,
  analysisConfidence,
}: QualityScoreProps) {
  const color = labelColor(label);
  const offset = CIRCUMFERENCE - (score / 100) * CIRCUMFERENCE;

  return (
    <div className="card card--highlight score-card">
      <div className="score-card__label">Overall Quality</div>

      <div className="score-ring" style={{ margin: "0 auto 12px" }}>
        <svg
          className="score-ring__svg"
          width="100%"
          height="100%"
          viewBox="0 0 180 180"
        >
          <circle className="score-ring__bg" cx="90" cy="90" r={RADIUS} />
          <circle
            className="score-ring__fill"
            cx="90"
            cy="90"
            r={RADIUS}
            stroke={color}
            strokeDasharray={CIRCUMFERENCE}
            strokeDashoffset={offset}
          />
        </svg>
        <div className="score-ring__label">
          <span className="score-ring__value" style={{ color }}>
            {score}
          </span>
          <span className="score-ring__max">/100</span>
        </div>
      </div>

      <div className="score-card__badge-row">
        <StatusBadge label={label} showIcon />
      </div>

      <p className="score-card__message">{labelMessage(label)}</p>

      <div className="score-card__confidence">
        <div className="score-card__confidence-label">
          Analysis Confidence
        </div>
        <ConfidenceBar value={analysisConfidence} />
      </div>
    </div>
  );
}
