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
    case "ACCEPTABLE":
      return "var(--green)";
    case "DEGRADED":
      return "var(--yellow)";
    case "DEFECTIVE":
      return "var(--red)";
  }
}

function labelMessage(label: QualityLabel): string {
  switch (label) {
    case "ACCEPTABLE":
      return "Image quality is suitable for downstream visual analysis.";
    case "DEGRADED":
      return "Quality issues detected that may affect downstream analysis.";
    case "DEFECTIVE":
      return "Significant visual degradation detected. Review or recapture is recommended.";
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
