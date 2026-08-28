import { ShieldCheck } from "lucide-react";
import type { AnalyticsReadinessDetails } from "../../types/analysis";
import ConfidenceBar from "./ConfidenceBar";

interface AnalyticsReadinessProps {
  score: number;
  status: string;
  details: AnalyticsReadinessDetails;
}

function statusColor(score: number): string {
  if (score >= 80) return "var(--green)";
  if (score >= 60) return "var(--blue, #3b82f6)";
  if (score >= 40) return "var(--yellow)";
  if (score >= 20) return "var(--orange, #f97316)";
  return "var(--red)";
}

interface PenaltyRowProps {
  label: string;
  value: number;
}

function PenaltyRow({ label, value }: PenaltyRowProps) {
  if (value === 0) return null;
  return (
    <div className="readiness-penalty">
      <span className="readiness-penalty__label">{label}</span>
      <span className="readiness-penalty__value">-{value.toFixed(1)}</span>
    </div>
  );
}

export default function AnalyticsReadiness({
  score,
  status,
  details,
}: AnalyticsReadinessProps) {
  const color = statusColor(score);
  const hasPenalties =
    details.blur_penalty > 0 ||
    details.exposure_penalty > 0 ||
    details.noise_penalty > 0 ||
    details.corruption_penalty > 0 ||
    details.information_penalty > 0;

  return (
    <div className="card readiness-card">
      <div className="card__heading">
        <ShieldCheck size={15} className="card__heading-icon" />
        <span className="card__heading-text">
          Analytics Readiness
          <span className="card__heading-sub">— downstream reliability</span>
        </span>
      </div>

      <div className="readiness-card__main">
        <div className="readiness-card__score">
          <span className="readiness-card__value" style={{ color }}>
            {Math.round(score)}
          </span>
          <span className="readiness-card__max">/100</span>
        </div>
        <div className="readiness-card__status">
          <span
            className="readiness-card__status-badge"
            style={{ color, borderColor: color }}
          >
            {status}
          </span>
        </div>
      </div>

      <div className="readiness-card__bar">
        <ConfidenceBar value={score} color={color} />
      </div>

      {hasPenalties && (
        <div className="readiness-card__penalties">
          <div className="readiness-card__penalties-title">
            Penalty Breakdown
          </div>
          <PenaltyRow label="Blur" value={details.blur_penalty} />
          <PenaltyRow label="Exposure" value={details.exposure_penalty} />
          <PenaltyRow label="Noise" value={details.noise_penalty} />
          <PenaltyRow label="Corruption" value={details.corruption_penalty} />
          <PenaltyRow label="Information" value={details.information_penalty} />
          <div className="readiness-card__base">
            Base quality: {details.base_quality_score.toFixed(1)}
          </div>
        </div>
      )}
    </div>
  );
}
