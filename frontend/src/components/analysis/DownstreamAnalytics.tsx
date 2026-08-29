import { Activity } from "lucide-react";
import type { DownstreamAnalytics as DownstreamAnalyticsType } from "../../types/analysis";

interface DownstreamAnalyticsProps {
  analytics: DownstreamAnalyticsType[];
}

const RELIABILITY_COLORS: Record<string, string> = {
  HIGH: "var(--green)",
  MEDIUM: "var(--yellow, #f59e0b)",
  LOW: "var(--orange, #f97316)",
  "VERY LOW": "var(--red)",
};

const RELIABILITY_BG: Record<string, string> = {
  HIGH: "rgba(34, 197, 94, 0.1)",
  MEDIUM: "rgba(245, 158, 11, 0.1)",
  LOW: "rgba(249, 115, 22, 0.1)",
  "VERY LOW": "rgba(239, 68, 68, 0.1)",
};

export default function DownstreamAnalytics({
  analytics,
}: DownstreamAnalyticsProps) {
  if (!analytics || analytics.length === 0) return null;

  return (
    <div className="card">
      <div className="card__heading">
        <Activity size={15} className="card__heading-icon" />
        <span className="card__heading-text">
          Downstream Analytics Impact
        </span>
      </div>

      <p
        style={{
          fontSize: 12,
          color: "var(--text-muted)",
          marginTop: 0,
          marginBottom: 16,
          fontStyle: "italic",
        }}
      >
        Estimated downstream reliability based on detected image-quality
        conditions; not measured model accuracy.
      </p>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
          gap: 12,
        }}
      >
        {analytics.map((item) => {
          const reliability = item.estimated_reliability;
          const color = RELIABILITY_COLORS[reliability] || "var(--text-muted)";
          const bgColor = RELIABILITY_BG[reliability] || "transparent";

          return (
            <div
              key={item.analytics_key}
              style={{
                border: `1px solid ${color}33`,
                borderRadius: 8,
                padding: "14px 16px",
                background: bgColor,
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: 8,
                }}
              >
                <span
                  style={{
                    fontWeight: 600,
                    fontSize: 14,
                    color: "var(--text)",
                  }}
                >
                  {item.analytics_type}
                </span>
                <span
                  style={{
                    fontSize: 12,
                    fontWeight: 700,
                    color,
                    padding: "2px 8px",
                    borderRadius: 4,
                    background: `${color}15`,
                    border: `1px solid ${color}30`,
                  }}
                >
                  {reliability}
                </span>
              </div>

              <p
                style={{
                  fontSize: 13,
                  color: "var(--text-secondary, var(--text-muted))",
                  margin: "0 0 8px 0",
                  lineHeight: 1.4,
                }}
              >
                {item.primary_reason}
              </p>

              {item.expected_impact && (
                <p
                  style={{
                    fontSize: 12,
                    color: "var(--text-muted)",
                    margin: 0,
                    fontStyle: "italic",
                  }}
                >
                  {item.expected_impact}
                </p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
