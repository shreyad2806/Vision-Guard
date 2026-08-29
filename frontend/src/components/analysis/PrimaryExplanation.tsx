import { AlertTriangle, CheckCircle, ArrowRight } from "lucide-react";
import type {
  PrimaryExplanation as PrimaryExplanationType,
  PrimaryIssue,
  SecondaryIssue,
} from "../../types/analysis";

interface PrimaryExplanationProps {
  explanation: PrimaryExplanationType;
}

const SEVERITY_COLORS: Record<string, string> = {
  critical: "var(--red)",
  high: "var(--orange, #f97316)",
  moderate: "var(--yellow, #f59e0b)",
  low: "var(--text-muted)",
};

function PrimaryIssueCard({ issue }: { issue: PrimaryIssue }) {
  const color = SEVERITY_COLORS[issue.severity] || "var(--text-muted)";

  return (
    <div
      style={{
        background: `${color}08`,
        border: `1px solid ${color}30`,
        borderRadius: 8,
        padding: "16px 20px",
        marginBottom: 16,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          marginBottom: 12,
        }}
      >
        <AlertTriangle size={18} style={{ color }} />
        <span
          style={{
            fontWeight: 700,
            fontSize: 16,
            color,
          }}
        >
          {issue.label}
        </span>
        <span
          style={{
            fontSize: 12,
            fontWeight: 600,
            color,
            padding: "2px 8px",
            borderRadius: 4,
            background: `${color}15`,
            border: `1px solid ${color}30`,
            textTransform: "uppercase",
          }}
        >
          {issue.severity}
        </span>
      </div>

      {/* Evidence */}
      <div style={{ marginBottom: 12 }}>
        <span
          style={{
            fontSize: 11,
            fontWeight: 600,
            color: "var(--text-muted)",
            textTransform: "uppercase",
            letterSpacing: 0.5,
          }}
        >
          Evidence
        </span>
        <div
          style={{
            fontSize: 14,
            color: "var(--text)",
            marginTop: 4,
          }}
        >
          <span style={{ fontWeight: 500 }}>
            {issue.evidence.metric.replace(/_/g, " ")}:
          </span>{" "}
          {typeof issue.evidence.value === "number"
            ? issue.evidence.value.toFixed(2)
            : issue.evidence.value}
          <span style={{ color: "var(--text-muted)", marginLeft: 8 }}>
            (threshold: {issue.evidence.threshold.toFixed(2)})
          </span>
        </div>
      </div>

      {/* Why this matters */}
      <div style={{ marginBottom: 12 }}>
        <span
          style={{
            fontSize: 11,
            fontWeight: 600,
            color: "var(--text-muted)",
            textTransform: "uppercase",
            letterSpacing: 0.5,
          }}
        >
          Why this matters
        </span>
        <p
          style={{
            fontSize: 14,
            color: "var(--text-secondary, var(--text))",
            margin: "4px 0 0 0",
            lineHeight: 1.5,
          }}
        >
          {issue.impact}
        </p>
      </div>
    </div>
  );
}

function NoIssuesCard() {
  return (
    <div
      style={{
        background: "rgba(34, 197, 94, 0.05)",
        border: "1px solid rgba(34, 197, 94, 0.2)",
        borderRadius: 8,
        padding: "16px 20px",
        marginBottom: 16,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
        }}
      >
        <CheckCircle size={18} style={{ color: "var(--green)" }} />
        <span
          style={{
            fontWeight: 600,
            fontSize: 15,
            color: "var(--green)",
          }}
        >
          No significant image-quality issues detected
        </span>
      </div>
    </div>
  );
}

function SecondaryIssues({ issues }: { issues: SecondaryIssue[] }) {
  if (issues.length === 0) return null;

  const SEVERITY_BADGE_COLORS: Record<string, string> = {
    critical: "var(--red)",
    high: "var(--orange, #f97316)",
    moderate: "var(--yellow, #f59e0b)",
    low: "var(--text-muted)",
  };

  return (
    <div style={{ marginBottom: 16 }}>
      <span
        style={{
          fontSize: 12,
          fontWeight: 600,
          color: "var(--text-muted)",
          textTransform: "uppercase",
          letterSpacing: 0.5,
        }}
      >
        Other Detected Issues
      </span>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 8,
          marginTop: 8,
        }}
      >
        {issues.map((issue, i) => {
          const color = SEVERITY_BADGE_COLORS[issue.severity] || "var(--text-muted)";
          return (
            <div
              key={`${issue.type}-${i}`}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                fontSize: 13,
              }}
            >
              <span style={{ color: "var(--text-muted)" }}>•</span>
              <span style={{ color: "var(--text)" }}>{issue.label}</span>
              <span
                style={{
                  fontSize: 11,
                  fontWeight: 600,
                  color,
                  padding: "1px 6px",
                  borderRadius: 3,
                  background: `${color}10`,
                }}
              >
                {issue.severity}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function DownstreamImpact({ explanation }: { explanation: PrimaryExplanationType }) {
  if (!explanation.downstream_summary || explanation.downstream_summary.length === 0) {
    return null;
  }

  const RELIABILITY_COLORS: Record<string, string> = {
    HIGH: "var(--green)",
    MEDIUM: "var(--yellow, #f59e0b)",
    LOW: "var(--orange, #f97316)",
    "VERY LOW": "var(--red)",
  };

  return (
    <div style={{ marginBottom: 16 }}>
      <span
        style={{
          fontSize: 12,
          fontWeight: 600,
          color: "var(--text-muted)",
          textTransform: "uppercase",
          letterSpacing: 0.5,
        }}
      >
        Downstream Impact
      </span>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 6,
          marginTop: 8,
        }}
      >
        {explanation.downstream_summary.map((item, i) => {
          const color = RELIABILITY_COLORS[item.estimated_reliability] || "var(--text-muted)";
          return (
            <div
              key={`${item.analytics_type}-${i}`}
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                fontSize: 13,
              }}
            >
              <span style={{ color: "var(--text)" }}>{item.analytics_type}</span>
              <span
                style={{
                  fontSize: 12,
                  fontWeight: 600,
                  color,
                }}
              >
                {item.estimated_reliability}
              </span>
            </div>
          );
        })}
      </div>
      <p
        style={{
          fontSize: 12,
          color: "var(--text-muted)",
          marginTop: 8,
          fontStyle: "italic",
        }}
      >
        {explanation.downstream_impact}
      </p>
    </div>
  );
}

function Recommendation({ text }: { text: string }) {
  return (
    <div
      style={{
        background: "rgba(59, 130, 246, 0.05)",
        border: "1px solid rgba(59, 130, 246, 0.15)",
        borderRadius: 8,
        padding: "12px 16px",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "flex-start",
          gap: 8,
        }}
      >
        <ArrowRight
          size={16}
          style={{ color: "var(--blue, #3b82f6)", marginTop: 2, flexShrink: 0 }}
        />
        <div>
          <span
            style={{
              fontSize: 12,
              fontWeight: 600,
              color: "var(--text-muted)",
              textTransform: "uppercase",
              letterSpacing: 0.5,
            }}
          >
            Recommendation
          </span>
          <p
            style={{
              fontSize: 14,
              color: "var(--text)",
              margin: "4px 0 0 0",
              lineHeight: 1.5,
            }}
          >
            {text}
          </p>
        </div>
      </div>
    </div>
  );
}

export default function PrimaryExplanation({
  explanation,
}: PrimaryExplanationProps) {
  if (!explanation) return null;

  return (
    <div className="card">
      <div className="card__heading">
        <span className="card__heading-text">Why this decision?</span>
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
        Issue-driven explanation based on detected image-quality conditions.
      </p>

      {/* Primary Issue */}
      {explanation.no_issues_detected ? (
        <NoIssuesCard />
      ) : explanation.primary_issue ? (
        <PrimaryIssueCard issue={explanation.primary_issue} />
      ) : null}

      {/* Why it matters */}
      <div style={{ marginBottom: 16 }}>
        <span
          style={{
            fontSize: 12,
            fontWeight: 600,
            color: "var(--text-muted)",
            textTransform: "uppercase",
            letterSpacing: 0.5,
          }}
        >
          Why This Matters
        </span>
        <p
          style={{
            fontSize: 14,
            color: "var(--text)",
            margin: "4px 0 0 0",
            lineHeight: 1.5,
          }}
        >
          {explanation.why_it_matters}
        </p>
      </div>

      {/* Downstream Impact */}
      <DownstreamImpact explanation={explanation} />

      {/* Secondary Issues */}
      <SecondaryIssues issues={explanation.secondary_issues} />

      {/* Recommendation */}
      <Recommendation text={explanation.recommendation} />
    </div>
  );
}
