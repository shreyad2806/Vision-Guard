interface ConfidenceBarProps {
  value: number;
  color?: string;
  label?: string;
}

function getConfidenceColor(value: number): string {
  if (value >= 80) return "var(--green)";
  if (value >= 60) return "var(--yellow)";
  return "var(--red)";
}

export default function ConfidenceBar({
  value,
  color,
  label,
}: ConfidenceBarProps) {
  const barColor = color ?? getConfidenceColor(value);

  return (
    <div className="confidence-bar">
      {label && (
        <span className="confidence-bar__label" style={{ fontSize: 11, color: "var(--text-muted)" }}>
          {label}
        </span>
      )}
      <div className="confidence-bar__track">
        <div
          className="confidence-bar__fill"
          style={{ width: `${value}%`, background: barColor }}
        />
      </div>
      <span className="confidence-bar__value" style={{ color: barColor }}>
        {value}%
      </span>
    </div>
  );
}
