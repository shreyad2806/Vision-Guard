import {
  CheckCircle2,
  AlertTriangle,
  XCircle,
} from "lucide-react";
import type { QualityLabel, Severity } from "../../types/analysis";

interface StatusBadgeProps {
  label: QualityLabel | Severity;
  size?: "sm" | "md";
  showIcon?: boolean;
}

function labelIcon(label: QualityLabel | Severity, size: number) {
  switch (label) {
    case "ACCEPTABLE":
    case "LOW":
      return <CheckCircle2 size={size} />;
    case "DEGRADED":
    case "MEDIUM":
      return <AlertTriangle size={size} />;
    case "DEFECTIVE":
    case "HIGH":
      return <XCircle size={size} />;
  }
}

export default function StatusBadge({
  label,
  size = "md",
  showIcon = false,
}: StatusBadgeProps) {
  const cls =
    label === "ACCEPTABLE" || label === "DEGRADED" || label === "DEFECTIVE"
      ? `badge badge--${label.toLowerCase()}`
      : `badge badge--${label.toLowerCase()}`;

  return (
    <span
      className={cls}
      style={size === "sm" ? { padding: "2px 8px", fontSize: 10 } : undefined}
    >
      {showIcon && labelIcon(label, size === "sm" ? 10 : 12)}
      {label}
    </span>
  );
}
