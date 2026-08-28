import {
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Info,
} from "lucide-react";
import type { QualityLabel, Severity } from "../../types/analysis";

interface StatusBadgeProps {
  label: QualityLabel | Severity;
  size?: "sm" | "md";
  showIcon?: boolean;
}

function labelIcon(label: QualityLabel | Severity, size: number) {
  if (label === "Excellent" || label === "Good" || label === "low") {
    return <CheckCircle2 size={size} />;
  }
  if (label === "Fair" || label === "moderate") {
    return <AlertTriangle size={size} />;
  }
  if (label === "Poor" || label === "Critical" || label === "high" || label === "critical") {
    return <XCircle size={size} />;
  }
  return <Info size={size} />;
}

function badgeClass(label: QualityLabel | Severity): string {
  // Quality labels use lowercase CSS classes: badge--excellent, badge--good, etc.
  const lower = label.toLowerCase();
  // Map old labels for backward compatibility
  const classMap: Record<string, string> = {
    acceptable: "badge--excellent",
    degraded: "badge--fair",
    defective: "badge--poor",
  };
  return classMap[lower] ?? `badge--${lower}`;
}

export default function StatusBadge({
  label,
  size = "md",
  showIcon = false,
}: StatusBadgeProps) {
  const cls = badgeClass(label);

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
