import { BarChart3 } from "lucide-react";
import type { ImageStatistics as ImageStats } from "../../types/analysis";

interface ImageStatisticsProps {
  statistics: ImageStats;
}

interface MetricDef {
  key: keyof ImageStats;
  label: string;
  max: number;
  hint: (v: number) => string;
  optional?: boolean;
}

const metrics: MetricDef[] = [
  {
    key: "sharpness",
    label: "Sharpness",
    max: 100,
    hint: (v) =>
      v >= 60
        ? "Good edge detail"
        : v >= 35
          ? "Below optimal threshold"
          : "Critically low — unusable",
  },
  {
    key: "brightness",
    label: "Brightness",
    max: 255,
    hint: (v) =>
      v >= 80 && v <= 180
        ? "Within acceptable range"
        : v < 80
          ? "Underexposed"
          : "Overexposed",
  },
  {
    key: "contrast",
    label: "Contrast",
    max: 100,
    hint: (v) =>
      v >= 40
        ? "Adequate dynamic range"
        : "Low contrast — limited detail",
  },
  {
    key: "noise_estimate",
    label: "Noise Estimate",
    max: 50,
    hint: (v) =>
      v <= 10
        ? "Clean signal"
        : v <= 20
          ? "Moderate noise present"
          : "High noise — degrades analysis",
  },
  {
    key: "entropy",
    label: "Entropy",
    max: 8,
    hint: (v) =>
      v >= 6
        ? "Rich visual information"
        : v >= 4
          ? "Limited information content"
          : "Very low information",
  },
  {
    key: "saturation",
    label: "Saturation",
    max: 100,
    hint: (v) =>
      v >= 25 && v <= 70
        ? "Natural color balance"
        : v < 25
          ? "Desaturated — may lack color data"
          : "Highly saturated",
  },
  {
    key: "underexposure_pct",
    label: "Underexposure",
    max: 100,
    optional: true,
    hint: (v) =>
      v <= 5
        ? "Minimal dark clipping"
        : v <= 15
          ? "Moderate dark regions"
          : "Significant underexposure",
  },
  {
    key: "overexposure_pct",
    label: "Overexposure",
    max: 100,
    optional: true,
    hint: (v) =>
      v <= 5
        ? "Minimal highlight clipping"
        : v <= 15
          ? "Moderate bright regions"
          : "Significant overexposure",
  },
  {
    key: "edge_density",
    label: "Edge Density",
    max: 30,
    optional: true,
    hint: (v) =>
      v >= 8
        ? "Good structural detail"
        : v >= 4
          ? "Moderate detail"
          : "Low structural content",
  },
  {
    key: "dynamic_range",
    label: "Dynamic Range",
    max: 255,
    optional: true,
    hint: (v) =>
      v >= 150
        ? "Wide tonal range"
        : v >= 80
          ? "Moderate range"
          : "Compressed range",
  },
  {
    key: "colorfulness",
    label: "Colorfulness",
    max: 100,
    optional: true,
    hint: (v) =>
      v >= 30 && v <= 70
        ? "Natural color diversity"
        : v < 30
          ? "Limited color variation"
          : "Highly colorful",
  },
  {
    key: "texture_complexity",
    label: "Texture",
    max: 100,
    optional: true,
    hint: (v) =>
      v >= 30
        ? "Rich texture detail"
        : v >= 15
          ? "Moderate texture"
          : "Low texture complexity",
  },
];

function getBarColor(value: number, max: number, key: string): string {
  const ratio = value / max;
  if (key === "noise_estimate") {
    return ratio <= 0.2
      ? "var(--green)"
      : ratio <= 0.4
        ? "var(--yellow)"
        : "var(--red)";
  }
  if (key === "brightness") {
    return ratio >= 0.31 && ratio <= 0.71
      ? "var(--green)"
      : "var(--yellow)";
  }
  return ratio >= 0.6
    ? "var(--green)"
    : ratio >= 0.35
      ? "var(--yellow)"
      : "var(--red)";
}

export default function ImageStatistics({ statistics }: ImageStatisticsProps) {
  return (
    <div className="card">
      <div className="card__heading">
        <BarChart3 size={15} className="card__heading-icon" />
        <span className="card__heading-text">
          Image Quality Metrics
          <span className="card__heading-sub">
            — interpretable visual characteristics
          </span>
        </span>
      </div>
      <div className="metrics-grid">
        {metrics.map((m) => {
          const value = statistics[m.key];
          // Skip optional metrics if not present
          if (m.optional && value === undefined) return null;
          const pct = Math.min(((value ?? 0) / m.max) * 100, 100);
          const color = getBarColor(value ?? 0, m.max, m.key);
          return (
            <div key={m.key} className="metric-card">
              <div className="metric-card__label">{m.label}</div>
              <div className="metric-card__value">{value ?? "—"}</div>
              <div className="metric-card__bar">
                <div
                  className="metric-card__bar-fill"
                  style={{ width: `${pct}%`, background: color }}
                />
              </div>
              <div className="metric-card__hint">{m.hint(value ?? 0)}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
