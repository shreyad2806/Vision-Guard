export type QualityLabel = "ACCEPTABLE" | "DEGRADED" | "DEFECTIVE";

export type Severity = "LOW" | "MEDIUM" | "HIGH";

export interface Issue {
  type: string;
  severity: Severity;
  confidence: number;
  explanation: string;
}

export interface ImageStatistics {
  sharpness: number;
  brightness: number;
  contrast: number;
  noise_estimate: number;
  entropy: number;
  saturation: number;
  // Extended ML features
  underexposure_pct?: number;
  overexposure_pct?: number;
  edge_density?: number;
  dynamic_range?: number;
  colorfulness?: number;
  texture_complexity?: number;
}

export interface AnalysisResult {
  analysis_id: string;
  filename: string;
  image_url?: string;
  quality_score: number;
  quality_label: QualityLabel;
  analysis_confidence: number;
  issues: Issue[];
  statistics: ImageStatistics;
  summary: string;
  created_at: string;
  processing_time_ms: number;
}
