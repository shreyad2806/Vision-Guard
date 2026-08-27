export type QualityLabel = "Excellent" | "Good" | "Fair" | "Poor" | "Critical";

export type Severity = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";

export interface Issue {
  type: string;
  title?: string;
  severity: Severity;
  confidence: number;
  description?: string;
  explanation?: string;
  feature_value?: number;
  threshold?: number;
}

export interface QualityAssessment {
  label: string;
  status: string;
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
  quality_assessment?: QualityAssessment;
  analysis_confidence: number;
  raw_prediction?: number | null;
  issues: Issue[];
  statistics: ImageStatistics;
  explanations?: string[];
  summary: string;
  recommendation?: string;
  created_at: string;
  processing_time_ms: number;
  model_version?: string;
}
