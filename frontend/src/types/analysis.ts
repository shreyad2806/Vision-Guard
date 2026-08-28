/** Quality label from calibration.py QUALITY_BANDS */
export type QualityLabel = "Excellent" | "Good" | "Fair" | "Poor" | "Critical";

/** Issue severity — lowercase as returned by the backend issue_detector */
export type Severity = "low" | "moderate" | "high" | "critical";

/** A single detected image-quality issue (Phase 2). */
export interface Issue {
  type: string;
  severity: Severity;
  metric: string;
  value: number;
  threshold: number;
  impact: string;
  confidence: number;
  description?: string | null;
}

/** Human-readable quality classification. */
export interface QualityAssessment {
  label: string;
  status: string;
}

/** All 12+ statistics extracted by the feature extractor. */
export interface ImageStatistics {
  sharpness: number;
  brightness: number;
  contrast: number;
  noise_estimate: number;
  entropy: number;
  saturation: number;
  underexposure_pct?: number;
  overexposure_pct?: number;
  edge_density?: number;
  dynamic_range?: number;
  colorfulness?: number;
  texture_complexity?: number;
}

/** Phase 3 — analytics readiness penalty breakdown. */
export interface AnalyticsReadinessDetails {
  base_quality_score: number;
  blur_penalty: number;
  exposure_penalty: number;
  noise_penalty: number;
  corruption_penalty: number;
  information_penalty: number;
}

/** Phase 4 — context-specific impact for a detected issue. */
export interface ContextImpact {
  issue_type: string;
  context: string;
  impact: string;
}

/** Phase 6 — quantitative evidence for an issue explanation. */
export interface IssueEvidence {
  metric: string;
  value: number;
  threshold: number;
}

/** Phase 6 — structured issue-driven explainability entry. */
export interface IssueExplanation {
  issue: string;
  evidence: IssueEvidence;
  why_it_matters: string;
  recommendation: string;
}

/** Complete analysis result matching the backend AnalysisResponse schema. */
export interface AnalysisResult {
  analysis_id: string;
  filename: string;
  image_path?: string | null;
  image_url?: string | null;
  quality_score: number;
  quality_label: QualityLabel;
  analysis_confidence: number;
  issues: Issue[];
  statistics: ImageStatistics;
  summary: string;
  recommendation?: string;
  created_at: string;
  processing_time_ms: number;
  model_version?: string;
  explanations?: string[];
  quality_assessment?: QualityAssessment | null;
  raw_prediction?: number | null;

  /** Phase 3 — analytics readiness */
  analytics_readiness_score: number;
  analytics_readiness_status: string;
  analytics_readiness_details: AnalyticsReadinessDetails;

  /** Phase 4 — smart-city context */
  context?: string;
  context_impacts?: ContextImpact[];

  /** Phase 6 — issue-driven explainability */
  issue_explanations?: IssueExplanation[];
}
