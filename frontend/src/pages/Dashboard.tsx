import { useCallback, useEffect, useRef, useState } from "react";
import { Sparkles, ArrowRight, ImageIcon } from "lucide-react";
import type { AnalysisResult as AnalysisResultType } from "../types/analysis";
import ImageUploader from "../components/upload/ImageUploader";
import AnalysisResult from "../components/analysis/AnalysisResult";
import QualityScore from "../components/analysis/QualityScore";
import AnalysisProgress from "../components/analysis/AnalysisProgress";
import ErrorState from "../components/common/ErrorState";
import EmptyState from "../components/common/EmptyState";
import { analyzeImage } from "../services/api";

const ANALYSIS_STAGES = [
  "Validating image",
  "Extracting visual features",
  "Running AI quality model",
  "Generating quality report",
];

const MOCK_STAGE_DURATION_MS = 600;

export default function Dashboard() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [stage, setStage] = useState(0);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<AnalysisResultType | null>(null);
  const [error, setError] = useState<string | null>(null);
  const stageTimers = useRef<ReturnType<typeof setTimeout>[]>([]);

  // Clean up timers on unmount
  useEffect(() => {
    return () => {
      stageTimers.current.forEach(clearTimeout);
    };
  }, []);

  const handleImageSelected = useCallback((file: File, url: string) => {
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return url;
    });
    setSelectedFile(file);
    setResult(null);
    setError(null);
  }, []);

  const handleRemove = useCallback(() => {
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    setSelectedFile(null);
    setResult(null);
    setError(null);
  }, []);

  const handleAnalyze = useCallback(async () => {
    if (!selectedFile) return;
    setLoading(true);
    setStage(0);
    setProgress(0);
    setError(null);
    setResult(null);

    // Clear previous timers
    stageTimers.current.forEach(clearTimeout);
    stageTimers.current = [];

    // Simulate staged progress
    for (let i = 0; i < ANALYSIS_STAGES.length; i++) {
      stageTimers.current.push(
        setTimeout(() => {
          setStage(i);
          setProgress(((i + 1) / ANALYSIS_STAGES.length) * 100);
        }, MOCK_STAGE_DURATION_MS * (i + 1)),
      );
    }

    try {
      const analysisResult = await analyzeImage(selectedFile);
      setProgress(100);
      // Brief pause at 100% before showing results
      await new Promise((r) => setTimeout(r, 300));
      setResult(analysisResult);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "An unexpected error occurred.";
      setError(message);
    } finally {
      stageTimers.current.forEach(clearTimeout);
      stageTimers.current = [];
      setLoading(false);
    }
  }, [selectedFile]);

  return (
    <div>
      {/* ── Hero ── */}
      <div className="page-header">
        <div className="page-header__top-row">
          <div>
            <h1 className="page-header__title">
              Visual Data Quality Intelligence
            </h1>
            <p className="page-header__subtitle">
              Upload an image to evaluate visual quality, detect degradation,
              and assess its reliability for downstream computer vision systems.
            </p>
          </div>
          <span className="page-header__badge">
            <Sparkles size={12} />
            AI-Powered Analysis
          </span>
        </div>
      </div>

      {/* ── Two-Column Workspace ── */}
      <div className="workspace-grid">
        {/* Left: Upload / Preview / Progress */}
        <div className="card">
          {loading ? (
            <AnalysisProgress
              stages={ANALYSIS_STAGES}
              activeStage={stage}
              progress={progress}
            />
          ) : (
            <>
              <ImageUploader
                onImageSelected={handleImageSelected}
                onRemove={handleRemove}
                selectedFile={selectedFile}
                previewUrl={previewUrl}
                disabled={loading}
              />
              <div style={{ display: "flex", justifyContent: "center", marginTop: 20 }}>
                <button
                  type="button"
                  className="btn btn--primary btn--primary-lg"
                  disabled={!selectedFile || loading}
                  onClick={handleAnalyze}
                >
                  Analyze Image
                  <ArrowRight size={16} />
                </button>
              </div>
            </>
          )}
        </div>

        {/* Right: Quality Score or Empty State */}
        <div className="card" style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
          {result && !loading ? (
            <QualityScore
              score={result.quality_score}
              label={result.quality_label}
              analysisConfidence={result.analysis_confidence}
            />
          ) : error && !loading ? (
            <ErrorState
              title="Unable to Analyze Image"
              message={error}
              onRetry={handleAnalyze}
            />
          ) : (
            <EmptyState
              icon={<ImageIcon size={40} style={{ color: "var(--text-muted)" }} />}
              title="No image selected"
              message="Upload an image to begin visual quality analysis."
            />
          )}
        </div>
      </div>

      {/* ── Full-Width Results Below Workspace ── */}
      {result && !loading && (
        <div className="results-section">
          <AnalysisResult result={result} />
        </div>
      )}

      {/* ── Error below workspace ── */}
      {error && !loading && (
        <div className="results-section">
          <ErrorState
            title="Unable to Analyze Image"
            message={error}
            onRetry={handleAnalyze}
          />
        </div>
      )}
    </div>
  );
}
