import { Loader2 } from "lucide-react";

interface LoadingStateProps {
  message?: string;
  stages?: string[];
  activeStage?: number;
}

export default function LoadingState({
  message = "Analyzing visual quality…",
  stages,
  activeStage,
}: LoadingStateProps) {
  return (
    <div className="state-view">
      <div className="spinner" />
      <p className="state-view__title" style={{ marginTop: 16 }}>
        {message}
      </p>
      {stages && activeStage !== undefined && (
        <div className="stages">
          {stages.map((stage, i) => (
            <div
              key={stage}
              className={`stages__item ${
                i < activeStage
                  ? "stages__item--done"
                  : i === activeStage
                    ? "stages__item--active"
                    : ""
              }`}
            >
              {i < activeStage ? (
                <Loader2 size={14} />
              ) : i === activeStage ? (
                <Loader2 size={14} className="spinner-small" />
              ) : (
                <Loader2 size={14} style={{ opacity: 0.3 }} />
              )}
              {stage}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
