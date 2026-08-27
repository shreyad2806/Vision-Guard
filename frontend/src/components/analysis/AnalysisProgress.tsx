import { Check } from "lucide-react";

interface AnalysisProgressProps {
  stages: string[];
  activeStage: number;
  progress: number;
}

export default function AnalysisProgress({
  stages,
  activeStage,
  progress,
}: AnalysisProgressProps) {
  return (
    <div className="analysis-progress">
      <div className="spinner" style={{ marginBottom: 20 }} />
      <p className="analysis-progress__title">Analyzing Visual Quality</p>
      <p className="analysis-progress__subtitle">
        This may take a few seconds
      </p>

      <div className="analysis-progress__stages">
        {stages.map((stage, i) => {
          let state: "done" | "active" | "pending";
          if (i < activeStage) state = "done";
          else if (i === activeStage) state = "active";
          else state = "pending";

          return (
            <div
              key={stage}
              className={`analysis-progress__stage analysis-progress__stage--${state}`}
            >
              <div className="analysis-progress__stage-icon">
                {state === "done" && (
                  <div className="analysis-progress__check">
                    <Check size={12} />
                  </div>
                )}
                {state === "active" && (
                  <div className="analysis-progress__spinner" />
                )}
                {state === "pending" && (
                  <div className="analysis-progress__dot" />
                )}
              </div>
              <span>{stage}</span>
            </div>
          );
        })}
      </div>

      <div className="analysis-progress__bar">
        <div className="progress-bar">
          <div
            className="progress-bar__fill"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>
    </div>
  );
}
