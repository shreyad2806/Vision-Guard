import { Lightbulb } from "lucide-react";
import type { IssueExplanation } from "../../types/analysis";

interface IssueExplanationsProps {
  explanations: IssueExplanation[];
}

export default function IssueExplanations({
  explanations,
}: IssueExplanationsProps) {
  if (!explanations || explanations.length === 0) return null;

  return (
    <div className="card">
      <div className="card__heading">
        <Lightbulb size={15} className="card__heading-icon" />
        <span className="card__heading-text">
          Explainability
          <span className="card__heading-sub">
            — why this analysis matters
          </span>
        </span>
      </div>

      <div className="explanations-list">
        {explanations.map((exp, i) => (
          <div key={`${exp.issue}-${i}`} className="explanation-card">
            <div className="explanation-card__header">
              <span className="explanation-card__issue">{exp.issue}</span>
            </div>

            {/* Evidence */}
            <div className="explanation-card__evidence">
              <span className="explanation-card__evidence-label">Evidence</span>
              <span className="explanation-card__evidence-detail">
                {exp.evidence.metric}: {exp.evidence.value.toFixed(2)}
                {" "}
                (threshold: {exp.evidence.threshold.toFixed(2)})
              </span>
            </div>

            {/* Why it matters */}
            <div className="explanation-card__section">
              <span className="explanation-card__section-label">
                Why this matters
              </span>
              <p className="explanation-card__text">{exp.why_it_matters}</p>
            </div>

            {/* Recommendation */}
            <div className="explanation-card__section">
              <span className="explanation-card__section-label">
                Recommendation
              </span>
              <p className="explanation-card__text">
                {exp.recommendation}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
