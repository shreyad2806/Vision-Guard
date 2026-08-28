import { MapPin } from "lucide-react";
import type { ContextImpact as ContextImpactType } from "../../types/analysis";

interface ContextImpactProps {
  context: string;
  impacts: ContextImpactType[];
}

export default function ContextImpact({
  context,
  impacts,
}: ContextImpactProps) {
  if (!context) return null;

  return (
    <div className="card">
      <div className="card__heading">
        <MapPin size={15} className="card__heading-icon" />
        <span className="card__heading-text">
          Smart-City Impact
          <span className="card__heading-sub">— {context}</span>
        </span>
      </div>

      <div className="context-impact__context-badge">
        <span className="context-impact__badge">{context}</span>
      </div>

      {impacts && impacts.length > 0 ? (
        <div className="context-impact__list">
          {impacts.map((imp, i) => (
            <div key={`${imp.issue_type}-${i}`} className="context-impact__item">
              <span className="context-impact__issue-type">
                {imp.issue_type.replace(/_/g, " ")}
              </span>
              <span className="context-impact__text">{imp.impact}</span>
            </div>
          ))}
        </div>
      ) : (
        <p className="context-impact__empty">
          No context-specific impacts — all quality metrics are within acceptable
          ranges for {context}.
        </p>
      )}
    </div>
  );
}
