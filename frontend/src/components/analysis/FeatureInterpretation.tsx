import type { AnalysisResult } from "../../types/analysis";

interface FeatureInterpretationProps {
  result: AnalysisResult;
}

interface Observation {
  text: string;
  level: "good" | "warn" | "bad";
}

function generateObservations(result: AnalysisResult): Observation[] {
  const { statistics: s, quality_label } = result;
  const obs: Observation[] = [];

  // Sharpness
  if (s.sharpness >= 60) {
    obs.push({ text: "Sharpness is within the optimal range.", level: "good" });
  } else if (s.sharpness >= 35) {
    obs.push({ text: "Sharpness is below the optimal threshold.", level: "warn" });
  } else {
    obs.push({ text: "Sharpness is critically low — edge detail is unrecoverable.", level: "bad" });
  }

  // Brightness
  if (s.brightness >= 80 && s.brightness <= 180) {
    obs.push({ text: "Brightness is balanced with no significant clipping.", level: "good" });
  } else if (s.brightness < 80) {
    obs.push({ text: "Image is underexposed — shadow detail is lost.", level: "bad" });
  } else {
    obs.push({ text: "Image is overexposed — highlight clipping detected.", level: "bad" });
  }

  // Noise
  if (s.noise_estimate <= 10) {
    obs.push({ text: "Noise levels are minimal and within clean thresholds.", level: "good" });
  } else if (s.noise_estimate <= 20) {
    obs.push({ text: "Moderate image noise was detected.", level: "warn" });
  } else {
    obs.push({ text: "High noise levels significantly degrade image quality.", level: "bad" });
  }

  // Entropy
  if (s.entropy >= 6) {
    obs.push({ text: "Rich visual information is present in the image.", level: "good" });
  } else if (s.entropy >= 4) {
    obs.push({ text: "Information content is limited.", level: "warn" });
  } else {
    obs.push({ text: "Very low information content — image may be corrupted.", level: "bad" });
  }

  // Overall verdict
  if (quality_label === "ACCEPTABLE") {
    obs.push({
      text: "Overall visual quality remains suitable for analysis.",
      level: "good",
    });
  } else if (quality_label === "DEGRADED") {
    obs.push({
      text: "Quality issues may impact downstream computer vision reliability.",
      level: "warn",
    });
  } else {
    obs.push({
      text: "Visual data is unreliable — recapture or manual review is needed.",
      level: "bad",
    });
  }

  return obs;
}

export default function FeatureInterpretation({
  result,
}: FeatureInterpretationProps) {
  const observations = generateObservations(result);

  return (
    <div className="card">
      <div className="card__heading">
        <span className="card__heading-text">Why this decision?</span>
      </div>
      <ul className="feature-interp__list">
        {observations.map((obs) => (
          <li key={obs.text} className="feature-interp__item">
            <span className={`feature-interp__bullet feature-interp__bullet--${obs.level}`} />
            <span>{obs.text}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
