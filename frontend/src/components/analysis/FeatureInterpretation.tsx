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

  // If the backend provides explanations, use them directly
  if (result.explanations && result.explanations.length > 0) {
    for (const explanation of result.explanations) {
      const lower = explanation.toLowerCase();
      let level: "good" | "warn" | "bad" = "good";
      if (lower.includes("below") || lower.includes("blur") || lower.includes("low") || lower.includes("desaturated") || lower.includes("overexposure") || lower.includes("underexposure")) {
        level = lower.includes("significantly") || lower.includes("well below") || lower.includes("very low") ? "bad" : "warn";
      }
      obs.push({ text: explanation, level });
    }
  } else {
    // Fallback to local observations
    // Sharpness
    if (s.sharpness >= 250) {
      obs.push({ text: "Sharpness is within the optimal range.", level: "good" });
    } else if (s.sharpness >= 100) {
      obs.push({ text: "Sharpness is below the optimal threshold.", level: "warn" });
    } else {
      obs.push({ text: "Sharpness is critically low — edge detail is unrecoverable.", level: "bad" });
    }

    // Brightness
    if (s.brightness >= 80 && s.brightness <= 170) {
      obs.push({ text: "Brightness is balanced with no significant clipping.", level: "good" });
    } else if (s.brightness < 50) {
      obs.push({ text: "Image is underexposed — shadow detail is lost.", level: "bad" });
    } else if (s.brightness > 200) {
      obs.push({ text: "Image is overexposed — highlight clipping detected.", level: "bad" });
    } else {
      obs.push({ text: "Brightness is somewhat outside the optimal range.", level: "warn" });
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
  }

  // Overall verdict (always append)
  if (quality_label === "Excellent" || quality_label === "Good") {
    obs.push({
      text: "Overall visual quality remains suitable for analysis.",
      level: "good",
    });
  } else if (quality_label === "Fair") {
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
