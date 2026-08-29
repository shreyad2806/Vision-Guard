"""Step 5 — Controlled-Condition Evaluation & Robustness Report

Runs the current v4.0 production pipeline on controlled visual conditions
and generates polished, reproducible robustness artifacts.
"""

import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
BENCH = PROJECT_ROOT / "benchmark" / "smart_city"

sys.path.insert(0, str(BACKEND_DIR))

from apps.ml.model_loader import load_model, get_model, get_calibrator, get_metadata, get_feature_names
from apps.ml.feature_extractor import extract_model_features, extract_all_features
from apps.ml.calibration import apply_calibration
from apps.ml.issue_detector import detect_issues
from apps.ml.readiness import calculate_analytics_readiness
from apps.ml.explainability import generate_issue_explanations
from apps.ml.context_definitions import get_context_impacts

load_model()
model = get_model()
calibrator = get_calibrator()
meta = get_metadata()
feature_names = get_feature_names()


def spearman_correlation(y_true, y_pred):
    from scipy import stats as scipy_stats
    if len(y_true) < 2:
        return 0.0
    corr, _ = scipy_stats.spearmanr(y_true, y_pred)
    return 0.0 if np.isnan(corr) else float(corr)


def generate_controlled_images():
    """Generate controlled-condition test images."""
    rng = np.random.RandomState(42)

    # Base clean image with structure
    img_clean = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img_clean, (50, 50), (590, 430), (180, 200, 160), -1)
    cv2.rectangle(img_clean, (80, 80), (560, 400), (140, 160, 120), 3)
    cv2.putText(img_clean, "TEST IMAGE", (160, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (60, 80, 50), 3)
    cv2.rectangle(img_clean, (100, 300), (300, 380), (100, 140, 200), -1)
    cv2.rectangle(img_clean, (350, 300), (550, 380), (200, 140, 100), -1)
    noise = rng.randint(0, 8, img_clean.shape, dtype=np.uint8)
    img_clean = cv2.add(img_clean, noise)

    # Blurred
    img_blur = cv2.GaussianBlur(img_clean.copy(), (51, 51), 25)

    # Underexposed
    img_under = np.clip(img_clean.copy().astype(float) * 0.12, 0, 255).astype(np.uint8)

    # Overexposed
    img_over = np.clip(img_clean.copy().astype(float) * 2.8 + 40, 0, 255).astype(np.uint8)

    # Noisy
    noise_arr = rng.normal(0, 50, img_clean.shape).astype(np.float32)
    img_noisy = np.clip(img_clean.copy().astype(float) + noise_arr, 0, 255).astype(np.uint8)

    # Severe degradation (very dark, flat, low information)
    img_severe = np.zeros((480, 640, 3), dtype=np.uint8)
    img_severe[:] = (15, 15, 15)
    severe_noise = rng.normal(0, 8, img_severe.shape).astype(np.float32)
    img_severe = np.clip(img_severe.astype(float) + severe_noise, 0, 255).astype(np.uint8)

    return {
        "clean": img_clean,
        "blurred": img_blur,
        "underexposed": img_under,
        "overexposed": img_over,
        "noisy": img_noisy,
        "severely_degraded": img_severe,
    }


def evaluate_condition(name, img):
    """Run the full production pipeline on a single image."""
    # Extract features
    model_feats = extract_model_features(img)
    all_feats = extract_all_features(img)

    # ML prediction
    vector = np.array([[model_feats.get(k, 0.0) for k in feature_names]])
    raw = float(model.predict(vector)[0])
    cal = apply_calibration(raw, calibrator)

    # Issue detection
    issues = detect_issues(all_feats)

    # Readiness
    readiness = calculate_analytics_readiness(cal, issues)

    # Explainability
    explanations = generate_issue_explanations(issues)

    # Context impacts (CCTV as representative context)
    impacts = get_context_impacts(issues, "CCTV Surveillance")

    return {
        "condition": name,
        "quality_score": round(cal, 2),
        "raw_prediction": round(raw, 4),
        "analytics_readiness_score": round(readiness["score"], 2),
        "readiness_status": readiness["status"],
        "issue_count": len(issues),
        "issues": issues,
        "primary_issue": issues[0]["type"] if issues else "none",
        "primary_severity": issues[0].get("severity", "none") if issues else "none",
        "primary_confidence": issues[0].get("confidence", 0.0) if issues else 0.0,
        "issue_types": "; ".join(i["type"] for i in issues),
        "issue_details": [{"type": i["type"], "severity": i.get("severity", "unknown"),
                          "confidence": i.get("confidence", 0.0),
                          "metric": i.get("metric", ""),
                          "value": i.get("value", 0),
                          "threshold": i.get("threshold", 0)} for i in issues],
        "explanations": explanations,
        "context_impacts": impacts,
        "model_features": model_feats,
        "all_features": all_feats,
    }


def main():
    print("=" * 70)
    print("  Step 5: Controlled-Condition Evaluation (v4.0)")
    print("=" * 70)

    print(f"\nModel: {type(model).__name__} v{meta.get('model_version', 'unknown')}")
    print(f"Features: {len(feature_names)}")
    print(f"Calibrator: {type(calibrator).__name__}")

    # Generate controlled images
    print("\n[1] Generating controlled-condition test images...")
    images = generate_controlled_images()
    print(f"  Created {len(images)} conditions: {list(images.keys())}")

    # Run evaluation
    print("\n[2] Running production pipeline on each condition...")
    results = []
    for name, img in images.items():
        result = evaluate_condition(name, img)
        results.append(result)
        print(f"  {name:25s} quality={result['quality_score']:6.1f}  "
              f"readiness={result['analytics_readiness_score']:6.1f}  "
              f"issues={result['issue_count']}  "
              f"primary={result['primary_issue']}")

    # Create polished CSV
    print("\n[3] Creating controlled-condition results CSV...")
    csv_path = BENCH / "phase10_controlled_condition_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "quality_score", "analytics_readiness_score",
                         "readiness_status", "primary_issue", "primary_severity",
                         "primary_confidence", "issue_count", "issue_types",
                         "raw_prediction"])
        for r in results:
            writer.writerow([
                r["condition"], r["quality_score"], r["analytics_readiness_score"],
                r["readiness_status"], r["primary_issue"], r["primary_severity"],
                r["primary_confidence"], r["issue_count"], r["issue_types"],
                r["raw_prediction"],
            ])
    print(f"  Saved: {csv_path}")

    # Create detailed JSON
    print("\n[4] Creating detailed JSON report...")
    json_data = {
        "model_version": meta.get("model_version", "unknown"),
        "model_type": type(model).__name__,
        "calibrator_type": type(calibrator).__name__,
        "feature_count": len(feature_names),
        "conditions": []
    }
    for r in results:
        json_data["conditions"].append({
            "condition": r["condition"],
            "quality_score": r["quality_score"],
            "raw_prediction": r["raw_prediction"],
            "analytics_readiness_score": r["analytics_readiness_score"],
            "readiness_status": r["readiness_status"],
            "issue_count": r["issue_count"],
            "primary_issue": r["primary_issue"],
            "issues": r["issue_details"],
            "model_features": r["model_features"],
        })
    json_path = BENCH / "phase10_controlled_condition_details.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  Saved: {json_path}")

    # Create markdown report
    print("\n[5] Creating robustness interpretation report...")
    md_path = BENCH / "phase10_controlled_condition_report.md"
    with open(md_path, "w") as f:
        f.write("# Controlled-Condition Robustness Report — VisionGuard v4.0\n\n")
        f.write("## Purpose\n\n")
        f.write("The controlled-condition evaluation tests whether VisionGuard responds differently ")
        f.write("to common visual-quality degradations. Each condition modifies a synthetic base image ")
        f.write("in a specific way, and the full production pipeline processes the result.\n\n")
        f.write("## Controlled-Condition Robustness Summary\n\n")
        f.write("| Condition | Quality | Readiness | Status | Primary Issue | Issues |\n")
        f.write("|---|---:|---:|---|---|---:|\n")
        for r in results:
            f.write(f"| {r['condition'].replace('_', ' ').title()} | {r['quality_score']:.1f} | "
                    f"{r['analytics_readiness_score']:.1f} | {r['readiness_status']} | "
                    f"{r['primary_issue'].replace('_', ' ').title()} | {r['issue_count']} |\n")

        f.write("\n## Detailed Results\n\n")
        for r in results:
            cond_title = r["condition"].replace("_", " ").title()
            f.write(f"### {cond_title}\n\n")
            f.write(f"- **Quality Score:** {r['quality_score']:.1f}\n")
            f.write(f"- **Raw Prediction:** {r['raw_prediction']:.4f}\n")
            f.write(f"- **Readiness Score:** {r['analytics_readiness_score']:.1f}\n")
            f.write(f"- **Readiness Status:** {r['readiness_status']}\n")
            f.write(f"- **Issues Detected:** {r['issue_count']}\n")
            if r["issues"]:
                f.write(f"\n| Issue | Severity | Confidence | Metric | Value | Threshold |\n")
                f.write(f"|---|---|---:|---|---:|---:|\n")
                for issue in r["issue_details"]:
                    f.write(f"| {issue['type'].replace('_', ' ').title()} | {issue['severity']} | "
                            f"{issue['confidence']:.2f} | {issue['metric']} | {issue['value']:.2f} | "
                            f"{issue['threshold']:.2f} |\n")
            f.write("\n")

            # Key features
            mf = r["model_features"]
            f.write("**Key Features:**\n")
            f.write(f"- brightness: {mf.get('brightness', 0):.1f}\n")
            f.write(f"- sharpness: {mf.get('sharpness', 0):.1f}\n")
            f.write(f"- contrast: {mf.get('contrast', 0):.1f}\n")
            f.write(f"- noise_estimate: {mf.get('noise_estimate', 0):.1f}\n")
            f.write(f"- edge_density: {mf.get('edge_density', 0):.4f}\n")
            f.write(f"- dark_pixel_pct: {mf.get('dark_pixel_pct', 0):.1f}\n")
            f.write(f"- bright_pixel_pct: {mf.get('bright_pixel_pct', 0):.1f}\n")
            f.write("\n")

        # Analysis section
        f.write("## Analysis\n\n")

        # Clean analysis
        clean = [r for r in results if r["condition"] == "clean"][0]
        f.write("### Clean Image\n\n")
        f.write(f"The clean image receives a quality score of **{clean['quality_score']:.1f}** ")
        f.write(f"with readiness **{clean['analytics_readiness_score']:.1f}**. ")
        if clean["issue_count"] == 0:
            f.write("No issues are detected, confirming the system correctly identifies good-quality imagery.\n\n")
        else:
            f.write(f"The system detects {clean['issue_count']} issue(s): {clean['issue_types']}. ")
            f.write("This may reflect the synthetic nature of the base image having limited texture/complexity.\n\n")

        # Blur analysis
        blur = [r for r in results if r["condition"] == "blurred"][0]
        blur_feat = blur["model_features"]
        f.write("### Blur\n\n")
        f.write(f"Blurring reduces quality to **{blur['quality_score']:.1f}** (vs clean: {clean['quality_score']:.1f}). ")
        f.write(f"Sharpness drops from {clean['model_features'].get('sharpness', 0):.1f} to {blur_feat.get('sharpness', 0):.1f}. ")
        f.write(f"Edge density changes from {clean['model_features'].get('edge_density', 0):.4f} to {blur_feat.get('edge_density', 0):.4f}. ")
        blur_issues = [i["type"] for i in blur["issues"]]
        f.write(f"The system detects: {', '.join(blur_issues)}.\n\n")

        # Underexposure analysis
        under = [r for r in results if r["condition"] == "underexposed"][0]
        under_feat = under["model_features"]
        f.write("### Underexposure\n\n")
        f.write(f"Underexposure reduces quality to **{under['quality_score']:.1f}**. ")
        f.write(f"Brightness drops from {clean['model_features'].get('brightness', 0):.1f} to {under_feat.get('brightness', 0):.1f}. ")
        f.write(f"Dark pixel percentage increases from {clean['model_features'].get('dark_pixel_pct', 0):.1f}% to {under_feat.get('dark_pixel_pct', 0):.1f}%. ")
        under_issues = [i["type"] for i in under["issues"]]
        f.write(f"The system detects: {', '.join(under_issues)}.\n\n")

        # Overexposure analysis
        over = [r for r in results if r["condition"] == "overexposed"][0]
        over_feat = over["model_features"]
        f.write("### Overexposure\n\n")
        f.write(f"Overexposure results in quality **{over['quality_score']:.1f}**. ")
        f.write(f"Brightness increases to {over_feat.get('brightness', 0):.1f} (from {clean['model_features'].get('brightness', 0):.1f}). ")
        f.write(f"Bright pixel percentage increases from {clean['model_features'].get('bright_pixel_pct', 0):.1f}% to {over_feat.get('bright_pixel_pct', 0):.1f}%. ")
        over_issues = [i["type"] for i in over["issues"]]
        f.write(f"The system detects: {', '.join(over_issues)}.\n\n")

        # Noise analysis
        noisy = [r for r in results if r["condition"] == "noisy"][0]
        noisy_feat = noisy["model_features"]
        f.write("### Noise\n\n")
        f.write(f"Noise results in quality **{noisy['quality_score']:.1f}**. ")
        f.write(f"Noise estimate increases from {clean['model_features'].get('noise_estimate', 0):.1f} to {noisy_feat.get('noise_estimate', 0):.1f}. ")
        noisy_issues = [i["type"] for i in noisy["issues"]]
        f.write(f"The system detects: {', '.join(noisy_issues)}.\n\n")

        # Severe degradation analysis
        severe = [r for r in results if r["condition"] == "severely_degraded"][0]
        severe_feat = severe["model_features"]
        f.write("### Severe Degradation\n\n")
        f.write(f"Severe degradation results in quality **{severe['quality_score']:.1f}**. ")
        f.write(f"Brightness drops to {severe_feat.get('brightness', 0):.1f}. ")
        f.write(f"Sharpness drops to {severe_feat.get('sharpness', 0):.1f}. ")
        f.write(f"Entropy drops to {severe_feat.get('entropy', 0):.2f} (indicating very low information content). ")
        severe_issues = [i["type"] for i in severe["issues"]]
        f.write(f"The system detects: {', '.join(severe_issues)}.\n\n")

        # Differentiation analysis
        f.write("## Quality/Readiness Differentiation\n\n")
        scores = [r["quality_score"] for r in results]
        readies = [r["analytics_readiness_score"] for r in results]
        conditions = [r["condition"] for r in results]
        f.write("| Condition | Quality | Readiness |\n")
        f.write("|---|---:|---:|\n")
        for c, s, r in zip(conditions, scores, readies):
            f.write(f"| {c.replace('_', ' ').title()} | {s:.1f} | {r:.1f} |\n")
        f.write(f"\n- **Quality range:** {min(scores):.1f} to {max(scores):.1f} (spread: {max(scores) - min(scores):.1f})\n")
        f.write(f"- **Readiness range:** {min(readies):.1f} to {max(readies):.1f} (spread: {max(readies) - min(readies):.1f})\n")
        all_same_quality = len(set(round(s, 1) for s in scores)) == 1
        all_same_readiness = len(set(round(r, 1) for r in readies)) == 1
        f.write(f"- **All conditions produce same quality:** {'YES (concern)' if all_same_quality else 'NO (expected)'}\n")
        f.write(f"- **All conditions produce same readiness:** {'YES (concern)' if all_same_readiness else 'NO (expected)'}\n\n")

        # Conclusion
        f.write("## Key Finding\n\n")
        if not all_same_quality and not all_same_readiness:
            f.write("VisionGuard produces **condition-sensitive** quality and readiness assessments. ")
            f.write("Different visual conditions produce meaningfully different quality scores, ")
            f.write("readiness levels, and detected issues. The system responds to measurable image ")
            f.write("characteristics (sharpness, brightness, noise, information content) rather than ")
            f.write("returning a fixed score regardless of input quality.\n\n")
        else:
            f.write("Some conditions produce similar scores, which may indicate limitations in the model's ")
            f.write("ability to distinguish certain degradation types.\n\n")

        f.write("## Limitations\n\n")
        f.write("- Synthetic test images may not fully represent real-world degradation patterns\n")
        f.write("- Controlled conditions test one degradation at a time; real images may have multiple simultaneous degradations\n")
        f.write("- Quality scores are model predictions, not ground-truth measurements\n")
        f.write("- The issue detector uses rule-based thresholds, not learned detection\n")
        f.write("- No issue-level ground truth exists to validate detection accuracy\n")

    print(f"  Saved: {md_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("  Controlled-Condition Robustness Summary")
    print("=" * 80)
    print(f"  {'Condition':<25} {'Quality':>8} {'Readiness':>10} {'Status':<20} {'Primary Issue':<30}")
    print("  " + "-" * 95)
    for r in results:
        cond = r["condition"].replace("_", " ").title()
        issue = r["primary_issue"].replace("_", " ").title() if r["primary_issue"] != "none" else "None"
        print(f"  {cond:<25} {r['quality_score']:>8.1f} {r['analytics_readiness_score']:>10.1f} "
              f"{r['readiness_status']:<20} {issue:<30}")

    scores = [r["quality_score"] for r in results]
    readies = [r["analytics_readiness_score"] for r in results]
    print(f"\n  Quality range:  {min(scores):.1f} to {max(scores):.1f} (spread: {max(scores) - min(scores):.1f})")
    print(f"  Readiness range: {min(readies):.1f} to {max(readies):.1f} (spread: {max(readies) - min(readies):.1f})")

    all_same_q = len(set(round(s, 1) for s in scores)) == 1
    all_same_r = len(set(round(r, 1) for r in readies)) == 1
    print(f"\n  Quality varies across conditions: {'NO' if all_same_q else 'YES'}")
    print(f"  Readiness varies across conditions: {'NO' if all_same_r else 'YES'}")

    print("\n" + "=" * 70)
    print("  Controlled-Condition Evaluation Complete")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
