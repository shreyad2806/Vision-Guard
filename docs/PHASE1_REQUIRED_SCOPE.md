# VisionGuard -- Phase 1: Required Scope

> **Day 1, Phase 1** -- Freeze the compulsory project scope before implementing further features.
>
> This document defines every mandatory detection capability, its current
> implementation status, and what remains before Phase 1 is complete.

---

## 1. Project Context

VisionGuard is an AI-powered visual data quality and analytics-readiness layer
for smart-city camera pipelines. It evaluates incoming CCTV, traffic, drone,
infrastructure, and smart-campus imagery before downstream computer-vision
systems consume the data.

The system must detect, quantify, and explain quality problems that would make
an image unreliable for automated analysis.

---

## 2. Extracted Visual Features (Current)

The following 12 features are extracted by
`backend/apps/ml/feature_extractor.py` via `extract_all_features()`:

| # | Feature Key            | Computation Method                                           | Used by Model | Used by Issue Detection |
|---|------------------------|--------------------------------------------------------------|:------------:|:-----------------------:|
| 1 | `sharpness`            | Variance of Laplacian (CV_64F) on grayscale                 | Yes (5 model features) | Yes             |
| 2 | `brightness`           | Mean grayscale intensity (0-255)                             | Yes          | Yes                     |
| 3 | `contrast`             | Standard deviation of grayscale intensity                    | Yes          | Yes                     |
| 4 | `noise_estimate`       | MAD-based sigma from Laplacian (`1.4826 * median(abs(L))`)  | No           | Yes                     |
| 5 | `entropy`              | Shannon entropy of grayscale histogram (256 bins)            | No           | Yes                     |
| 6 | `saturation`           | Mean HSV S-channel                                           | Yes          | Yes                     |
| 7 | `underexposure_pct`    | `mean(gray < 30) * 100`                                     | No           | **No -- GAP**           |
| 8 | `overexposure_pct`     | `mean(gray > 225) * 100`                                    | No           | **No -- GAP**           |
| 9 | `edge_density`         | `mean(Canny(gray, 50, 150) > 0) * 100`                      | Yes          | Yes                     |
| 10 | `dynamic_range`        | `percentile(gray, 95) - percentile(gray, 5)`                | No           | Yes                     |
| 11 | `colorfulness`         | Hasler & Susstrunk colorfulness metric (RG-YB)               | No           | **No -- GAP**           |
| 12 | `texture_complexity`   | High-frequency energy ratio via FFT                           | No           | **No -- GAP**           |

> **Note:** The 5 model features (brightness, contrast, sharpness, saturation,
> edge_density) are also extracted at 224x224 resolution by
> `extract_model_features()` and must stay in sync with `ml/train.py`.

---

## 3. Mandatory Detection Capabilities

### 3.1 Blur / Insufficient Sharpness

| Column | Value |
|--------|-------|
| **Requirement** | Detect images that are blurred, out of focus, or have critically low edge detail, preventing reliable object detection or scene understanding. |
| **Current feature(s)** | `sharpness` (Laplacian variance), `edge_density` (Canny edge ratio) |
| **Detection logic** | `issue_detector.py`: sharpness < p5 (30.0) triggers "Severe Blur" (HIGH); sharpness < p10 (80.0) triggers "Low Sharpness" (MEDIUM). Thresholds come from `DEFAULT_THRESHOLDS` or training metadata percentiles. |
| **Threshold / evidence** | Severe: sharpness < 30.0; Low: sharpness < 80.0; Optimal: sharpness >= 200.0. Edge density < 5.0% also triggers "Low Structural Content" (MEDIUM). |
| **Backend output** | `Issue(type="severe_blur", severity="HIGH")` or `Issue(type="low_sharpness", severity="MEDIUM")` with confidence, feature_value, threshold, description. |
| **Frontend explanation** | "Sharpness is significantly below the calibrated threshold. The image may be blurry." / "Image appears soft or slightly blurry. Edge detail is below the expected threshold." |
| **Status** | **IMPLEMENTED** -- Two-tier detection (severe + moderate) via Laplacian variance. No motion-blur vs. defocus-blur distinction. |

---

### 3.2 Underexposure

| Column | Value |
|--------|-------|
| **Requirement** | Detect images that are too dark for downstream analysis, where shadow detail is lost and object detection will fail in dark regions. |
| **Current feature(s)** | `brightness` (mean grayscale), `underexposure_pct` (pixels < 30) |
| **Detection logic** | `issue_detector.py`: brightness < p5 (35.0) triggers "Severe Underexposure" (HIGH); brightness < p10 (55.0) triggers "Low Brightness" (MEDIUM). The `underexposure_pct` feature is **extracted but never used** in issue detection. |
| **Threshold / evidence** | Severe: brightness < 35.0; Low: brightness < 55.0; Optimal low: 80.0. The `underexposure_pct` feature (percentage of pixels < 30) exists but has **no detection rule**. |
| **Backend output** | `Issue(type="severe_underexposure", severity="HIGH")` or `Issue(type="low_brightness", severity="MEDIUM")` with confidence, feature_value, threshold, description. |
| **Frontend explanation** | "Image is extremely dark. Most shadow detail is lost and the image is unreliable for visual analysis." / "Image appears darker than the expected range. Shadow detail may be partially lost." |
| **Status** | **PARTIALLY IMPLEMENTED** -- Brightness-based detection works. `underexposure_pct` (pixel-level dark ratio) is extracted but has **no corresponding issue detection rule**. This is a gap. |

---

### 3.3 Overexposure

| Column | Value |
|--------|-------|
| **Requirement** | Detect images that are too bright, where highlight clipping has occurred and bright-region detail is lost. |
| **Current feature(s)** | `brightness` (mean grayscale), `overexposure_pct` (pixels > 225) |
| **Detection logic** | `issue_detector.py`: brightness > p95 (210.0) triggers "Severe Overexposure" (HIGH); brightness > p90 (190.0) triggers "High Brightness" (MEDIUM). The `overexposure_pct` feature is **extracted but never used** in issue detection. |
| **Threshold / evidence** | Severe: brightness > 210.0; Moderate: brightness > 190.0; Optimal high: 170.0. The `overexposure_pct` feature (percentage of pixels > 225) exists but has **no detection rule**. |
| **Backend output** | `Issue(type="severe_overexposure", severity="HIGH")` or `Issue(type="high_brightness", severity="MEDIUM")` with confidence, feature_value, threshold, description. |
| **Frontend explanation** | "Image is extremely bright with significant highlight clipping. Detail in bright areas is lost." / "Image may be overexposed. Some highlight clipping may have occurred." |
| **Status** | **PARTIALLY IMPLEMENTED** -- Brightness-based detection works. `overexposure_pct` (pixel-level bright ratio) is extracted but has **no corresponding issue detection rule**. This is a gap. |

---

### 3.4 Image Noise

| Column | Value |
|--------|-------|
| **Requirement** | Detect images with high sensor noise, compression artifacts, or grain that degrades visual quality and feature extraction reliability. |
| **Current feature(s)** | `noise_estimate` (MAD sigma from Laplacian) |
| **Detection logic** | `issue_detector.py`: noise > p95 (40.0) triggers "High Noise" (HIGH); noise > p90 (25.0) triggers "Moderate Noise" (MEDIUM). |
| **Threshold / evidence** | High: noise_estimate > 40.0; Moderate: noise_estimate > 25.0. Reference range baseline: 10.0. |
| **Backend output** | `Issue(type="high_noise", severity="HIGH")` or `Issue(type="moderate_noise", severity="MEDIUM")` with confidence, feature_value, threshold, description. |
| **Frontend explanation** | "Image contains very high levels of noise. This significantly degrades visual quality and feature extraction." / "Image contains elevated noise levels which may affect the reliability of visual analysis." |
| **Status** | **IMPLEMENTED** -- Two-tier detection via MAD-based noise estimate. No distinction between sensor noise, JPEG artifacts, or color noise. |

---

### 3.5 Image Corruption or Severe Degradation

| Column | Value |
|--------|-------|
| **Requirement** | Detect images that are corrupted (truncated JPEG, broken headers, pixel-shift artifacts, salt-and-pepper damage) or severely degraded beyond simple quality metrics. |
| **Current feature(s)** | `entropy` (Shannon entropy), `texture_complexity` (FFT high-freq ratio), general low-value fallback in multiple features |
| **Detection logic** | `issue_detector.py`: entropy < p10 (4.5) triggers "Low Information Content" (LOW). Low entropy can correlate with corruption but is **not specific** to it. There is **no dedicated corruption detection** -- no checks for: truncated file structure, abnormal histogram shapes (bimodal from salt-and-pepper), zero-variance blocks, or byte-level integrity. |
| **Threshold / evidence** | entropy < 4.5 triggers low information content. **No corruption-specific thresholds exist.** |
| **Backend output** | `Issue(type="low_entropy", severity="LOW")` -- generic, not corruption-specific. |
| **Frontend explanation** | "Image has low information content. This may indicate a featureless or heavily compressed scene." |
| **Status** | **NOT IMPLEMENTED** -- No dedicated corruption/degradation detector. Current entropy check is a weak proxy. Missing: file integrity validation, pixel-level artifact detection, block artifact analysis, truncated-image detection. |

---

### 3.6 Potential Visual Defect

| Column | Value |
|--------|-------|
| **Requirement** | Detect localized visual defects: dead pixels, lens flare, water droplets on lens, condensation, obstruction (cobwebs, branches), banding artifacts, or vignetting. |
| **Current feature(s)** | None directly. `texture_complexity` (FFT ratio) could partially detect banding. `dynamic_range` could detect flat regions from obstruction. |
| **Detection logic** | **No detection logic exists.** All 12 features are global statistics -- none perform spatial or local-region analysis. No defect-specific detectors (e.g., dead-pixel cluster detection, lens-flare color histogram analysis, banding FFT peak detection, vignetting radial falloff). |
| **Threshold / evidence** | **None defined.** |
| **Backend output** | **None.** |
| **Frontend explanation** | **None.** |
| **Status** | **NOT IMPLEMENTED** -- No visual defect detection capability exists. This requires new spatial analysis modules beyond the current global-feature approach. |

---

### 3.7 Additional Justified Quality Issues

| Column | Value |
|--------|-------|
| **Requirement** | Detect other quality issues that affect analytics readiness: low contrast, desaturation, oversaturation, limited dynamic range, low structural content, and color balance problems. |
| **Current feature(s)** | `contrast`, `saturation`, `dynamic_range`, `edge_density`, `colorfulness` |
| **Detection logic** | `issue_detector.py` handles: (1) low_contrast -- contrast < p10 (18.0), MEDIUM; (2) low_saturation -- saturation < p10 (15.0), MEDIUM; (3) oversaturation -- saturation > p95 (120.0), MEDIUM; (4) limited_dynamic_range -- dynamic_range < p10 (60.0), LOW; (5) low_structural_content -- edge_density < p10 (5.0%), MEDIUM. **NOT handled:** `colorfulness` (extracted but no threshold/rule), `texture_complexity` (extracted but no threshold/rule). |
| **Threshold / evidence** | contrast < 18.0; saturation < 15.0 or > 120.0; dynamic_range < 60.0; edge_density < 5.0%. **No thresholds for colorfulness or texture_complexity.** |
| **Backend output** | Issues: `low_contrast`, `low_saturation`, `oversaturation`, `limited_dynamic_range`, `low_structural_content` -- each with severity, confidence, description. |
| **Frontend explanation** | Per-feature explanations generated in `_generate_explanations()` covering brightness, contrast, sharpness, saturation, and edge density ranges. |
| **Status** | **PARTIALLY IMPLEMENTED** -- 5 of 7 additional issues detected. `colorfulness` and `texture_complexity` are extracted but have **no detection rules**. |

---

## 4. Gap Summary

| # | Gap | Current State | Required for Phase 1 |
|---|-----|---------------|----------------------|
| 1 | `underexposure_pct` unused | Feature extracted, no rule | Add pixel-level dark-region detection rule |
| 2 | `overexposure_pct` unused | Feature extracted, no rule | Add pixel-level highlight-clipping detection rule |
| 3 | `colorfulness` unused | Feature extracted, no rule | Add color-balance / desaturation-via-colorfulness rule |
| 4 | `texture_complexity` unused | Feature extracted, no rule | Add texture-anomaly detection rule |
| 5 | No corruption detector | Only generic entropy check | Add file-integrity + artifact-level corruption detection |
| 6 | No visual defect detector | Nothing implemented | Add spatial/local-region defect analysis (lens artifacts, dead pixels, banding, obstruction) |
| 7 | No blur-type distinction | Single blur type detected | Optional: separate motion blur vs. defocus vs. soft focus |

---

## 5. Current Pipeline Architecture

```
Image Upload
    |
    v
analysis_service.py::run_analysis()
    |
    +---> extract_model_features()    [5 features for ML model]
    +---> extract_all_features()      [12 features for UI + issue detection]
    |
    v
inference.py::predict_quality()
    |
    +---> model.predict()             [raw quality prediction]
    +---> apply_calibration()         [IsotonicRegression -> 0-100 score]
    +---> get_quality_assessment()    [score -> label]
    +---> detect_issues()             [feature thresholds -> issue list]
    +---> _generate_explanations()    [per-feature text explanations]
    +---> get_quality_recommendation() [score -> recommendation text]
    |
    v
AnalysisResponse (JSON)
    - quality_score (0-100)
    - quality_label (Excellent/Good/Fair/Poor/Critical)
    - issues: list[Issue]
    - statistics: ImageStatistics (12 features)
    - explanations: list[str]
    - summary, recommendation
```

---

## 6. Phase 1 Definition of Done

- [ ] **Blur detection**: Detect and classify blur severity with confidence score
- [ ] **Underexposure detection**: Detect using both brightness AND pixel-level dark ratio (`underexposure_pct`)
- [ ] **Overexposure detection**: Detect using both brightness AND pixel-level bright ratio (`overexposure_pct`)
- [ ] **Noise detection**: Detect with severity tiers (already implemented)
- [ ] **Corruption detection**: Validate file integrity and detect common corruption patterns (truncated data, abnormal histograms, block artifacts)
- [ ] **Visual defect detection**: Detect at minimum: dead-pixel clusters, lens artifacts (flare, vignetting), and banding/striping
- [ ] **All 12 extracted features** must be used in at least one detection rule or scoring pathway
- [ ] **Every issue type** must produce a severity (LOW/MEDIUM/HIGH/CRITICAL), confidence (0-1), human-readable title, and frontend explanation
- [ ] **Quality score** (0-100) must be computed by the ML model with IsotonicRegression calibration, with rule-based fallback when no model is loaded
- [ ] **No regression** in existing KADID-10K / KonIQ-10K training pipeline (`python -m ml.train` must still work)
- [ ] **Backend tests** must pass for all issue detector edge cases
- [ ] **Frontend** must render all issue types with appropriate severity indicators and explanations

---

## 7. Explicitly Out of Scope (Phase 1)

- Model retraining or new dataset ingestion
- Real-time video stream analysis
- Multi-frame temporal quality tracking
- Camera-specific calibration profiles
- Advanced defect classification (e.g., distinguishing cobweb from condensation)
- Automated image enhancement / restoration

---

*Document created: 2026-08-28*
*Last inspected code: `backend/apps/ml/feature_extractor.py`, `backend/apps/ml/issue_detector.py`, `backend/apps/ml/inference.py`, `backend/apps/ml/calibration.py`, `backend/apps/schemas/analysis.py`, `backend/apps/services/analysis_service.py`*
