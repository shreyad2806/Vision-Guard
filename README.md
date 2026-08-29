## Mandatory Quality Detection Capabilities

VisionGuard explicitly detects the following required visual quality issues:

| Requirement | Detection Method |
|---|---|
| Blur / Insufficient Sharpness | Laplacian variance and edge density |
| Underexposure | Dark pixel percentage and brightness analysis |
| Overexposure | Bright pixel percentage and brightness analysis |
| Image Noise | Laplacian-based noise estimate |
| Image Corruption / Severe Degradation | Multi-signal degradation detection |
| Potential Visual Defect | Abnormal feature combination detection |
| Additional Quality Issues | Contrast, saturation, entropy, dynamic range, structural content |

## Model Evaluation (v4.0 — Leakage-Free, Feature-Engineered)

The VisionGuard image quality model uses a **HistGradientBoosting regressor** trained on 16,300 images from KADID-10K and KonIQ-10K datasets. The calibration (IsotonicRegression) is fitted on a **validation set only** — test data is never used for training or calibration.

### Split

| Split | Samples | Percentage |
|-------|---------|------------|
| Train | 11,414 | 70.0% |
| Validation | 2,443 | 15.0% |
| Test | 2,443 | 15.0% |

### Test Set Metrics

| Metric | Raw | Calibrated |
|--------|-----|------------|
| MAE | 13.42 | 13.27 |
| RMSE | 16.93 | 16.91 |
| R² | 0.54 | 0.54 |
| Spearman | 0.71 | 0.71 |

### Key Properties

- **20 features:** brightness, contrast, sharpness, saturation, edge_density, noise_estimate, entropy, colorfulness, luminance_p10, luminance_p90, dark_pixel_pct, bright_pixel_pct, luminance_median, gradient_magnitude, dynamic_range, percentile_range, noise_to_signal, channel_std, hue_entropy, texture_complexity
- **Calibration:** IsotonicRegression fitted on validation set (never on test data)
- **Test set:** Untouched during training and calibration
- **Model version:** v4.0.0
- **Random seed:** 42
- **Latency:** ~112ms total pipeline (P95: 137ms)

### Feature Importance

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | sharpness | 0.952 |
| 2 | gradient_magnitude | 0.282 |
| 3 | colorfulness | 0.157 |
| 4 | entropy | 0.140 |
| 5 | saturation | 0.092 |
| 6 | noise_to_signal | 0.085 |
| 7 | edge_density | 0.083 |
| 8 | hue_entropy | 0.072 |
| 9 | luminance_p10 | 0.069 |
| 10 | dark_pixel_pct | 0.068 |

The model relies primarily on sharpness, structural information (gradient magnitude, edge density), and color features for quality prediction.

### Honest Limitations

- R²=0.54: Model explains 54% of quality variance
- 20 hand-crafted features cannot capture all perceptual quality dimensions
- Training domain (photography) differs from deployment domain (smart-city)
- Score compression: calibrated predictions range [7.6, 83.0] vs. target range [0, 100]
- No issue-level ground truth for detector evaluation
- Single split (no cross-validation)

## Controlled-Condition Evaluation

VisionGuard was evaluated under controlled visual conditions to verify that the system produces condition-sensitive assessments.

| Condition | Quality | Readiness | Status | Primary Issue |
|---|---:|---:|---|---|
| Clean | 60.8 | 45.8 | LIMITED READINESS | Underexposure (minor) |
| Blurred | 13.7 | 0.0 | CRITICAL / REJECT | Severe Blur |
| Underexposed | 24.7 | 0.0 | CRITICAL / REJECT | Severe Blur |
| Overexposed | 66.7 | 21.7 | NOT READY | Overexposure |
| Noisy | 29.6 | 0.0 | CRITICAL / REJECT | Underexposure |
| Severe Degradation | 13.7 | 0.0 | CRITICAL / REJECT | Underexposure |

These controlled tests demonstrate that changes in image quality produce corresponding changes in detected issues, quality assessment, and downstream analytics readiness. Quality scores range from 13.7 (degraded) to 66.7 (overexposed), and readiness scores range from 0.0 (critical) to 45.8 (limited).

## ML Model Performance

| Metric | v4.0 |
|---|---:|
| MAE | 13.27 |
| RMSE | 16.91 |
| R² | 0.54 |
| Spearman | 0.71 |

These are ML predictive metrics measured on the held-out test set (leakage-free evaluation).

## Smart-City Benchmark Results

VisionGuard was evaluated on 600 images spanning five smart-city contexts.

| Context | Images | Avg Quality | Avg Readiness |
|---|---:|---:|---:|
| CCTV Surveillance | 100 | 48.73 | 45.97 |
| Drone Imagery | 200 | 38.00 | 23.29 |
| Infrastructure Inspection | 100 | 48.34 | 28.53 |
| Low-Light Evaluation | 100 | 25.61 | 0.59 |
| Traffic Monitoring | 100 | 55.61 | 42.28 |

> **Note:** Average quality and analytics readiness scores are benchmark dataset statistics, not model accuracy metrics. Predictive model performance is reported separately using MAE, RMSE, R², and Spearman correlation on the held-out test set.

## Downstream Analytics Simulation

VisionGuard estimates how detected image-quality problems may affect downstream computer-vision analytics. This provides an **estimated downstream reliability** assessment — not measured downstream model accuracy.

### How It Works

```
Image
  ↓
Feature Extraction
  ↓
Quality Assessment
  ↓
Issue Detection
  ↓
Analytics Readiness
  ↓
Estimated Downstream Reliability
```

### Supported Analytics

| Analytics | Description |
|---|---|
| Object Detection | General object detection and classification |
| Vehicle Detection | Vehicle identification and tracking |
| Person Detection | Pedestrian and person detection |
| OCR / License Plate | Optical character recognition and license plate reading |
| Infrastructure / Defect Detection | Structural defect and damage detection |

### Example Result

| Analytics | Estimated Reliability | Reason |
|---|---|---|
| Object Detection | HIGH | No major quality issues detected |
| Vehicle Detection | MEDIUM | Underexposure may reduce visibility |
| Person Detection | MEDIUM | Low-light conditions affect detection |
| OCR / License Plate | LOW | Blur and noise degrade text recognition |

### Context Awareness

The system tailors analytics to the selected pipeline context:

- **CCTV Surveillance:** Object, Vehicle, Person, OCR detection
- **Traffic Monitoring:** Vehicle, Object, OCR detection
- **Infrastructure Inspection:** Infrastructure/Defect, Object detection
- **Drone Imagery:** Object, Vehicle, Person detection
- **Smart Campus:** Object, Vehicle, Person, OCR detection

**Important:** These are estimated reliability assessments derived from VisionGuard's image-quality analysis. They are not measured downstream model accuracies. No downstream models (YOLO, OCR, etc.) are actually executed or evaluated.

## Explainable Decisions

VisionGuard does not only return a score. It identifies the primary image-quality issue and provides a structured explanation:

```
PRIMARY ISSUE
  → Underexposure — HIGH

EVIDENCE
  → Dark pixel ratio: 24.18% (threshold: 25.00%)

WHY THIS MATTERS
  → Underexposure reduces contrast in shadow areas, lowering detection
    confidence for people, vehicles, and infrastructure elements.

DOWNSTREAM IMPACT
  → Person Detection: MEDIUM
  → Vehicle Detection: MEDIUM
  → OCR / License Plate: LOW

RECOMMENDATION
  → Increase camera exposure or gain, add supplementary lighting,
    or apply adaptive histogram equalisation.
```

Each explanation is driven by the actual detected issues, their severity, and measured evidence from the image. No values are hard-coded.