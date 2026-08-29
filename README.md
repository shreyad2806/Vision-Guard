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