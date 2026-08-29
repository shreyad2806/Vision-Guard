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

## Model Evaluation (v2.1 — Leakage-Free)

The VisionGuard image quality model uses a **RandomForest regressor** trained on 16,300 images from KADID-10K and KonIQ-10K datasets. The calibration (IsotonicRegression) is fitted on a **validation set only** — test data is never used for training or calibration.

### Split

| Split | Samples | Percentage |
|-------|---------|------------|
| Train | 11,414 | 70.0% |
| Validation | 2,443 | 15.0% |
| Test | 2,443 | 15.0% |

### Test Set Metrics

| Metric | Raw | Calibrated |
|--------|-----|------------|
| MAE | 16.71 | 16.60 |
| RMSE | 20.30 | 20.32 |
| R² | 0.33 | 0.33 |
| Spearman | 0.55 | 0.55 |

### Key Properties

- **5 features:** brightness, contrast, sharpness, saturation, edge_density
- **Calibration:** IsotonicRegression fitted on validation set (never on test data)
- **Test set:** Untouched during training and calibration
- **Model version:** v2.1.0
- **Random seed:** 42
- **Latency:** ~115ms total pipeline (P95: 161ms)

### Honest Limitations

- R²=0.33: Model explains 33% of quality variance
- 5 features may not capture all perceptual quality dimensions
- Training domain (photography) differs from deployment domain (smart-city)
- Score compression: calibrated predictions range [4.3, 79.4] vs. target range [0, 100]
- No issue-level ground truth for detector evaluation
- Single split (no cross-validation)