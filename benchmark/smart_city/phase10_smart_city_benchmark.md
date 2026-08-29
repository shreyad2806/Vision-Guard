# VisionGuard Smart-City Benchmark Results

This benchmark evaluates VisionGuard across 600 images spanning five smart-city visual contexts. It measures the quality and downstream analytics readiness assigned by the VisionGuard pipeline.

**Important:** These are benchmark dataset averages, not ML accuracy measurements. The ML predictive performance metrics (MAE, RMSE, R², Spearman) are reported separately from the leakage-free test evaluation.

## Dataset Coverage

| Context | Dataset | Images |
|---|---|---:|
| CCTV Surveillance | MOT17 | 100 |
| Drone Imagery | VisDrone | 200 |
| Infrastructure Inspection | RoadDefects-ISeg | 100 |
| Low-Light Evaluation | LOL | 100 |
| Traffic Monitoring | BDD100K | 100 |
| **Total** | | **600** |

## Benchmark Results

### Quality and Readiness Summary

| Context | Images | Avg Quality | Avg Readiness |
|---|---:|---:|---:|
| CCTV Surveillance | 100 | 48.73 | 45.97 |
| Drone Imagery | 200 | 38.00 | 23.29 |
| Infrastructure Inspection | 100 | 48.34 | 28.53 |
| Low-Light Evaluation | 100 | 25.61 | 0.59 |
| Traffic Monitoring | 100 | 55.61 | 42.28 |

### Readiness Distribution

| Context | Ready | Limited | Not Ready | Critical |
|---|---:|---:|---:|---:|
| CCTV Surveillance | 23 | 44 | 23 | 10 |
| Drone Imagery | 0 | 0 | 200 | 0 |
| Infrastructure Inspection | 1 | 13 | 53 | 33 |
| Low-Light Evaluation | 0 | 0 | 0 | 100 |
| Traffic Monitoring | 14 | 43 | 32 | 11 |

### Issue Distribution

| Context | Issue | Count |
|---|---|---:|
| CCTV Surveillance | Overexposure | 14 |
| CCTV Surveillance | Insufficient sharpness | 4 |
| Drone Imagery | Overexposure | 200 |
| Infrastructure Inspection | Image noise | 72 |
| Infrastructure Inspection | Overexposure | 19 |
| Infrastructure Inspection | Low color information | 17 |
| Infrastructure Inspection | Low contrast | 2 |
| Low-Light Evaluation | Underexposure | 100 |
| Low-Light Evaluation | Severe visual degradation | 84 |
| Low-Light Evaluation | Insufficient sharpness | 71 |
| Low-Light Evaluation | Low contrast | 10 |
| Low-Light Evaluation | Visual degradation | 2 |
| Low-Light Evaluation | Limited dynamic range | 1 |
| Traffic Monitoring | Underexposure | 48 |
| Traffic Monitoring | Insufficient sharpness | 8 |
| Traffic Monitoring | Overexposure | 8 |

## Interpretation

### CCTV Surveillance (MOT17)

The CCTV Surveillance benchmark subset produced an average quality score of 48.73/100 and average analytics readiness of 45.97/100. The readiness distribution shows 23% READY, 44% LIMITED READINESS, 23% NOT READY, and 10% CRITICAL/REJECT. The most common issues were overexposure (14 images) and insufficient sharpness (4 images). This suggests that while many CCTV frames are suitable for analytics, a significant portion have quality limitations that may affect downstream performance.

### Drone Imagery (VisDrone)

The Drone Imagery subset produced an average quality score of 38.00/100 and average readiness of 23.29/100. All 200 images were classified as NOT READY, with overexposure detected in all images. This indicates that the VisDrone dataset contains consistently challenging imaging conditions for downstream analytics, likely due to aerial perspective, lighting conditions, and small object sizes.

### Infrastructure Inspection (RoadDefects-ISeg)

The Infrastructure Inspection subset produced an average quality score of 48.34/100 and average readiness of 28.53/100. The readiness distribution shows 1% READY, 13% LIMITED READINESS, 53% NOT READY, and 33% CRITICAL/REJECT. The dominant issue was image noise (72 images), followed by overexposure (19) and low color information (17). This suggests infrastructure inspection imagery often contains noise that may affect defect detection reliability.

### Low-Light Evaluation (LOL)

The Low-Light Evaluation subset produced an average quality score of 25.61/100 and average readiness of 0.59/100. All 100 images were classified as CRITICAL/REJECT. The dominant issues were underexposure (100 images), severe visual degradation (84), and insufficient sharpness (71). This confirms that low-light imagery presents substantial downstream suitability challenges.

### Traffic Monitoring (BDD100K)

The Traffic Monitoring subset produced an average quality score of 55.61/100 and average readiness of 42.28/100. The readiness distribution shows 14% READY, 43% LIMITED READINESS, 32% NOT READY, and 11% CRITICAL/REJECT. The most common issue was underexposure (48 images), followed by insufficient sharpness (8) and overexposure (8). This suggests traffic monitoring imagery has mixed quality with significant lighting-related challenges.

## Three Types of Metrics

### A. ML Predictive Performance

Measured on the held-out test set (leakage-free evaluation):

| Metric | Value |
|---|---:|
| MAE | 13.27 |
| RMSE | 16.91 |
| R² | 0.54 |
| Spearman | 0.71 |

These evaluate how well the model predicts the quality target.

### B. Smart-City Benchmark Results

Measured across the 600 benchmark images:

| Metric | Value |
|---|---:|
| Total images | 600 |
| Average quality score | 43.26 |
| Average readiness score | 28.13 |

These describe the evaluated image population.

### C. System Performance

Measured during inference:

| Metric | Value |
|---|---:|
| Mean latency | ~112ms |
| P95 latency | ~137ms |

These measure runtime performance.

## Limitations

- The benchmark uses 600 images from 5 datasets, which may not represent all smart-city imaging conditions
- Quality scores are model predictions, not ground-truth measurements
- Readiness scores reflect VisionGuard's assessment, not actual downstream model performance
- The benchmark does not evaluate actual downstream model accuracy
- Different datasets have different imaging characteristics that may affect results
- The benchmark results are dataset-specific and should not be generalized without additional evaluation
