# Phase 10 — ML Training Pipeline Audit (v2.1 — Leakage-Free)

**Date:** 2026-08-29  
**Auditor:** Buffy (automated)  
**Scope:** Complete ML pipeline audit for VisionGuard image quality model  
**Version:** v2.1.0 — Calibration leakage removed

---

## 1. Training Data Source

| Property | Value |
|----------|-------|
| Dataset 1 | **KADID-10K** (Kadid Image Database) |
| Dataset 2 | **KonIQ-10K** (KonIQ-10K Database) |
| CSV rows (KADID) | 10,125 |
| CSV rows (KonIQ) | 10,073 |
| Images available (KADID) | 10,206 |
| Images available (KonIQ) | 6,425 |
| Training samples used | 16,300 combined (10,125 KADID + 6,175 KonIQ) |

**Ground-truth labels:** Both datasets provide continuous quality scores:
- KADID: DMOS (Difference Mean Opinion Score), higher = better quality
- KonIQ: MOS (Mean Opinion Score), higher = better quality

Both are normalized independently to 0–100 scale (higher = better).

---

## 2. Dataset Composition

**KADID-10K:** Distorted versions of 81 reference images. Each reference has 5 distortion types at 5 severity levels = ~25 distortions per reference. This means many training images share the same underlying scene, which introduces **potential within-reference correlation**.

**KonIQ-10K:** Independent images from various sources with crowdsourced MOS ratings. Lower correlation between images.

**Combined distribution:**
| Quality Bin | Count |
|-------------|-------|
| [0–20) | 1,750 |
| [20–40) | 3,094 |
| [40–60) | 3,434 |
| [60–80) | 4,845 |
| [80–100) | 3,177 |
| **Total** | **16,300** |

---

## 3. Feature Extraction

5 engineered features extracted from OpenCV at 224×224 resize:

| Feature | Computation | Unit |
|---------|------------|------|
| brightness | mean(grayscale) | 0–255 |
| contrast | std(grayscale) | 0–~100 |
| sharpness | Laplacian variance | 0–∞ |
| saturation | mean(HSV S channel) | 0–255 |
| edge_density | Canny(100,200) > 0 mean | 0–1 |

**Training/inference consistency:** Both `ml/train.py` and `backend/apps/ml/feature_extractor.py` use identical preprocessing: resize 224×224, grayscale for brightness/contrast/sharpness/edge_density, HSV for saturation, Canny(100,200). This is verified by code inspection.

**Limitation:** Only 5 features are used by the ML model. The system also extracts 12 features for UI display and issue detection, but these are NOT used by the model.

---

## 4. Model Architecture

| Property | Value |
|----------|-------|
| Model type | RandomForestRegressor |
| n_estimators | 300 |
| max_depth | 15 |
| min_samples_leaf | 5 |
| n_jobs | -1 |
| random_state | 42 |

**Why this model:** Selected by highest Spearman rank correlation on validation set. RandomForest was preferred over HistGradientBoosting (Spearman: 0.56 vs 0.55) because rank correlation better reflects the model's ability to correctly order samples — more important for a quality scoring system than raw RMSE.

---

## 5. Train/Validation/Test Split

| Property | Value |
|----------|-------|
| Method | Stratified split by quality bins |
| Train ratio | 70% (11,414 samples) |
| Validation ratio | 15% (2,443 samples) |
| Test ratio | 15% (2,443 samples) |
| Random seed | 42 |
| Number of bins | 5 (edges: 0, 20, 40, 60, 80, 100) |

**No overlap:** All three splits are verified to contain disjoint samples. The sum of all splits equals the total dataset size.

---

## 6. Calibration

| Property | Value |
|----------|-------|
| Method | IsotonicRegression |
| y_min | 0.0 |
| y_max | 100.0 |
| increasing | True |
| out_of_bounds | clip |
| **Fitted on** | **Validation set predictions (15% split)** |
| **Never fitted on** | **Test set** |

**Mapping:** Raw prediction range [5.21, 90.81] → Calibrated score [0–100].

**v2.1 correction:** The calibrator is fitted on validation predictions only. Test data is never used for calibration fitting. This eliminates the calibration leakage that existed in v2.0.

---

## 7. Reported Metrics (from model_metadata.json)

### Raw Model (before calibration)
| Metric | Value | Interpretation |
|--------|-------|----------------|
| MAE | 16.71 | Average error of ~17 points on 0–100 scale |
| RMSE | 20.30 | Root mean squared error |
| R² | 0.33 | Model explains ~33% of quality variance |
| Spearman | 0.55 | Moderate rank correlation |

### Calibrated Model (after calibration)
| Metric | Value | Interpretation |
|--------|-------|----------------|
| MAE | 16.60 | Average error of ~17 points on 0–100 scale |
| RMSE | 20.32 | Root mean squared error |
| R² | 0.33 | Model explains ~33% of quality variance |
| Spearman | 0.55 | Moderate rank correlation |

**Honest assessment:** These are **moderate** metrics. R²=0.33 indicates the model captures some but not all quality variance. The MAE of 17 points means predictions are often ±17 points from the true normalized quality score. This is acceptable for a screening/triage tool but not for precision quality grading.

**Calibration effect:** Calibration slightly improves MAE (16.71→16.60) but marginally worsens R² (0.335→0.333) and Spearman (0.548→0.546). This suggests the calibration mapping is mostly redistributing scores without significantly changing the model's ranking ability.

---

## 8. Strengths

1. **Hybrid architecture:** Deterministic CV features + learned model + rule-based issue detection = interpretable pipeline
2. **Training/inference consistency:** Feature extraction code matches between training and production
3. **Stratified 3-way split:** Quality bins are balanced across train/validation/test
4. **Multiple model comparison:** RandomForest, HistGradientBoosting, ExtraTrees were compared; best selected by Spearman
5. **Isotonic calibration:** Monotonic mapping ensures score ordering is preserved
6. **Independent issue detection:** Not derived from model — rule-based with real metrics
7. **Reproducible:** Fixed random seed (42), deterministic feature extraction
8. **No calibration leakage:** Calibrator fitted on validation set only, never on test data

## 9. Weaknesses and Risks

1. **5-feature limitation:** Only brightness/contrast/sharpness/saturation/edge_density feed the model; 12 features exist but 7 are unused by ML
2. **KADID within-reference correlation:** 10,125 images derive from 81 references; if not properly split, this leaks information
3. **Narrow raw prediction range:** Raw predictions [5–91] mapped to [0–100] — the model rarely predicts extreme values
4. **No cross-validation:** Single 70/15/15 split; no k-fold or nested CV for robust metric estimation
5. **Confidence from tree variance is rudimentary:** `_estimate_confidence` uses `std(estimator.predict)` which is not well-calibrated; returns fixed 75.0 on failure
6. **No data augmentation:** Training uses only original images
7. **Feature distribution mismatch risk:** Training on KADID/KonIQ (indoor/outdoor photography) may not generalize to smart-city imagery (CCTV, drone, traffic)
8. **Score compression:** Model outputs tend toward middle range; very high or low quality images may not be distinguished well
9. **R²=0.33:** Model explains only 33% of quality variance — moderate, not high, predictive power

## 10. Reproducibility

| Aspect | Status |
|--------|--------|
| Random seed | ✅ Fixed (42) |
| Feature extraction | ✅ Deterministic (OpenCV) |
| Model training | ✅ Deterministic (scikit-learn with seed) |
| Dataset availability | ✅ Verified on disk |
| Artifact serialization | ✅ joblib with version pinned |
| Environment | ⚠️ Python 3.12, scikit-learn version-dependent |
| Three-way split | ✅ Verified no overlap between train/val/test |

---

## Summary

The VisionGuard ML pipeline v2.1 is a **technically defensible classical ML system** with a clear pipeline, reproducible training, and honest limitations. The calibration leakage present in v2.0 has been **removed**: the IsotonicRegression calibrator is now fitted on a dedicated validation set (15%), and final metrics are computed on an untouched test set (15%).

The R²=0.33 and Spearman=0.55 indicate moderate predictive power, which is consistent with using only 5 visual features to predict a subjective quality score. The system is appropriate as a **screening/triage tool** for smart-city image quality assessment, not as a precision quality measurement instrument.
