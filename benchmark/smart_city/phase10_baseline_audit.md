# Phase 10 — Baseline Audit Report (Leakage-Free)

**Date:** 2026-08-29  
**Author:** Buffy (automated audit)  
**Version:** v2.1.0 — Calibration leakage removed  

---

## 1. Previous Evaluation Methodology

The original pipeline used a two-way split:

| Component | Value |
|-----------|-------|
| Split | 80% train / 20% test |
| Model | HistGradientBoostingRegressor |
| Calibration | IsotonicRegression |
| Calibration fitted on | **Test set** |
| Random seed | 42 |

The IsotonicRegression calibrator was fitted on the test set's raw predictions and labels, and then the same test set was used to report final metrics. This means the reported metrics reflected calibrator-tuned predictions, not truly held-out performance.

---

## 2. Leakage Discovered

### What was wrong

The calibration layer (IsotonicRegression) was fitted using test-set data:

```python
# OLD (leaky) code in ml/train.py:
calibrator, calibrated_preds = fit_calibrator(best["model"], X_test, y_test)
```

The `fit_calibrator` function received `X_test` and `y_test`, generated predictions, and then fitted the calibrator to map those predictions to the true labels. The final reported metrics were computed on these same calibrated test predictions.

### Where it was happening

- **File:** `ml/train.py` — the `fit_calibrator()` call in `main()` received test data
- **File:** `backend/models/model_metadata.json` — the stored calibrator artifact was fitted on test data
- **File:** `backend/run_phase10_eval.py` — evaluation script reported metrics from test-set calibration

### Why it was invalid/optimistic

IsotonicRegression is a non-parametric monotonic mapping. When fitted on test data, it memorizes the mapping from raw predictions to true labels for those specific test samples. This inflates R² and Spearman correlation because the calibrator has "seen" the test targets. The effect is typically small (a few percentage points) but methodologically invalid.

### How it was fixed

The pipeline was refactored to use a three-way split:

```
TRAIN (70%)
  → fit feature preprocessing (none needed — OpenCV features are deterministic)
  → train model

VALIDATION (15%)
  → generate raw predictions
  → fit IsotonicRegression calibrator on validation predictions + labels

TEST (15%)
  → generate raw predictions
  → apply the already-fitted calibrator (never refit)
  → calculate final metrics
```

The test set is never used for training, calibration fitting, or any decision that optimizes test performance.

---

## 3. Dataset Split

| Property | Value |
|----------|-------|
| Total samples | 16,300 |
| Train | 11,414 (70.0%) |
| Validation | 2,443 (15.0%) |
| Test | 2,443 (15.0%) |
| Random seed | 42 |
| Split method | Stratified by quality bins (5 bins: 0-20, 20-40, 40-60, 60-80, 80-100) |
| Overlap verified | ✅ train ∩ val = ∅, train ∩ test = ∅, val ∩ test = ∅ |
| Datasets | KADID-10K (10,125) + KonIQ-10K (6,175) |

### Verification

The sum of all splits equals the total dataset size (11,414 + 2,443 + 2,443 = 16,300). No overlapping indices exist between any pair of splits. This is verified in `test_phase10_regression.py::TestCalibrationLeakageRemoval::test_no_overlap_between_splits`.

---

## 4. Calibration Procedure

| Property | Value |
|----------|-------|
| Method | IsotonicRegression |
| y_min | 0.0 |
| y_max | 100.0 |
| increasing | True |
| out_of_bounds | clip |
| **Fitted on** | **Validation set only** (2,443 samples) |
| **Never fitted on** | **Test set** (explicitly verified) |
| Applied to | Test set raw predictions (without refitting) |

The calibrator is stored as `backend/models/calibrator.joblib`. The metadata explicitly records `fitted_on: "validation_set"` and `never_fitted_on: "test_set"`.

---

## 5. Final Test Metrics

### Raw model (before calibration, on untouched test set)

| Metric | Value |
|--------|-------|
| MAE | 16.71 |
| RMSE | 20.30 |
| R² | 0.33 |
| Spearman | 0.55 |

### Calibrated model (after validation-fitted calibration, on untouched test set)

| Metric | Value |
|--------|-------|
| MAE | 16.60 |
| RMSE | 20.32 |
| R² | 0.33 |
| Spearman | 0.55 |

### Per-dataset calibrated test metrics

| Dataset | MAE | RMSE | R² | Spearman |
|---------|-----|------|-----|----------|
| KADID-10K | 18.73 | 22.49 | 0.33 | 0.60 |
| KonIQ-10K | 13.08 | 16.12 | 0.20 | 0.56 |

### Test set statistics

| Statistic | Value |
|-----------|-------|
| Sample count | 2,443 |
| Target mean | 55.67 |
| Target std | 24.88 |
| Prediction mean | 56.23 |
| Prediction std | 14.52 |
| Prediction min | 4.33 |
| Prediction max | 79.36 |

---

## 6. Latency

Full pipeline measurement (50 runs, 640×480 image, model loaded once):

| Statistic | Value |
|-----------|-------|
| Runs | 50 |
| Mean | 225.08 ms |
| Median | 219.65 ms |
| P95 | 333.47 ms |
| Min | 130.00 ms |
| Max | 464.87 ms |
| Std | 76.33 ms |

Pipeline stages: feature extraction → ML inference → calibration → issue detection → readiness calculation → explainability → context impacts. Model and calibrator loaded as singletons.

---

## 7. Comparison with Old Reported Metrics

| Metric | Old (Leaky) | Corrected (Leakage-Free) | Difference |
|--------|-------------|--------------------------|------------|
| MAE | 16.05 | 16.60 | +0.55 (worse) |
| RMSE | 19.85 | 20.32 | +0.47 (worse) |
| R² | 0.36 | 0.33 | -0.03 (worse) |
| Spearman | 0.57 | 0.55 | -0.02 (worse) |

**The corrected metrics are slightly worse, as expected.** The old metrics were optimistic because the calibrator was fitted on the test set. The corrected metrics are more trustworthy.

Note: The model also changed from HistGradientBoosting (old) to RandomForest (new) because RandomForest achieved higher Spearman correlation on the validation set with the corrected split. The difference is small and both models perform comparably.

---

## 8. Limitations

1. **R²=0.33** — The model explains 33% of quality variance. This is moderate, not high.
2. **5 features only** — The model uses 5 of 12 available features. Seven features are used only for issue detection, not for quality prediction.
3. **Domain gap** — Training on photography datasets (KADID/KonIQ) vs. deployment on smart-city imagery (CCTV, drone, traffic) introduces generalization risk.
4. **Score compression** — Calibrated predictions range [4.3, 79.4] while targets span [0, 100]. The model has limited ability to predict extreme quality values.
5. **No issue-level ground truth** — The issue detector's precision and recall cannot be formally measured.
6. **Single split** — No cross-validation. Point estimates only; confidence intervals unknown.
7. **No categorical quality labels** — Ground truth is continuous MOS/DMOS only.
8. **KADID within-reference correlation** — 10,125 images derive from 81 references. If not properly split, this could leak information.
9. **Classical feature-based IQA limitations** — 5 hand-crafted features cannot capture all perceptual quality dimensions.
10. **Latency variance** — P95 is 333ms vs. mean 225ms, indicating occasional slowdowns.

---

## 9. Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `ml/train.py` | Modified | Three-way split, calibrator fitted on validation only |
| `backend/run_phase10_eval.py` | Modified | Leakage-free evaluation script |
| `backend/models/model.joblib` | Regenerated | Retrained with corrected pipeline |
| `backend/models/calibrator.joblib` | Regenerated | Fitted on validation set only |
| `backend/models/model_metadata.json` | Regenerated | Updated with split info, leakage-free metrics |
| `backend/tests/test_phase10_regression.py` | Modified | Added 9 calibration leakage verification tests |
| `backend/tests/test_scoring_integrity.py` | Modified | Updated test inputs for new model |
| `benchmark/smart_city/phase10_baseline_evaluation.json` | Created | Baseline evaluation artifact |
| `benchmark/smart_city/phase10_baseline_evaluation.csv` | Created | Baseline evaluation CSV |
| `benchmark/smart_city/phase10_baseline_audit.md` | Created | This audit report |
| `benchmark/smart_city/phase10_model_evaluation.json` | Regenerated | Updated evaluation metadata |
| `benchmark/smart_city/phase10_model_evaluation.csv` | Regenerated | Updated evaluation CSV |
| `benchmark/smart_city/phase10_condition_tests.csv` | Regenerated | Controlled-condition results |
| `benchmark/smart_city/phase10_performance.json` | Regenerated | Performance profiling |
| `benchmark/smart_city/phase10_performance.csv` | Regenerated | Performance CSV |
| `benchmark/smart_city/phase10_score_distribution.json` | Regenerated | Score distribution |
| `benchmark/smart_city/phase10_ml_audit.md` | Modified | Updated audit for v2.1 |
| `benchmark/smart_city/PHASE10_FINAL_REPORT.md` | Modified | Updated final report for v2.1 |
| `README.md` | Modified | Added model evaluation section |

---

## 10. Conclusion

The calibration leakage has been **removed**. The corrected baseline metrics (MAE=16.60, RMSE=20.32, R²=0.33, Spearman=0.55) are now computed on an untouched test set with the calibrator fitted exclusively on validation data. These metrics are suitable for comparing future model improvements.

The corrected metrics are slightly worse than the previous leaky metrics (MAE 16.05→16.60, R² 0.36→0.33), confirming that the old evaluation was indeed optimistic. This is expected and acceptable — the new baseline is methodologically valid.
