"""VisionGuard Step 2 - Model Improvement Experiment

Uses ONLY train/val for all decisions. Test set is touched exactly once at the end.

Pipeline:
  1. Load datasets and extract extended features
  2. Create train/val/test split (seed=42, same as baseline)
  3. Compare models on validation (5=>8 features)
  4. Hyperparameter tuning on validation
  5. Select best model
  6. Fit calibrator on validation
  7. Final evaluation on untouched test
  8. Generate all artifacts
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from scipy import stats as scipy_stats
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
BACKEND_MODELS = BACKEND_DIR / "models"
BENCH = PROJECT_ROOT / "benchmark" / "smart_city"
BENCH.mkdir(parents=True, exist_ok=True)

from ml.training.dataset_loader import load_kadid, load_koniq, get_quality_bins, QUALITY_BIN_EDGES

# ---------------------------------------------------------------------------
# Extended feature extraction (8 features)
# ---------------------------------------------------------------------------
# The new feature set adds 3 meaningful features to the original 5:
#   noise_estimate  - MAD of Laplacian, captures noise level
#   entropy         - Shannon entropy, captures information content
#   colorfulness    - RGB channel diversity, captures color richness
#
# These are all computable from the same 224x224 grayscale/color image,
# require no extra passes, and are available during production inference.

FEATURE_NAMES_V2 = [
    "brightness",      # mean of grayscale
    "contrast",        # std of grayscale
    "sharpness",       # Laplacian variance
    "saturation",      # mean of HSV S channel
    "edge_density",    # Canny(100,200) > 0 fraction
    "noise_estimate",  # MAD of Laplacian (robust noise metric)
    "entropy",         # Shannon entropy of grayscale histogram
    "colorfulness",    # RGB channel diversity metric
]

FEATURE_NAMES_V1 = [
    "brightness",
    "contrast",
    "sharpness",
    "saturation",
    "edge_density",
]


def extract_features_v2(image_path: str) -> list[float] | None:
    """Extract 8 model features from an image path.

    Preprocessing matches production exactly:
      - Resize to 224x224
      - Grayscale for brightness, contrast, sharpness, edge_density, noise, entropy
      - HSV for saturation
      - BGR for colorfulness
      - Canny(100, 200) for edge_density
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    resized = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    saturation = float(np.mean(hsv[:, :, 1]))

    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges > 0))

    # New features
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    noise_estimate = float(np.median(np.abs(laplacian)) * 1.4826)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = float(abs(-np.sum(hist * np.log2(hist))))

    b, g, r = cv2.split(resized.astype(float))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    rg_mean, rg_std = np.mean(rg), np.std(rg)
    yb_mean, yb_std = np.mean(yb), np.std(yb)
    colorfulness = float(np.sqrt(rg_std**2 + yb_std**2) + 0.3 * np.sqrt(rg_mean**2 + yb_mean**2))

    return [brightness, contrast, sharpness, saturation, edge_density,
            noise_estimate, entropy, colorfulness]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    corr, _ = scipy_stats.spearmanr(y_true, y_pred)
    return 0.0 if np.isnan(corr) else float(corr)


def evaluate(y_true, y_pred, label=""):
    y_pred = np.clip(y_pred, 0, 100)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    sp = spearman_correlation(y_true, y_pred)
    print(f"  {label:30s}  MAE={mae:7.4f}  RMSE={rmse:7.4f}  R2={r2:7.4f}  Spearman={sp:7.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2, "spearman": sp}


def print_bin_distribution(scores, name):
    print(f"  {name} quality distribution:")
    for i in range(len(QUALITY_BIN_EDGES) - 1):
        lo, hi = QUALITY_BIN_EDGES[i], QUALITY_BIN_EDGES[i + 1]
        if i == len(QUALITY_BIN_EDGES) - 2:
            count = int(np.sum(scores >= lo))
        else:
            count = int(np.sum((scores >= lo) & (scores < hi)))
        pct = count / len(scores) * 100 if len(scores) > 0 else 0
        print(f"    [{lo:3d}-{hi:3d}): {count:5d} ({pct:5.1f}%)")
    print(f"    Total: {len(scores)}")


# ---------------------------------------------------------------------------
# Stratified split (same seed as baseline)
# ---------------------------------------------------------------------------
def stratified_split(X, y, ds_labels, test_size=0.15, val_size=0.15, random_state=42):
    rng = np.random.RandomState(random_state)
    bins = get_quality_bins(y)
    train_idx, val_idx, test_idx = [], [], []
    for bin_id in range(5):
        bin_indices = np.where(bins == bin_id)[0]
        if len(bin_indices) == 0:
            continue
        rng.shuffle(bin_indices)
        n_test = max(1, int(len(bin_indices) * test_size))
        n_val = max(1, int(len(bin_indices) * val_size))
        test_idx.extend(bin_indices[:n_test])
        val_idx.extend(bin_indices[n_test:n_test + n_val])
        train_idx.extend(bin_indices[n_test + n_val:])
    train_idx = np.array(sorted(train_idx))
    val_idx = np.array(sorted(val_idx))
    test_idx = np.array(sorted(test_idx))
    return (X[train_idx], X[val_idx], X[test_idx],
            y[train_idx], y[val_idx], y[test_idx],
            ds_labels[train_idx], ds_labels[val_idx], ds_labels[test_idx],
            train_idx, val_idx, test_idx)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("  VisionGuard Step 2 - Model Improvement Experiment")
    print("=" * 80)

    # ===== PHASE 1: Load and extract extended features =====
    print("\n[PHASE 1] Loading datasets and extracting extended features...")
    all_samples = load_kadid() + load_koniq()
    if not all_samples:
        print("ERROR: No samples loaded!")
        return

    X_list, y_list, ds_names = [], [], []
    skipped = 0
    for i, sample in enumerate(all_samples):
        if (i + 1) % 2000 == 0:
            print(f"  Progress: {i + 1}/{len(all_samples)} ({skipped} skipped)")
        features = extract_features_v2(sample["image_path"])
        if features is None:
            skipped += 1
            continue
        X_list.append(features)
        y_list.append(sample["normalized_quality_score"])
        ds_names.append(sample["dataset_name"])

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.float64)
    ds_labels = np.array(ds_names)

    # Remove invalid
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y) | np.isinf(X).any(axis=1) | np.isinf(y))
    X, y, ds_labels = X[valid_mask], y[valid_mask], ds_labels[valid_mask]
    print(f"  Total valid samples: {len(X)}")
    print(f"  Features: {len(FEATURE_NAMES_V2)} - {FEATURE_NAMES_V2}")

    # ===== PHASE 2: Split =====
    print("\n[PHASE 2] Stratified train/validation/test split (70/15/15, seed=42)...")
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     ds_train, ds_val, ds_test,
     idx_train, idx_val, idx_test) = stratified_split(X, y, ds_labels)
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    all_idx = set(idx_train) | set(idx_val) | set(idx_test)
    assert len(idx_train) + len(idx_val) + len(idx_test) == len(all_idx), "Overlap!"
    print(f"  No overlap: OK")

    # Correlation matrix for new features
    print("\n  Feature correlation matrix:")
    header = f"{'':>15s}" + "".join(f"{fn:>14s}" for fn in FEATURE_NAMES_V2)
    print(f"  {header}")
    corr_matrix = np.corrcoef(X_train.T)
    for i, fn in enumerate(FEATURE_NAMES_V2):
        row = f"{fn:>15s}" + "".join(f"{corr_matrix[i,j]:14.3f}" for j in range(len(FEATURE_NAMES_V2)))
        print(f"  {row}")

    # ===== PHASE 3: Model comparison (8 features) =====
    print("\n[PHASE 3] Model comparison on VALIDATION set (8 features)...")

    base_models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300, max_depth=8, learning_rate=0.1, min_samples_leaf=10,
            l2_regularization=0.1, random_state=42),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.1, min_samples_leaf=10,
            subsample=0.8, random_state=42),
    }

    model_results = {}
    for name, model in base_models.items():
        print(f"\n  Training {name}...")
        t0 = time.time()
        model.fit(X_train, y_train)
        t_train = time.time() - t0

        y_val_pred = np.clip(model.predict(X_val), 0, 100)
        m = evaluate(y_val, y_val_pred, f"{name} (val)")
        m["train_time"] = t_train
        m["model"] = model
        model_results[name] = m

    # ===== PHASE 4: Hyperparameter tuning on top models =====
    print("\n[PHASE 4] Hyperparameter tuning on validation set...")

    # Tune HistGradientBoosting (usually strongest on tabular data)
    print("\n  --- Tuning HistGradientBoosting ---")
    hgb_configs = [
        {"max_iter": 500, "max_depth": 6, "learning_rate": 0.05, "min_samples_leaf": 15,
         "l2_regularization": 0.5, "random_state": 42},
        {"max_iter": 500, "max_depth": 8, "learning_rate": 0.05, "min_samples_leaf": 10,
         "l2_regularization": 0.1, "random_state": 42},
        {"max_iter": 800, "max_depth": 6, "learning_rate": 0.03, "min_samples_leaf": 20,
         "l2_regularization": 1.0, "random_state": 42},
        {"max_iter": 400, "max_depth": 10, "learning_rate": 0.08, "min_samples_leaf": 8,
         "l2_regularization": 0.1, "random_state": 42},
        {"max_iter": 600, "max_depth": 5, "learning_rate": 0.05, "min_samples_leaf": 20,
         "l2_regularization": 0.5, "random_state": 42},
    ]

    best_hgb_score = (-1.0, 0.0, 0.0)
    best_hgb_model = None
    best_hgb_name = ""
    for i, cfg in enumerate(hgb_configs):
        name = f"HGB_cfg{i}"
        model = HistGradientBoostingRegressor(**cfg)
        t0 = time.time()
        model.fit(X_train, y_train)
        t_train = time.time() - t0
        y_val_pred = np.clip(model.predict(X_val), 0, 100)
        m = evaluate(y_val, y_val_pred, name)
        # Rank by Spearman => -RMSE => -MAE
        score = (m["spearman"], -m["rmse"], -m["mae"])
        if score > best_hgb_score:
            best_hgb_score = score
            best_hgb_model = model
            best_hgb_name = name
            best_hgb_cfg = cfg
    print(f"\n  Best HGB config: {best_hgb_name} => Spearman={best_hgb_score[0]:.4f}")

    # Tune RandomForest
    print("\n  --- Tuning RandomForest ---")
    rf_configs = [
        {"n_estimators": 500, "max_depth": 20, "min_samples_leaf": 5, "n_jobs": -1, "random_state": 42},
        {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 10, "n_jobs": -1, "random_state": 42},
        {"n_estimators": 500, "max_depth": 15, "min_samples_leaf": 8, "max_features": "sqrt",
         "n_jobs": -1, "random_state": 42},
    ]

    best_rf_score = (-1.0, 0.0, 0.0)
    best_rf_model = None
    for i, cfg in enumerate(rf_configs):
        name = f"RF_cfg{i}"
        model = RandomForestRegressor(**cfg)
        t0 = time.time()
        model.fit(X_train, y_train)
        t_train = time.time() - t0
        y_val_pred = np.clip(model.predict(X_val), 0, 100)
        m = evaluate(y_val, y_val_pred, name)
        score = (m["spearman"], -m["rmse"], -m["mae"])
        if score > best_rf_score:
            best_rf_score = score
            best_rf_model = model

    # ===== PHASE 5: Select best overall model =====
    print("\n[PHASE 5] Selecting best model from all candidates...")

    all_candidates = {}
    # Base models
    for name, m in model_results.items():
        all_candidates[name] = m
    # Tuned models
    m = evaluate(y_val, np.clip(best_hgb_model.predict(X_val), 0, 100), "Best Tuned HGB")
    m["model"] = best_hgb_model
    all_candidates["HistGradientBoosting_tuned"] = m
    m = evaluate(y_val, np.clip(best_rf_model.predict(X_val), 0, 100), "Best Tuned RF")
    m["model"] = best_rf_model
    all_candidates["RandomForest_tuned"] = m

    best_name = max(all_candidates.keys(),
                    key=lambda k: (all_candidates[k]["spearman"],
                                   -all_candidates[k]["rmse"],
                                   -all_candidates[k]["mae"]))
    best = all_candidates[best_name]
    print(f"\n  >> SELECTED: {best_name}")
    print(f"     Val MAE={best['mae']:.4f}  RMSE={best['rmse']:.4f}  R2={best['r2']:.4f}  Spearman={best['spearman']:.4f}")

    # Train final selected model on TRAIN+VAL for the best model
    # Actually, we should only use train for final model. But the hyperparameters
    # were selected on validation. So retrain on train only.
    final_model = best["model"].__class__(**best["model"].get_params())
    t0 = time.time()
    final_model.fit(X_train, y_train)
    t_train_final = time.time() - t0
    y_val_final = np.clip(final_model.predict(X_val), 0, 100)
    print(f"\n  Final model retrained on TRAIN only (time: {t_train_final:.1f}s):")
    final_val = evaluate(y_val, y_val_final, "Final model (val)")

    # ===== PHASE 6: Overfitting check =====
    print("\n[PHASE 6] Overfitting check...")
    y_train_pred = np.clip(final_model.predict(X_train), 0, 100)
    train_metrics = evaluate(y_train, y_train_pred, "Train")
    print(f"  Validation:")
    val_metrics = final_val
    print(f"\n  Train-Val gap:")
    for k in ["mae", "rmse", "r2", "spearman"]:
        diff = train_metrics[k] - val_metrics[k]
        print(f"    {k:10s}: train={train_metrics[k]:.4f}  val={val_metrics[k]:.4f}  gap={diff:+.4f}")

    # ===== PHASE 7: Calibration on validation =====
    print("\n[PHASE 7] Fitting calibrator on VALIDATION predictions...")
    raw_val_preds = np.clip(final_model.predict(X_val), 0, 100)
    calibrator = IsotonicRegression(y_min=0.0, y_max=100.0, increasing=True, out_of_bounds="clip")
    calibrator.fit(raw_val_preds, y_val)
    calibrated_val_preds = np.clip(calibrator.predict(raw_val_preds), 0, 100)
    print("  Raw validation:")
    raw_val_m = evaluate(y_val, raw_val_preds, "Raw (val)")
    print("  Calibrated validation:")
    cal_val_m = evaluate(y_val, calibrated_val_preds, "Calibrated (val)")

    # ===== PHASE 8: FINAL TEST EVALUATION (once, never retouched) =====
    print("\n" + "=" * 80)
    print("[PHASE 8] FINAL TEST EVALUATION - UNTOUCHED TEST SET")
    print("=" * 80)
    raw_test_preds = np.clip(final_model.predict(X_test), 0, 100)
    calibrated_test_preds = np.clip(calibrator.predict(raw_test_preds), 0, 100)

    print("\n  Raw model (test):")
    raw_test_m = evaluate(y_test, raw_test_preds, "Raw (test)")
    print("\n  Calibrated model (test):")
    cal_test_m = evaluate(y_test, calibrated_test_preds, "Calibrated (test)")

    # Per-dataset
    print("\n  Per-dataset calibrated test metrics:")
    for ds_name in ["KADID-10K", "KonIQ-10K"]:
        mask = ds_test == ds_name
        if mask.sum() == 0:
            continue
        m = evaluate(y_test[mask], calibrated_test_preds[mask], f"  {ds_name}")

    # Test statistics
    test_stats = {
        "sample_count": int(len(X_test)),
        "target_mean": round(float(np.mean(y_test)), 4),
        "target_std": round(float(np.std(y_test)), 4),
        "prediction_mean": round(float(np.mean(calibrated_test_preds)), 4),
        "prediction_std": round(float(np.std(calibrated_test_preds)), 4),
        "prediction_min": round(float(np.min(calibrated_test_preds)), 4),
        "prediction_max": round(float(np.max(calibrated_test_preds)), 4),
        "target_min": round(float(np.min(y_test)), 4),
        "target_max": round(float(np.max(y_test)), 4),
    }

    # ===== Comparison with baseline =====
    print("\n" + "=" * 80)
    print("  COMPARISON WITH BASELINE")
    print("=" * 80)
    baseline = {"mae": 16.5986, "rmse": 20.3239, "r2": 0.3327, "spearman": 0.5464}
    print(f"\n  {'Metric':<12} {'Baseline':>10} {'Improved':>10} {'Change':>10} {'%':>8}")
    print(f"  {'-'*52}")
    for k in ["mae", "rmse", "r2", "spearman"]:
        old = baseline[k]
        new = cal_test_m[k]
        diff = new - old
        if k in ["mae", "rmse"]:
            pct = ((old - new) / old) * 100  # improvement is negative diff
        else:
            pct = ((new - old) / abs(old)) * 100 if old != 0 else 0
        better = "*" if (diff < 0 and k in ["mae", "rmse"]) or (diff > 0 and k in ["r2", "spearman"]) else ""
        print(f"  {k:<12} {old:10.4f} {new:10.4f} {diff:+10.4f} {pct:+7.2f}% {better}")

    # ===== Feature importance =====
    print("\n[FEATURE IMPORTANCE]")
    if hasattr(final_model, "feature_importances_"):
        importances = final_model.feature_importances_
    elif hasattr(final_model, "estimators_"):
        # For HistGradientBoosting, use permutation-like approach
        importances = np.zeros(len(FEATURE_NAMES_V2))
        # Use built-in feature_importances_ if available
        if hasattr(final_model, "feature_importances_"):
            importances = final_model.feature_importances_
        else:
            # Manual permutation importance on validation
            base_score = cal_test_m["spearman"]
            for fi in range(len(FEATURE_NAMES_V2)):
                X_test_perm = X_test.copy()
                rng = np.random.RandomState(42)
                rng.shuffle(X_test_perm[:, fi])
                perm_pred = np.clip(final_model.predict(X_test_perm), 0, 100)
                perm_cal = np.clip(calibrator.predict(perm_pred), 0, 100)
                perm_score = spearman_correlation(y_test, perm_cal)
                importances[fi] = max(0, base_score - perm_score)
    else:
        importances = np.zeros(len(FEATURE_NAMES_V2))

    # Sort by importance
    sorted_idx = np.argsort(importances)[::-1]
    print(f"\n  {'Rank':<5} {'Feature':<20} {'Importance':>12}")
    print(f"  {'-'*40}")
    feat_imp_list = []
    for rank, idx in enumerate(sorted_idx, 1):
        print(f"  {rank:<5} {FEATURE_NAMES_V2[idx]:<20} {importances[idx]:12.6f}")
        feat_imp_list.append({"feature": FEATURE_NAMES_V2[idx], "importance": round(float(importances[idx]), 6), "rank": rank})

    # ===== Latency =====
    print("\n[LATENCY MEASUREMENT]")
    # Create a synthetic test image
    img_test = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img_test, (0, 0), (640, 480), (120, 140, 160), -1)
    cv2.putText(img_test, "Test", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (200, 220, 240), 3)

    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        resized = cv2.resize(img_test, (224, 224))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        saturation = float(np.mean(hsv[:, :, 1]))
        edges = cv2.Canny(gray, 100, 200)
        edge_density = float(np.mean(edges > 0))
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_estimate = float(np.median(np.abs(laplacian)) * 1.4826)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        entropy_val = float(abs(-np.sum(hist * np.log2(hist))))
        b_ch, g_ch, r_ch = cv2.split(resized.astype(float))
        rg = np.abs(r_ch - g_ch)
        yb = np.abs(0.5 * (r_ch + g_ch) - b_ch)
        colorfulness = float(np.sqrt(np.std(rg)**2 + np.std(yb)**2) + 0.3 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2))

        vector = np.array([[brightness, contrast, sharpness, saturation, edge_density,
                           noise_estimate, entropy_val, colorfulness]])
        raw = float(final_model.predict(vector)[0])
        cal = float(calibrator.predict(np.array([[raw]]))[0])
        latencies.append((time.perf_counter() - t0) * 1000)

    lat_arr = np.array(latencies)
    latency_stats = {
        "runs": 50,
        "mean_ms": round(float(np.mean(lat_arr)), 2),
        "median_ms": round(float(np.median(lat_arr)), 2),
        "p95_ms": round(float(np.percentile(lat_arr, 95)), 2),
        "min_ms": round(float(np.min(lat_arr)), 2),
        "max_ms": round(float(np.max(lat_arr)), 2),
    }
    print(f"  Runs: {latency_stats['runs']}")
    print(f"  Mean: {latency_stats['mean_ms']}ms")
    print(f"  Median: {latency_stats['median_ms']}ms")
    print(f"  P95: {latency_stats['p95_ms']}ms")

    # ===== Save model artifacts =====
    print("\n[SAVING ARTIFACTS]")
    import joblib
    model_path = BACKEND_MODELS / "model.joblib"
    joblib.dump({"model": final_model, "feature_names": FEATURE_NAMES_V2}, model_path)
    calibrator_path = BACKEND_MODELS / "calibrator.joblib"
    joblib.dump(calibrator, calibrator_path)

    feature_statistics = {}
    for i, fname in enumerate(FEATURE_NAMES_V2):
        col = X[:, i]
        feature_statistics[fname] = {
            "min": round(float(np.min(col)), 4),
            "max": round(float(np.max(col)), 4),
            "mean": round(float(np.mean(col)), 4),
            "std": round(float(np.std(col)), 4),
            "p5": round(float(np.percentile(col, 5)), 4),
            "p10": round(float(np.percentile(col, 10)), 4),
            "p25": round(float(np.percentile(col, 25)), 4),
            "p50": round(float(np.percentile(col, 50)), 4),
            "p75": round(float(np.percentile(col, 75)), 4),
            "p90": round(float(np.percentile(col, 90)), 4),
            "p95": round(float(np.percentile(col, 95)), 4),
        }

    raw_preds_all = np.clip(final_model.predict(X), 0, 100)

    metadata = {
        "model_type": best_name,
        "model_version": "v3.0.0",
        "training_datasets": ["KADID-10K", "KonIQ-10K"],
        "feature_names": FEATURE_NAMES_V2,
        "feature_count": len(FEATURE_NAMES_V2),
        "target_scale": "0-100 higher-is-better",
        "split": {
            "method": "stratified_by_quality_bins",
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "random_seed": 42,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
        },
        "calibration": {
            "method": "IsotonicRegression",
            "fitted_on": "validation_set",
            "never_fitted_on": "test_set",
            "y_min": 0.0,
            "y_max": 100.0,
            "increasing": True,
            "pred_min": round(float(np.min(raw_preds_all)), 4),
            "pred_max": round(float(np.max(raw_preds_all)), 4),
            "pred_mean": round(float(np.mean(raw_preds_all)), 4),
            "pred_std": round(float(np.std(raw_preds_all)), 4),
        },
        "metrics": {
            "note": "Computed on untouched test set. Calibrator fitted on validation data only.",
            "raw": {k: round(v, 4) for k, v in raw_test_m.items()},
            "calibrated": {k: round(v, 4) for k, v in cal_test_m.items()},
            "mae": round(cal_test_m["mae"], 4),
            "rmse": round(cal_test_m["rmse"], 4),
            "r2": round(cal_test_m["r2"], 4),
            "spearman": round(cal_test_m["spearman"], 4),
        },
        "test_statistics": test_stats,
        "dataset_samples": {
            "kadid": int((ds_labels == "KADID-10K").sum()),
            "koniq": int((ds_labels == "KonIQ-10K").sum()),
            "combined": len(X),
        },
        "feature_statistics": feature_statistics,
        "quality_bins": {
            "edges": QUALITY_BIN_EDGES,
            "train_distribution": {
                str(QUALITY_BIN_EDGES[i]): int(np.sum((y_train >= QUALITY_BIN_EDGES[i]) & (y_train < QUALITY_BIN_EDGES[i + 1])))
                for i in range(len(QUALITY_BIN_EDGES) - 1)
            },
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    metadata_path = BACKEND_MODELS / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {model_path}")
    print(f"  Saved: {calibrator_path}")
    print(f"  Saved: {metadata_path}")

    # ===== Save comparison CSV =====
    comparison_rows = []
    for name, m in all_candidates.items():
        comparison_rows.append({
            "model": name,
            "mae": round(m["mae"], 4),
            "rmse": round(m["rmse"], 4),
            "r2": round(m["r2"], 4),
            "spearman": round(m["spearman"], 4),
            "train_time_s": round(m.get("train_time", 0), 2),
        })
    comparison_csv = BENCH / "phase10_model_comparison.csv"
    with open(comparison_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "mae", "rmse", "r2", "spearman", "train_time_s"])
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(f"  Saved: {comparison_csv}")

    # ===== Save feature importance =====
    feat_imp_path = BENCH / "phase10_feature_importance.json"
    with open(feat_imp_path, "w") as f:
        json.dump({"features": feat_imp_list, "method": "tree_feature_importances" if hasattr(final_model, "feature_importances_") else "permutation_importance"}, f, indent=2)
    print(f"  Saved: {feat_imp_path}")

    # ===== Save improvement JSON =====
    improvement = {
        "evaluation_type": "model_improvement",
        "baseline_version": "v2.1.0",
        "improved_version": "v3.0.0",
        "baseline_metrics": baseline,
        "improved_raw_metrics": {k: round(v, 4) for k, v in raw_test_m.items()},
        "improved_calibrated_metrics": {k: round(v, 4) for k, v in cal_test_m.items()},
        "improvement_pct": {
            k: round(((baseline[k] - cal_test_m[k]) / abs(baseline[k])) * 100, 2)
            for k in ["mae", "rmse"]
        } | {
            k: round(((cal_test_m[k] - baseline[k]) / abs(baseline[k])) * 100, 2)
            for k in ["r2", "spearman"]
        },
        "selected_model": best_name,
        "selected_features": FEATURE_NAMES_V2,
        "feature_count": len(FEATURE_NAMES_V2),
        "hyperparameters": {k: v for k, v in final_model.get_params().items()},
        "train_val_gap": {k: round(train_metrics[k] - val_metrics[k], 4) for k in ["mae", "rmse", "r2", "spearman"]},
        "latency": latency_stats,
        "feature_importance": feat_imp_list,
        "test_statistics": test_stats,
        "overfitting_analysis": {
            "train_mae": round(train_metrics["mae"], 4),
            "val_mae": round(val_metrics["mae"], 4),
            "test_mae": round(cal_test_m["mae"], 4),
            "assessment": "Moderate gap" if abs(train_metrics["mae"] - val_metrics["mae"]) < 3 else "Large gap - possible overfitting",
        },
    }
    improvement_path = BENCH / "phase10_model_improvement.json"
    with open(improvement_path, "w") as f:
        json.dump(improvement, f, indent=2)
    print(f"  Saved: {improvement_path}")

    print("\n" + "=" * 80)
    print("  EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
