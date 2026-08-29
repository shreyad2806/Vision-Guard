"""VisionGuard Image Quality Model Training — v2.1 (Leakage-Free)

Multi-dataset training pipeline:
  1. Load KADID-10K and KonIQ-10K datasets
  2. Normalize quality labels independently to 0-100
  3. Extract identical features from both datasets
  4. Stratified train/validation/test split (70/15/15)
  5. Train and compare: RandomForest, HistGradientBoosting, ExtraTrees
  6. Evaluate on validation set
  7. Fit IsotonicRegression calibrator on VALIDATION set only
  8. Final evaluation on untouched TEST set (raw + calibrated)
  9. Save complete artifact bundle

v2.1 fixes:
  - Calibration leakage removed: IsotonicRegression is fitted on
    validation data, never on test data.
  - Three-way split: 70% train / 15% validation / 15% test.
  - Test set is never seen by model training OR calibration fitting.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import joblib
import numpy as np
from scipy import stats as scipy_stats
from sklearn.ensemble import (
    ExtraTreesRegressor,
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
BACKEND_MODELS.mkdir(parents=True, exist_ok=True)

from ml.training.dataset_loader import load_kadid, load_koniq, get_quality_bins, QUALITY_BIN_EDGES

# ---------------------------------------------------------------------------
# Feature extraction (must match backend/apps/ml/feature_extractor.py exactly)
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "brightness", "contrast", "sharpness", "saturation", "edge_density",
    "noise_estimate", "entropy", "colorfulness",
    "luminance_p10", "luminance_p90", "dark_pixel_pct", "bright_pixel_pct",
    "luminance_median", "gradient_magnitude", "dynamic_range", "percentile_range",
    "noise_to_signal", "channel_std", "hue_entropy", "texture_complexity",
]


def extract_features(image_path: str) -> list[float] | None:
    """Extract the 20 model features from an image path.

    Preprocessing matches the production feature extractor exactly.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    resized = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    b_ch, g_ch, r_ch = cv2.split(resized.astype(float))

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    saturation = float(np.mean(hsv[:, :, 1]))
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges > 0))
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    noise_estimate = float(np.median(np.abs(laplacian)) * 1.4826)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = float(abs(-np.sum(hist * np.log2(hist))))
    rg = np.abs(r_ch - g_ch)
    yb = np.abs(0.5 * (r_ch + g_ch) - b_ch)
    colorfulness = float(np.sqrt(np.std(rg)**2 + np.std(yb)**2) + 0.3 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2))
    luminance_p10 = float(np.percentile(gray, 10))
    luminance_p90 = float(np.percentile(gray, 90))
    dark_pixel_pct = float(np.mean(gray < 30) * 100)
    bright_pixel_pct = float(np.mean(gray > 225) * 100)
    luminance_median = float(np.median(gray))
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = float(np.mean(np.sqrt(gx**2 + gy**2)))
    dynamic_range = float(np.percentile(gray, 95) - np.percentile(gray, 5))
    percentile_range = float(luminance_p90 - luminance_p10)
    noise_to_signal = noise_estimate / max(brightness, 1.0)
    channel_std = float(np.std([np.mean(r_ch), np.mean(g_ch), np.mean(b_ch)]))
    hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()
    hue_hist = hue_hist / hue_hist.sum()
    hue_hist = hue_hist[hue_hist > 0]
    hue_entropy = float(abs(-np.sum(hue_hist * np.log2(hue_hist))))
    f = np.fft.fft2(gray.astype(float))
    fshift = np.fft.fftshift(f)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    radius = min(h, w) * 0.2
    fft_mask = ((Y - cy)**2 + (X - cx)**2) > radius**2
    mag = np.abs(fshift)
    total_energy = np.sum(mag**2)
    texture_complexity = float(np.sum(mag[fft_mask]**2) / total_energy * 100) if total_energy > 0 else 0.0

    return [brightness, contrast, sharpness, saturation, edge_density,
            noise_estimate, entropy, colorfulness, luminance_p10, luminance_p90,
            dark_pixel_pct, bright_pixel_pct, luminance_median, gradient_magnitude,
            dynamic_range, percentile_range, noise_to_signal, channel_std,
            hue_entropy, texture_complexity]


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------
def print_bin_distribution(scores: np.ndarray, name: str) -> None:
    """Print quality bin distribution."""
    print(f"  {name} quality distribution:")
    for i in range(len(QUALITY_BIN_EDGES) - 1):
        lo = QUALITY_BIN_EDGES[i]
        hi = QUALITY_BIN_EDGES[i + 1]
        if i == 0:
            count = int(np.sum((scores >= lo) & (scores < hi)))
        elif i == len(QUALITY_BIN_EDGES) - 2:
            count = int(np.sum(scores >= lo))
        else:
            count = int(np.sum((scores >= lo) & (scores < hi)))
        pct = count / len(scores) * 100 if len(scores) > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"    [{lo:3d}-{hi:3d}): {count:5d} ({pct:5.1f}%) {bar}")
    print(f"    Total: {len(scores)}")


def spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Spearman rank correlation."""
    if len(y_true) < 2:
        return 0.0
    corr, _ = scipy_stats.spearmanr(y_true, y_pred)
    corr_val = float(corr)
    return 0.0 if np.isnan(corr_val) else corr_val


# ---------------------------------------------------------------------------
# Stratified train/validation/test split (70/15/15)
# ---------------------------------------------------------------------------
def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    dataset_labels: np.ndarray,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified 3-way split by quality bins.

    Returns: X_train, X_val, X_test, y_train, y_val, y_test,
             ds_train, ds_val, ds_test, idx_train, idx_val, idx_test
    """
    rng = np.random.RandomState(random_state)
    bins = get_quality_bins(y)

    train_idx = []
    val_idx = []
    test_idx = []

    for bin_id in range(5):
        bin_mask = bins == bin_id
        bin_indices = np.where(bin_mask)[0]
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
            dataset_labels[train_idx], dataset_labels[val_idx], dataset_labels[test_idx],
            train_idx, val_idx, test_idx)


# ---------------------------------------------------------------------------
# Model training and comparison
# ---------------------------------------------------------------------------
def train_and_compare(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """Train multiple regressors and compare on validation set."""
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300,
            max_depth=8,
            learning_rate=0.1,
            min_samples_leaf=10,
            l2_regularization=0.1,
            random_state=42,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\n  Training {name}...")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        print(f"    Training time: {train_time:.1f}s")

        y_val_pred = model.predict(X_val)
        y_val_pred = np.clip(y_val_pred, 0, 100)

        mae = mean_absolute_error(y_val, y_val_pred)
        rmse = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))
        r2 = r2_score(y_val, y_val_pred)
        spearman = spearman_correlation(y_val, y_val_pred)

        print(f"    Val MAE:      {mae:.4f}")
        print(f"    Val RMSE:     {rmse:.4f}")
        print(f"    Val R²:       {r2:.4f}")
        print(f"    Val Spearman: {spearman:.4f}")

        results[name] = {
            "model": model,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "spearman": spearman,
            "train_time": train_time,
            "val_pred": y_val_pred,
        }

    return results


def select_best_model(results: dict) -> str:
    """Select best model by: highest Spearman → lowest RMSE → lowest MAE.

    Prefer models with better rank correlation, as this reflects the
    model's ability to correctly order samples — more important for
    a quality scoring system than raw RMSE.
    """
    best_name = max(
        results.keys(),
        key=lambda k: (results[k]["spearman"], -results[k]["rmse"], -results[k]["mae"]),
    )
    return best_name


def print_comparison_table(results: dict) -> None:
    """Print a formatted comparison table."""
    print(f"\n{'Model':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'Spearman':>10} {'Time':>8}")
    print("-" * 75)
    for name, r in results.items():
        print(
            f"{name:<25} {r['mae']:>8.4f} {r['rmse']:>8.4f} "
            f"{r['r2']:>8.4f} {r['spearman']:>10.4f} {r['train_time']:>7.1f}s"
        )


# ---------------------------------------------------------------------------
# Fit calibrator on VALIDATION set (NOT test set)
# ---------------------------------------------------------------------------
def fit_calibrator(
    model, X_val: np.ndarray, y_val: np.ndarray
) -> tuple[IsotonicRegression, np.ndarray]:
    """Fit IsotonicRegression calibrator on VALIDATION predictions.

    Returns (calibrator, calibrated_predictions on validation).
    This is the corrected version: calibrator is NEVER fitted on test data.
    """
    raw_preds = model.predict(X_val)
    raw_preds = np.clip(raw_preds, 0, 100)

    # Fit isotonic regression: maps raw prediction → calibrated score
    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=100.0,
        increasing=True,
        out_of_bounds="clip",
    )
    calibrator.fit(raw_preds, y_val)
    calibrated = calibrator.predict(raw_preds)

    return calibrator, calibrated


# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------
def save_artifacts(
    model,
    calibrator: IsotonicRegression,
    feature_names: list[str],
    metadata: dict,
) -> None:
    """Save model, calibrator, and metadata to backend/models/."""
    # Save model
    model_path = BACKEND_MODELS / "model.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
        },
        model_path,
    )

    # Save calibrator
    calibrator_path = BACKEND_MODELS / "calibrator.joblib"
    joblib.dump(calibrator, calibrator_path)

    # Save metadata
    metadata_path = BACKEND_MODELS / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Artifacts saved:")
    print(f"    {model_path}")
    print(f"    {calibrator_path}")
    print(f"    {metadata_path}")


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  VisionGuard — Image Quality Model Training v2.1 (Leakage-Free)")
    print("  Multi-Dataset: KADID-10K + KonIQ-10K")
    print("  Split: 70% train / 15% validation / 15% test")
    print("=" * 70)

    # --- Step 1: Load datasets ---
    print("\n[1/9] Loading datasets...")
    all_samples = load_kadid() + load_koniq()

    if not all_samples:
        print("ERROR: No samples loaded!")
        sys.exit(1)

    # --- Step 2: Extract features ---
    print(f"\n[2/9] Extracting features from {len(all_samples)} images...")
    X_list = []
    y_list = []
    dataset_names = []
    skipped = 0

    for i, sample in enumerate(all_samples):
        if (i + 1) % 1000 == 0:
            print(f"  Progress: {i + 1}/{len(all_samples)} ({skipped} skipped)")

        features = extract_features(sample["image_path"])
        if features is None:
            skipped += 1
            continue

        X_list.append(features)
        y_list.append(sample["normalized_quality_score"])
        dataset_names.append(sample["dataset_name"])

    print(f"  Extracted: {len(X_list)} valid samples, {skipped} skipped")

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.float64)
    ds_labels = np.array(dataset_names)

    # --- Step 3: Remove invalid samples ---
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y) | np.isinf(X).any(axis=1) | np.isinf(y))
    X = X[valid_mask]
    y = y[valid_mask]
    ds_labels = ds_labels[valid_mask]
    print(f"\n[3/9] After removing invalid: {len(X)} samples")

    # --- Step 4: Dataset statistics ---
    print(f"\n[4/9] Dataset statistics:")
    kadid_mask = ds_labels == "KADID-10K"
    koniq_mask = ds_labels == "KonIQ-10K"
    print(f"  KADID-10K:  {kadid_mask.sum()} samples")
    print(f"  KonIQ-10K:  {koniq_mask.sum()} samples")
    print(f"  Combined:   {len(X)} samples")
    print_bin_distribution(y, "Combined")
    if kadid_mask.sum() > 0:
        print_bin_distribution(y[kadid_mask], "KADID-10K")
    if koniq_mask.sum() > 0:
        print_bin_distribution(y[koniq_mask], "KonIQ-10K")

    # --- Step 5: Stratified 3-way split (70/15/15) ---
    print(f"\n[5/9] Stratified train/validation/test split (70/15/15)...")
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     ds_train, ds_val, ds_test,
     idx_train, idx_val, idx_test) = stratified_split(
        X, y, ds_labels, test_size=0.15, val_size=0.15, random_state=42
    )
    print(f"  Train:      {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:       {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

    # Check for duplicate indices across splits
    all_idx = set(idx_train) | set(idx_val) | set(idx_test)
    assert len(idx_train) + len(idx_val) + len(idx_test) == len(all_idx), \
        "Splits contain overlapping indices!"
    print(f"  No overlapping samples between splits: OK")

    # --- Step 6: Train and compare ---
    print(f"\n[6/9] Training and comparing models on TRAIN, evaluating on VALIDATION...")
    results = train_and_compare(X_train, y_train, X_val, y_val)
    print_comparison_table(results)

    best_name = select_best_model(results)
    best = results[best_name]
    print(f"\n  >> Best model: {best_name}")
    print(f"    Val RMSE:     {best['rmse']:.4f}")
    print(f"    Val MAE:      {best['mae']:.4f}")
    print(f"    Val R²:       {best['r2']:.4f}")
    print(f"    Val Spearman: {best['spearman']:.4f}")

    # --- Step 7: Fit calibrator on VALIDATION set only ---
    print(f"\n[7/9] Fitting IsotonicRegression calibrator on VALIDATION predictions...")
    calibrator, calibrated_val_preds = fit_calibrator(best["model"], X_val, y_val)

    val_mae = mean_absolute_error(y_val, calibrated_val_preds)
    val_rmse = float(np.sqrt(mean_squared_error(y_val, calibrated_val_preds)))
    val_r2 = r2_score(y_val, calibrated_val_preds)
    val_spearman = spearman_correlation(y_val, calibrated_val_preds)
    print(f"  Calibrated Validation MAE:      {val_mae:.4f}")
    print(f"  Calibrated Validation RMSE:     {val_rmse:.4f}")
    print(f"  Calibrated Validation R²:       {val_r2:.4f}")
    print(f"  Calibrated Validation Spearman: {val_spearman:.4f}")

    # --- Step 8: Final evaluation on UNTOUCHED TEST set ---
    print(f"\n[8/9] Final evaluation on UNTOUCHED TEST set...")
    raw_test_preds = np.clip(best["model"].predict(X_test), 0, 100)
    calibrated_test_preds = calibrator.predict(raw_test_preds)
    calibrated_test_preds = np.clip(calibrated_test_preds, 0, 100)

    # Raw metrics (before calibration)
    raw_mae = mean_absolute_error(y_test, raw_test_preds)
    raw_rmse = float(np.sqrt(mean_squared_error(y_test, raw_test_preds)))
    raw_r2 = r2_score(y_test, raw_test_preds)
    raw_spearman = spearman_correlation(y_test, raw_test_preds)

    # Calibrated metrics
    cal_mae = mean_absolute_error(y_test, calibrated_test_preds)
    cal_rmse = float(np.sqrt(mean_squared_error(y_test, calibrated_test_preds)))
    cal_r2 = r2_score(y_test, calibrated_test_preds)
    cal_spearman = spearman_correlation(y_test, calibrated_test_preds)

    print(f"\n  --- Raw Model (Test Set) ---")
    print(f"  MAE:      {raw_mae:.4f}")
    print(f"  RMSE:     {raw_rmse:.4f}")
    print(f"  R²:       {raw_r2:.4f}")
    print(f"  Spearman: {raw_spearman:.4f}")

    print(f"\n  --- Calibrated Model (Test Set) ---")
    print(f"  MAE:      {cal_mae:.4f}")
    print(f"  RMSE:     {cal_rmse:.4f}")
    print(f"  R²:       {cal_r2:.4f}")
    print(f"  Spearman: {cal_spearman:.4f}")

    # Per-dataset test metrics
    print(f"\n  Per-dataset calibrated test metrics:")
    for ds_name, mask_fn in [("KADID-10K", lambda d: d == "KADID-10K"), ("KonIQ-10K", lambda d: d == "KonIQ-10K")]:
        ds_mask = mask_fn(ds_test)
        if ds_mask.sum() == 0:
            print(f"    {ds_name}: no test samples")
            continue
        ds_y_true = y_test[ds_mask]
        ds_y_pred = calibrated_test_preds[ds_mask]
        ds_mae = mean_absolute_error(ds_y_true, ds_y_pred)
        ds_rmse = float(np.sqrt(mean_squared_error(ds_y_true, ds_y_pred)))
        ds_r2 = r2_score(ds_y_true, ds_y_pred)
        ds_spearman = spearman_correlation(ds_y_true, ds_y_pred)
        print(f"    {ds_name}: MAE={ds_mae:.4f}, RMSE={ds_rmse:.4f}, R²={ds_r2:.4f}, Spearman={ds_spearman:.4f}")

    # --- Step 9: Compute metadata ---
    raw_preds_all = best["model"].predict(X)
    raw_preds_all = np.clip(raw_preds_all, 0, 100)

    feature_statistics = {}
    for i, fname in enumerate(FEATURE_NAMES):
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

    metadata = {
        "model_type": best_name,
        "model_version": "v2.1.0",
        "training_datasets": ["KADID-10K", "KonIQ-10K"],
        "feature_names": FEATURE_NAMES,
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
            "raw": {
                "mae": round(raw_mae, 4),
                "rmse": round(raw_rmse, 4),
                "r2": round(raw_r2, 4),
                "spearman": round(raw_spearman, 4),
            },
            "calibrated": {
                "mae": round(cal_mae, 4),
                "rmse": round(cal_rmse, 4),
                "r2": round(cal_r2, 4),
                "spearman": round(cal_spearman, 4),
            },
            # Legacy fields for backward compatibility
            "mae": round(cal_mae, 4),
            "rmse": round(cal_rmse, 4),
            "r2": round(cal_r2, 4),
            "spearman": round(cal_spearman, 4),
        },
        "test_statistics": {
            "test_sample_count": len(X_test),
            "test_target_mean": round(float(np.mean(y_test)), 4),
            "test_target_std": round(float(np.std(y_test)), 4),
            "test_prediction_mean": round(float(np.mean(calibrated_test_preds)), 4),
            "test_prediction_std": round(float(np.std(calibrated_test_preds)), 4),
            "test_prediction_min": round(float(np.min(calibrated_test_preds)), 4),
            "test_prediction_max": round(float(np.max(calibrated_test_preds)), 4),
        },
        "dataset_samples": {
            "kadid": int(kadid_mask.sum()),
            "koniq": int(koniq_mask.sum()),
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

    # --- Save artifacts ---
    print(f"\n[9/9] Saving artifacts...")
    save_artifacts(best["model"], calibrator, FEATURE_NAMES, metadata)

    print(f"\n{'=' * 70}")
    print(f"  Training complete! (v2.1 — Leakage-Free)")
    print(f"  Best model: {best_name}")
    print(f"  Split: {len(X_train)}/{len(X_val)}/{len(X_test)} (train/val/test)")
    print(f"  Calibrator fitted on: VALIDATION set ({len(X_val)} samples)")
    print(f"  Final test metrics: MAE={cal_mae:.4f}, RMSE={cal_rmse:.4f}, R²={cal_r2:.4f}, Spearman={cal_spearman:.4f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
