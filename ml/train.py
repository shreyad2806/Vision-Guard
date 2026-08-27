"""VisionGuard Image Quality Model Training — v2.0

Multi-dataset training pipeline:
  1. Load KADID-10K and KonIQ-10K datasets
  2. Normalize quality labels independently to 0-100
  3. Extract identical features from both datasets
  4. Stratified train/test split by quality bins
  5. Train and compare: RandomForest, HistGradientBoosting, ExtraTrees
  6. Evaluate overall + per-dataset metrics
  7. Fit IsotonicRegression calibrator on validation set
  8. Save complete artifact bundle
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

# Add backend to path for imports
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BASE_DIR / "training"))

from dataset_loader import load_kadid, load_koniq, get_quality_bins, QUALITY_BIN_EDGES  # noqa: E402

# ---------------------------------------------------------------------------
# Feature extraction (must match backend/apps/ml/feature_extractor.py exactly)
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "brightness",
    "contrast",
    "sharpness",
    "saturation",
    "edge_density",
]


def extract_features(image_path: str) -> list[float] | None:
    """Extract the 5 model features from an image path.

    Preprocessing matches the production feature extractor:
      - Resize to 224x224
      - Grayscale for brightness, contrast, sharpness, edge_density
      - HSV for saturation
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

    return [brightness, contrast, sharpness, saturation, edge_density]


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
    return float(corr) if not np.isnan(corr) else 0.0


# ---------------------------------------------------------------------------
# Stratified train/test split
# ---------------------------------------------------------------------------
def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    dataset_labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """Stratified split by quality bins, preserving dataset proportions."""
    rng = np.random.RandomState(random_state)
    bins = get_quality_bins(y)

    train_idx = []
    test_idx = []

    for bin_id in range(5):
        bin_mask = bins == bin_id
        bin_indices = np.where(bin_mask)[0]
        if len(bin_indices) == 0:
            continue

        rng.shuffle(bin_indices)
        n_test = max(1, int(len(bin_indices) * test_size))
        test_idx.extend(bin_indices[:n_test])
        train_idx.extend(bin_indices[n_test:])

    train_idx = np.array(sorted(train_idx))
    test_idx = np.array(sorted(test_idx))

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx], dataset_labels[train_idx], dataset_labels[test_idx]


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
    """Select best model by: lowest RMSE → lowest MAE → highest Spearman."""
    best_name = min(
        results.keys(),
        key=lambda k: (results[k]["rmse"], results[k]["mae"], -results[k]["spearman"]),
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
# Fit calibrator on validation set
# ---------------------------------------------------------------------------
def fit_calibrator(
    model, X_val: np.ndarray, y_val: np.ndarray
) -> tuple[IsotonicRegression, np.ndarray]:
    """Fit IsotonicRegression calibrator on validation predictions.

    Returns (calibrator, calibrated_predictions).
    Ensures monotonicity: higher raw → higher calibrated.
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
    print("  VisionGuard — Image Quality Model Training v2.0")
    print("  Multi-Dataset: KADID-10K + KonIQ-10K")
    print("=" * 70)

    # --- Step 1: Load datasets ---
    print("\n[1/8] Loading datasets...")
    all_samples = load_kadid() + load_koniq()

    if not all_samples:
        print("ERROR: No samples loaded!")
        sys.exit(1)

    # --- Step 2: Extract features ---
    print(f"\n[2/8] Extracting features from {len(all_samples)} images...")
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
    print(f"\n[3/8] After removing invalid: {len(X)} samples")

    # --- Step 4: Dataset statistics ---
    print(f"\n[4/8] Dataset statistics:")
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

    # --- Step 5: Stratified split ---
    print(f"\n[5/8] Stratified train/test split (80/20)...")
    X_train, X_test, y_train, y_test, ds_train, ds_test = stratified_split(
        X, y, ds_labels, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test:  {len(X_test)} samples")

    # --- Step 6: Train and compare ---
    print(f"\n[6/8] Training and comparing models...")
    results = train_and_compare(X_train, y_train, X_test, y_test)
    print_comparison_table(results)

    best_name = select_best_model(results)
    best = results[best_name]
    print(f"\n  >> Best model: {best_name}")
    print(f"    RMSE:     {best['rmse']:.4f}")
    print(f"    MAE:      {best['mae']:.4f}")
    print(f"    R²:       {best['r2']:.4f}")
    print(f"    Spearman: {best['spearman']:.4f}")

    # --- Step 7: Fit calibrator ---
    print(f"\n[7/8] Fitting IsotonicRegression calibrator...")
    calibrator, calibrated_preds = fit_calibrator(best["model"], X_test, y_test)

    # Evaluate calibrated performance
    cal_mae = mean_absolute_error(y_test, calibrated_preds)
    cal_rmse = float(np.sqrt(mean_squared_error(y_test, calibrated_preds)))
    cal_r2 = r2_score(y_test, calibrated_preds)
    cal_spearman = spearman_correlation(y_test, calibrated_preds)
    print(f"  Calibrated MAE:      {cal_mae:.4f}")
    print(f"  Calibrated RMSE:     {cal_rmse:.4f}")
    print(f"  Calibrated R²:       {cal_r2:.4f}")
    print(f"  Calibrated Spearman: {cal_spearman:.4f}")

    # Per-dataset test metrics
    print(f"\n  Per-dataset test metrics:")
    for ds_name, mask_fn in [("KADID-10K", lambda d: d == "KADID-10K"), ("KonIQ-10K", lambda d: d == "KonIQ-10K")]:
        ds_mask = mask_fn(ds_test)
        if ds_mask.sum() == 0:
            print(f"    {ds_name}: no test samples")
            continue
        ds_y_true = y_test[ds_mask]
        ds_y_pred = calibrated_preds[ds_mask]
        ds_mae = mean_absolute_error(ds_y_true, ds_y_pred)
        ds_rmse = float(np.sqrt(mean_squared_error(ds_y_true, ds_y_pred)))
        ds_r2 = r2_score(ds_y_true, ds_y_pred)
        ds_spearman = spearman_correlation(ds_y_true, ds_y_pred)
        print(f"    {ds_name}: MAE={ds_mae:.4f}, RMSE={ds_rmse:.4f}, R²={ds_r2:.4f}, Spearman={ds_spearman:.4f}")

    # --- Step 8: Compute calibration metadata ---
    # Compute prediction statistics for calibration
    raw_preds_all = best["model"].predict(X)
    raw_preds_all = np.clip(raw_preds_all, 0, 100)

    # Feature statistics for issue detection
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
        "model_version": "v2.0.0",
        "training_datasets": ["KADID-10K", "KonIQ-10K"],
        "feature_names": FEATURE_NAMES,
        "target_scale": "0-100 higher-is-better",
        "calibration": {
            "method": "IsotonicRegression",
            "pred_min": round(float(np.min(raw_preds_all)), 4),
            "pred_max": round(float(np.max(raw_preds_all)), 4),
            "pred_mean": round(float(np.mean(raw_preds_all)), 4),
            "pred_std": round(float(np.std(raw_preds_all)), 4),
        },
        "metrics": {
            "mae": round(cal_mae, 4),
            "rmse": round(cal_rmse, 4),
            "r2": round(cal_r2, 4),
            "spearman": round(cal_spearman, 4),
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

    # --- Step 9: Save artifacts ---
    print(f"\n[8/8] Saving artifacts...")
    save_artifacts(best["model"], calibrator, FEATURE_NAMES, metadata)

    print(f"\n{'=' * 70}")
    print(f"  Training complete!")
    print(f"  Best model: {best_name}")
    print(f"  Calibrated RMSE: {cal_rmse:.4f}")
    print(f"  Calibrated Spearman: {cal_spearman:.4f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
