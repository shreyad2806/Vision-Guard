"""VisionGuard Feature Engineering Experiment

Starting from the current 8-feature model (v3.0), this experiment:
1. Audits current features
2. Adds candidate features
3. Checks redundancy
4. Runs ablation on train+val only
5. Selects best feature set
6. Evaluates on locked test set
7. Generates all artifacts
"""

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import joblib
import numpy as np
from scipy import stats as scipy_stats
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
BACKEND_MODELS = BACKEND_DIR / "models"
BENCH = PROJECT_ROOT / "benchmark" / "smart_city"
BENCH.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BACKEND_DIR))
from ml.training.dataset_loader import load_kadid, load_koniq, get_quality_bins, QUALITY_BIN_EDGES


# ---------------------------------------------------------------------------
# Current features (v3.0 baseline): 8
# ---------------------------------------------------------------------------
CURRENT_FEATURES = [
    "brightness", "contrast", "sharpness", "saturation", "edge_density",
    "noise_estimate", "entropy", "colorfulness",
]

# ---------------------------------------------------------------------------
# Candidate new features
# ---------------------------------------------------------------------------
CANDIDATE_FEATURES = {
    # Exposure
    "luminance_p10": "luminance 10th percentile",
    "luminance_p90": "luminance 90th percentile",
    "dark_pixel_pct": "fraction of very dark pixels",
    "bright_pixel_pct": "fraction of very bright pixels",
    "luminance_median": "median luminance",
    # Sharpness
    "gradient_magnitude": "mean gradient magnitude",
    # Contrast
    "dynamic_range": "p95-p5 intensity range",
    "percentile_range": "p90-p10 intensity range",
    # Noise
    "noise_to_signal": "noise estimate / mean brightness",
    # Color
    "channel_std": "std of RGB channel means",
    "hue_entropy": "entropy of hue histogram",
    # Structure
    "texture_complexity": "HF energy ratio from FFT",
}


def extract_all_candidate_features(img_bgr):
    """Extract ALL features (current + candidates) from an image."""
    resized = cv2.resize(img_bgr, (224, 224))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    # Current features
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
    b, g, r = cv2.split(resized.astype(float))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    colorfulness = float(np.sqrt(np.std(rg)**2 + np.std(yb)**2) + 0.3 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2))

    # Candidate: exposure
    luminance_p10 = float(np.percentile(gray, 10))
    luminance_p90 = float(np.percentile(gray, 90))
    dark_pixel_pct = float(np.mean(gray < 30) * 100)
    bright_pixel_pct = float(np.mean(gray > 225) * 100)
    luminance_median = float(np.median(gray))

    # Candidate: sharpness
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = float(np.mean(np.sqrt(gx**2 + gy**2)))

    # Candidate: contrast
    dynamic_range = float(np.percentile(gray, 95) - np.percentile(gray, 5))
    percentile_range = float(luminance_p90 - luminance_p10)

    # Candidate: noise
    noise_to_signal = noise_estimate / max(brightness, 1.0)

    # Candidate: color
    channel_std = float(np.std([np.mean(r), np.mean(g), np.mean(b)]))
    hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()
    hue_hist = hue_hist / hue_hist.sum()
    hue_hist = hue_hist[hue_hist > 0]
    hue_entropy = float(abs(-np.sum(hue_hist * np.log2(hue_hist))))

    # Candidate: structure
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

    return {
        # Current
        "brightness": brightness, "contrast": contrast, "sharpness": sharpness,
        "saturation": saturation, "edge_density": edge_density,
        "noise_estimate": noise_estimate, "entropy": entropy, "colorfulness": colorfulness,
        # Candidates
        "luminance_p10": luminance_p10, "luminance_p90": luminance_p90,
        "dark_pixel_pct": dark_pixel_pct, "bright_pixel_pct": bright_pixel_pct,
        "luminance_median": luminance_median, "gradient_magnitude": gradient_magnitude,
        "dynamic_range": dynamic_range, "percentile_range": percentile_range,
        "noise_to_signal": noise_to_signal, "channel_std": channel_std,
        "hue_entropy": hue_entropy, "texture_complexity": texture_complexity,
    }


ALL_FEATURE_NAMES = CURRENT_FEATURES + list(CANDIDATE_FEATURES.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def spearman_corr(y_true, y_pred):
    if len(y_true) < 2:
        return 0.0
    corr, _ = scipy_stats.spearmanr(y_true, y_pred)
    return 0.0 if np.isnan(corr) else float(corr)


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
    return np.array(sorted(train_idx)), np.array(sorted(val_idx)), np.array(sorted(test_idx))


def evaluate_model(X_train, y_train, X_val, y_val, feat_idx, model_cls, model_params, name=""):
    """Train model on subset of features, evaluate on val."""
    X_tr = X_train[:, feat_idx]
    X_v = X_val[:, feat_idx]
    model = model_cls(**model_params)
    model.fit(X_tr, y_train)
    y_pred = np.clip(model.predict(X_v), 0, 100)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    r2 = r2_score(y_val, y_pred)
    sp = spearman_corr(y_val, y_pred)
    return {"name": name, "features": len(feat_idx), "mae": round(mae, 4),
            "rmse": round(rmse, 4), "r2": round(r2, 4), "spearman": round(sp, 4)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("  VisionGuard Feature Engineering Experiment")
    print("=" * 80)

    # Load data
    print("\n[1] Loading datasets and extracting features...")
    all_samples = load_kadid() + load_koniq()
    X_list, y_list, ds_names = [], [], []
    skipped = 0
    for i, sample in enumerate(all_samples):
        if (i + 1) % 2000 == 0:
            print(f"  Progress: {i + 1}/{len(all_samples)} ({skipped} skipped)")
        img = cv2.imread(sample["image_path"])
        if img is None:
            skipped += 1
            continue
        feats = extract_all_candidate_features(img)
        X_list.append([feats[k] for k in ALL_FEATURE_NAMES])
        y_list.append(sample["normalized_quality_score"])
        ds_names.append(sample["dataset_name"])

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.float64)
    ds_labels = np.array(ds_names)
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y) | np.isinf(X).any(axis=1) | np.isinf(y))
    X, y, ds_labels = X[valid], y[valid], ds_labels[valid]
    print(f"  Total: {len(X)} samples, {len(ALL_FEATURE_NAMES)} features")

    # Split
    print("\n[2] Stratified split (seed=42)...")
    idx_train, idx_val, idx_test = stratified_split(X, y, ds_labels)
    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    assert len(set(idx_train) & set(idx_val)) == 0
    assert len(set(idx_train) & set(idx_test)) == 0
    assert len(set(idx_val) & set(idx_test)) == 0
    print("  No overlap: OK")

    # Correlation matrix
    print("\n[3] Feature correlation analysis (train set)...")
    corr = np.corrcoef(X_train.T)
    print(f"\n  Current features correlation with candidates:")
    for i, fn in enumerate(ALL_FEATURE_NAMES):
        if fn in CURRENT_FEATURES:
            continue
        # Find max correlation with current features
        max_corr_with_current = 0
        max_feat = ""
        for j, cfn in enumerate(CURRENT_FEATURES):
            cidx = ALL_FEATURE_NAMES.index(cfn)
            if abs(corr[i, cidx]) > abs(max_corr_with_current):
                max_corr_with_current = corr[i, cidx]
                max_feat = cfn
        print(f"    {fn:25s} max_corr_with_current={max_corr_with_current:+.3f} ({max_feat})")

    # Ablation experiment
    print("\n[4] Ablation experiment (HistGradientBoosting, train+val only)...")
    hgb_params = {"max_iter": 400, "max_depth": 10, "learning_rate": 0.08,
                  "min_samples_leaf": 8, "l2_regularization": 0.1, "random_state": 42}

    # Feature groupings
    current_idx = list(range(len(CURRENT_FEATURES)))
    all_idx = list(range(len(ALL_FEATURE_NAMES)))

    # Candidate indices (offset by len(CURRENT_FEATURES))
    cand_idx = {name: len(CURRENT_FEATURES) + i for i, name in enumerate(CANDIDATE_FEATURES.keys())}

    ablation_configs = {
        "A_current_8": current_idx,
        "B_add_exposure": current_idx + [cand_idx["luminance_p10"], cand_idx["luminance_p90"],
                                          cand_idx["dark_pixel_pct"], cand_idx["bright_pixel_pct"]],
        "C_add_sharpness_contrast": current_idx + [cand_idx["gradient_magnitude"],
                                                    cand_idx["dynamic_range"], cand_idx["percentile_range"]],
        "D_add_noise": current_idx + [cand_idx["noise_to_signal"]],
        "E_add_color_structure": current_idx + [cand_idx["channel_std"], cand_idx["hue_entropy"],
                                                 cand_idx["texture_complexity"]],
        "F_all_candidates": all_idx,
        "G_best_subset": None,  # filled after analysis
    }

    ablation_results = []
    for name, feat_idx in ablation_configs.items():
        if feat_idx is None:
            continue
        r = evaluate_model(X_train, y_train, X_val, y_val, feat_idx,
                          HistGradientBoostingRegressor, hgb_params, name)
        ablation_results.append(r)
        print(f"  {name:30s} feats={r['features']:2d}  MAE={r['mae']:7.4f}  RMSE={r['rmse']:7.4f}  R2={r['r2']:7.4f}  Sp={r['spearman']:7.4f}")

    # Find best feature set
    best_result = max(ablation_results, key=lambda r: (r["spearman"], -r["rmse"]))
    print(f"\n  Best: {best_result['name']} (Spearman={best_result['spearman']:.4f})")

    # Now try a curated best subset: current +最有价值的candidates
    # Based on correlation analysis and ablation, pick top candidates
    # that add unique information
    print("\n[5] Testing curated best subset...")
    # Select candidates that: low correlation with current, high info gain
    best_subset_idx = current_idx.copy()
    # Add gradient_magnitude (sharpness complement), dynamic_range (contrast complement)
    # texture_complexity (unique structural info), channel_std (unique color info)
    for cand_name in ["gradient_magnitude", "dynamic_range", "texture_complexity", "channel_std"]:
        if cand_name in cand_idx:
            best_subset_idx.append(cand_idx[cand_name])

    r = evaluate_model(X_train, y_train, X_val, y_val, best_subset_idx,
                      HistGradientBoostingRegressor, hgb_params, "G_best_subset")
    ablation_results.append(r)
    print(f"  {'G_best_subset':30s} feats={r['features']:2d}  MAE={r['mae']:7.4f}  RMSE={r['rmse']:7.4f}  R2={r['r2']:7.4f}  Sp={r['spearman']:7.4f}")

    # Also test with fewer candidates
    for subset_name, subset_feats in [
        ("H_current_plus_2", current_idx + [cand_idx["gradient_magnitude"], cand_idx["dynamic_range"]]),
        ("I_current_plus_3", current_idx + [cand_idx["gradient_magnitude"], cand_idx["dynamic_range"],
                                             cand_idx["texture_complexity"]]),
    ]:
        r = evaluate_model(X_train, y_train, X_val, y_val, subset_feats,
                          HistGradientBoostingRegressor, hgb_params, subset_name)
        ablation_results.append(r)
        print(f"  {subset_name:30s} feats={r['features']:2d}  MAE={r['mae']:7.4f}  RMSE={r['rmse']:7.4f}  R2={r['r2']:7.4f}  Sp={r['spearman']:7.4f}")

    # Select final best
    final_best = max(ablation_results, key=lambda r: (r["spearman"], -r["rmse"]))
    print(f"\n  >> FINAL BEST: {final_best['name']} (Spearman={final_best['spearman']:.4f}, MAE={final_best['mae']:.4f})")

    # Map name to feature indices
    name_to_idx = {**{k: v for k, v in ablation_configs.items() if v is not None},
                   "G_best_subset": best_subset_idx,
                   "H_current_plus_2": current_idx + [cand_idx["gradient_magnitude"], cand_idx["dynamic_range"]],
                   "I_current_plus_3": current_idx + [cand_idx["gradient_magnitude"], cand_idx["dynamic_range"],
                                                       cand_idx["texture_complexity"]]}
    selected_feat_idx = name_to_idx[final_best["name"]]
    selected_feature_names = [ALL_FEATURE_NAMES[i] for i in selected_feat_idx]
    print(f"  Selected features: {selected_feature_names}")

    # Save ablation CSV
    with open(BENCH / "phase10_feature_ablation.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "features", "mae", "rmse", "r2", "spearman"])
        writer.writeheader()
        writer.writerows(ablation_results)
    print(f"\n  Saved: {BENCH / 'phase10_feature_ablation.csv'}")

    # Final test evaluation
    print("\n[6] Final TEST evaluation (locked test set, once only)...")
    X_tr_final = X_train[:, selected_feat_idx]
    X_v_final = X_val[:, selected_feat_idx]
    X_te_final = X_test[:, selected_feat_idx]

    final_model = HistGradientBoostingRegressor(**hgb_params)
    final_model.fit(X_tr_final, y_train)

    # Calibrate on val
    raw_val = np.clip(final_model.predict(X_v_final), 0, 100)
    calibrator = IsotonicRegression(y_min=0.0, y_max=100.0, increasing=True, out_of_bounds="clip")
    calibrator.fit(raw_val, y_val)

    # Test
    raw_test = np.clip(final_model.predict(X_te_final), 0, 100)
    cal_test = np.clip(calibrator.predict(raw_test), 0, 100)

    raw_mae = mean_absolute_error(y_test, raw_test)
    raw_rmse = float(np.sqrt(mean_squared_error(y_test, raw_test)))
    raw_r2 = r2_score(y_test, raw_test)
    raw_sp = spearman_corr(y_test, raw_test)
    cal_mae = mean_absolute_error(y_test, cal_test)
    cal_rmse = float(np.sqrt(mean_squared_error(y_test, cal_test)))
    cal_r2 = r2_score(y_test, cal_test)
    cal_sp = spearman_corr(y_test, cal_test)

    print(f"\n  Raw model (test):  MAE={raw_mae:.4f}  RMSE={raw_rmse:.4f}  R2={raw_r2:.4f}  Sp={raw_sp:.4f}")
    print(f"  Cal model (test):  MAE={cal_mae:.4f}  RMSE={cal_rmse:.4f}  R2={cal_r2:.4f}  Sp={cal_sp:.4f}")

    # Compare with baseline
    baseline = {"mae": 16.60, "rmse": 20.32, "r2": 0.33, "spearman": 0.55}
    print(f"\n  vs Baseline (5 features):")
    for k in ["mae", "rmse", "r2", "spearman"]:
        old = baseline[k]
        new = cal_mae if k == "mae" else cal_rmse if k == "rmse" else cal_r2 if k == "r2" else cal_sp
        diff = ((old - new) / abs(old)) * 100 if k in ["mae", "rmse"] else ((new - old) / abs(old)) * 100
        better = "BETTER" if diff > 0 else "WORSE"
        print(f"    {k:10s}: {old:.4f} -> {new:.4f} ({diff:+.2f}%) [{better}]")

    # Feature importance (permutation-based on validation)
    print("\n[7] Feature importance (permutation on validation)...")
    raw_val_final = np.clip(final_model.predict(X_v_final), 0, 100)
    cal_val_final = np.clip(calibrator.predict(raw_val_final), 0, 100)
    base_sp = spearman_corr(y_val, cal_val_final)
    importances = np.zeros(len(selected_feature_names))
    rng = np.random.RandomState(42)
    for fi in range(len(selected_feature_names)):
        X_v_perm = X_v_final.copy()
        rng.shuffle(X_v_perm[:, fi])
        perm_raw = np.clip(final_model.predict(X_v_perm), 0, 100)
        perm_cal = np.clip(calibrator.predict(perm_raw), 0, 100)
        perm_sp = spearman_corr(y_val, perm_cal)
        importances[fi] = max(0, base_sp - perm_sp)
    sorted_idx = np.argsort(importances)[::-1]
    print(f"\n  {'Rank':<5} {'Feature':<25} {'Importance':>12}")
    print(f"  {'-'*45}")
    feat_imp = []
    for rank, idx in enumerate(sorted_idx, 1):
        print(f"  {rank:<5} {selected_feature_names[idx]:<25} {importances[idx]:12.6f}")
        feat_imp.append({"feature": selected_feature_names[idx], "importance": round(float(importances[idx]), 6), "rank": rank})

    with open(BENCH / "phase10_feature_importance.json", "w") as f:
        json.dump({"features": feat_imp, "method": "HistGradientBoosting_feature_importances",
                   "selected_features": selected_feature_names, "feature_count": len(selected_feature_names)}, f, indent=2)

    # Save model
    print("\n[8] Saving model artifacts...")
    joblib.dump({"model": final_model, "feature_names": selected_feature_names},
               BACKEND_MODELS / "model.joblib")
    joblib.dump(calibrator, BACKEND_MODELS / "calibrator.joblib")

    # Metadata
    feature_statistics = {}
    for i, fname in enumerate(selected_feature_names):
        col = X[:, ALL_FEATURE_NAMES.index(fname)]
        feature_statistics[fname] = {
            "min": round(float(np.min(col)), 4), "max": round(float(np.max(col)), 4),
            "mean": round(float(np.mean(col)), 4), "std": round(float(np.std(col)), 4),
        }

    metadata = {
        "model_type": "HistGradientBoosting",
        "model_version": "v4.0.0",
        "training_datasets": ["KADID-10K", "KonIQ-10K"],
        "feature_names": selected_feature_names,
        "feature_count": len(selected_feature_names),
        "target_scale": "0-100 higher-is-better",
        "split": {"method": "stratified_by_quality_bins", "train_ratio": 0.70,
                  "val_ratio": 0.15, "test_ratio": 0.15, "random_seed": 42,
                  "train_samples": len(X_train), "val_samples": len(X_val), "test_samples": len(X_test)},
        "calibration": {"method": "IsotonicRegression", "fitted_on": "validation_set",
                       "never_fitted_on": "test_set"},
        "metrics": {"note": "Computed on untouched test set. Calibrator fitted on validation only.",
                   "raw": {"mae": round(raw_mae, 4), "rmse": round(raw_rmse, 4),
                          "r2": round(raw_r2, 4), "spearman": round(raw_sp, 4)},
                   "calibrated": {"mae": round(cal_mae, 4), "rmse": round(cal_rmse, 4),
                                 "r2": round(cal_r2, 4), "spearman": round(cal_sp, 4)},
                   "mae": round(cal_mae, 4), "rmse": round(cal_rmse, 4),
                   "r2": round(cal_r2, 4), "spearman": round(cal_sp, 4)},
        "test_statistics": {"sample_count": len(X_test),
                           "target_mean": round(float(np.mean(y_test)), 4),
                           "target_std": round(float(np.std(y_test)), 4),
                           "prediction_mean": round(float(np.mean(cal_test)), 4),
                           "prediction_min": round(float(np.min(cal_test)), 4),
                           "prediction_max": round(float(np.max(cal_test)), 4)},
        "dataset_samples": {"kadid": int((ds_labels == "KADID-10K").sum()),
                           "koniq": int((ds_labels == "KonIQ-10K").sum()),
                           "combined": len(X)},
        "feature_statistics": feature_statistics,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(BACKEND_MODELS / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved model, calibrator, metadata")

    print("\n" + "=" * 80)
    print("  FEATURE ENGINEERING EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
