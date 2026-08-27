"""Train image quality prediction models.

Compares RandomForestRegressor and HistGradientBoostingRegressor,
selects the best model on validation metrics, and saves the final
pipeline to backend/models/.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent / "backend"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from dataset_loader import load_kadid, split_dataset  # noqa: E402
from build_features import (  # noqa: E402
    batch_extract,
    features_to_array,
    DEFAULT_FEATURE_NAMES,
)

MODELS_DIR = BACKEND_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Train multiple regressors and return comparison results."""

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
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
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        t0 = time.time()

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model.fit(X_train_scaled, y_train)

        train_time = time.time() - t0
        print(f"  Training time: {train_time:.1f}s")

        # Evaluate on validation set
        y_val_pred = model.predict(X_val_scaled)
        y_val_pred = np.clip(y_val_pred, 0, 100)

        mae = mean_absolute_error(y_val, y_val_pred)
        rmse = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))
        r2 = r2_score(y_val, y_val_pred)

        print(f"  Validation MAE:  {mae:.3f}")
        print(f"  Validation RMSE: {rmse:.3f}")
        print(f"  Validation R²:   {r2:.4f}")

        results[name] = {
            "model": model,
            "scaler": scaler,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "train_time": train_time,
        }

    return results


def evaluate_on_test(
    model,
    scaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Evaluate the final model on the held-out test set."""
    X_test_scaled = scaler.transform(X_test)
    y_test_pred = model.predict(X_test_scaled)
    y_test_pred = np.clip(y_test_pred, 0, 100)

    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
    r2 = r2_score(y_test, y_test_pred)

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "predictions": y_test_pred,
    }


def save_artifacts(
    best_model,
    best_scaler,
    feature_names: list[str],
    val_metrics: dict,
    test_metrics: dict,
    training_info: dict,
):
    """Save model, scaler, and metadata to backend/models/."""
    joblib.dump(best_model, MODELS_DIR / "quality_model.joblib")
    joblib.dump(best_scaler, MODELS_DIR / "feature_scaler.joblib")

    metadata = {
        "model_name": training_info["model_name"],
        "model_version": "visionguard-iqa-v1",
        "feature_names": feature_names,
        "training_datasets": training_info["datasets"],
        "training_mode": training_info["mode"],
        "training_samples": training_info["n_train"],
        "validation_metrics": {
            "mae": round(float(val_metrics["mae"]), 4),
            "rmse": round(float(val_metrics["rmse"]), 4),
            "r2": round(float(val_metrics["r2"]), 4),
        },
        "test_metrics": {
            k: v for k, v in test_metrics.items() if k != "predictions"
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel artifacts saved to {MODELS_DIR}/")
    print(f"  quality_model.joblib")
    print(f"  feature_scaler.joblib")
    print(f"  model_metadata.json")


def main():
    parser = argparse.ArgumentParser(description="Train VisionGuard IQA model")
    parser.add_argument(
        "--mode",
        choices=["fast", "full"],
        default="fast",
        help="Training mode: 'fast' uses a subset, 'full' uses all images",
    )
    parser.add_argument(
        "--max-samples-fast",
        type=int,
        default=3000,
        help="Max samples for fast mode",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--max-image-dim",
        type=int,
        default=512,
        help="Max image dimension for feature extraction",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("VisionGuard — Image Quality Model Training")
    print("=" * 60)
    print(f"  Mode: {args.mode}")
    print(f"  Random state: {args.random_state}")
    print(f"  Max image dim: {args.max_image_dim}")
    print()

    # ── Load dataset ──────────────────────────────────────────────
    print("Loading KADID-10K dataset...")
    max_samples = args.max_samples_fast if args.mode == "fast" else None
    dataset = load_kadid(max_samples=max_samples, random_state=args.random_state)

    print(f"  Total images: {len(dataset['images'])}")
    print(f"  Reference groups: {len(dataset['ref_groups'])}")
    print(f"  Score range: {dataset['scores'].min():.1f} - {dataset['scores'].max():.1f}")
    print(f"  Score mean: {dataset['scores'].mean():.1f}")
    print(f"  Score std: {dataset['scores'].std():.1f}")
    print()

    # ── Split ─────────────────────────────────────────────────────
    splits = split_dataset(dataset, random_state=args.random_state)
    print("Dataset splits:")
    print(f"  Train: {len(splits['train']['images'])} images ({len(splits['train_refs'])} refs)")
    print(f"  Val:   {len(splits['val']['images'])} images ({len(splits['val_refs'])} refs)")
    print(f"  Test:  {len(splits['test']['images'])} images ({len(splits['test_refs'])} refs)")
    print()

    # ── Feature extraction ────────────────────────────────────────
    print("Extracting features from training set...")
    train_idx, train_features, train_skipped = batch_extract(
        splits["train"]["images"], max_dim=args.max_image_dim
    )
    train_scores = splits["train"]["scores"][train_idx]

    print("Extracting features from validation set...")
    val_idx, val_features, val_skipped = batch_extract(
        splits["val"]["images"], max_dim=args.max_image_dim
    )
    val_scores = splits["val"]["scores"][val_idx]

    print("Extracting features from test set...")
    test_idx, test_features, test_skipped = batch_extract(
        splits["test"]["images"], max_dim=args.max_image_dim
    )
    test_scores = splits["test"]["scores"][test_idx]

    # Convert to arrays
    X_train = features_to_array(train_features)
    X_val = features_to_array(val_features)
    X_test = features_to_array(test_features)
    y_train = np.array(train_scores)
    y_val = np.array(val_scores)
    y_test = np.array(test_scores)

    print(f"\nFeature matrix shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")
    print()

    # ── Train and compare ─────────────────────────────────────────
    print("=" * 60)
    print("Model Training & Comparison")
    print("=" * 60)

    results = train_and_evaluate(X_train, y_train, X_val, y_val, DEFAULT_FEATURE_NAMES)

    # Select best model by MAE
    best_name = min(results, key=lambda k: results[k]["mae"])
    best = results[best_name]

    print(f"\n{'=' * 60}")
    print(f"Best model: {best_name}")
    print(f"  Validation MAE:  {best['mae']:.3f}")
    print(f"  Validation RMSE: {best['rmse']:.3f}")
    print(f"  Validation R²:   {best['r2']:.4f}")
    print(f"{'=' * 60}")

    # ── Final test evaluation ─────────────────────────────────────
    print("\nFinal evaluation on held-out test set...")
    test_results = evaluate_on_test(
        best["model"], best["scaler"], X_test, y_test
    )
    print(f"  Test MAE:  {test_results['mae']:.3f}")
    print(f"  Test RMSE: {test_results['rmse']:.3f}")
    print(f"  Test R²:   {test_results['r2']:.4f}")

    # ── Save artifacts ────────────────────────────────────────────
    total_samples = len(train_features) + len(val_features) + len(test_features)
    total_skipped = train_skipped + val_skipped + test_skipped

    save_artifacts(
        best_model=best["model"],
        best_scaler=best["scaler"],
        feature_names=DEFAULT_FEATURE_NAMES,
        val_metrics={"mae": best["mae"], "rmse": best["rmse"], "r2": best["r2"]},
        test_metrics=test_results,
        training_info={
            "model_name": best_name,
            "datasets": ["KADID-10K"],
            "mode": args.mode,
            "n_train": total_samples,
        },
    )

    print(f"\nTraining complete.")
    print(f"  Total images processed: {total_samples}")
    print(f"  Total skipped: {total_skipped}")
    print(f"  Best model: {best_name}")


if __name__ == "__main__":
    main()
