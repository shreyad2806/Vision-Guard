"""Evaluate a trained model and generate diagnostic visualisations.

Run after train_model.py has saved artifacts to backend/models/.
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from dataset_loader import load_kadid, split_dataset  # noqa: E402
from build_features import (  # noqa: E402
    batch_extract,
    features_to_array,
)

MODELS_DIR = BACKEND_ROOT / "models"


def load_saved_model():
    """Load the trained model, scaler, and metadata."""
    model = joblib.load(MODELS_DIR / "quality_model.joblib")
    scaler = joblib.load(MODELS_DIR / "feature_scaler.joblib")
    with open(MODELS_DIR / "model_metadata.json") as f:
        metadata = json.load(f)
    return model, scaler, metadata


def plot_actual_vs_predicted(
    y_actual: np.ndarray,
    y_predicted: np.ndarray,
    save_path: Path,
):
    """Create a scatter plot of actual vs predicted quality scores."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot
    ax = axes[0]
    ax.scatter(y_actual, y_predicted, alpha=0.3, s=10, c="#3b82f6")
    lims = [0, 100]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Quality Score", fontsize=11)
    ax.set_ylabel("Predicted Quality Score", fontsize=11)
    ax.set_title("Actual vs Predicted", fontsize=13, fontweight="bold")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Residual distribution
    ax2 = axes[1]
    residuals = y_predicted - y_actual
    ax2.hist(residuals, bins=50, color="#6366f1", alpha=0.7, edgecolor="white")
    ax2.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax2.set_xlabel("Prediction Error (Predicted - Actual)", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Error Distribution", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {save_path}")


def plot_feature_importance(
    model,
    feature_names: list[str],
    save_path: Path,
):
    """Plot feature importance for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        print("  Model does not support feature_importances_, skipping.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        range(len(feature_names)),
        importances[indices],
        color="#6366f1",
        alpha=0.8,
    )
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right")
    ax.set_ylabel("Importance", fontsize=11)
    ax.set_title("Feature Importance", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {save_path}")


def main():
    print("=" * 60)
    print("VisionGuard — Model Evaluation")
    print("=" * 60)

    # Load model
    print("\nLoading saved model...")
    model, scaler, metadata = load_saved_model()
    print(f"  Model: {metadata['model_name']}")
    print(f"  Version: {metadata['model_version']}")
    print(f"  Features: {metadata['feature_names']}")
    print(f"  Training samples: {metadata['training_samples']}")

    # Load test data
    print("\nLoading KADID-10K test set...")
    dataset = load_kadid(random_state=42)
    splits = split_dataset(dataset, random_state=42)

    print("Extracting test features...")
    test_idx, test_features, skipped = batch_extract(
        splits["test"]["images"], verbose=True
    )
    test_scores = splits["test"]["scores"][test_idx]

    X_test = features_to_array(test_features, metadata["feature_names"])
    y_test = test_scores

    # Predict
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_pred = np.clip(y_pred, 0, 100)

    # Metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    print(f"\nTest Set Metrics:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")

    # Per-label breakdown
    print("\nPer-label breakdown:")
    for label, lo, hi in [("ACCEPTABLE", 75, 100), ("DEGRADED", 45, 74), ("DEFECTIVE", 0, 44)]:
        mask = (y_test >= lo) & (y_test <= hi)
        if mask.sum() > 0:
            label_mae = mean_absolute_error(y_test[mask], y_pred[mask])
            print(f"  {label:12s}: n={mask.sum():5d}, MAE={label_mae:.3f}")

    # Save plots
    plot_dir = Path(__file__).resolve().parent / "plots"
    plot_dir.mkdir(exist_ok=True)

    plot_actual_vs_predicted(y_test, y_pred, plot_dir / "actual_vs_predicted.png")
    plot_feature_importance(model, metadata["feature_names"], plot_dir / "feature_importance.png")

    print(f"\nEvaluation complete.")


if __name__ == "__main__":
    main()
