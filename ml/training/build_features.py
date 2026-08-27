"""Batch feature extraction for training datasets.

Uses the same feature extractor as the backend to ensure consistency.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add backend to path so we can import the production feature extractor
BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from apps.ml.feature_extractor import extract_features  # noqa: E402

# Maximum dimension for feature extraction (preserves aspect ratio)
MAX_DIM = 512


def resize_if_needed(img: np.ndarray, max_dim: int = MAX_DIM) -> np.ndarray:
    """Resize image if any dimension exceeds max_dim, preserving aspect ratio."""
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img

    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def extract_features_from_path(
    image_path: str,
    max_dim: int = MAX_DIM,
) -> dict | None:
    """Load an image and extract features.

    Returns None if the image cannot be loaded.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    img = resize_if_needed(img, max_dim)
    return extract_features(img)


def batch_extract(
    image_paths: list[str],
    max_dim: int = MAX_DIM,
    verbose: bool = True,
) -> tuple[list[int], list[dict], int]:
    """Extract features from a list of image paths.

    Returns:
        valid_indices: indices of successfully processed images
        features: list of feature dicts (one per valid image)
        skipped: count of skipped images
    """
    features = []
    valid_indices = []
    skipped = 0
    total = len(image_paths)

    t0 = time.time()

    for i, img_path in enumerate(image_paths):
        if verbose and (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(
                f"  Progress: {i+1}/{total} "
                f"({skipped} skipped) "
                f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]"
            )

        feat = extract_features_from_path(img_path, max_dim)
        if feat is None:
            skipped += 1
            continue

        features.append(feat)
        valid_indices.append(i)

    elapsed = time.time() - t0
    if verbose:
        print(
            f"  Done: {len(features)} extracted, "
            f"{skipped} skipped in {elapsed:.1f}s"
        )

    return valid_indices, features, skipped


def features_to_array(
    features: list[dict],
    feature_names: list[str] | None = None,
) -> np.ndarray:
    """Convert list of feature dicts to a numpy array.

    Parameters
    ----------
    features : list of dicts
    feature_names : optional list of feature names to use (determines column order)

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)
    """
    if feature_names is None:
        feature_names = sorted(features[0].keys())

    arr = np.array([[f[name] for name in feature_names] for f in features])
    return arr


# Default feature names matching the production feature extractor
DEFAULT_FEATURE_NAMES = [
    "brightness",
    "colorfulness",
    "contrast",
    "dynamic_range",
    "edge_density",
    "entropy",
    "noise_estimate",
    "overexposure_pct",
    "saturation",
    "sharpness",
    "texture_complexity",
    "underexposure_pct",
]


if __name__ == "__main__":
    # Quick test with a sample image
    sample_images = list(Path(BACKEND_ROOT / "data" / "uploads").glob("*"))
    if sample_images:
        print(f"Testing feature extraction on: {sample_images[0].name}")
        feat = extract_features_from_path(str(sample_images[0]))
        if feat:
            print(f"  Features: {feat}")
        else:
            print("  Could not extract features")
    else:
        print("No sample images found for testing.")
