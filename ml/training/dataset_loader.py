"""Unified dataset loader for KADID-10K and KonIQ-10K.

Each dataset is normalized independently to a 0-100 quality scale where:
  100 = excellent visual quality
  0   = extremely poor visual quality

Output per sample:
  - image_path: absolute path to the image file
  - dataset_name: "KADID-10K" or "KonIQ-10K"
  - raw_quality_score: original score from the dataset
  - normalized_quality_score: normalized to 0-100, higher = better
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "datasets" / "extracted"

# Quality bin edges for stratified splitting
QUALITY_BIN_EDGES = [0, 20, 40, 60, 80, 100]


def get_quality_bins(scores: np.ndarray) -> np.ndarray:
    """Assign quality scores to bin indices [0, 1, 2, 3, 4]."""
    bins = np.digitize(scores, bins=QUALITY_BIN_EDGES[1:-1])  # 4 bins from 5 edges
    return bins


def print_quality_distribution(
    scores: np.ndarray, dataset_name: str
) -> dict[int, int]:
    """Print the quality bin distribution for a dataset."""
    bins = get_quality_bins(scores)
    counts: dict[int, int] = {}
    for i in range(5):
        lo = QUALITY_BIN_EDGES[i]
        hi = QUALITY_BIN_EDGES[i + 1]
        count = int(np.sum((scores >= lo) if i == 0 else (scores >= lo)))
        # More precise count
        if i == 0:
            count = int(np.sum((scores >= lo) & (scores < hi)))
        elif i == 4:
            count = int(np.sum(scores >= lo))
        else:
            count = int(np.sum((scores >= lo) & (scores < hi)))
        counts[i] = count
        print(f"    [{lo:3d}-{hi:3d}): {count:5d} samples")

    total = sum(counts.values())
    print(f"    Total: {total} samples")
    return counts


def load_kadid() -> list[dict]:
    """Load KADID-10K dataset.

    KADID-10K DMOS convention:
      - higher DMOS = better visual quality (less distortion)
      - DMOS range is approximately 1.0 (worst) to ~5.0 (best)

    Normalized score:
      normalized = 100 * (dmos - min_dmos) / (max_dmos - min_dmos)
    """
    kadid_root = DATA_ROOT / "kadid10k" / "kadid10k"
    csv_path = kadid_root / "dmos.csv"
    img_dir = kadid_root / "images"

    if not csv_path.exists():
        raise FileNotFoundError(f"KADID CSV not found: {csv_path}")

    print(f"\nLoading KADID-10K from: {kadid_root}")

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"  CSV rows: {len(rows)}")

    # Parse all DMOS scores
    raw_scores = []
    samples = []

    for row in rows:
        img_path = img_dir / row["dist_img"]
        if not img_path.exists():
            continue

        raw_dmos = float(row["dmos"])
        raw_scores.append(raw_dmos)
        samples.append({
            "image_path": str(img_path),
            "dataset_name": "KADID-10K",
            "raw_quality_score": raw_dmos,
        })

    raw_arr = np.array(raw_scores)
    min_dmos = float(np.min(raw_arr))
    max_dmos = float(np.max(raw_arr))

    # KADID: higher DMOS = better quality
    # Normalize: 0-100 scale
    for sample in samples:
        dmos = sample["raw_quality_score"]
        if max_dmos > min_dmos:
            norm = 100.0 * (dmos - min_dmos) / (max_dmos - min_dmos)
        else:
            norm = 50.0
        sample["normalized_quality_score"] = round(max(0.0, min(100.0, norm)), 2)

    # Summary
    norm_scores = np.array([s["normalized_quality_score"] for s in samples])
    print(f"  Raw DMOS range: [{min_dmos:.4f}, {max_dmos:.4f}]")
    print(f"  DMOS direction: higher = better quality")
    print(f"  Normalized score range: [{norm_scores.min():.2f}, {norm_scores.max():.2f}]")
    print(f"  Normalized score mean: {norm_scores.mean():.2f}, std: {norm_scores.std():.2f}")
    print(f"  Valid samples: {len(samples)}")
    print(f"  Quality distribution:")
    print_quality_distribution(norm_scores, "KADID-10K")

    return samples


def load_koniq() -> list[dict]:
    """Load KonIQ-10K dataset.

    KonIQ-10K MOS convention:
      - MOS is a Mean Opinion Score on a 5-point ACR scale
      - higher MOS = better quality
      - Range: 1.0 (worst) to 5.0 (best)

    Normalized score:
      normalized = 100 * (mos - min_mos) / (max_mos - min_mos)
    """
    # Try multiple possible locations
    possible_roots = [
        DATA_ROOT / "koniq10k",
        DATA_ROOT / "koniq-10k",
    ]

    koniq_root = None
    for root in possible_roots:
        if root.exists():
            koniq_root = root
            break

    if koniq_root is None:
        raise FileNotFoundError(
            f"KonIQ-10K dataset not found. Searched: {[str(r) for r in possible_roots]}"
        )

    # Find CSV
    csv_files = list(koniq_root.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {koniq_root}")

    csv_path = csv_files[0]
    print(f"\nLoading KonIQ-10K from: {koniq_root}")
    print(f"  CSV file: {csv_path.name}")

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"  CSV rows: {len(rows)}")
    print(f"  Columns: {list(rows[0].keys()) if rows else 'N/A'}")

    # Find image directory and score column
    image_dirs = [d for d in koniq_root.iterdir() if d.is_dir()]
    img_dir = None
    for d in image_dirs:
        # Check if this dir contains images
        try:
            for f in list(d.iterdir())[:5]:
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    img_dir = d
                    break
        except PermissionError:
            pass
        if img_dir:
            break

    if img_dir is None:
        # Maybe images are directly in the root
        for f in list(koniq_root.iterdir())[:5]:
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                img_dir = koniq_root
                break

    if img_dir is None:
        raise FileNotFoundError(f"No image directory found in {koniq_root}")

    print(f"  Image directory: {img_dir}")

    # Find score column
    score_col = None
    candidates = ["MOS", "mos", "dmos", "score", "quality_score", "quality"]
    for col in candidates:
        if col in rows[0]:
            score_col = col
            break
    if not score_col:
        for col in rows[0]:
            if col.lower() in ["mos", "dmos", "score", "quality_score"]:
                score_col = col
                break

    if score_col is None:
        raise ValueError(f"Could not find quality score column in CSV. Columns: {list(rows[0].keys())}")

    # Find image name column
    img_col = None
    for col in ["image_name", "filename", "image", "image_file", "dist_img"]:
        if col in rows[0]:
            img_col = col
            break

    if img_col is None:
        for col in rows[0]:
            if "name" in col.lower() or "file" in col.lower() or "image" in col.lower():
                img_col = col
                break

    if img_col is None:
        raise ValueError(f"Could not find image name column. Columns: {list(rows[0].keys())}")

    print(f"  Score column: '{score_col}'")
    print(f"  Image column: '{img_col}'")

    # Parse samples
    raw_scores = []
    samples = []
    missing = 0

    for row in rows:
        img_name = row[img_col]
        # Try to find the image
        img_path = img_dir / img_name
        if not img_path.exists():
            missing += 1
            continue

        try:
            raw_mos = float(row[score_col])
        except (ValueError, KeyError):
            continue

        raw_scores.append(raw_mos)
        samples.append({
            "image_path": str(img_path),
            "dataset_name": "KonIQ-10K",
            "raw_quality_score": raw_mos,
        })

    if not raw_scores:
        raise ValueError("No valid samples found in KonIQ-10K dataset")

    raw_arr = np.array(raw_scores)
    min_mos = float(np.min(raw_arr))
    max_mos = float(np.max(raw_arr))

    # KonIQ: higher MOS = better quality
    # Normalize: 0-100 scale
    for sample in samples:
        mos = sample["raw_quality_score"]
        if max_mos > min_mos:
            norm = 100.0 * (mos - min_mos) / (max_mos - min_mos)
        else:
            norm = 50.0
        sample["normalized_quality_score"] = round(max(0.0, min(100.0, norm)), 2)

    # Summary
    norm_scores = np.array([s["normalized_quality_score"] for s in samples])
    print(f"  Raw MOS range: [{min_mos:.4f}, {max_mos:.4f}]")
    print(f"  MOS direction: higher = better quality")
    print(f"  Normalized score range: [{norm_scores.min():.2f}, {norm_scores.max():.2f}]")
    print(f"  Normalized score mean: {norm_scores.mean():.2f}, std: {norm_scores.std():.2f}")
    print(f"  Valid samples: {len(samples)}")
    print(f"  Missing images: {missing}")
    print(f"  Quality distribution:")
    print_quality_distribution(norm_scores, "KonIQ-10K")

    return samples


def load_all_datasets() -> list[dict]:
    """Load and combine all available datasets."""
    all_samples = []

    # Load KADID-10K
    try:
        kadid_samples = load_kadid()
        all_samples.extend(kadid_samples)
        print(f"\n  KADID-10K loaded: {len(kadid_samples)} samples")
    except Exception as e:
        print(f"\n  ✗ Failed to load KADID-10K: {e}")

    # Load KonIQ-10K
    try:
        koniq_samples = load_koniq()
        all_samples.extend(koniq_samples)
        print(f"  KonIQ-10K loaded: {len(koniq_samples)} samples")
    except Exception as e:
        print(f"  ✗ Failed to load KonIQ-10K: {e}")

    if not all_samples:
        raise ValueError("No datasets could be loaded!")

    # Combined statistics
    all_scores = np.array([s["normalized_quality_score"] for s in all_samples])
    print(f"\n{'=' * 70}")
    print(f"  Combined Dataset Statistics")
    print(f"{'=' * 70}")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Normalized score range: [{all_scores.min():.2f}, {all_scores.max():.2f}]")
    print(f"  Normalized score mean: {all_scores.mean():.2f}, std: {all_scores.std():.2f}")
    print(f"\n  Quality distribution (combined):")
    print_quality_distribution(all_scores, "Combined")

    return all_samples


if __name__ == "__main__":
    print("=" * 70)
    print("  VisionGuard — Unified Dataset Loader")
    print("=" * 70)
    samples = load_all_datasets()
    print(f"\n  Total samples loaded: {len(samples)}")
