"""Load KADID-10K dataset with group-aware train/val/test splitting."""

import csv
from pathlib import Path
from typing import Optional

import numpy as np

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def load_kadid(
    max_samples: Optional[int] = None,
    random_state: int = 42,
) -> dict:
    """Load KADID-10K dataset.

    Returns dict with:
        - images: list of image paths
        - scores: numpy array of quality scores (0-100)
        - ref_groups: dict mapping ref_image -> list of indices
        - metadata: list of dicts with distortion info
    """
    kadid_root = DATA_ROOT / "kadid10k"
    csv_path = kadid_root / "dmos.csv"
    img_dir = kadid_root / "images"

    if not csv_path.exists():
        raise FileNotFoundError(f"KADID CSV not found: {csv_path}")

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Parse all rows
    images = []
    raw_scores = []
    ref_images = []
    metadata = []

    for row in rows:
        img_path = img_dir / row["dist_img"]
        if not img_path.exists():
            continue

        images.append(str(img_path))
        raw_scores.append(float(row["dmos"]))
        ref_images.append(row["ref_img"])
        metadata.append({
            "dist_img": row["dist_img"],
            "ref_img": row["ref_img"],
            "var": float(row["var"]),
        })

    # Convert DMOS to quality score 0-100
    # DMOS range: 1.0 (worst) to ~5.0 (best)
    # Quality = (dmos - 1) / 4 * 100
    raw_scores = np.array(raw_scores)
    scores = np.clip((raw_scores - 1.0) / 4.0 * 100, 0, 100)

    # Group by reference image for leakage-aware splitting
    ref_groups = {}
    for i, ref in enumerate(ref_images):
        if ref not in ref_groups:
            ref_groups[ref] = []
        ref_groups[ref].append(i)

    # Subsample if requested
    if max_samples and max_samples < len(images):
        rng = np.random.RandomState(random_state)
        indices = rng.choice(len(images), max_samples, replace=False)
        images = [images[i] for i in indices]
        scores = scores[indices]
        ref_images = [ref_images[i] for i in indices]
        metadata = [metadata[i] for i in indices]

        # Rebuild ref_groups
        ref_groups = {}
        for i, ref in enumerate(ref_images):
            if ref not in ref_groups:
                ref_groups[ref] = []
            ref_groups[ref].append(i)

    return {
        "images": images,
        "scores": scores,
        "ref_images": ref_images,
        "ref_groups": ref_groups,
        "metadata": metadata,
        "dataset_name": "KADID-10K",
    }


def split_dataset(
    dataset: dict,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_state: int = 42,
) -> dict:
    """Split dataset by reference groups to prevent leakage.

    Returns dict with train/val/test splits, each containing
    images, scores, and indices.
    """
    rng = np.random.RandomState(random_state)
    ref_names = list(dataset["ref_groups"].keys())
    rng.shuffle(ref_names)

    n = len(ref_names)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_refs = ref_names[:n_train]
    val_refs = ref_names[n_train:n_train + n_val]
    test_refs = ref_names[n_train + n_val:]

    def collect_indices(refs):
        indices = []
        for ref in refs:
            indices.extend(dataset["ref_groups"][ref])
        return sorted(indices)

    train_idx = collect_indices(train_refs)
    val_idx = collect_indices(val_refs)
    test_idx = collect_indices(test_refs)

    def make_split(indices):
        return {
            "images": [dataset["images"][i] for i in indices],
            "scores": dataset["scores"][indices],
            "indices": indices,
        }

    return {
        "train": make_split(train_idx),
        "val": make_split(val_idx),
        "test": make_split(test_idx),
        "train_refs": train_refs,
        "val_refs": val_refs,
        "test_refs": test_refs,
    }


if __name__ == "__main__":
    print("Loading KADID-10K...")
    ds = load_kadid()
    print(f"  Total images: {len(ds['images'])}")
    print(f"  Reference groups: {len(ds['ref_groups'])}")
    print(f"  Score range: {ds['scores'].min():.1f} - {ds['scores'].max():.1f}")

    splits = split_dataset(ds)
    print(f"\nSplit sizes:")
    print(f"  Train: {len(splits['train']['images'])} images ({len(splits['train_refs'])} refs)")
    print(f"  Val:   {len(splits['val']['images'])} images ({len(splits['val_refs'])} refs)")
    print(f"  Test:  {len(splits['test']['images'])} images ({len(splits['test_refs'])} refs)")
