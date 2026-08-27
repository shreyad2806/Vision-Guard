"""Inspect available datasets and report structure, columns, score ranges."""

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "datasets" / "extracted"


def inspect_kadid():
    """Inspect the KADID-10K dataset."""
    kadid_root = DATA_ROOT / "kadid10k" / "kadid10k"
    csv_path = kadid_root / "dmos.csv"
    img_dir = kadid_root / "images"

    print("=" * 60)
    print("KADID-10K Dataset Inspection")
    print("=" * 60)

    if not csv_path.exists():
        print(f"  CSV not found: {csv_path}")
        return

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"  Dataset root: {kadid_root}")
    print(f"  Image directory: {img_dir}")
    print(f"  Metadata file: {csv_path}")
    print(f"  Columns: {list(rows[0].keys())}")
    print(f"  Total rows: {len(rows)}")

    # Score statistics
    scores = [float(r["dmos"]) for r in rows]
    mean_score = sum(scores) / len(scores)
    print(f"  DMOS score range: {min(scores):.3f} - {max(scores):.3f}")
    print(f"  DMOS score mean: {mean_score:.3f}")
    print(f"  DMOS score std: {(sum((s - mean_score)**2 for s in scores) / len(scores))**0.5:.3f}")

    # Normalized quality scores
    norm_scores = [max(0, min(100, (s - 1.0) / 4.0 * 100)) for s in scores]
    print(f"  Normalized quality range: {min(norm_scores):.1f} - {max(norm_scores):.1f}")

    # Reference images
    ref_imgs = set(r["ref_img"] for r in rows)
    print(f"  Unique reference images: {len(ref_imgs)}")

    # Distortion types
    dist_types: set[str] = set()
    for r in rows:
        fname = r["dist_img"].replace(".png", "")
        parts = fname.split("_")
        if len(parts) >= 3:
            dist_types.add(parts[1])
    print(f"  Distortion types: {len(dist_types)} ({sorted(dist_types)[:5]}...)")

    # Check image existence
    missing = sum(1 for r in rows if not (img_dir / r["dist_img"]).exists())
    print(f"  Missing images: {missing}")

    print(f"\n  Interpretation: DMOS 1.0 = low quality, DMOS ~5.0 = high quality")
    print(f"  Quality = (DMOS - 1) / 4 * 100  ->  maps to 0-100 scale")
    print()


def inspect_koniq():
    """Inspect the KonIQ-10K dataset."""
    koniq_root = DATA_ROOT / "koniq-10k"

    print("=" * 60)
    print("KonIQ-10K Dataset Inspection")
    print("=" * 60)

    if not koniq_root.exists():
        print(f"  Dataset not found at: {koniq_root}")
        print(f"  Download from: https://database.mmsp-kn.de/koniq-10k-database.html")
        print(f"  Expected structure: koniq-10k/images/ and koniq-10k/koniq10k.csv")
        print()
        return False

    csv_path = koniq_root / "koniq10k.csv"
    img_dir = koniq_root / "images"

    if not csv_path.exists():
        print(f"  CSV not found: {csv_path}")
        return False

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"  Dataset root: {koniq_root}")
    print(f"  Image directory: {img_dir}")
    print(f"  Metadata file: {csv_path}")
    print(f"  Columns: {list(rows[0].keys())}")
    print(f"  Total rows: {len(rows)}")

    # Try common score column names
    score_col = None
    for col in ["MOS", "mos", "score", "quality_score", "dmos", "quality"]:
        if col in rows[0]:
            score_col = col
            break

    if score_col:
        scores = [float(r[score_col]) for r in rows]
        print(f"  Score column: {score_col}")
        print(f"  Score range: {min(scores):.3f} - {max(scores):.3f}")
        print(f"  Score mean: {sum(scores)/len(scores):.3f}")

    print()


def main():
    inspect_kadid()
    has_koniq = inspect_koniq()

    if not has_koniq:
        print("Note: KonIQ-10K not available. Training will use KADID-10K only.")


if __name__ == "__main__":
    main()
