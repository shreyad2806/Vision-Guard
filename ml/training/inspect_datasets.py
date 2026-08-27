"""Inspect available datasets and report structure, columns, score ranges."""

import csv
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "datasets" / "extracted"


def find_csv_files(root: Path) -> list[Path]:
    """Find all CSV files in a directory tree."""
    csv_files = []
    for f in root.rglob("*.csv"):
        csv_files.append(f)
    return csv_files


def find_image_dirs(root: Path) -> list[Path]:
    """Find directories containing image files."""
    image_dirs = []
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    for dirpath in root.rglob("*"):
        if dirpath.is_dir():
            try:
                for f in list(dirpath.iterdir())[:10]:
                    if f.suffix.lower() in extensions:
                        image_dirs.append(dirpath)
                        break
            except PermissionError:
                pass
    return image_dirs


def inspect_dataset(name: str, root: Path) -> dict | None:
    """Inspect a single dataset and return summary info."""
    print("=" * 70)
    print(f"  {name} Dataset Inspection")
    print("=" * 70)

    if not root.exists():
        print(f"  ✗ Dataset root not found: {root}")
        return None

    print(f"  Dataset root: {root}")

    # Find CSV files
    csv_files = find_csv_files(root)
    if csv_files:
        print(f"  CSV/metadata files found:")
        for f in csv_files:
            print(f"    - {f.relative_to(root)}")
    else:
        print(f"  ✗ No CSV files found")

    # Find image directories
    image_dirs = find_image_dirs(root)
    if image_dirs:
        print(f"  Image directories found:")
        for d in image_dirs:
            img_count = sum(1 for _ in d.iterdir() if _.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
            print(f"    - {d.relative_to(root)} ({img_count} images)")
    else:
        print(f"  ✗ No image directories found")

    # Parse each CSV
    results = {}
    for csv_path in csv_files:
        print(f"\n  --- Inspecting: {csv_path.name} ---")
        try:
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception as e:
            print(f"  ✗ Error reading CSV: {e}")
            continue

        if not rows:
            print(f"  ✗ CSV is empty")
            continue

        columns = list(rows[0].keys())
        print(f"  Column names: {columns}")
        print(f"  Total metadata rows: {len(rows)}")

        # Show sample rows
        print(f"  Sample rows (first 3):")
        for i, row in enumerate(rows[:3]):
            print(f"    Row {i}: {dict(row)}")

        # Try to find quality score column
        score_col = None
        candidates = ["dmos", "MOS", "mos", "score", "quality_score", "quality"]
        for col in candidates:
            if col in columns:
                score_col = col
                break

        if not score_col:
            # Try case-insensitive match
            for col in columns:
                if col.lower() in ["dmos", "mos", "score", "quality_score", "quality"]:
                    score_col = col
                    break

        if score_col:
            scores = []
            for row in rows:
                try:
                    scores.append(float(row[score_col]))
                except (ValueError, KeyError):
                    pass

            if scores:
                import statistics
                print(f"  Quality score column: '{score_col}'")
                print(f"  Quality score range: {min(scores):.4f} - {max(scores):.4f}")
                print(f"  Quality score mean: {statistics.mean(scores):.4f}")
                print(f"  Quality score std: {statistics.stdev(scores):.4f}")
                print(f"  Quality score min: {min(scores):.4f}")
                print(f"  Quality score max: {max(scores):.4f}")
                results["score_col"] = score_col
                results["scores"] = scores
                results["columns"] = columns
                results["total_rows"] = len(rows)
        else:
            print(f"  ✗ Could not identify quality score column")

        # Count images that exist
        if image_dirs:
            img_dir = image_dirs[0]
            # Find image filename column
            img_col = None
            for col in ["dist_img", "image_name", "filename", "image", "image_file"]:
                if col in columns:
                    img_col = col
                    break

            if img_col:
                missing = 0
                found = 0
                for row in rows:
                    img_path = img_dir / row[img_col]
                    if img_path.exists():
                        found += 1
                    else:
                        missing += 1
                print(f"  Available images: {found}")
                print(f"  Missing images: {missing}")
                results["available_images"] = found
                results["missing_images"] = missing
                results["img_col"] = img_col
                results["img_dir"] = str(img_dir)

    results["csv_files"] = [str(f) for f in csv_files]
    results["image_dirs"] = [str(d) for d in image_dirs]
    results["dataset_root"] = str(root)
    return results


def inspect_kadid():
    """Inspect the KADID-10K dataset."""
    kadid_root = DATA_ROOT / "kadid10k" / "kadid10k"
    return inspect_dataset("KADID-10K", kadid_root)


def inspect_koniq():
    """Inspect the KonIQ-10K dataset."""
    # Try multiple possible locations
    possible_roots = [
        DATA_ROOT / "koniq10k",
        DATA_ROOT / "koniq-10k",
        DATA_ROOT / "KONIQ-10k",
    ]

    for root in possible_roots:
        if root.exists():
            return inspect_dataset("KonIQ-10K", root)

    print("  ✗ KonIQ-10K not found in any expected location")
    print(f"    Searched: {[str(r) for r in possible_roots]}")
    return None


def main():
    print("VisionGuard — Dataset Inspection")
    print("=" * 70)
    print()

    kadid_info = inspect_kadid()
    print()
    koniq_info = inspect_koniq()
    print()

    # Summary
    print("=" * 70)
    print("  Summary")
    print("=" * 70)

    if kadid_info:
        print(f"  KADID-10K:")
        print(f"    Score column: {kadid_info.get('score_col', 'N/A')}")
        print(f"    Total rows: {kadid_info.get('total_rows', 'N/A')}")
        print(f"    Available images: {kadid_info.get('available_images', 'N/A')}")
        if kadid_info.get("scores"):
            s = kadid_info["scores"]
            print(f"    Score range: {min(s):.4f} - {max(s):.4f}")
    else:
        print(f"  KADID-10K: NOT FOUND")

    if koniq_info:
        print(f"  KonIQ-10K:")
        print(f"    Score column: {koniq_info.get('score_col', 'N/A')}")
        print(f"    Total rows: {koniq_info.get('total_rows', 'N/A')}")
        print(f"    Available images: {koniq_info.get('available_images', 'N/A')}")
        if koniq_info.get("scores"):
            s = koniq_info["scores"]
            print(f"    Score range: {min(s):.4f} - {max(s):.4f}")
    else:
        print(f"  KonIQ-10K: NOT FOUND")


if __name__ == "__main__":
    main()
