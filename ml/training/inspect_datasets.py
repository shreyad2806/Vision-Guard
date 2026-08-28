"""Inspect available datasets and report structure, columns, score ranges.

Run with:
    python -m ml.training.inspect_datasets
"""

from __future__ import annotations

import csv
import os
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "datasets"
EXTRACTED_ROOT = DATA_ROOT / "extracted"
RAW_ROOT = DATA_ROOT / "raw"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
SCORE_COLUMN_CANDIDATES = [
    "dmos", "mos", "MOS", "DMOS", "score", "quality_score", "quality",
    "MOS_zscore", "mos_zscore",
]
IMAGE_COLUMN_CANDIDATES = [
    "image_name", "dist_img", "filename", "image", "image_file",
    "img_name", "img_file",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_tree(root: Path, max_depth: int = 3, prefix: str = "") -> None:
    """Print a directory tree summary (limited depth)."""
    if max_depth < 0:
        print(f"{prefix}... (truncated)")
        return
    try:
        entries = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name))
    except PermissionError:
        print(f"{prefix}[permission denied]")
        return

    dirs = [e for e in entries if e.is_dir()]
    files = [e for e in entries if e.is_file()]

    # Summarize if too many entries
    if len(files) > 20:
        file_groups: dict[str, int] = {}
        for f in files:
            ext = f.suffix.lower() or "(no ext)"
            file_groups[ext] = file_groups.get(ext, 0) + 1
        for d in dirs:
            print(f"{prefix}{d.name}/")
            print_tree(d, max_depth - 1, prefix + "  ")
        summary = ", ".join(f"{count} {ext}" for ext, count in sorted(file_groups.items()))
        print(f"{prefix}  ({len(files)} files: {summary})")
    else:
        for d in dirs:
            print(f"{prefix}{d.name}/")
            print_tree(d, max_depth - 1, prefix + "  ")
        for f in files:
            size_mb = f.stat().st_size / (1024 * 1024)
            if size_mb > 1:
                print(f"{prefix}{f.name}  ({size_mb:.1f} MB)")
            else:
                print(f"{prefix}{f.name}")


def find_csv_files(root: Path) -> list[Path]:
    """Find all CSV files in a directory tree."""
    return sorted(root.rglob("*.csv"))


def count_images_in_dir(d: Path) -> dict[str, int]:
    """Count images in a directory by extension."""
    counts: dict[str, int] = {}
    try:
        for f in d.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                ext = f.suffix.lower()
                counts[ext] = counts.get(ext, 0) + 1
    except PermissionError:
        pass
    return counts


def find_image_dirs(root: Path) -> list[Path]:
    """Find directories that contain image files."""
    image_dirs: list[Path] = []
    for dirpath in root.rglob("*"):
        if dirpath.is_dir():
            try:
                for f in list(dirpath.iterdir())[:10]:
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                        image_dirs.append(dirpath)
                        break
            except PermissionError:
                pass
    return image_dirs


def read_csv_info(csv_path: Path) -> dict:
    """Read a CSV file and extract metadata info."""
    info: dict = {"path": str(csv_path), "name": csv_path.name}
    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        info["error"] = str(e)
        return info

    if not rows:
        info["empty"] = True
        return info

    columns = list(rows[0].keys())
    info["columns"] = columns
    info["total_rows"] = len(rows)
    info["sample_rows"] = [dict(r) for r in rows[:3]]

    # Find score column
    score_col = None
    for candidate in SCORE_COLUMN_CANDIDATES:
        if candidate in columns:
            score_col = candidate
            break
    if not score_col:
        for col in columns:
            if col.lower() in [c.lower() for c in SCORE_COLUMN_CANDIDATES]:
                score_col = col
                break

    if score_col:
        scores: list[float] = []
        for row in rows:
            try:
                scores.append(float(row[score_col]))
            except (ValueError, KeyError):
                pass
        if scores:
            info["score_col"] = score_col
            info["num_scores"] = len(scores)
            info["score_min"] = min(scores)
            info["score_max"] = max(scores)
            info["score_mean"] = statistics.mean(scores)
            info["score_std"] = statistics.stdev(scores) if len(scores) > 1 else 0.0

    # Find image name column
    img_col = None
    for candidate in IMAGE_COLUMN_CANDIDATES:
        if candidate in columns:
            img_col = candidate
            break
    if not img_col:
        for col in columns:
            if any(kw in col.lower() for kw in ["name", "file", "image"]):
                img_col = col
                break
    if img_col:
        info["img_col"] = img_col

    return info


def check_image_match(rows: list[dict], img_col: str, image_dirs: list[Path]) -> dict:
    """Check how many images referenced in CSV exist on disk."""
    if not image_dirs:
        return {"available": 0, "missing": 0, "total": 0}

    img_dir = image_dirs[0]
    available = 0
    missing = 0
    for row in rows:
        img_name = row.get(img_col, "")
        if img_name and (img_dir / img_name).exists():
            available += 1
        else:
            missing += 1

    return {
        "available": available,
        "missing": missing,
        "total": available + missing,
        "img_dir": str(img_dir),
    }


# ---------------------------------------------------------------------------
# KADID inspection
# ---------------------------------------------------------------------------

def inspect_kadid() -> dict | None:
    """Inspect the KADID-10K dataset."""
    kadid_root = EXTRACTED_ROOT / "kadid10k" / "kadid10k"
    print("=" * 70)
    print("  KADID-10K Dataset Inspection")
    print("=" * 70)

    if not kadid_root.exists():
        print(f"  [X] Dataset root not found: {kadid_root}")
        return None

    print(f"  Root: {kadid_root}")
    print()

    # Tree
    print("  Directory tree:")
    print_tree(kadid_root, max_depth=2, prefix="    ")
    print()

    # CSV info
    csv_files = find_csv_files(kadid_root)
    if not csv_files:
        print("  [X] No CSV files found")
        return None

    kadid_info: dict = {}
    for csv_path in csv_files:
        print(f"  CSV file: {csv_path.name}")
        info = read_csv_info(csv_path)
        kadid_info.update(info)

        if "columns" in info:
            print(f"    Columns: {info['columns']}")
            print(f"    Total rows: {info['total_rows']}")
        if "sample_rows" in info:
            print("    Sample rows:")
            for i, row in enumerate(info["sample_rows"]):
                print(f"      Row {i}: {row}")
        if "score_col" in info:
            print(f"    Score column: '{info['score_col']}'")
            print(f"    Score range: [{info['score_min']:.4f}, {info['score_max']:.4f}]")
            print(f"    Score mean: {info['score_mean']:.4f}, std: {info['score_std']:.4f}")
        print()

    # Image dirs
    image_dirs = find_image_dirs(kadid_root)
    if image_dirs:
        for d in image_dirs:
            img_counts = count_images_in_dir(d)
            total = sum(img_counts.values())
            print(f"  Image directory: {d.relative_to(kadid_root)}")
            print(f"    Total images: {total}")
            if img_counts:
                print(f"    Extensions: {dict(img_counts)}")

    else:
        print("  [X] No image directories found")

    # Re-read rows for matching check
    csv_path = csv_files[0]
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    if "img_col" in kadid_info and image_dirs:
        match = check_image_match(all_rows, kadid_info["img_col"], image_dirs)
        kadid_info["available_images"] = match["available"]
        kadid_info["missing_images"] = match["missing"]
        print(f"\n  Image match: {match['available']}/{match['total']} available, {match['missing']} missing")

    return kadid_info


# ---------------------------------------------------------------------------
# KonIQ inspection
# ---------------------------------------------------------------------------

def inspect_koniq() -> dict | None:
    """Inspect the KonIQ-10K dataset."""
    print()
    print("=" * 70)
    print("  KonIQ-10K Dataset Inspection")
    print("=" * 70)

    # Search in multiple locations
    koniq_locations = [
        EXTRACTED_ROOT / "koniq10k",
        EXTRACTED_ROOT / "koniq-10k",
        EXTRACTED_ROOT / "KONIQ-10k",
    ]

    koniq_root = None
    for loc in koniq_locations:
        if loc.exists():
            koniq_root = loc
            break

    if koniq_root is None:
        print("  [X] KonIQ-10K dataset not found in any expected location:")
        for loc in koniq_locations:
            print(f"    - {loc}")
        print()
        print("  KONIQ IMAGE DATA NOT FOUND — TRAINING CANNOT USE THIS DATASET YET")
        return None

    print(f"  Root: {koniq_root}")
    print()

    # Tree
    print("  Directory tree:")
    print_tree(koniq_root, max_depth=2, prefix="    ")
    print()

    # CSV info
    csv_files = find_csv_files(koniq_root)
    if not csv_files:
        print("  [X] No CSV files found")
        return None

    koniq_info: dict = {}
    for csv_path in csv_files:
        print(f"  CSV file: {csv_path.name}")
        info = read_csv_info(csv_path)
        koniq_info.update(info)

        if "columns" in info:
            print(f"    Columns: {info['columns']}")
            print(f"    Total rows: {info['total_rows']}")
        if "sample_rows" in info:
            print("    Sample rows:")
            for i, row in enumerate(info["sample_rows"]):
                print(f"      Row {i}: {row}")
        if "score_col" in info:
            print(f"    Score column: '{info['score_col']}'")
            print(f"    Score range: [{info['score_min']:.4f}, {info['score_max']:.4f}]")
            print(f"    Score mean: {info['score_mean']:.4f}, std: {info['score_std']:.4f}")
        print()

    # Image dirs
    image_dirs = find_image_dirs(koniq_root)
    if image_dirs:
        for d in image_dirs:
            img_counts = count_images_in_dir(d)
            total = sum(img_counts.values())
            print(f"  Image directory: {d.relative_to(koniq_root)}")
            print(f"    Total images: {total}")
            if img_counts:
                print(f"    Extensions: {dict(img_counts)}")
    else:
        print("  [X] No image directories found")
        print()
        print("  KONIQ IMAGE DATA NOT FOUND — TRAINING CANNOT USE THIS DATASET YET")
        koniq_info["images_found"] = False
        return koniq_info

    koniq_info["images_found"] = True

    # Re-read rows for matching check
    csv_path = csv_files[0]
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    if "img_col" in koniq_info and image_dirs:
        match = check_image_match(all_rows, koniq_info["img_col"], image_dirs)
        koniq_info["available_images"] = match["available"]
        koniq_info["missing_images"] = match["missing"]
        print(f"\n  Image match: {match['available']}/{match['total']} available, {match['missing']} missing")

    return koniq_info


# ---------------------------------------------------------------------------
# Raw data inspection
# ---------------------------------------------------------------------------

def inspect_raw() -> None:
    """Inspect the datasets/raw/ directory."""
    print()
    print("=" * 70)
    print("  Raw Data Inspection (datasets/raw/)")
    print("=" * 70)

    if not RAW_ROOT.exists():
        print("  [X] datasets/raw/ not found")
        return

    print(f"  Root: {RAW_ROOT}")
    print()

    print("  Directory tree:")
    print_tree(RAW_ROOT, max_depth=2, prefix="    ")
    print()

    # Check zip files
    zip_files = list(RAW_ROOT.rglob("*.zip"))
    if zip_files:
        print("  ZIP files:")
        for zf in zip_files:
            size_gb = zf.stat().st_size / (1024**3)
            print(f"    {zf.relative_to(RAW_ROOT)} ({size_gb:.2f} GB)")
        print()

    # Check koniq_scores specifically
    koniq_scores_dir = RAW_ROOT / "koniq_scores"
    if koniq_scores_dir.exists():
        print("  KonIQ scores directory:")
        csv_files = find_csv_files(koniq_scores_dir)
        for csv_path in csv_files:
            info = read_csv_info(csv_path)
            print(f"    CSV: {csv_path.name}")
            if "columns" in info:
                print(f"      Columns: {info['columns']}")
                print(f"      Total rows: {info['total_rows']}")
            if "score_col" in info:
                print(f"      Score column: '{info['score_col']}'")
                print(f"      Score range: [{info['score_min']:.4f}, {info['score_max']:.4f}]")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("  VisionGuard — Dataset Inspection")
    print("=" * 70)
    print()

    # 1. Inspect raw directory
    inspect_raw()

    # 2. Inspect extracted KADID
    kadid_info = inspect_kadid()

    # 3. Inspect extracted KonIQ
    koniq_info = inspect_koniq()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    # KADID
    print()
    if kadid_info:
        kadid_count = kadid_info.get("total_rows", kadid_info.get("available_images", "N/A"))
        kadid_available = kadid_info.get("available_images", kadid_info.get("total_rows", 0))
        print(f"  KADID-10K sample count:    {kadid_count}")
        print(f"  KADID-10K available images: {kadid_available}")
        print(f"  KADID-10K score column:    {kadid_info.get('score_col', 'N/A')}")
        if kadid_info.get("score_min") is not None:
            print(f"  KADID-10K score range:     [{kadid_info['score_min']:.4f}, {kadid_info['score_max']:.4f}]")
    else:
        print("  KADID-10K: NOT FOUND")

    # KonIQ
    print()
    if koniq_info:
        koniq_count = koniq_info.get("total_rows", "N/A")
        koniq_available = koniq_info.get("available_images", 0)
        koniq_images_found = koniq_info.get("images_found", False)
        print(f"  KonIQ-10K sample count:    {koniq_count}")
        print(f"  KonIQ-10K available images: {koniq_available}")
        print(f"  KonIQ-10K image availability: {'YES' if koniq_images_found else 'NO'}")
        print(f"  KonIQ-10K metadata file:   {koniq_info.get('name', 'N/A')}")
        print(f"  KonIQ-10K score column:    {koniq_info.get('score_col', 'N/A')}")
        if koniq_info.get("score_min") is not None:
            print(f"  KonIQ-10K score range:     [{koniq_info['score_min']:.4f}, {koniq_info['score_max']:.4f}]")

        if not koniq_images_found:
            print()
            print("  [!] KONIQ IMAGE DATA NOT FOUND -- TRAINING CANNOT USE THIS DATASET YET")
    else:
        print("  KonIQ-10K: NOT FOUND")
        print()
        print("  [!] KONIQ IMAGE DATA NOT FOUND -- TRAINING CANNOT USE THIS DATASET YET")

    # Readiness
    print()
    print("-" * 70)
    datasets_ready = False
    if kadid_info and kadid_info.get("available_images", 0) > 0:
        if koniq_info and koniq_info.get("images_found", False) and koniq_info.get("available_images", 0) > 0:
            print("  [OK] Both datasets are ready to combine for training.")
            datasets_ready = True
        elif koniq_info and koniq_info.get("images_found", False):
            print("  [~] KADID ready, KonIQ has images but low match. Partial training possible.")
            datasets_ready = True
        else:
            print("  [~] Only KADID is available. Training will use KADID only.")
            datasets_ready = True
    else:
        print("  [X] No datasets are ready for training.")

    if datasets_ready:
        print()
        print("  Combined dataset can be loaded with:")
        print("    python -m ml.train")


if __name__ == "__main__":
    main()
