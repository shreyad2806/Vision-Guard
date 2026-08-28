"""Phase 5 — Smart-City Benchmark Evaluator.

Reads a manifest of images, runs the full VisionGuard pipeline on each,
aggregates results by context and dataset, and writes:
  - benchmark/smart_city/phase5_results.csv   (per-image detail)
  - benchmark/smart_city/phase5_summary.json  (aggregated summary)

Usage (from project root)::

    python -m benchmark.smart_city.evaluate_benchmark

Or import and call ``run_benchmark()`` / ``load_manifest()`` directly.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Resolve imports — the evaluator lives in benchmark/, the ML code in backend/
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent          # …/VisionGuard
_BACKEND_DIR = _PROJECT_ROOT / "backend"

# Make backend importable (apps.ml, apps.services, …)
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import cv2  # noqa: E402  (must come after sys.path manipulation)

from apps.ml.feature_extractor import extract_model_features, extract_all_features  # noqa: E402
from apps.ml.inference import predict_quality  # noqa: E402
from apps.ml.readiness import calculate_analytics_readiness  # noqa: E402
from apps.ml.context_definitions import get_context_impacts, SUPPORTED_CONTEXTS  # noqa: E402

logger = logging.getLogger(__name__)

# Output paths (relative to project root)
_RESULTS_CSV = _PROJECT_ROOT / "benchmark" / "smart_city" / "phase5_results.csv"
_SUMMARY_JSON = _PROJECT_ROOT / "benchmark" / "smart_city" / "phase5_summary.json"
_MANIFEST_CSV = _PROJECT_ROOT / "datasets" / "smart_city_benchmark" / "manifest.csv"


# ===================================================================
# 1. Manifest loading
# ===================================================================

def load_manifest(manifest_path: str | Path | None = None) -> list[dict]:
    """Read the benchmark manifest CSV.

    Each row has: ``image_path``, ``context``, ``dataset``.

    Returns a list of dicts.  Paths are normalised to forward slashes
    and resolved relative to the project root so they work on every OS.
    """
    path = Path(manifest_path) if manifest_path else _MANIFEST_CSV
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw_path = row["image_path"].strip().strip('"')
            # Normalise Windows back-slashes → forward slashes
            raw_path = raw_path.replace("\\", "/")
            # Resolve relative to project root
            resolved = (_PROJECT_ROOT / raw_path).resolve()
            rows.append({
                "image_path": str(resolved),
                "context": row["context"].strip().strip('"'),
                "dataset": row["dataset"].strip().strip('"'),
            })
    return rows


# ===================================================================
# 2. Single-image evaluation
# ===================================================================

def evaluate_image(
    image_path: str,
    context: str = "CCTV Surveillance",
) -> dict[str, Any]:
    """Run the VisionGuard pipeline on a single image.

    Returns a dict with the result fields required by Phase 5.
    Raises ``FileNotFoundError`` if the image does not exist, or
    ``ValueError`` if OpenCV cannot decode it.
    """
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {image_path}")

    model_features = extract_model_features(img)
    all_features = extract_all_features(img)

    result = predict_quality(model_features, all_features, context=context)

    return {
        "image_path": str(p),
        "context": context,
        "dataset": "",  # filled by the caller from the manifest
        "quality_score": result["quality_score"],
        "analytics_readiness_score": result["analytics_readiness_score"],
        "analytics_readiness_status": result["analytics_readiness_status"],
        "detected_issues": result["issues"],
    }


# ===================================================================
# 3. Aggregation
# ===================================================================

def aggregate_results(results: list[dict]) -> dict[str, Any]:
    """Aggregate per-image results by context and dataset.

    Returns a dict with two keys: ``by_context`` and ``by_dataset``.
    Each maps a label to a summary dict with counts, averages, status
    distribution, and issue counts.
    """
    VALID_STATUSES = {
        "HIGHLY READY", "READY", "LIMITED READINESS",
        "NOT READY", "CRITICAL / REJECT",
    }

    # --- by context ---
    ctx_groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        ctx_groups[r["context"]].append(r)

    by_context: dict[str, dict] = {}
    for ctx, items in sorted(ctx_groups.items()):
        q_scores = [i["quality_score"] for i in items]
        a_scores = [i["analytics_readiness_score"] for i in items]
        status_dist: dict[str, int] = defaultdict(int)
        issue_counts: dict[str, int] = defaultdict(int)
        for i in items:
            status_dist[i["analytics_readiness_status"]] += 1
            for iss in i["detected_issues"]:
                issue_counts[iss["type"]] += 1
        by_context[ctx] = {
            "image_count": len(items),
            "average_quality_score": round(sum(q_scores) / len(q_scores), 2) if q_scores else 0.0,
            "average_analytics_readiness_score": round(sum(a_scores) / len(a_scores), 2) if a_scores else 0.0,
            "readiness_status_distribution": dict(status_dist),
            "issue_counts": dict(issue_counts),
        }

    # --- by dataset ---
    ds_groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        ds_groups[r["dataset"]].append(r)

    by_dataset: dict[str, dict] = {}
    for ds, items in sorted(ds_groups.items()):
        q_scores = [i["quality_score"] for i in items]
        a_scores = [i["analytics_readiness_score"] for i in items]
        status_dist2: dict[str, int] = defaultdict(int)
        issue_counts2: dict[str, int] = defaultdict(int)
        for i in items:
            status_dist2[i["analytics_readiness_status"]] += 1
            for iss in i["detected_issues"]:
                issue_counts2[iss["type"]] += 1
        by_dataset[ds] = {
            "image_count": len(items),
            "average_quality_score": round(sum(q_scores) / len(q_scores), 2) if q_scores else 0.0,
            "average_analytics_readiness_score": round(sum(a_scores) / len(a_scores), 2) if a_scores else 0.0,
            "readiness_status_distribution": dict(status_dist2),
            "issue_counts": dict(issue_counts2),
        }

    return {"by_context": by_context, "by_dataset": by_dataset}


# ===================================================================
# 4. Output writers
# ===================================================================

def _serialize_issues(issues: list[dict]) -> str:
    """Compact JSON serialisation of detected issues for the CSV."""
    return json.dumps(issues, separators=(",", ":"))


def write_results_csv(results: list[dict], output_path: str | Path | None = None) -> Path:
    """Write per-image results to CSV."""
    out = Path(output_path) if output_path else _RESULTS_CSV
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_path", "context", "dataset",
        "quality_score", "analytics_readiness_score",
        "analytics_readiness_status", "detected_issues",
    ]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = dict(r)
            row["detected_issues"] = _serialize_issues(row["detected_issues"])
            writer.writerow(row)
    return out


def write_summary_json(summary: dict, output_path: str | Path | None = None) -> Path:
    """Write aggregated summary to JSON."""
    out = Path(output_path) if output_path else _SUMMARY_JSON
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    return out


# ===================================================================
# 5. Main benchmark runner
# ===================================================================

def run_benchmark(
    manifest_path: str | Path | None = None,
    results_csv: str | Path | None = None,
    summary_json: str | Path | None = None,
) -> dict[str, Any]:
    """Execute the full benchmark and return the summary dict.

    Parameters
    ----------
    manifest_path : path to the manifest CSV (default: datasets/…/manifest.csv)
    results_csv   : output path for per-image CSV (default: benchmark/smart_city/phase5_results.csv)
    summary_json  : output path for summary JSON  (default: benchmark/smart_city/phase5_summary.json)

    Returns
    -------
    dict with keys: ``total``, ``processed``, ``skipped``, ``summary``.
    """
    manifest = load_manifest(manifest_path)
    total = len(manifest)
    logger.info("Loaded %d images from manifest", total)

    results: list[dict] = []
    skipped: list[dict] = []

    for idx, entry in enumerate(manifest, start=1):
        img_path = entry["image_path"]
        ctx = entry["context"]
        ds = entry["dataset"]

        try:
            row = evaluate_image(img_path, context=ctx)
            row["dataset"] = ds
            results.append(row)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("[%d/%d] SKIP %s — %s", idx, total, img_path, exc)
            skipped.append({"image_path": img_path, "error": str(exc)})

        if idx % 100 == 0 or idx == total:
            logger.info("  processed %d / %d  (skipped %d)", idx, total, len(skipped))

    # Aggregate
    summary = aggregate_results(results)
    summary["total_images"] = total
    summary["processed_images"] = len(results)
    summary["skipped_images"] = len(skipped)
    summary["skipped_details"] = skipped

    # Write outputs
    csv_path = write_results_csv(results, results_csv)
    json_path = write_summary_json(summary, summary_json)

    logger.info("Results CSV  → %s  (%d rows)", csv_path, len(results))
    logger.info("Summary JSON → %s", json_path)

    return {
        "total": total,
        "processed": len(results),
        "skipped": len(skipped),
        "summary": summary,
        "csv_path": str(csv_path),
        "json_path": str(json_path),
    }


# ===================================================================
# CLI entry point
# ===================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    t0 = time.perf_counter()
    report = run_benchmark()
    elapsed = time.perf_counter() - t0
    print(f"\nBenchmark complete: {report['processed']}/{report['total']} images "
          f"({report['skipped']} skipped) in {elapsed:.1f}s")
    sys.exit(0 if report["skipped"] == 0 else 1)
