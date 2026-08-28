"""Tests for Phase 5 — Smart-City Benchmark Evaluator.

Covers:
  - Manifest loading and parsing
  - Correct image / context / dataset mapping
  - One-image evaluation (pipeline end-to-end)
  - Score ranges 0–100
  - Valid readiness statuses
  - Aggregation by context
  - Missing-image handling (graceful skip)
  - Output CSV and JSON creation
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure the project root is importable so we can reach benchmark.smart_city
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmark.smart_city.evaluate_benchmark import (
    aggregate_results,
    evaluate_image,
    load_manifest,
    run_benchmark,
    write_results_csv,
    write_summary_json,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MANIFEST = _PROJECT_ROOT / "datasets" / "smart_city_benchmark" / "manifest.csv"
_VALID_STATUSES = {
    "HIGHLY READY", "READY", "LIMITED READINESS",
    "NOT READY", "CRITICAL / REJECT",
}


# ===================================================================
# 1. Manifest loading
# ===================================================================

class TestManifestLoading:
    def test_manifest_exists(self):
        assert _MANIFEST.exists(), f"Manifest not found at {_MANIFEST}"

    def test_load_manifest_returns_list(self):
        rows = load_manifest()
        assert isinstance(rows, list)
        assert len(rows) == 600

    def test_load_manifest_rows_have_required_keys(self):
        rows = load_manifest()
        for row in rows:
            assert "image_path" in row
            assert "context" in row
            assert "dataset" in row

    def test_load_manifest_paths_are_resolved_to_absolute(self):
        rows = load_manifest()
        for row in rows:
            # On Windows, Path.resolve() may produce backslashes — that's
            # fine.  The important property is that the path is absolute.
            assert os.path.isabs(row["image_path"]), (
                f"Path not absolute: {row['image_path']}"
            )

    def test_load_manifest_paths_are_resolved(self):
        rows = load_manifest()
        for row in rows:
            assert os.path.isabs(row["image_path"]), (
                f"Path not absolute: {row['image_path']}"
            )

    def test_load_manifest_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_manifest(tmp_path / "nonexistent.csv")


# ===================================================================
# 2. Correct image / context / dataset mapping
# ===================================================================

class TestManifestMapping:
    def test_context_distribution(self):
        rows = load_manifest()
        ctx_counts: dict[str, int] = {}
        for row in rows:
            ctx_counts[row["context"]] = ctx_counts.get(row["context"], 0) + 1

        assert ctx_counts.get("CCTV Surveillance") == 100
        assert ctx_counts.get("Drone Imagery") == 200
        assert ctx_counts.get("Infrastructure Inspection") == 100
        assert ctx_counts.get("Traffic Monitoring") == 100
        assert ctx_counts.get("Low-Light Evaluation") == 100

    def test_dataset_distribution(self):
        rows = load_manifest()
        ds_counts: dict[str, int] = {}
        for row in rows:
            ds_counts[row["dataset"]] = ds_counts.get(row["dataset"], 0) + 1

        assert ds_counts.get("MOT17") == 100
        assert ds_counts.get("VisDrone") == 200
        assert ds_counts.get("RoadDefects-ISeg") == 100
        assert ds_counts.get("BDD100K") == 100
        assert ds_counts.get("LOL") == 100

    def test_cctv_images_are_jpg(self):
        rows = load_manifest()
        cctv_rows = [r for r in rows if r["context"] == "CCTV Surveillance"]
        for row in cctv_rows:
            assert row["image_path"].endswith(".jpg")

    def test_drone_images_are_jpg(self):
        rows = load_manifest()
        drone_rows = [r for r in rows if r["context"] == "Drone Imagery"]
        for row in drone_rows:
            assert row["image_path"].endswith(".jpg")

    def test_low_light_images_are_png(self):
        rows = load_manifest()
        ll_rows = [r for r in rows if r["context"] == "Low-Light Evaluation"]
        for row in ll_rows:
            assert row["image_path"].endswith(".png")


# ===================================================================
# 3. One-image evaluation
# ===================================================================

class TestSingleImageEvaluation:
    """Run the full pipeline on one real benchmark image and verify the contract."""

    @pytest.fixture(scope="class")
    def first_cctv_image(self):
        rows = load_manifest()
        cctv = [r for r in rows if r["context"] == "CCTV Surveillance"]
        return cctv[0]

    @pytest.fixture(scope="class")
    def first_drone_image(self):
        rows = load_manifest()
        drone = [r for r in rows if r["context"] == "Drone Imagery"]
        return drone[0]

    @pytest.fixture(scope="class")
    def first_infra_image(self):
        rows = load_manifest()
        infra = [r for r in rows if r["context"] == "Infrastructure Inspection"]
        return infra[0]

    @pytest.fixture(scope="class")
    def first_traffic_image(self):
        rows = load_manifest()
        traffic = [r for r in rows if r["context"] == "Traffic Monitoring"]
        return traffic[0]

    @pytest.fixture(scope="class")
    def first_lowlight_image(self):
        rows = load_manifest()
        ll = [r for r in rows if r["context"] == "Low-Light Evaluation"]
        return ll[0]

    # --- CCTV ---
    def test_cctv_result_has_required_keys(self, first_cctv_image):
        result = evaluate_image(first_cctv_image["image_path"], context="CCTV Surveillance")
        required = {
            "image_path", "context", "dataset",
            "quality_score", "analytics_readiness_score",
            "analytics_readiness_status", "detected_issues",
        }
        assert required.issubset(result.keys())

    def test_cctv_quality_score_in_range(self, first_cctv_image):
        result = evaluate_image(first_cctv_image["image_path"], context="CCTV Surveillance")
        assert 0 <= result["quality_score"] <= 100

    def test_cctv_readiness_score_in_range(self, first_cctv_image):
        result = evaluate_image(first_cctv_image["image_path"], context="CCTV Surveillance")
        assert 0 <= result["analytics_readiness_score"] <= 100

    def test_cctv_readiness_status_valid(self, first_cctv_image):
        result = evaluate_image(first_cctv_image["image_path"], context="CCTV Surveillance")
        assert result["analytics_readiness_status"] in _VALID_STATUSES

    # --- Drone ---
    def test_drone_quality_score_in_range(self, first_drone_image):
        result = evaluate_image(first_drone_image["image_path"], context="Drone Imagery")
        assert 0 <= result["quality_score"] <= 100

    def test_drone_readiness_status_valid(self, first_drone_image):
        result = evaluate_image(first_drone_image["image_path"], context="Drone Imagery")
        assert result["analytics_readiness_status"] in _VALID_STATUSES

    # --- Infrastructure ---
    def test_infra_quality_score_in_range(self, first_infra_image):
        result = evaluate_image(first_infra_image["image_path"], context="Infrastructure Inspection")
        assert 0 <= result["quality_score"] <= 100

    def test_infra_readiness_status_valid(self, first_infra_image):
        result = evaluate_image(first_infra_image["image_path"], context="Infrastructure Inspection")
        assert result["analytics_readiness_status"] in _VALID_STATUSES

    # --- Traffic ---
    def test_traffic_quality_score_in_range(self, first_traffic_image):
        result = evaluate_image(first_traffic_image["image_path"], context="Traffic Monitoring")
        assert 0 <= result["quality_score"] <= 100

    def test_traffic_readiness_status_valid(self, first_traffic_image):
        result = evaluate_image(first_traffic_image["image_path"], context="Traffic Monitoring")
        assert result["analytics_readiness_status"] in _VALID_STATUSES

    # --- Low-Light ---
    def test_lowlight_quality_score_in_range(self, first_lowlight_image):
        result = evaluate_image(first_lowlight_image["image_path"], context="CCTV Surveillance")
        assert 0 <= result["quality_score"] <= 100

    def test_lowlight_readiness_status_valid(self, first_lowlight_image):
        result = evaluate_image(first_lowlight_image["image_path"], context="CCTV Surveillance")
        assert result["analytics_readiness_status"] in _VALID_STATUSES


# ===================================================================
# 4. Score ranges 0-100
# ===================================================================

class TestScoreRanges:
    """Evaluate a sample of images and verify scores stay in [0, 100]."""

    @pytest.fixture(scope="class")
    def sample_results(self):
        """Evaluate one image per context (5 total)."""
        rows = load_manifest()
        seen_contexts: set[str] = set()
        sample = []
        for row in rows:
            if row["context"] not in seen_contexts:
                seen_contexts.add(row["context"])
                result = evaluate_image(row["image_path"], context=row["context"])
                result["dataset"] = row["dataset"]
                sample.append(result)
                if len(seen_contexts) == 5:
                    break
        return sample

    def test_all_quality_scores_in_range(self, sample_results):
        for r in sample_results:
            assert 0 <= r["quality_score"] <= 100, (
                f"quality_score out of range: {r['quality_score']} for {r['image_path']}"
            )

    def test_all_readiness_scores_in_range(self, sample_results):
        for r in sample_results:
            assert 0 <= r["analytics_readiness_score"] <= 100, (
                f"readiness_score out of range: {r['analytics_readiness_score']}"
            )

    def test_all_readiness_statuses_valid(self, sample_results):
        for r in sample_results:
            assert r["analytics_readiness_status"] in _VALID_STATUSES


# ===================================================================
# 5. Valid readiness statuses
# ===================================================================

class TestReadinessStatus:
    """Readiness status must map correctly to the score."""

    @pytest.fixture(scope="class")
    def sample_result(self):
        rows = load_manifest()
        return evaluate_image(rows[0]["image_path"], context=rows[0]["context"])

    def test_status_matches_score(self, sample_result):
        score = sample_result["analytics_readiness_score"]
        status = sample_result["analytics_readiness_status"]
        if score >= 80:
            assert status == "HIGHLY READY"
        elif score >= 60:
            assert status == "READY"
        elif score >= 40:
            assert status == "LIMITED READINESS"
        elif score >= 20:
            assert status == "NOT READY"
        else:
            assert status == "CRITICAL / REJECT"


# ===================================================================
# 6. Aggregation by context
# ===================================================================

class TestAggregation:
    def test_aggregate_produces_by_context_and_by_dataset(self):
        rows = load_manifest()
        # Evaluate only a small subset for speed
        results = []
        seen: set[str] = set()
        for row in rows:
            ctx = row["context"]
            if ctx not in seen:
                seen.add(ctx)
                r = evaluate_image(row["image_path"], context=ctx)
                r["dataset"] = row["dataset"]
                results.append(r)
                if len(seen) == 5:
                    break

        agg = aggregate_results(results)
        assert "by_context" in agg
        assert "by_dataset" in agg
        assert len(agg["by_context"]) == 5

    def test_aggregate_average_scores_are_in_range(self):
        rows = load_manifest()
        results = []
        seen: set[str] = set()
        for row in rows:
            ctx = row["context"]
            if ctx not in seen:
                seen.add(ctx)
                r = evaluate_image(row["image_path"], context=ctx)
                r["dataset"] = row["dataset"]
                results.append(r)
                if len(seen) == 5:
                    break

        agg = aggregate_results(results)
        for ctx, stats in agg["by_context"].items():
            assert 0 <= stats["average_quality_score"] <= 100, (
                f"{ctx}: avg quality {stats['average_quality_score']}"
            )
            assert 0 <= stats["average_analytics_readiness_score"] <= 100, (
                f"{ctx}: avg readiness {stats['average_analytics_readiness_score']}"
            )

    def test_aggregate_image_counts_sum_correctly(self):
        results = [
            {"context": "A", "dataset": "X", "quality_score": 50,
             "analytics_readiness_score": 40, "analytics_readiness_status": "LIMITED READINESS",
             "detected_issues": []},
            {"context": "A", "dataset": "X", "quality_score": 60,
             "analytics_readiness_score": 50, "analytics_readiness_status": "LIMITED READINESS",
             "detected_issues": []},
            {"context": "B", "dataset": "Y", "quality_score": 90,
             "analytics_readiness_score": 85, "analytics_readiness_status": "HIGHLY READY",
             "detected_issues": [{"type": "severe_blur"}]},
        ]
        agg = aggregate_results(results)
        assert agg["by_context"]["A"]["image_count"] == 2
        assert agg["by_context"]["B"]["image_count"] == 1
        assert agg["by_dataset"]["X"]["image_count"] == 2
        assert agg["by_dataset"]["Y"]["image_count"] == 1

    def test_aggregate_status_distribution(self):
        results = [
            {"context": "A", "dataset": "X", "quality_score": 90,
             "analytics_readiness_score": 90, "analytics_readiness_status": "HIGHLY READY",
             "detected_issues": []},
            {"context": "A", "dataset": "X", "quality_score": 70,
             "analytics_readiness_score": 65, "analytics_readiness_status": "READY",
             "detected_issues": []},
        ]
        agg = aggregate_results(results)
        dist = agg["by_context"]["A"]["readiness_status_distribution"]
        assert dist["HIGHLY READY"] == 1
        assert dist["READY"] == 1

    def test_aggregate_issue_counts(self):
        results = [
            {"context": "A", "dataset": "X", "quality_score": 50,
             "analytics_readiness_score": 40, "analytics_readiness_status": "LIMITED READINESS",
             "detected_issues": [
                 {"type": "severe_blur"},
                 {"type": "underexposure"},
             ]},
            {"context": "A", "dataset": "X", "quality_score": 60,
             "analytics_readiness_score": 50, "analytics_readiness_status": "LIMITED READINESS",
             "detected_issues": [
                 {"type": "severe_blur"},
             ]},
        ]
        agg = aggregate_results(results)
        counts = agg["by_context"]["A"]["issue_counts"]
        assert counts["severe_blur"] == 2
        assert counts["underexposure"] == 1


# ===================================================================
# 7. Missing image handling
# ===================================================================

class TestMissingImageHandling:
    def test_missing_image_returns_file_not_found(self, tmp_path):
        fake = str(tmp_path / "nonexistent.jpg")
        with pytest.raises(FileNotFoundError):
            evaluate_image(fake, context="CCTV Surveillance")

    def test_invalid_image_returns_value_error(self, tmp_path):
        bad = tmp_path / "bad.jpg"
        bad.write_bytes(b"not an image data")
        with pytest.raises((ValueError, Exception)):
            evaluate_image(str(bad), context="CCTV Surveillance")

    def test_run_benchmark_skips_missing_images(self, tmp_path):
        """Build a tiny manifest with one real and one missing image."""
        rows = load_manifest()
        real_path = rows[0]["image_path"]
        fake_path = str(tmp_path / "does_not_exist.jpg")

        # Create a temp manifest
        manifest_file = tmp_path / "test_manifest.csv"
        with open(manifest_file, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["image_path", "context", "dataset"])
            writer.writerow([real_path, "CCTV Surveillance", "MOT17"])
            writer.writerow([fake_path, "CCTV Surveillance", "MOT17"])

        results_csv = tmp_path / "results.csv"
        summary_json = tmp_path / "summary.json"

        report = run_benchmark(
            manifest_path=str(manifest_file),
            results_csv=str(results_csv),
            summary_json=str(summary_json),
        )
        assert report["total"] == 2
        assert report["processed"] == 1
        assert report["skipped"] == 1


# ===================================================================
# 8. Output CSV/JSON creation
# ===================================================================

class TestOutputCreation:
    def test_write_results_csv_creates_file(self, tmp_path):
        out = tmp_path / "results.csv"
        results = [
            {
                "image_path": "/fake/img.jpg",
                "context": "CCTV Surveillance",
                "dataset": "MOT17",
                "quality_score": 85,
                "analytics_readiness_score": 80,
                "analytics_readiness_status": "HIGHLY READY",
                "detected_issues": [{"type": "severe_blur"}],
            },
        ]
        path = write_results_csv(results, out)
        assert path.exists()
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["context"] == "CCTV Surveillance"
        assert rows[0]["quality_score"] == "85"

    def test_write_summary_json_creates_file(self, tmp_path):
        out = tmp_path / "summary.json"
        summary = {
            "by_context": {"CCTV Surveillance": {"image_count": 1}},
            "by_dataset": {"MOT17": {"image_count": 1}},
        }
        path = write_summary_json(summary, out)
        assert path.exists()
        with open(path) as fh:
            loaded = json.load(fh)
        assert "by_context" in loaded
        assert loaded["by_context"]["CCTV Surveillance"]["image_count"] == 1

    def test_run_benchmark_creates_both_files(self, tmp_path):
        rows = load_manifest()
        # Use only 1 image to keep the test fast
        manifest_file = tmp_path / "mini_manifest.csv"
        with open(manifest_file, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["image_path", "context", "dataset"])
            writer.writerow([rows[0]["image_path"], rows[0]["context"], rows[0]["dataset"]])

        results_csv = tmp_path / "results.csv"
        summary_json = tmp_path / "summary.json"

        report = run_benchmark(
            manifest_path=str(manifest_file),
            results_csv=str(results_csv),
            summary_json=str(summary_json),
        )

        assert report["processed"] == 1
        assert results_csv.exists()
        assert summary_json.exists()

        # Verify CSV content
        with open(results_csv, newline="") as fh:
            reader = csv.DictReader(fh)
            csv_rows = list(reader)
        assert len(csv_rows) == 1
        assert csv_rows[0]["context"] == rows[0]["context"]

        # Verify JSON content
        with open(summary_json) as fh:
            loaded = json.load(fh)
        assert loaded["processed_images"] == 1
        assert "by_context" in loaded


# ===================================================================
# 9. All six contexts produce results
# ===================================================================

class TestAllContexts:
    """Each of the six supported contexts should produce valid results."""

    @pytest.mark.parametrize("ctx", [
        "CCTV Surveillance",
        "Traffic Monitoring",
        "Crowd Monitoring",
        "Drone Imagery",
        "Infrastructure Inspection",
        "Smart Campus",
    ])
    def test_context_accepted_and_scored(self, ctx):
        """Use a generic benchmark image; the context is a pipeline parameter."""
        rows = load_manifest()
        img_path = rows[0]["image_path"]
        result = evaluate_image(img_path, context=ctx)
        assert result["context"] == ctx
        assert 0 <= result["quality_score"] <= 100
        assert 0 <= result["analytics_readiness_score"] <= 100
        assert result["analytics_readiness_status"] in _VALID_STATUSES


# ===================================================================
# 10. End-to-end run on first few images only
# ===================================================================

class TestMiniEndToEnd:
    """Run the full benchmark on a tiny manifest (3 images) to verify the
    entire pipeline without waiting for all 600 images."""

    def test_mini_benchmark_completes(self, tmp_path):
        rows = load_manifest()
        manifest_file = tmp_path / "mini.csv"
        with open(manifest_file, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["image_path", "context", "dataset"])
            # One from each of 3 different contexts
            writer.writerow([rows[0]["image_path"], rows[0]["context"], rows[0]["dataset"]])
            cctv_done = False
            drone_done = False
            infra_done = False
            for r in rows:
                if r["context"] == "Drone Imagery" and not drone_done:
                    writer.writerow([r["image_path"], r["context"], r["dataset"]])
                    drone_done = True
                elif r["context"] == "Infrastructure Inspection" and not infra_done:
                    writer.writerow([r["image_path"], r["context"], r["dataset"]])
                    infra_done = True
                elif r["context"] == "Traffic Monitoring" and not cctv_done:
                    writer.writerow([r["image_path"], r["context"], r["dataset"]])
                    cctv_done = True
                if cctv_done and drone_done and infra_done:
                    break

        results_csv = tmp_path / "results.csv"
        summary_json = tmp_path / "summary.json"

        report = run_benchmark(
            manifest_path=str(manifest_file),
            results_csv=str(results_csv),
            summary_json=str(summary_json),
        )

        assert report["total"] == 4
        assert report["processed"] == 4
        assert report["skipped"] == 0
        assert results_csv.exists()
        assert summary_json.exists()

        # Verify summary structure
        summary = report["summary"]
        assert "by_context" in summary
        assert "by_dataset" in summary
        assert summary["processed_images"] == 4
        assert summary["skipped_images"] == 0
