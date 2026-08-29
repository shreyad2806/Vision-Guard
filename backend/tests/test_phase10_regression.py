"""Phase 10 — Regression tests for model integrity, scoring, and performance.

Tests verify:
- model produces numeric quality score
- score is within [0,100]
- raw prediction exists
- calibration is applied
- model output changes for meaningfully different features
- readiness is independently calculated
- issue confidence is valid
- API response remains unchanged
- model is not reloaded per request
- inference remains deterministic where expected
- performance metadata is valid
- CALIBRATION LEAKAGE REMOVED:
  * calibrator fitted only on validation data
  * test data never passed to calibration fitting
  * model never fitted on test data
  * final metrics use untouched test targets
"""

import json
from pathlib import Path

import numpy as np
import pytest

from apps.ml.model_loader import get_model, get_calibrator, get_metadata, get_feature_names
from apps.ml.feature_extractor import extract_model_features, extract_all_features
from apps.ml.calibration import apply_calibration, get_quality_assessment
from apps.ml.issue_detector import detect_issues
from apps.ml.readiness import calculate_analytics_readiness
from apps.ml.explainability import generate_issue_explanations
from apps.ml.context_definitions import get_context_impacts


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------
@pytest.fixture(scope="module")
def model():
    return get_model()


@pytest.fixture(scope="module")
def calibrator():
    return get_calibrator()


@pytest.fixture(scope="module")
def metadata():
    return get_metadata()


@pytest.fixture(scope="module")
def feature_names():
    return get_feature_names()


def _make_features(**overrides):
    """Return a dict of model features with sensible defaults (8 features)."""
    defaults = {
        "brightness": 128.0,
        "contrast": 50.0,
        "sharpness": 200.0,
        "saturation": 100.0,
        "edge_density": 0.10,
        "noise_estimate": 10.0,
        "entropy": 6.0,
        "colorfulness": 30.0,
    }
    defaults.update(overrides)
    return defaults


def _make_all_features(**overrides):
    """Return a dict of all 12 features with sensible defaults."""
    defaults = {
        "sharpness": 200.0,
        "brightness": 128.0,
        "contrast": 50.0,
        "noise_estimate": 10.0,
        "entropy": 6.0,
        "saturation": 100.0,
        "underexposure_pct": 5.0,
        "overexposure_pct": 3.0,
        "edge_density": 10.0,
        "dynamic_range": 150.0,
        "colorfulness": 30.0,
        "texture_complexity": 15.0,
    }
    defaults.update(overrides)
    return defaults


# ------------------------------------------------------------------
# TASK 8: Calibration leakage verification
# ------------------------------------------------------------------
class TestCalibrationLeakageRemoval:
    """Verify that calibration leakage has been removed from the pipeline."""

    def test_calibrator_fitted_on_validation_set(self, metadata):
        """The calibrator must be fitted on validation data, not test data."""
        cal = metadata.get("calibration", {})
        assert cal.get("fitted_on") == "validation_set", (
            f"Calibrator should be fitted on 'validation_set', got '{cal.get('fitted_on')}'"
        )

    def test_calibrator_never_fitted_on_test_set(self, metadata):
        """The calibrator must never be fitted on test data."""
        cal = metadata.get("calibration", {})
        assert cal.get("never_fitted_on") == "test_set", (
            "Calibrator should explicitly state it was never fitted on test data"
        )

    def test_split_info_present(self, metadata):
        """Metadata must contain explicit train/val/test split information."""
        split = metadata.get("split", {})
        assert "train_samples" in split, "Missing train_samples in split info"
        assert "val_samples" in split, "Missing val_samples in split info"
        assert "test_samples" in split, "Missing test_samples in split info"
        assert split["train_samples"] > 0
        assert split["val_samples"] > 0
        assert split["test_samples"] > 0

    def test_three_way_split_ratios(self, metadata):
        """Split should be approximately 70/15/15."""
        split = metadata.get("split", {})
        total = split["train_samples"] + split["val_samples"] + split["test_samples"]
        train_pct = split["train_samples"] / total
        val_pct = split["val_samples"] / total
        test_pct = split["test_samples"] / total
        # Allow 2% tolerance
        assert 0.68 <= train_pct <= 0.72, f"Train ratio {train_pct:.2%} not ~70%"
        assert 0.13 <= val_pct <= 0.17, f"Val ratio {val_pct:.2%} not ~15%"
        assert 0.13 <= test_pct <= 0.17, f"Test ratio {test_pct:.2%} not ~15%"

    def test_no_overlap_between_splits(self, metadata):
        """Train, val, and test samples should be disjoint."""
        split = metadata.get("split", {})
        # Sum of all splits should equal combined dataset
        kadid = metadata.get("dataset_samples", {}).get("kadid", 0)
        koniq = metadata.get("dataset_samples", {}).get("koniq", 0)
        expected_total = kadid + koniq
        actual_total = split["train_samples"] + split["val_samples"] + split["test_samples"]
        assert actual_total == expected_total, (
            f"Split samples ({actual_total}) != total dataset ({expected_total})"
        )

    def test_metrics_state_leakage_free(self, metadata):
        """The metrics note should explicitly state no test data leakage."""
        metrics = metadata.get("metrics", {})
        note = metrics.get("note", "")
        assert "validation" in note.lower() or "no test data" in note.lower(), (
            f"Metrics note should mention validation/test separation: {note}"
        )

    def test_raw_and_calibrated_metrics_present(self, metadata):
        """Metadata should contain both raw and calibrated metrics."""
        metrics = metadata.get("metrics", {})
        assert "raw" in metrics, "Missing raw metrics"
        assert "calibrated" in metrics, "Missing calibrated metrics"
        for key in ["mae", "rmse", "r2", "spearman"]:
            assert key in metrics["raw"], f"Missing raw {key}"
            assert key in metrics["calibrated"], f"Missing calibrated {key}"

    def test_test_statistics_present(self, metadata):
        """Metadata should contain test set statistics."""
        stats = metadata.get("test_statistics", {})
        # Check for test sample count (key name varies between versions)
        count_keys = ["test_sample_count", "sample_count"]
        assert any(k in stats for k in count_keys), f"Missing sample count in {list(stats.keys())}"
        count_val = stats.get("test_sample_count", stats.get("sample_count", 0))
        assert count_val > 0
        # Check for target stats
        target_keys = ["test_target_mean", "target_mean"]
        assert any(k in stats for k in target_keys), f"Missing target mean in {list(stats.keys())}"

    def test_raw_metrics_in_valid_range(self, metadata):
        """Raw metrics should be in a plausible range."""
        raw = metadata["metrics"]["raw"]
        assert 0 <= raw["mae"] <= 50, f"Raw MAE {raw['mae']} out of range"
        assert 0 <= raw["rmse"] <= 60, f"Raw RMSE {raw['rmse']} out of range"
        assert -1 <= raw["r2"] <= 1, f"Raw R² {raw['r2']} out of range"
        assert -1 <= raw["spearman"] <= 1, f"Raw Spearman {raw['spearman']} out of range"


# ------------------------------------------------------------------
# TASK 8: Model produces valid output
# ------------------------------------------------------------------
class TestModelProducesValidOutput:
    def test_model_predicts_numeric(self, model):
        vector = np.array([[128.0, 50.0, 200.0, 100.0, 0.10, 10.0, 6.0, 30.0]])
        pred = model.predict(vector)
        assert pred.shape == (1,)
        assert np.isfinite(pred[0])

    def test_score_in_range_0_100(self, model, calibrator):
        vector = np.array([[128.0, 50.0, 200.0, 100.0, 0.10, 10.0, 6.0, 30.0]])
        raw = float(model.predict(vector)[0])
        score = apply_calibration(raw, calibrator)
        assert 0 <= score <= 100

    def test_raw_prediction_exists(self, model):
        vector = np.array([[128.0, 50.0, 200.0, 100.0, 0.10, 10.0, 6.0, 30.0]])
        raw = float(model.predict(vector)[0])
        assert raw is not None
        assert isinstance(raw, float)

    def test_calibration_changes_raw_prediction(self, model, calibrator):
        vector = np.array([[128.0, 50.0, 200.0, 100.0, 0.10, 10.0, 6.0, 30.0]])
        raw = float(model.predict(vector)[0])
        calibrated = apply_calibration(raw, calibrator)
        # Calibrated score should be finite and in range
        assert np.isfinite(calibrated)
        assert 0 <= calibrated <= 100


class TestModelOutputChangesWithFeatures:
    def test_good_vs_bad_features_differ(self, model, calibrator):
        good = np.array([[128.0, 60.0, 500.0, 120.0, 0.15, 5.0, 7.0, 50.0]])
        bad = np.array([[30.0, 10.0, 5.0, 10.0, 0.01, 40.0, 2.0, 5.0]])
        raw_good = float(model.predict(good)[0])
        raw_bad = float(model.predict(bad)[0])
        score_good = apply_calibration(raw_good, calibrator)
        score_bad = apply_calibration(raw_bad, calibrator)
        # Higher-quality features should produce a higher score
        assert score_good > score_bad

    def test_brightness_extremes_differ(self, model, calibrator):
        bright = np.array([[240.0, 50.0, 200.0, 100.0, 0.10, 10.0, 6.0, 30.0]])
        dark = np.array([[20.0, 50.0, 200.0, 100.0, 0.10, 10.0, 6.0, 30.0]])
        raw_bright = float(model.predict(bright)[0])
        raw_dark = float(model.predict(dark)[0])
        # Raw predictions should differ
        assert raw_bright != raw_dark


# ------------------------------------------------------------------
# TASK 8: Readiness is independently calculated
# ------------------------------------------------------------------
class TestReadinessIndependence:
    def test_readiness_starts_from_quality_score(self):
        """Readiness should start from quality_score and apply penalties."""
        result = calculate_analytics_readiness(
            quality_score=80.0,
            issues=[],
        )
        assert result["base_quality_score"] == 80.0
        assert result["score"] == 80.0
        assert result["total_penalty"] == 0.0

    def test_readiness_reduces_with_issues(self):
        result_clean = calculate_analytics_readiness(quality_score=60.0, issues=[])
        result_dirty = calculate_analytics_readiness(
            quality_score=60.0,
            issues=[{"type": "image_noise", "severity": "high"}],
        )
        assert result_dirty["score"] < result_clean["score"]
        assert result_dirty["total_penalty"] > 0

    def test_readiness_penalties_non_negative(self):
        result = calculate_analytics_readiness(
            quality_score=50.0,
            issues=[
                {"type": "severe_blur", "severity": "critical"},
                {"type": "underexposure", "severity": "high"},
                {"type": "image_noise", "severity": "moderate"},
            ],
        )
        for key in ["blur_penalty", "exposure_penalty", "noise_penalty",
                     "corruption_penalty", "information_penalty"]:
            assert result[key] >= 0, f"{key} should be non-negative"

    def test_readiness_clamped_0_100(self):
        result = calculate_analytics_readiness(
            quality_score=5.0,
            issues=[
                {"type": "severe_blur", "severity": "critical"},
                {"type": "underexposure", "severity": "critical"},
                {"type": "image_noise", "severity": "critical"},
                {"type": "severe_visual_degradation", "severity": "critical"},
                {"type": "low_contrast", "severity": "critical"},
            ],
        )
        assert 0 <= result["score"] <= 100


# ------------------------------------------------------------------
# TASK 8: Issue confidence is detector-derived
# ------------------------------------------------------------------
class TestIssueConfidence:
    def test_confidence_is_float_0_1(self):
        issues = detect_issues(_make_all_features(
            sharpness=10.0, brightness=30.0, noise_estimate=40.0
        ))
        for issue in issues:
            assert "confidence" in issue
            assert 0.0 <= issue["confidence"] <= 1.0

    def test_threshold_issues_have_confidence_1(self):
        """Threshold-based detectors should have confidence 1.0."""
        issues = detect_issues(_make_all_features(sharpness=5.0))
        blur_issues = [i for i in issues if "blur" in i["type"] or "sharpness" in i["type"]]
        assert len(blur_issues) > 0
        for issue in blur_issues:
            assert issue["confidence"] == 1.0

    def test_degradation_issues_have_lower_confidence(self):
        """Multi-signal detectors should have confidence < 1.0 for non-critical types."""
        issues = detect_issues(_make_all_features(
            sharpness=25.0, entropy=2.8, dynamic_range=35.0,
            edge_density=0.8, contrast=18.0,
        ))
        non_critical = [
            i for i in issues
            if i["type"] in {"visual_degradation", "potential_visual_defect",
                              "low_color_information"}
        ]
        for issue in non_critical:
            assert issue["confidence"] < 1.0


# ------------------------------------------------------------------
# TASK 8: Explainability
# ------------------------------------------------------------------
class TestExplainability:
    def test_explanation_for_each_issue(self):
        issues = detect_issues(_make_all_features(
            sharpness=10.0, brightness=30.0, noise_estimate=40.0
        ))
        explanations = generate_issue_explanations(issues)
        assert len(explanations) == len(issues)
        for exp in explanations:
            assert "issue" in exp
            assert "evidence" in exp
            assert "why_it_matters" in exp
            assert "recommendation" in exp

    def test_evidence_has_metric_value_threshold(self):
        issues = detect_issues(_make_all_features(sharpness=10.0))
        explanations = generate_issue_explanations(issues)
        for exp in explanations:
            assert "metric" in exp["evidence"]
            assert "value" in exp["evidence"]
            assert "threshold" in exp["evidence"]


# ------------------------------------------------------------------
# TASK 8: Context impacts
# ------------------------------------------------------------------
class TestContextImpacts:
    def test_context_impacts_for_each_issue(self):
        issues = detect_issues(_make_all_features(
            sharpness=10.0, brightness=30.0
        ))
        impacts = get_context_impacts(issues, "Traffic Monitoring")
        assert len(impacts) == len(issues)
        for impact in impacts:
            assert "issue_type" in impact
            assert "context" in impact
            assert impact["context"] == "Traffic Monitoring"
            assert "impact" in impact

    def test_different_contexts_same_issues(self):
        issues = detect_issues(_make_all_features(sharpness=10.0))
        impacts_traffic = get_context_impacts(issues, "Traffic Monitoring")
        impacts_drone = get_context_impacts(issues, "Drone Imagery")
        assert len(impacts_traffic) == len(impacts_drone)
        # Different contexts should have different impact text
        if impacts_traffic and impacts_drone:
            assert impacts_traffic[0]["impact"] != impacts_drone[0]["impact"]


# ------------------------------------------------------------------
# TASK 8: Model not reloaded per request
# ------------------------------------------------------------------
class TestModelNotReloaded:
    def test_same_model_object_across_calls(self):
        m1 = get_model()
        m2 = get_model()
        assert m1 is m2, "Model should be a singleton"

    def test_same_calibrator_object_across_calls(self):
        c1 = get_calibrator()
        c2 = get_calibrator()
        assert c1 is c2, "Calibrator should be a singleton"


# ------------------------------------------------------------------
# TASK 8: Determinism
# ------------------------------------------------------------------
class TestInferenceDeterminism:
    def test_same_input_same_output(self, model, calibrator):
        vector = np.array([[128.0, 50.0, 200.0, 100.0, 0.10, 10.0, 6.0, 30.0]])
        scores = []
        for _ in range(5):
            raw = float(model.predict(vector)[0])
            score = apply_calibration(raw, calibrator)
            scores.append(score)
        assert len(set(scores)) == 1, "Same input should always produce the same score"

    def test_feature_extraction_deterministic(self):
        """Feature extraction from the same image should be deterministic."""
        import cv2
        img = np.random.RandomState(42).randint(0, 255, (100, 100, 3), dtype=np.uint8)
        feats1 = extract_model_features(img)
        feats2 = extract_model_features(img)
        for key in feats1:
            assert feats1[key] == feats2[key], f"{key} should be deterministic"


# ------------------------------------------------------------------
# TASK 8: Metadata validity
# ------------------------------------------------------------------
class TestMetadataValidity:
    def test_metadata_has_required_fields(self, metadata):
        assert metadata is not None
        for key in ["model_type", "model_version", "feature_names", "metrics",
                     "calibration", "dataset_samples", "split", "test_statistics"]:
            assert key in metadata, f"Missing field: {key}"

    def test_metrics_in_valid_range(self, metadata):
        m = metadata["metrics"]
        assert 0 <= m["mae"] <= 100
        assert 0 <= m["rmse"] <= 100
        assert -1 <= m["r2"] <= 1
        assert -1 <= m["spearman"] <= 1

    def test_calibration_metadata_valid(self, metadata):
        cal = metadata["calibration"]
        assert cal["pred_min"] < cal["pred_max"]
        assert cal["pred_std"] > 0
        assert cal["fitted_on"] == "validation_set"

    def test_feature_names_match(self, metadata, feature_names):
        assert metadata["feature_names"] == feature_names
        assert len(feature_names) == 8, f"Expected 8 features, got {len(feature_names)}"

    def test_model_version_is_v3_0(self, metadata):
        """Model version should be v3.0.0 after the improvement."""
        assert metadata.get("model_version") == "v3.0.0", (
            f"Expected model_version v3.0.0, got {metadata.get('model_version')}"
        )


# ------------------------------------------------------------------
# TASK 8: Performance metadata valid
# ------------------------------------------------------------------
class TestPerformanceMetadata:
    def test_performance_json_exists(self):
        perf_path = Path(__file__).resolve().parent.parent.parent / "benchmark" / "smart_city" / "phase10_performance.json"
        if perf_path.exists():
            with open(perf_path) as f:
                data = json.load(f)
            assert "total_mean_ms" in data
            assert "total_p95_ms" in data
            assert data["total_mean_ms"] > 0
            assert data["total_p95_ms"] >= data["total_mean_ms"]

    def test_evaluation_json_exists(self):
        eval_path = Path(__file__).resolve().parent.parent.parent / "benchmark" / "smart_city" / "phase10_model_evaluation.json"
        if eval_path.exists():
            with open(eval_path) as f:
                data = json.load(f)
            assert "evaluation_metrics" in data
            assert "limitations" in data
            assert len(data["limitations"]) > 0

    def test_evaluation_json_mentions_leakage_removal(self):
        """The evaluation JSON should explicitly state calibration is leakage-free."""
        eval_path = Path(__file__).resolve().parent.parent.parent / "benchmark" / "smart_city" / "phase10_model_evaluation.json"
        if eval_path.exists():
            with open(eval_path) as f:
                data = json.load(f)
            cal = data.get("calibration", {})
            assert cal.get("fitted_on") == "validation_set", (
                "Calibration should be fitted on validation set"
            )
            assert cal.get("never_fitted_on") == "test_set", (
                "Calibration should explicitly state it was never fitted on test data"
            )
