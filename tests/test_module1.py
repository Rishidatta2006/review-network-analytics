"""
test_module1.py — Comprehensive unit tests for Module 1.

Run with:
    python -m pytest tests/test_module1.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_validation import validate_dataset
from src.feature_engineering import (
    engineer_features,
    extract_ground_truth,
    GROUND_TRUTH_COLUMNS,
)
from src.preprocessing import (
    get_numerical_feature_columns,
    handle_missing_values,
    scale_features,
    run_module1,
)


# ---------------------------------------------------------------------------
# Fixtures — small synthetic datasets for testing
# ---------------------------------------------------------------------------

def _make_reviews(n: int = 10, **overrides) -> pd.DataFrame:
    """Create a minimal valid reviews DataFrame."""
    data = {
        "review_id": [f"REV{i:06d}" for i in range(1, n + 1)],
        "reviewer_id": [f"R{(i % 3) + 1:04d}" for i in range(n)],
        "product_id": [f"P{(i % 2) + 1:04d}" for i in range(n)],
        "rating": [1, 2, 3, 4, 5, 1, 3, 5, 2, 4][:n],
        "timestamp": pd.date_range("2026-03-01", periods=n, freq="6h").strftime(
            "%Y-%m-%d %H:%M:%S"
        ).tolist(),
        "review_text": [f"Review text {i}" for i in range(n)],
        "verified_purchase": [True, False] * (n // 2) + [True] * (n % 2),
    }
    data.update(overrides)
    return pd.DataFrame(data)


def _make_products() -> pd.DataFrame:
    return pd.DataFrame({
        "product_id": ["P0001", "P0002"],
        "product_category": ["electronics", "books"],
    })


def _make_ground_truth() -> pd.DataFrame:
    return pd.DataFrame({
        "reviewer_id": ["R0001", "R0002", "R0003"],
        "ground_truth_category": ["normal", "positive_manipulation", "negative_manipulation"],
        "campaign_id": [pd.NA, "C_POS_01", "C_NEG_01"],
        "campaign_direction": ["none", "positive", "negative"],
        "is_coordinated": [False, True, True],
    })


# ---------------------------------------------------------------------------
# TEST 1: Valid dataset acceptance
# ---------------------------------------------------------------------------

class TestValidDatasetAcceptance:
    def test_valid_dataset_no_errors(self):
        reviews = _make_reviews()
        products = _make_products()
        gt = _make_ground_truth()
        report = validate_dataset(reviews, products, gt)
        assert not report.has_fatal_errors, f"Unexpected errors: {report.errors}"


# ---------------------------------------------------------------------------
# TEST 2: Duplicate review IDs
# ---------------------------------------------------------------------------

class TestDuplicateReviewIds:
    def test_duplicate_review_ids_detected(self):
        reviews = _make_reviews()
        reviews.loc[1, "review_id"] = reviews.loc[0, "review_id"]  # duplicate
        report = validate_dataset(reviews)
        assert any("duplicate" in e.lower() for e in report.errors), \
            "Expected error about duplicate review IDs"


# ---------------------------------------------------------------------------
# TEST 3: Invalid ratings
# ---------------------------------------------------------------------------

class TestInvalidRatings:
    def test_out_of_range_rating(self):
        reviews = _make_reviews()
        reviews.loc[0, "rating"] = 6
        report = validate_dataset(reviews)
        assert any("outside range" in e.lower() for e in report.errors)

    def test_non_numeric_rating(self):
        reviews = _make_reviews()
        reviews["rating"] = reviews["rating"].astype(object)
        reviews.loc[0, "rating"] = "bad"
        report = validate_dataset(reviews)
        assert any("non-numeric" in e.lower() for e in report.errors)


# ---------------------------------------------------------------------------
# TEST 4: Missing reviewer IDs
# ---------------------------------------------------------------------------

class TestMissingReviewerIds:
    def test_null_reviewer_id(self):
        reviews = _make_reviews()
        reviews.loc[0, "reviewer_id"] = None
        report = validate_dataset(reviews)
        assert any("null" in e.lower() and "reviewer_id" in e for e in report.errors)


# ---------------------------------------------------------------------------
# TEST 5: Unknown product IDs
# ---------------------------------------------------------------------------

class TestUnknownProductIds:
    def test_unknown_product_flagged(self):
        reviews = _make_reviews()
        reviews.loc[0, "product_id"] = "P9999"
        products = _make_products()
        report = validate_dataset(reviews, products)
        assert any("unknown" in e.lower() for e in report.errors)


# ---------------------------------------------------------------------------
# TEST 6: Missing review text
# ---------------------------------------------------------------------------

class TestMissingReviewText:
    def test_missing_text_does_not_crash(self):
        reviews = _make_reviews()
        reviews.loc[0, "review_text"] = None
        reviews.loc[1, "review_text"] = ""
        features = engineer_features(reviews)
        assert "avg_review_length_words" in features.columns
        assert not features["avg_review_length_words"].isna().all()

    def test_all_text_missing(self):
        reviews = _make_reviews()
        reviews["review_text"] = None
        features = engineer_features(reviews)
        assert (features["avg_review_length_words"] == 0).all()


# ---------------------------------------------------------------------------
# TEST 7: Invalid timestamps
# ---------------------------------------------------------------------------

class TestInvalidTimestamps:
    def test_unparseable_timestamp(self):
        reviews = _make_reviews()
        reviews["timestamp"] = reviews["timestamp"].astype(object)
        reviews.loc[0, "timestamp"] = "not-a-date"
        report = validate_dataset(reviews)
        assert any("timestamp" in e.lower() for e in report.errors)


# ---------------------------------------------------------------------------
# TEST 8: Reviewers with only one review
# ---------------------------------------------------------------------------

class TestSingleReviewReviewer:
    def test_single_review_features(self):
        reviews = pd.DataFrame({
            "review_id": ["REV1"],
            "reviewer_id": ["R0001"],
            "product_id": ["P0001"],
            "rating": [4],
            "timestamp": ["2026-05-01 12:00:00"],
            "review_text": ["Good product"],
            "verified_purchase": [True],
        })
        features = engineer_features(reviews)
        assert len(features) == 1
        row = features.iloc[0]
        assert row["review_count"] == 1
        assert row["reviews_per_day"] == 1.0  # Convention for single review
        assert np.isnan(row["avg_inter_review_hours"])  # Explicitly NaN


# ---------------------------------------------------------------------------
# TEST 9: Reviewers with multiple reviews
# ---------------------------------------------------------------------------

class TestMultipleReviewReviewer:
    def test_multiple_reviews(self):
        reviews = pd.DataFrame({
            "review_id": ["REV1", "REV2", "REV3"],
            "reviewer_id": ["R0001", "R0001", "R0001"],
            "product_id": ["P0001", "P0002", "P0001"],
            "rating": [5, 3, 4],
            "timestamp": [
                "2026-05-01 12:00:00",
                "2026-05-02 12:00:00",
                "2026-05-03 12:00:00",
            ],
            "review_text": ["Great", "Okay", "Good"],
            "verified_purchase": [True, False, True],
        })
        features = engineer_features(reviews)
        assert len(features) == 1
        row = features.iloc[0]
        assert row["review_count"] == 3
        assert row["avg_rating"] == 4.0
        assert row["unique_products_reviewed"] == 2
        assert row["avg_inter_review_hours"] == pytest.approx(24.0, abs=0.1)


# ---------------------------------------------------------------------------
# TEST 10: Positive and negative rating deviations
# ---------------------------------------------------------------------------

class TestRatingDeviations:
    def test_signed_deviation_preserved(self):
        # Reviewer R0001 rates above product mean; R0002 rates below.
        reviews = pd.DataFrame({
            "review_id": ["R1", "R2", "R3", "R4"],
            "reviewer_id": ["R0001", "R0001", "R0002", "R0002"],
            "product_id": ["P0001", "P0001", "P0001", "P0001"],
            "rating": [5, 5, 1, 1],
            "timestamp": ["2026-05-01"] * 4,
            "review_text": ["a", "b", "c", "d"],
            "verified_purchase": [True] * 4,
        })
        features = engineer_features(reviews)
        dev = features["mean_rating_deviation"]
        assert dev.loc["R0001"] > 0, "R0001 should have positive deviation"
        assert dev.loc["R0002"] < 0, "R0002 should have negative deviation"


# ---------------------------------------------------------------------------
# TEST 11: Ground-truth label leakage prevention
# ---------------------------------------------------------------------------

class TestLeakagePrevention:
    def test_no_gt_columns_in_features(self):
        reviews = _make_reviews()
        # Deliberately add a ground-truth column to reviews
        reviews["campaign_id"] = "C_POS_01"
        features = engineer_features(reviews)
        leaked = set(features.columns) & GROUND_TRUTH_COLUMNS
        assert not leaked, f"Ground-truth columns leaked into features: {leaked}"

    def test_features_exclude_is_coordinated(self):
        reviews = _make_reviews()
        reviews["is_coordinated"] = True
        features = engineer_features(reviews)
        assert "is_coordinated" not in features.columns


# ---------------------------------------------------------------------------
# TEST 12: Unique reviewer rows in feature matrix
# ---------------------------------------------------------------------------

class TestUniqueReviewerRows:
    def test_one_row_per_reviewer(self):
        reviews = _make_reviews()
        features = engineer_features(reviews)
        assert features.index.is_unique, "Feature matrix has duplicate reviewer IDs"
        reviewer_ids_in_reviews = reviews["reviewer_id"].nunique()
        assert len(features) == reviewer_ids_in_reviews


# ---------------------------------------------------------------------------
# TEST 13: Missing-value handling
# ---------------------------------------------------------------------------

class TestMissingValueHandling:
    def test_temporal_nan_filled(self):
        reviews = pd.DataFrame({
            "review_id": ["REV1"],
            "reviewer_id": ["R0001"],
            "product_id": ["P0001"],
            "rating": [4],
            "timestamp": ["2026-05-01 12:00:00"],
            "review_text": ["Good"],
            "verified_purchase": [True],
        })
        features = engineer_features(reviews)
        num_cols = get_numerical_feature_columns(features)
        filled, treatment = handle_missing_values(features, num_cols)
        # After filling, temporal columns should not be NaN
        assert not filled["avg_inter_review_hours"].isna().any()
        assert not filled["review_time_std_hours"].isna().any()
        assert filled["avg_inter_review_hours"].iloc[0] == 0.0

    def test_no_nans_after_handling(self):
        reviews = _make_reviews(10)
        features = engineer_features(reviews)
        num_cols = get_numerical_feature_columns(features)
        filled, _ = handle_missing_values(features, num_cols)
        total_missing = filled[num_cols].isna().sum().sum()
        assert total_missing == 0, f"Still {total_missing} NaN values after handling"


# ---------------------------------------------------------------------------
# TEST 14: Scaled numerical output
# ---------------------------------------------------------------------------

class TestScaledOutput:
    def test_scaled_mean_zero_std_one(self):
        reviews = _make_reviews(10)
        features = engineer_features(reviews)
        num_cols = get_numerical_feature_columns(features)
        filled, _ = handle_missing_values(features, num_cols)
        scaled = scale_features(filled, num_cols)

        for col in num_cols:
            vals = scaled[col].values
            if np.std(vals) > 0:  # non-constant
                assert abs(np.mean(vals)) < 1e-10, f"{col} mean not ~0"
                assert abs(np.std(vals) - 1.0) < 0.1, f"{col} std not ~1"

    def test_scaled_has_no_gt_columns(self):
        reviews = _make_reviews(10)
        features = engineer_features(reviews)
        num_cols = get_numerical_feature_columns(features)
        filled, _ = handle_missing_values(features, num_cols)
        scaled = scale_features(filled, num_cols)
        leaked = set(scaled.columns) & GROUND_TRUTH_COLUMNS
        assert not leaked


# ---------------------------------------------------------------------------
# TEST 15: Reproducible output
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_identical_runs(self):
        reviews = _make_reviews(10)
        products = _make_products()
        f1 = engineer_features(reviews, products)
        f2 = engineer_features(reviews, products)
        pd.testing.assert_frame_equal(f1, f2)

    def test_pipeline_reproducible(self, tmp_path):
        # Write test data to disk
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        reviews = _make_reviews(10)
        reviews.to_csv(data_dir / "reviews.csv", index=False)
        _make_products().to_csv(data_dir / "products.csv", index=False)
        _make_ground_truth().to_csv(data_dir / "ground_truth.csv", index=False)

        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"

        run_module1(data_dir, out1)
        run_module1(data_dir, out2)

        df1 = pd.read_csv(out1 / "reviewer_features.csv")
        df2 = pd.read_csv(out2 / "reviewer_features.csv")
        pd.testing.assert_frame_equal(df1, df2)

        s1 = pd.read_csv(out1 / "reviewer_features_scaled.csv")
        s2 = pd.read_csv(out2 / "reviewer_features_scaled.csv")
        pd.testing.assert_frame_equal(s1, s2)


# ---------------------------------------------------------------------------
# Additional validation tests
# ---------------------------------------------------------------------------

class TestGroundTruthCategories:
    def test_expected_categories_accepted(self):
        gt = _make_ground_truth()
        report = validate_dataset(_make_reviews(), gt_df=gt)
        assert not report.has_fatal_errors

    def test_unexpected_category_warned(self):
        gt = _make_ground_truth()
        gt.loc[0, "ground_truth_category"] = "unknown_type"
        report = validate_dataset(_make_reviews(), gt_df=gt)
        assert any("unexpected" in w.lower() for w in report.warnings)


class TestVerifiedPurchaseRatio:
    def test_ratio_computed(self):
        reviews = _make_reviews()
        features = engineer_features(reviews)
        assert "verified_purchase_ratio" in features.columns
        # All ratios should be between 0 and 1
        assert (features["verified_purchase_ratio"] >= 0).all()
        assert (features["verified_purchase_ratio"] <= 1).all()

    def test_missing_vp_column(self):
        reviews = _make_reviews()
        reviews = reviews.drop(columns=["verified_purchase"])
        features = engineer_features(reviews)
        assert "verified_purchase_ratio" not in features.columns


class TestProductCategoryEntropy:
    def test_single_category_zero_entropy(self):
        reviews = pd.DataFrame({
            "review_id": ["R1", "R2"],
            "reviewer_id": ["R0001", "R0001"],
            "product_id": ["P0001", "P0002"],
            "rating": [4, 5],
            "timestamp": ["2026-05-01", "2026-05-02"],
            "review_text": ["Good", "Great"],
            "verified_purchase": [True, True],
            "product_category": ["electronics", "electronics"],
        })
        features = engineer_features(reviews)
        assert features.loc["R0001", "product_category_entropy"] == 0.0

    def test_multi_category_positive_entropy(self):
        reviews = pd.DataFrame({
            "review_id": ["R1", "R2"],
            "reviewer_id": ["R0001", "R0001"],
            "product_id": ["P0001", "P0002"],
            "rating": [4, 5],
            "timestamp": ["2026-05-01", "2026-05-02"],
            "review_text": ["Good", "Great"],
            "verified_purchase": [True, True],
            "product_category": ["electronics", "books"],
        })
        features = engineer_features(reviews)
        assert features.loc["R0001", "product_category_entropy"] > 0


class TestExtractGroundTruth:
    def test_extraction(self):
        gt = _make_ground_truth()
        gt_eval = extract_ground_truth(gt)
        assert "reviewer_id" in gt_eval.columns
        assert "ground_truth_category" in gt_eval.columns
        assert len(gt_eval) == len(gt)
