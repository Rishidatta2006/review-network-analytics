"""
data_validation.py — Reusable validation functions for the fake-review-network dataset.

Validates reviews.csv, products.csv, and ground_truth.csv, producing a structured
validation report with fatal errors, warnings, and informational statistics.

This module never modifies the original data files.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REQUIRED_REVIEW_COLUMNS = [
    "review_id", "reviewer_id", "product_id", "rating",
    "timestamp", "review_text", "verified_purchase",
]
REQUIRED_GROUND_TRUTH_COLUMNS = [
    "reviewer_id", "ground_truth_category",
]
REQUIRED_PRODUCT_COLUMNS = [
    "product_id", "product_category",
]
EXPECTED_CATEGORIES = {
    "normal", "positive_manipulation", "negative_manipulation", "ambiguous",
}
VALID_CAMPAIGN_DIRECTIONS = {"positive", "negative", "none"}


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------
class ValidationReport:
    """Accumulates errors, warnings and statistics during validation."""

    def __init__(self) -> None:
        self.errors: list[str] = []       # Fatal — processing cannot continue
        self.warnings: list[str] = []     # Non-fatal — processing can continue
        self.info: list[str] = []         # Informational statistics
        self.stats: dict[str, Any] = {}

    @property
    def has_fatal_errors(self) -> bool:
        return len(self.errors) > 0

    def add_error(self, msg: str) -> None:
        logger.error(msg)
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        logger.warning(msg)
        self.warnings.append(msg)

    def add_info(self, msg: str) -> None:
        logger.info(msg)
        self.info.append(msg)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fatal_errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
            "statistics": self.stats,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def to_text(self) -> str:
        lines = ["=" * 70, "MODULE 1 — VALIDATION REPORT", "=" * 70, ""]

        lines.append("— FATAL ERRORS —")
        if self.errors:
            for e in self.errors:
                lines.append(f"  [ERROR] {e}")
        else:
            lines.append("  None")
        lines.append("")

        lines.append("— WARNINGS —")
        if self.warnings:
            for w in self.warnings:
                lines.append(f"  [WARN]  {w}")
        else:
            lines.append("  None")
        lines.append("")

        lines.append("— INFO —")
        for i in self.info:
            lines.append(f"  [INFO]  {i}")
        lines.append("")

        lines.append("— STATISTICS —")
        for key, val in self.stats.items():
            if isinstance(val, dict):
                lines.append(f"  {key}:")
                for k2, v2 in val.items():
                    lines.append(f"    {k2}: {v2}")
            else:
                lines.append(f"  {key}: {val}")
        lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------

def _check_required_columns(
    df: pd.DataFrame,
    required: list[str],
    file_label: str,
    report: ValidationReport,
) -> bool:
    """Check that *required* columns exist. Returns True if all present."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        report.add_error(
            f"{file_label} is missing required columns: {missing}"
        )
        return False
    report.add_info(f"{file_label} has all required columns: {required}")
    return True


def _check_no_null_ids(
    df: pd.DataFrame,
    id_cols: list[str],
    file_label: str,
    report: ValidationReport,
) -> None:
    """Flag null values in identifier columns as fatal errors."""
    for col in id_cols:
        if col not in df.columns:
            continue
        n_null = df[col].isna().sum()
        if n_null > 0:
            report.add_error(
                f"{file_label}.{col} has {n_null} null values "
                f"({n_null / len(df) * 100:.2f}%)"
            )


def _check_unique_ids(
    df: pd.DataFrame,
    col: str,
    file_label: str,
    report: ValidationReport,
) -> None:
    """Flag duplicate values in a column that should be unique."""
    n_dup = df[col].duplicated().sum()
    if n_dup > 0:
        report.add_error(
            f"{file_label}.{col} has {n_dup} duplicate values"
        )
    else:
        report.add_info(f"{file_label}.{col} is unique ({len(df)} values)")


def _check_ratings(
    df: pd.DataFrame,
    report: ValidationReport,
) -> None:
    """Validate ratings are integers 1–5."""
    if "rating" not in df.columns:
        return
    # Coerce to numeric, flag non-numeric
    ratings = pd.to_numeric(df["rating"], errors="coerce")
    n_non_numeric = ratings.isna().sum() - df["rating"].isna().sum()
    if n_non_numeric > 0:
        report.add_error(
            f"reviews: {n_non_numeric} non-numeric rating values"
        )

    valid = ratings.dropna()
    non_int = valid[valid != valid.astype(int)]
    if len(non_int) > 0:
        report.add_error(
            f"reviews: {len(non_int)} ratings are not integers"
        )

    out_of_range = valid[(valid < 1) | (valid > 5)]
    if len(out_of_range) > 0:
        report.add_error(
            f"reviews: {len(out_of_range)} ratings outside range [1, 5]"
        )

    if n_non_numeric == 0 and len(non_int) == 0 and len(out_of_range) == 0:
        report.add_info("All ratings are integers in [1, 5]")


def _check_timestamps(
    df: pd.DataFrame,
    report: ValidationReport,
) -> None:
    """Validate timestamps can be parsed as datetimes."""
    if "timestamp" not in df.columns:
        return
    parsed = pd.to_datetime(df["timestamp"], errors="coerce")
    n_invalid = parsed.isna().sum() - df["timestamp"].isna().sum()
    if n_invalid > 0:
        report.add_error(
            f"reviews: {n_invalid} timestamps cannot be parsed as datetime"
        )
    else:
        ts_min = parsed.min()
        ts_max = parsed.max()
        report.add_info(f"Timestamps valid: {ts_min} -> {ts_max}")
        report.stats["timestamp_range"] = {"min": str(ts_min), "max": str(ts_max)}


def _check_product_references(
    reviews_df: pd.DataFrame,
    products_df: pd.DataFrame | None,
    report: ValidationReport,
) -> None:
    """Check that product IDs in reviews reference known products."""
    if products_df is None:
        report.add_warning("No products catalogue provided; skipping product reference check")
        return
    if "product_id" not in reviews_df.columns:
        return
    known = set(products_df["product_id"])
    review_pids = set(reviews_df["product_id"].dropna())
    unknown = review_pids - known
    if unknown:
        report.add_error(
            f"reviews: {len(unknown)} unknown product IDs: "
            f"{sorted(unknown)[:10]}{'...' if len(unknown) > 10 else ''}"
        )
    else:
        report.add_info(
            f"All {len(review_pids)} product IDs in reviews are defined in the catalogue"
        )


def _check_reviewer_ground_truth_coverage(
    reviews_df: pd.DataFrame,
    gt_df: pd.DataFrame | None,
    report: ValidationReport,
) -> None:
    """Check that reviewer IDs in reviews have ground-truth records."""
    if gt_df is None:
        report.add_warning("No ground-truth file provided; skipping coverage check")
        return
    review_rids = set(reviews_df["reviewer_id"].dropna())
    gt_rids = set(gt_df["reviewer_id"].dropna())
    missing = review_rids - gt_rids
    if missing:
        report.add_warning(
            f"{len(missing)} reviewer(s) in reviews have no ground-truth record: "
            f"{sorted(missing)[:10]}"
        )
    extra = gt_rids - review_rids
    if extra:
        report.add_warning(
            f"{len(extra)} ground-truth reviewer(s) have no reviews: "
            f"{sorted(extra)[:10]}"
        )
    if not missing and not extra:
        report.add_info(
            "All reviewer IDs match between reviews and ground-truth "
            f"({len(review_rids)} reviewers)"
        )


def _check_ground_truth_categories(
    gt_df: pd.DataFrame,
    report: ValidationReport,
) -> None:
    """Check ground-truth categories use expected values."""
    if "ground_truth_category" not in gt_df.columns:
        report.add_warning("ground_truth.csv has no 'ground_truth_category' column")
        return
    categories = set(gt_df["ground_truth_category"].dropna().unique())
    unexpected = categories - EXPECTED_CATEGORIES
    if unexpected:
        report.add_warning(
            f"Unexpected ground-truth categories found: {unexpected}"
        )
    report.add_info(f"Ground-truth categories present: {sorted(categories)}")

    # Distribution
    dist = gt_df["ground_truth_category"].value_counts().to_dict()
    report.stats["reviewer_category_distribution"] = dist


def _check_campaign_consistency(
    gt_df: pd.DataFrame,
    report: ValidationReport,
) -> None:
    """Check campaign IDs follow conventions and are consistent with labels."""
    if "campaign_id" not in gt_df.columns:
        report.add_info("No campaign_id column in ground-truth")
        return

    campaigns = gt_df["campaign_id"].dropna().unique()
    pos_campaigns = [c for c in campaigns if c.startswith("C_POS_")]
    neg_campaigns = [c for c in campaigns if c.startswith("C_NEG_")]
    other_campaigns = [
        c for c in campaigns
        if not c.startswith("C_POS_") and not c.startswith("C_NEG_")
    ]

    report.stats["positive_campaign_count"] = len(pos_campaigns)
    report.stats["negative_campaign_count"] = len(neg_campaigns)
    report.add_info(
        f"Campaigns: {len(pos_campaigns)} positive, {len(neg_campaigns)} negative"
    )

    if other_campaigns:
        report.add_warning(
            f"Non-standard campaign IDs found: {other_campaigns}"
        )

    # Check direction consistency
    if "campaign_direction" in gt_df.columns:
        for _, row in gt_df.dropna(subset=["campaign_id"]).iterrows():
            cid = row["campaign_id"]
            direction = row.get("campaign_direction", "")
            if cid.startswith("C_POS_") and direction != "positive":
                report.add_warning(
                    f"Campaign {cid} has direction '{direction}' (expected 'positive') "
                    f"for reviewer {row['reviewer_id']}"
                )
            elif cid.startswith("C_NEG_") and direction != "negative":
                report.add_warning(
                    f"Campaign {cid} has direction '{direction}' (expected 'negative') "
                    f"for reviewer {row['reviewer_id']}"
                )


def _check_missing_values(
    df: pd.DataFrame,
    file_label: str,
    report: ValidationReport,
) -> dict[str, int]:
    """Report missing values per column."""
    missing = df.isna().sum()
    missing_dict = {col: int(n) for col, n in missing.items() if n > 0}
    if missing_dict:
        report.add_info(
            f"{file_label} missing values: {missing_dict}"
        )
    else:
        report.add_info(f"{file_label} has no missing values")
    return missing_dict


def _check_duplicate_records(
    df: pd.DataFrame,
    file_label: str,
    report: ValidationReport,
) -> int:
    """Report duplicate full rows."""
    n_dup = df.duplicated().sum()
    if n_dup > 0:
        report.add_warning(
            f"{file_label} has {n_dup} fully duplicate rows"
        )
    else:
        report.add_info(f"{file_label} has no fully duplicate rows")
    return int(n_dup)


def _check_rating_distribution(
    reviews_df: pd.DataFrame,
    report: ValidationReport,
) -> None:
    """Record rating distribution and check for both high and low activity."""
    if "rating" not in reviews_df.columns:
        return
    # Coerce to numeric to handle mixed-type columns safely
    numeric_ratings = pd.to_numeric(reviews_df["rating"], errors="coerce").dropna()
    dist = numeric_ratings.astype(int).value_counts().sort_index().to_dict()
    report.stats["rating_distribution"] = dist

    has_low = any(k in dist and dist[k] > 0 for k in [1, 2])
    has_high = any(k in dist and dist[k] > 0 for k in [4, 5])
    if not has_low:
        report.add_warning("No low-rating (1–2) reviews found in the dataset")
    if not has_high:
        report.add_warning("No high-rating (4–5) reviews found in the dataset")
    if has_low and has_high:
        report.add_info("Both high-rating and low-rating reviews are present")


# ---------------------------------------------------------------------------
# Main validation entry point
# ---------------------------------------------------------------------------

def validate_dataset(
    reviews_df: pd.DataFrame,
    products_df: pd.DataFrame | None = None,
    gt_df: pd.DataFrame | None = None,
) -> ValidationReport:
    """
    Run all validation checks on the provided DataFrames.

    Parameters
    ----------
    reviews_df : pd.DataFrame
        The reviews dataset (one row per review).
    products_df : pd.DataFrame or None
        Product catalogue, if available.
    gt_df : pd.DataFrame or None
        Ground-truth labels, if available.

    Returns
    -------
    ValidationReport
        Structured report with errors, warnings, info, and statistics.
    """
    report = ValidationReport()

    # --- Basic statistics ---
    report.stats["review_count"] = len(reviews_df)
    report.stats["unique_reviewer_count"] = int(reviews_df["reviewer_id"].nunique()) if "reviewer_id" in reviews_df.columns else 0
    report.stats["unique_product_count"] = int(reviews_df["product_id"].nunique()) if "product_id" in reviews_df.columns else 0
    report.add_info(
        f"Review count: {report.stats['review_count']}, "
        f"Unique reviewers: {report.stats['unique_reviewer_count']}, "
        f"Unique products: {report.stats['unique_product_count']}"
    )

    # --- Required columns ---
    reviews_ok = _check_required_columns(
        reviews_df, REQUIRED_REVIEW_COLUMNS, "reviews.csv", report
    )

    if products_df is not None:
        _check_required_columns(
            products_df, REQUIRED_PRODUCT_COLUMNS, "products.csv", report
        )

    if gt_df is not None:
        _check_required_columns(
            gt_df, REQUIRED_GROUND_TRUTH_COLUMNS, "ground_truth.csv", report
        )

    if not reviews_ok:
        report.add_error("Cannot continue validation — reviews.csv missing required columns")
        return report

    # --- ID checks ---
    _check_no_null_ids(reviews_df, ["review_id", "reviewer_id"], "reviews.csv", report)
    _check_unique_ids(reviews_df, "review_id", "reviews.csv", report)

    if gt_df is not None:
        _check_no_null_ids(gt_df, ["reviewer_id"], "ground_truth.csv", report)
        _check_unique_ids(gt_df, "reviewer_id", "ground_truth.csv", report)

    # --- Ratings ---
    _check_ratings(reviews_df, report)

    # --- Timestamps ---
    _check_timestamps(reviews_df, report)

    # --- Product references ---
    _check_product_references(reviews_df, products_df, report)

    # --- Reviewer ↔ ground-truth coverage ---
    _check_reviewer_ground_truth_coverage(reviews_df, gt_df, report)

    # --- Ground-truth categories ---
    if gt_df is not None:
        _check_ground_truth_categories(gt_df, report)

    # --- Campaign consistency ---
    if gt_df is not None:
        _check_campaign_consistency(gt_df, report)

    # --- Missing values ---
    mv_reviews = _check_missing_values(reviews_df, "reviews.csv", report)
    report.stats["missing_values_reviews"] = mv_reviews

    if products_df is not None:
        mv_products = _check_missing_values(products_df, "products.csv", report)
        report.stats["missing_values_products"] = mv_products

    if gt_df is not None:
        mv_gt = _check_missing_values(gt_df, "ground_truth.csv", report)
        report.stats["missing_values_ground_truth"] = mv_gt

    # --- Duplicate records ---
    dup_reviews = _check_duplicate_records(reviews_df, "reviews.csv", report)
    report.stats["duplicate_review_rows"] = dup_reviews

    # --- Rating distribution ---
    _check_rating_distribution(reviews_df, report)

    return report


def load_and_validate(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, ValidationReport]:
    """
    Load CSV files from *data_dir* and run validation.

    Returns
    -------
    tuple of (reviews_df, products_df, gt_df, report)
    """
    reviews_path = data_dir / "reviews.csv"
    products_path = data_dir / "products.csv"
    gt_path = data_dir / "ground_truth.csv"

    if not reviews_path.exists():
        raise FileNotFoundError(f"Required file not found: {reviews_path}")

    logger.info("Loading reviews from %s", reviews_path)
    reviews_df = pd.read_csv(reviews_path)

    products_df = None
    if products_path.exists():
        logger.info("Loading products from %s", products_path)
        products_df = pd.read_csv(products_path)
    else:
        logger.warning("products.csv not found in %s", data_dir)

    gt_df = None
    if gt_path.exists():
        logger.info("Loading ground truth from %s", gt_path)
        gt_df = pd.read_csv(gt_path)
    else:
        logger.warning("ground_truth.csv not found in %s", data_dir)

    report = validate_dataset(reviews_df, products_df, gt_df)
    return reviews_df, products_df, gt_df, report
