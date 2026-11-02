"""
feature_engineering.py — Reviewer-level feature aggregation.

Aggregates review-level records into one feature row per reviewer.
Ground-truth / campaign columns are NEVER included in the output features.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns that must never appear in the feature matrix (data-leakage prevention)
GROUND_TRUTH_COLUMNS = {
    "ground_truth_category",
    "campaign_id",
    "campaign_direction",
    "is_coordinated",
}


def compute_avg_rating(reviews: pd.DataFrame) -> pd.Series:
    """Feature 1: Mean rating per reviewer."""
    return reviews.groupby("reviewer_id")["rating"].mean().rename("avg_rating")


def compute_rating_deviation(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 2: Rating deviation relative to product mean rating.

    Returns per-reviewer:
        mean_rating_deviation   — signed mean (preserves positive/negative patterns)
        mean_absolute_rating_deviation — mean of absolute deviations
    """
    product_means = reviews.groupby("product_id")["rating"].transform("mean")
    reviews = reviews.assign(_deviation=reviews["rating"] - product_means)

    grouped = reviews.groupby("reviewer_id")["_deviation"]
    result = pd.DataFrame({
        "mean_rating_deviation": grouped.mean(),
        "mean_absolute_rating_deviation": grouped.apply(lambda s: s.abs().mean()),
    })
    return result


def compute_review_frequency(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 3: Review count and reviews-per-day.

    Convention for single-review reviewers: reviews_per_day is set to the
    review_count (1) because the active observation period is undefined.
    This avoids division by zero and infinity while remaining conservative.
    """
    ts = pd.to_datetime(reviews["timestamp"], errors="coerce")
    df = reviews.assign(_ts=ts)

    grouped = df.groupby("reviewer_id")
    review_count = grouped.size().rename("review_count")

    def _reviews_per_day(group: pd.DataFrame) -> float:
        if len(group) < 2:
            return float(len(group))
        span = (group["_ts"].max() - group["_ts"].min()).total_seconds() / 86400.0
        if span == 0:
            return float(len(group))
        return len(group) / span

    rpd = grouped.apply(_reviews_per_day).rename("reviews_per_day")
    return pd.DataFrame({"review_count": review_count, "reviews_per_day": rpd})


def compute_review_length(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 4: Average review text length in words and characters.

    Missing or empty review text is treated as 0 words / 0 characters for that
    review, preserving the reviewer in the output.
    """
    text = reviews["review_text"].fillna("")
    word_counts = text.str.split().str.len().fillna(0)
    char_counts = text.str.len().fillna(0)

    df = reviews.assign(_words=word_counts, _chars=char_counts)
    grouped = df.groupby("reviewer_id")
    return pd.DataFrame({
        "avg_review_length_words": grouped["_words"].mean(),
        "avg_review_length_chars": grouped["_chars"].mean(),
    })


def compute_product_concentration(
    reviews: pd.DataFrame,
    products_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Feature 5: Product overlap and concentration.

    - unique_products_reviewed: count of distinct products per reviewer.
    - product_category_count: number of distinct product categories reviewed
      (only if product_category is available in reviews or products).
    - product_category_entropy: Shannon entropy of the reviewer's category
      distribution (only when ≥ 2 categories and sufficient data).
    """
    grouped = reviews.groupby("reviewer_id")

    unique_products = grouped["product_id"].nunique().rename("unique_products_reviewed")
    result = pd.DataFrame({"unique_products_reviewed": unique_products})

    # Determine product category source
    cat_col = None
    if "product_category" in reviews.columns:
        cat_col = "product_category"
    elif products_df is not None and "product_category" in products_df.columns:
        # Merge category into reviews
        reviews = reviews.merge(
            products_df[["product_id", "product_category"]],
            on="product_id",
            how="left",
            suffixes=("", "_catalogue"),
        )
        cat_col = "product_category" if "product_category" in reviews.columns else "product_category_catalogue"

    if cat_col is not None and cat_col in reviews.columns:
        cat_grouped = reviews.groupby("reviewer_id")[cat_col]
        result["product_category_count"] = cat_grouped.nunique()

        def _entropy(cats: pd.Series) -> float:
            counts = cats.value_counts()
            if len(counts) < 2:
                return 0.0
            probs = counts / counts.sum()
            return float(-np.sum(probs * np.log(probs)))

        result["product_category_entropy"] = cat_grouped.apply(_entropy)
    else:
        logger.info(
            "product_category not available — "
            "product_category_count and product_category_entropy not computed"
        )

    return result


def compute_temporal_behavior(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 6: Temporal behaviour per reviewer.

    - avg_inter_review_hours: mean time between consecutive reviews (hours).
    - review_time_std_hours: std of inter-review intervals (hours).

    Reviewers with < 2 reviews: both values are NaN (explicitly handled by
    the preprocessing step rather than imputing arbitrary values here).
    """
    ts = pd.to_datetime(reviews["timestamp"], errors="coerce")
    df = reviews.assign(_ts=ts).sort_values(["reviewer_id", "_ts"])

    def _temporal_stats(group: pd.DataFrame) -> pd.Series:
        if len(group) < 2:
            return pd.Series({
                "avg_inter_review_hours": np.nan,
                "review_time_std_hours": np.nan,
            })
        diffs = group["_ts"].diff().dropna().dt.total_seconds() / 3600.0
        return pd.Series({
            "avg_inter_review_hours": diffs.mean(),
            "review_time_std_hours": diffs.std() if len(diffs) >= 2 else np.nan,
        })

    result = df.groupby("reviewer_id").apply(_temporal_stats)
    # Flatten if MultiIndex result
    if isinstance(result.index, pd.MultiIndex):
        result = result.droplevel(-1)
    return result


def compute_verified_purchase_ratio(reviews: pd.DataFrame) -> pd.Series | None:
    """
    Feature 7: Verified-purchase ratio.

    Returns None if the column is absent. Normalizes True/False, 'True'/'False',
    1/0 representations.
    """
    if "verified_purchase" not in reviews.columns:
        logger.info("verified_purchase column not found — skipping feature")
        return None

    vp = reviews["verified_purchase"]
    # Normalize string representations
    if vp.dtype == object:
        vp = vp.map(
            {"True": True, "False": False, "true": True, "false": False,
             "1": True, "0": False, "yes": True, "no": False}
        )
    vp = vp.astype(float)
    return reviews.assign(_vp=vp).groupby("reviewer_id")["_vp"].mean().rename(
        "verified_purchase_ratio"
    )


def engineer_features(
    reviews_df: pd.DataFrame,
    products_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build the complete reviewer-level feature matrix.

    Parameters
    ----------
    reviews_df : pd.DataFrame
        Raw reviews (one row per review).
    products_df : pd.DataFrame or None
        Product catalogue (for category features).

    Returns
    -------
    pd.DataFrame
        One row per reviewer_id. Contains ONLY behavioural features and the
        reviewer_id index. No ground-truth columns.
    """
    # Drop any ground-truth / leakage columns from the working copy
    safe_cols = [c for c in reviews_df.columns if c not in GROUND_TRUTH_COLUMNS]
    reviews = reviews_df[safe_cols].copy()

    logger.info("Computing reviewer-level features for %d reviews …",
                len(reviews))

    parts: list[pd.DataFrame | pd.Series] = []

    # Feature 1
    parts.append(compute_avg_rating(reviews))

    # Feature 2
    parts.append(compute_rating_deviation(reviews))

    # Feature 3
    parts.append(compute_review_frequency(reviews))

    # Feature 4
    parts.append(compute_review_length(reviews))

    # Feature 5
    parts.append(compute_product_concentration(reviews, products_df))

    # Feature 6
    parts.append(compute_temporal_behavior(reviews))

    # Feature 7
    vp = compute_verified_purchase_ratio(reviews)
    if vp is not None:
        parts.append(vp)

    # Combine all features
    features = pd.concat(parts, axis=1)
    features.index.name = "reviewer_id"

    # Final leakage safety check
    leaked = set(features.columns) & GROUND_TRUTH_COLUMNS
    if leaked:
        raise RuntimeError(
            f"DATA LEAKAGE DETECTED — the following ground-truth columns "
            f"appeared in the feature matrix: {leaked}"
        )

    logger.info(
        "Feature matrix: %d reviewers × %d features",
        len(features), len(features.columns),
    )
    return features


def extract_ground_truth(gt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract ground-truth columns into a separate evaluation-only DataFrame.

    Only keeps reviewer_id and ground-truth / campaign label columns.
    """
    keep_cols = ["reviewer_id"]
    for col in ["ground_truth_category", "campaign_id", "campaign_direction", "is_coordinated"]:
        if col in gt_df.columns:
            keep_cols.append(col)
    return gt_df[keep_cols].copy()
