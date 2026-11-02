"""
preprocessing.py — Module 1 entry point: validation, feature engineering, scaling.

Run with:
    python -m src.preprocessing [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR]

Produces:
    outputs/reviewer_features.csv          — reviewer IDs + raw engineered features
    outputs/reviewer_features_scaled.csv   — reviewer IDs + StandardScaler-transformed features
    outputs/reviewer_ground_truth.csv      — evaluation-only labels (separate from features)
    outputs/reports/module1_validation_report.json
    outputs/reports/module1_validation_report.txt
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data_validation import load_and_validate, ValidationReport
from src.feature_engineering import (
    engineer_features,
    extract_ground_truth,
    GROUND_TRUTH_COLUMNS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature columns expected in the output
# ---------------------------------------------------------------------------
EXPECTED_FEATURE_COLUMNS = [
    "avg_rating",
    "mean_rating_deviation",
    "mean_absolute_rating_deviation",
    "review_count",
    "reviews_per_day",
    "avg_review_length_words",
    "avg_review_length_chars",
    "unique_products_reviewed",
    "product_category_count",
    "product_category_entropy",
    "avg_inter_review_hours",
    "review_time_std_hours",
    "verified_purchase_ratio",
]


def get_numerical_feature_columns(features: pd.DataFrame) -> list[str]:
    """
    Identify numerical feature columns, excluding identifiers and ground-truth.

    Parameters
    ----------
    features : pd.DataFrame
        The raw feature matrix with reviewer_id as index.

    Returns
    -------
    list[str]
        Column names suitable for scaling.
    """
    exclude = GROUND_TRUTH_COLUMNS | {"reviewer_id"}
    num_cols = [
        c for c in features.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(features[c])
    ]
    return num_cols


def handle_missing_values(
    features: pd.DataFrame,
    num_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Handle missing values in numerical features using documented, defensible rules.

    Rules applied:
    - avg_inter_review_hours: NaN for single-review reviewers → filled with 0
      (no inter-review interval exists).
    - review_time_std_hours: NaN for ≤2 reviews → filled with 0
      (insufficient data to compute variance).
    - Other numerical columns: NaN filled with column median
      (robust to outliers, preserves distribution shape).

    Parameters
    ----------
    features : pd.DataFrame
        Raw feature matrix.
    num_cols : list[str]
        Numerical column names.

    Returns
    -------
    tuple of (filled DataFrame, dict describing treatment per column)
    """
    filled = features.copy()
    treatment: dict[str, str] = {}

    for col in num_cols:
        n_missing = filled[col].isna().sum()
        if n_missing == 0:
            treatment[col] = "no_missing"
            continue

        if col == "avg_inter_review_hours":
            filled[col] = filled[col].fillna(0.0)
            treatment[col] = f"filled_with_0 ({n_missing} values; single-review reviewers)"
        elif col == "review_time_std_hours":
            filled[col] = filled[col].fillna(0.0)
            treatment[col] = f"filled_with_0 ({n_missing} values; ≤2-review reviewers)"
        else:
            median_val = filled[col].median()
            filled[col] = filled[col].fillna(median_val)
            treatment[col] = f"filled_with_median={median_val:.4f} ({n_missing} values)"

    return filled, treatment


def scale_features(
    features: pd.DataFrame,
    num_cols: list[str],
) -> pd.DataFrame:
    """
    Apply StandardScaler to numerical feature columns.

    Constant-valued columns are scaled safely (StandardScaler produces 0 for
    zero-variance features).

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix with missing values already handled.
    num_cols : list[str]
        Columns to scale.

    Returns
    -------
    pd.DataFrame
        DataFrame with reviewer_id as index and scaled numerical features.
    """
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(features[num_cols].values)
    scaled_df = pd.DataFrame(
        scaled_values,
        index=features.index,
        columns=num_cols,
    )
    scaled_df.index.name = "reviewer_id"

    # Log constant features
    for i, col in enumerate(num_cols):
        if scaler.scale_[i] == 0:
            logger.warning(
                "Feature '%s' is constant — scaled to 0 for all reviewers", col
            )

    return scaled_df


def save_outputs(
    features_raw: pd.DataFrame,
    features_scaled: pd.DataFrame,
    gt_eval: pd.DataFrame | None,
    report: ValidationReport,
    missing_treatment: dict[str, str],
    num_cols: list[str],
    output_dir: Path,
) -> dict[str, Path]:
    """Save all Module 1 output files and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    # Raw features
    raw_path = output_dir / "reviewer_features.csv"
    features_raw.to_csv(raw_path)
    paths["reviewer_features"] = raw_path
    logger.info("Saved raw features → %s (%d rows × %d cols)",
                raw_path, len(features_raw), len(features_raw.columns))

    # Scaled features
    scaled_path = output_dir / "reviewer_features_scaled.csv"
    features_scaled.to_csv(scaled_path)
    paths["reviewer_features_scaled"] = scaled_path
    logger.info("Saved scaled features → %s (%d rows × %d cols)",
                scaled_path, len(features_scaled), len(features_scaled.columns))

    # Ground truth (evaluation only)
    if gt_eval is not None:
        gt_path = output_dir / "reviewer_ground_truth.csv"
        gt_eval.to_csv(gt_path, index=False)
        paths["reviewer_ground_truth"] = gt_path
        logger.info("Saved ground truth → %s", gt_path)

    # Validation report — enrich with feature-engineering details
    report.stats["feature_columns"] = list(features_raw.columns)
    report.stats["feature_count"] = len(features_raw.columns)
    report.stats["numerical_feature_columns"] = num_cols
    report.stats["missing_value_treatment"] = missing_treatment
    report.stats["scaling_method"] = "StandardScaler (zero mean, unit variance)"
    report.stats["reviewer_count_in_features"] = len(features_raw)

    # JSON report
    json_path = reports_dir / "module1_validation_report.json"
    json_path.write_text(report.to_json(), encoding="utf-8")
    paths["report_json"] = json_path

    # Text report
    txt_path = reports_dir / "module1_validation_report.txt"
    txt_path.write_text(report.to_text(), encoding="utf-8")
    paths["report_txt"] = txt_path
    logger.info("Saved reports → %s, %s", json_path, txt_path)

    return paths


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_module1(data_dir: Path, output_dir: Path) -> dict:
    """
    Execute the full Module 1 pipeline.

    Parameters
    ----------
    data_dir : Path
        Directory containing reviews.csv, products.csv, ground_truth.csv.
    output_dir : Path
        Directory to write output files.

    Returns
    -------
    dict with keys: report, features_raw, features_scaled, gt_eval, paths
    """
    # Step 1: Load and validate
    reviews_df, products_df, gt_df, report = load_and_validate(data_dir)

    if report.has_fatal_errors:
        logger.error("Fatal validation errors found — cannot proceed.")
        # Still save the report
        reports_dir = output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "module1_validation_report.json").write_text(
            report.to_json(), encoding="utf-8"
        )
        (reports_dir / "module1_validation_report.txt").write_text(
            report.to_text(), encoding="utf-8"
        )
        print("\n" + report.to_text())
        raise SystemExit(
            "Module 1 aborted due to fatal validation errors. "
            "See the report above."
        )

    # Step 2: Feature engineering
    features_raw = engineer_features(reviews_df, products_df)

    # Data-leakage guard
    leaked = set(features_raw.columns) & GROUND_TRUTH_COLUMNS
    if leaked:
        raise RuntimeError(
            f"DATA LEAKAGE: ground-truth columns in feature matrix: {leaked}"
        )

    # Step 3: Identify numerical columns and handle missing values
    num_cols = get_numerical_feature_columns(features_raw)
    features_filled, missing_treatment = handle_missing_values(features_raw, num_cols)

    # Step 4: Scale
    features_scaled = scale_features(features_filled, num_cols)

    # Step 5: Extract ground truth
    gt_eval = None
    if gt_df is not None:
        gt_eval = extract_ground_truth(gt_df)

    # Step 6: Save
    paths = save_outputs(
        features_filled, features_scaled, gt_eval,
        report, missing_treatment, num_cols, output_dir,
    )

    return {
        "report": report,
        "features_raw": features_filled,
        "features_scaled": features_scaled,
        "gt_eval": gt_eval,
        "paths": paths,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Module 1: Data Validation and Feature Engineering",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing the input CSV files (default: data/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for output files (default: outputs/)",
    )
    args = parser.parse_args()

    result = run_module1(args.data_dir, args.output_dir)

    # Summary to stdout
    report = result["report"]
    features = result["features_raw"]
    scaled = result["features_scaled"]

    print("\n" + "=" * 70)
    print("MODULE 1 — EXECUTION COMPLETE")
    print("=" * 70)
    print(f"Reviewers in feature matrix : {len(features)}")
    print(f"Feature columns             : {len(features.columns)}")
    print(f"  {list(features.columns)}")
    print(f"Scaled feature columns      : {len(scaled.columns)}")
    print(f"Validation errors           : {len(report.errors)}")
    print(f"Validation warnings         : {len(report.warnings)}")

    # Missing value summary
    missing_after = features[get_numerical_feature_columns(features)].isna().sum()
    total_missing = int(missing_after.sum())
    print(f"Missing values after impute : {total_missing}")

    # Check signed deviation retention
    if "mean_rating_deviation" in features.columns:
        has_pos = (features["mean_rating_deviation"] > 0).any()
        has_neg = (features["mean_rating_deviation"] < 0).any()
        print(f"Positive rating deviations  : {'YES' if has_pos else 'NO'}")
        print(f"Negative rating deviations  : {'YES' if has_neg else 'NO'}")

    # Confirm no leakage
    leaked = set(features.columns) & GROUND_TRUTH_COLUMNS
    print(f"Ground-truth leakage        : {'DETECTED — ' + str(leaked) if leaked else 'NONE'}")

    print("\nOutput files:")
    for label, p in result["paths"].items():
        print(f"  {label}: {p}")

    print("=" * 70)


if __name__ == "__main__":
    main()
