"""
clustering_evaluation.py — Evaluation utilities for spectral clustering results.

Builds contingency tables, cluster behavior summaries, and formatted reports.
Ground-truth labels are used ONLY for evaluation, never as clustering input.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_contingency_table(
    labels: np.ndarray,
    reviewer_ids: list[str],
    gt_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a contingency table: rows = predicted cluster, columns = ground-truth category.

    Uses reviewer_id for matching, not positional index.
    """
    gt_indexed = gt_df.set_index("reviewer_id")

    rows = []
    for rid, cluster in zip(reviewer_ids, labels):
        cat = gt_indexed.loc[rid, "ground_truth_category"] if rid in gt_indexed.index else "unknown"
        rows.append({"reviewer_id": rid, "cluster_id": int(cluster), "ground_truth": cat})

    df = pd.DataFrame(rows)
    contingency = pd.crosstab(df["cluster_id"], df["ground_truth"])

    # Reorder columns for readability
    desired_order = ["normal", "positive_manipulation", "negative_manipulation", "ambiguous"]
    present = [c for c in desired_order if c in contingency.columns]
    extra = [c for c in contingency.columns if c not in desired_order]
    contingency = contingency[present + extra]

    return contingency


def build_cluster_behavior_summary(
    labels: np.ndarray,
    reviewer_ids: list[str],
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a per-cluster behavioral summary from reviewer features.

    Analyses:
    - Average rating and rating deviation patterns
    - Review activity (count, frequency)
    - High/low rating fractions
    - Verified purchase ratio
    """
    assign = pd.DataFrame({
        "reviewer_id": reviewer_ids,
        "cluster_id": labels.astype(int),
    }).set_index("reviewer_id")

    # Join with features
    merged = assign.join(features_df, how="left")

    # Compute per-cluster statistics
    rows = []
    for cid in sorted(merged["cluster_id"].unique()):
        cluster = merged[merged["cluster_id"] == cid]
        n = len(cluster)

        row: dict[str, Any] = {
            "cluster_id": int(cid),
            "size": n,
            "pct_of_total": round(100 * n / len(merged), 1),
        }

        # Rating features
        if "avg_rating" in cluster.columns:
            row["avg_rating"] = round(cluster["avg_rating"].mean(), 3)
        if "mean_rating_deviation" in cluster.columns:
            row["mean_rating_deviation"] = round(cluster["mean_rating_deviation"].mean(), 4)
        if "mean_absolute_rating_deviation" in cluster.columns:
            row["mean_abs_rating_deviation"] = round(cluster["mean_absolute_rating_deviation"].mean(), 4)

        # Rating distribution proxies
        if "avg_rating" in cluster.columns:
            row["frac_high_rating"] = round((cluster["avg_rating"] >= 4.0).mean(), 3)
            row["frac_low_rating"] = round((cluster["avg_rating"] <= 2.0).mean(), 3)

        # Activity
        if "review_count" in cluster.columns:
            row["review_count"] = round(cluster["review_count"].mean(), 1)
        if "reviews_per_day" in cluster.columns:
            row["reviews_per_day"] = round(cluster["reviews_per_day"].mean(), 4)

        # Product diversity
        if "unique_products_reviewed" in cluster.columns:
            row["unique_products"] = round(cluster["unique_products_reviewed"].mean(), 1)
        if "product_category_entropy" in cluster.columns:
            row["category_entropy"] = round(cluster["product_category_entropy"].mean(), 3)

        # Temporal
        if "avg_inter_review_hours" in cluster.columns:
            row["avg_inter_review_hours"] = round(cluster["avg_inter_review_hours"].mean(), 1)

        # Verified purchase
        if "verified_purchase_ratio" in cluster.columns:
            row["verified_purchase_ratio"] = round(cluster["verified_purchase_ratio"].mean(), 3)

        rows.append(row)

    return pd.DataFrame(rows)


def format_evaluation_report(
    primary: dict[str, Any],
    all_results: list[dict[str, Any]],
    contingency: pd.DataFrame,
    behavior_df: pd.DataFrame,
    config: dict[str, Any],
) -> str:
    """Generate a human-readable evaluation report."""
    lines = [
        "=" * 70,
        "MODULE 3 — SPECTRAL CLUSTERING EVALUATION REPORT",
        "=" * 70, "",
        "— CONFIGURATION —",
        f"  Laplacian formulation     : {config['laplacian']}",
        f"  K range evaluated         : {config['k_min']}..{config['k_max']}",
        f"  Primary k selected        : {config['k_primary_selected']}",
        f"  Selection method          : {config['selection_method']}",
        f"  Random seed               : {config['seed']}",
        f"  K-Means initializations   : {config['kmeans_n_init']}",
        "",
        "— PRIMARY EXPERIMENT RESULTS —",
        f"  k                         : {primary['k']}",
        f"  ARI                       : {primary['ari']:.4f}",
        f"  NMI                       : {primary['nmi']:.4f}",
    ]

    sil = primary.get("silhouette")
    lines.append(f"  Silhouette Score          : {f'{sil:.4f}' if sil is not None else 'N/A'}")
    lines.append(f"  Cluster sizes             : {primary['cluster_sizes']}")
    lines.append("")

    # All experiments
    lines.append("— ALL EXPERIMENTS —")
    lines.append(f"  {'k':>3}  {'ARI':>8}  {'NMI':>8}  {'Silhouette':>11}  {'Runtime':>8}  Status")
    for r in all_results:
        if r["status"] == "success":
            sil_str = f"{r['silhouette']:.4f}" if r["silhouette"] is not None else "N/A"
            marker = " <--" if r["k"] == primary["k"] else ""
            lines.append(
                f"  {r['k']:>3}  {r['ari']:>8.4f}  {r['nmi']:>8.4f}  "
                f"{sil_str:>11}  {r['runtime_seconds']:>7.2f}s  {r['status']}{marker}"
            )
        else:
            lines.append(f"  {r['k']:>3}  {'—':>8}  {'—':>8}  {'—':>11}  {r['runtime_seconds']:>7.2f}s  FAILED")
    lines.append("")

    # Contingency table
    lines.append("— CONTINGENCY TABLE (predicted cluster × ground-truth category) —")
    lines.append("  " + contingency.to_string().replace("\n", "\n  "))
    lines.append("")

    # Cluster behavior
    lines.append("— CLUSTER BEHAVIORAL PROFILES —")
    lines.append("  Note: Patterns are descriptive, not fraud classifications.")
    lines.append("  A cluster's behavioral profile alone does not prove coordination.")
    lines.append("")
    for _, row in behavior_df.iterrows():
        cid = int(row["cluster_id"])
        lines.append(f"  Cluster {cid} (n={int(row['size'])}, {row['pct_of_total']}% of reviewers):")
        if "avg_rating" in row:
            lines.append(f"    Avg rating              : {row['avg_rating']:.3f}")
        if "mean_rating_deviation" in row:
            lines.append(f"    Mean rating deviation    : {row['mean_rating_deviation']:+.4f}")
        if "mean_abs_rating_deviation" in row:
            lines.append(f"    Mean abs rating dev      : {row['mean_abs_rating_deviation']:.4f}")
        if "frac_high_rating" in row:
            lines.append(f"    Fraction high-rating     : {row['frac_high_rating']:.3f}")
        if "frac_low_rating" in row:
            lines.append(f"    Fraction low-rating      : {row['frac_low_rating']:.3f}")
        if "review_count" in row:
            lines.append(f"    Avg review count         : {row['review_count']:.1f}")
        if "verified_purchase_ratio" in row:
            lines.append(f"    Verified purchase ratio  : {row['verified_purchase_ratio']:.3f}")
        if "category_entropy" in row:
            lines.append(f"    Category entropy         : {row['category_entropy']:.3f}")
        lines.append("")

    lines.extend([
        "— INTERPRETATION NOTES —",
        "  • Cluster IDs are arbitrary and do not map to specific fraud types.",
        "  • ARI and NMI measure agreement with synthetic labels, not real-world fraud.",
        "  • Silhouette Score measures structural quality without labels.",
        "  • High-rating clusters are not automatically positive manipulation.",
        "  • Low-rating clusters are not automatically negative manipulation.",
        "  • Multiple behavioral signals should be considered together.",
        "",
        "=" * 70,
    ])

    return "\n".join(lines)
