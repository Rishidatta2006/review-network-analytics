"""
spectral_clustering.py — Module 3: Spectral Clustering and Evaluation.

Loads the reviewer similarity graph from Module 2, performs spectral embedding
and K-Means clustering, evaluates against ground-truth labels, and analyses
positive/negative rating manipulation patterns.

Run with:
    python -m src.spectral_clustering [OPTIONS]

Options:
    --graph-dir       Directory with Module 2 outputs (default: outputs/graph/)
    --features-dir    Directory with Module 1 outputs (default: outputs/)
    --output-dir      Directory for clustering outputs (default: outputs/clustering/)
    --k-min           Minimum cluster count to evaluate (default: 2)
    --k-max           Maximum cluster count to evaluate (default: 8)
    --k-primary       Primary cluster count (default: auto-select by silhouette)
    --laplacian       'normalized' or 'unnormalized' (default: normalized)
    --seed            Random seed (default: 42)
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from src.feature_engineering import GROUND_TRUTH_COLUMNS
from src.clustering_evaluation import (
    build_contingency_table,
    build_cluster_behavior_summary,
    format_evaluation_report,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------------

def load_adjacency_matrix(graph_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Load the weighted adjacency matrix and extract reviewer ordering."""
    adj_path = graph_dir / "graph_adjacency_matrix.csv"
    if not adj_path.exists():
        raise FileNotFoundError(f"Adjacency matrix not found: {adj_path}")

    df = pd.read_csv(adj_path, index_col=0)
    W = df.values.astype(np.float64)
    reviewer_ids = list(df.index)

    # Validate
    n = W.shape[0]
    if W.shape[0] != W.shape[1]:
        raise ValueError(f"Adjacency matrix is not square: {W.shape}")
    if not np.all(np.isfinite(W)):
        raise ValueError("Adjacency matrix contains non-finite values")
    if np.any(W < 0):
        raise ValueError("Adjacency matrix contains negative values")
    if not np.allclose(W, W.T, atol=1e-10):
        raise ValueError("Adjacency matrix is not symmetric")
    if np.max(np.abs(np.diag(W))) > 1e-10:
        raise ValueError("Adjacency matrix diagonal is not zero")

    logger.info("Loaded adjacency matrix: %d x %d, %d non-zero entries",
                n, n, np.count_nonzero(W))
    return W, reviewer_ids


def load_node_mapping(graph_dir: Path) -> list[str]:
    """Load the reviewer-to-matrix-index mapping."""
    path = graph_dir / "reviewer_node_mapping.csv"
    if not path.exists():
        raise FileNotFoundError(f"Node mapping not found: {path}")
    df = pd.read_csv(path)
    return df.sort_values("matrix_index")["reviewer_id"].tolist()


def load_ground_truth(features_dir: Path) -> pd.DataFrame:
    """Load ground-truth labels (evaluation only — never used as features)."""
    gt_path = features_dir / "reviewer_ground_truth.csv"
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")
    return pd.read_csv(gt_path)


def load_reviewer_features(features_dir: Path) -> pd.DataFrame:
    """Load raw reviewer features for cluster behavior analysis."""
    path = features_dir / "reviewer_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Reviewer features not found: {path}")
    # Try loading with named index first; fall back to positional index
    try:
        df = pd.read_csv(path, index_col="reviewer_id")
    except (ValueError, KeyError):
        df = pd.read_csv(path, index_col=0)
        df.index.name = "reviewer_id"
    return df


def validate_ordering(
    adj_ids: list[str],
    mapping_ids: list[str],
) -> None:
    """Verify adjacency matrix and node mapping have identical ordering."""
    if adj_ids != mapping_ids:
        raise ValueError(
            "Reviewer ordering mismatch between adjacency matrix and node mapping. "
            f"First difference at index {next(i for i, (a, b) in enumerate(zip(adj_ids, mapping_ids)) if a != b)}"
        )
    logger.info("Reviewer ordering validated: %d reviewers", len(adj_ids))


# ---------------------------------------------------------------------------
# Spectral embedding
# ---------------------------------------------------------------------------

def compute_normalized_laplacian(W: np.ndarray) -> np.ndarray:
    """Compute L_sym = I - D^{-1/2} W D^{-1/2}, handling zero-degree nodes."""
    n = W.shape[0]
    degrees = W.sum(axis=1)
    d_inv_sqrt = np.zeros(n)
    nonzero = degrees > 0
    d_inv_sqrt[nonzero] = 1.0 / np.sqrt(degrees[nonzero])
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_sym = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt
    L_sym = (L_sym + L_sym.T) / 2.0  # enforce symmetry
    return L_sym


def compute_unnormalized_laplacian(W: np.ndarray) -> np.ndarray:
    """Compute L = D - W."""
    D = np.diag(W.sum(axis=1))
    return D - W


def spectral_embedding(
    W: np.ndarray,
    k: int,
    laplacian_type: str = "normalized",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the k-dimensional spectral embedding from the adjacency matrix.

    Parameters
    ----------
    W : np.ndarray
        Weighted adjacency matrix.
    k : int
        Number of clusters (embedding dimensions = k).
    laplacian_type : str
        'normalized' or 'unnormalized'.

    Returns
    -------
    tuple of (embedding, eigenvalues)
        embedding: np.ndarray of shape (n, k) — the k eigenvectors
        eigenvalues: np.ndarray of shape (n,) — all eigenvalues sorted ascending
    """
    if laplacian_type == "normalized":
        L = compute_normalized_laplacian(W)
    elif laplacian_type == "unnormalized":
        L = compute_unnormalized_laplacian(W)
    else:
        raise ValueError(f"Unknown laplacian_type: {laplacian_type}")

    # Dense eigen decomposition (appropriate for n=1000)
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # Sort by ascending eigenvalue (eigh already does this, but be explicit)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select the first k eigenvectors (smallest eigenvalues)
    U = eigenvectors[:, :k].copy()

    # For normalized spectral clustering (Ng-Jordan-Weiss):
    # row-normalize the embedding
    if laplacian_type == "normalized":
        row_norms = np.linalg.norm(U, axis=1, keepdims=True)
        # Avoid division by zero for isolated nodes
        row_norms = np.maximum(row_norms, 1e-10)
        U = U / row_norms

    return U, eigenvalues


# ---------------------------------------------------------------------------
# K-Means clustering
# ---------------------------------------------------------------------------

def run_kmeans(
    embedding: np.ndarray,
    k: int,
    seed: int = 42,
    n_init: int = 20,
) -> tuple[np.ndarray, float]:
    """
    Apply K-Means to the spectral embedding.

    Returns
    -------
    tuple of (labels, inertia)
    """
    kmeans = KMeans(
        n_clusters=k,
        random_state=seed,
        n_init=n_init,
        max_iter=300,
    )
    labels = kmeans.fit_predict(embedding)
    return labels, float(kmeans.inertia_)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_clustering(
    labels: np.ndarray,
    reviewer_ids: list[str],
    gt_df: pd.DataFrame,
    embedding: np.ndarray,
) -> dict[str, Any]:
    """
    Evaluate predicted clusters against ground-truth labels.

    Ground-truth is used ONLY for evaluation scoring, never as clustering input.
    """
    # Match ground truth to predicted labels by reviewer_id
    gt_indexed = gt_df.set_index("reviewer_id")
    gt_categories = []
    for rid in reviewer_ids:
        if rid in gt_indexed.index:
            gt_categories.append(gt_indexed.loc[rid, "ground_truth_category"])
        else:
            gt_categories.append("unknown")
    gt_array = np.array(gt_categories)

    # ARI and NMI
    ari = adjusted_rand_score(gt_array, labels)
    nmi = normalized_mutual_info_score(gt_array, labels)

    # Silhouette Score on the spectral embedding
    n_labels = len(set(labels))
    silhouette = np.nan
    if 2 <= n_labels < len(labels):
        try:
            silhouette = silhouette_score(embedding, labels)
        except ValueError as e:
            logger.warning("Silhouette score failed: %s", e)

    # Cluster size distribution
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = {int(c): int(n) for c, n in zip(unique, counts)}

    return {
        "ari": float(ari),
        "nmi": float(nmi),
        "silhouette": float(silhouette) if not np.isnan(silhouette) else None,
        "cluster_sizes": cluster_sizes,
        "n_clusters": n_labels,
    }


# ---------------------------------------------------------------------------
# Multi-k experiment
# ---------------------------------------------------------------------------

def run_experiments(
    W: np.ndarray,
    reviewer_ids: list[str],
    gt_df: pd.DataFrame,
    k_min: int = 2,
    k_max: int = 8,
    laplacian_type: str = "normalized",
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Run spectral clustering for a range of k values."""
    results = []
    for k in range(k_min, k_max + 1):
        logger.info("Running experiment: k=%d, laplacian=%s", k, laplacian_type)
        t0 = time.time()

        try:
            embedding, eigenvalues = spectral_embedding(W, k, laplacian_type)
            labels, inertia = run_kmeans(embedding, k, seed=seed)
            metrics = evaluate_clustering(labels, reviewer_ids, gt_df, embedding)
            elapsed = time.time() - t0

            result = {
                "k": k,
                "laplacian": laplacian_type,
                "ari": metrics["ari"],
                "nmi": metrics["nmi"],
                "silhouette": metrics["silhouette"],
                "inertia": inertia,
                "cluster_sizes": metrics["cluster_sizes"],
                "runtime_seconds": round(elapsed, 3),
                "labels": labels,
                "embedding": embedding,
                "eigenvalues": eigenvalues[:k + 2].tolist(),
                "status": "success",
            }
        except Exception as e:
            elapsed = time.time() - t0
            logger.error("Experiment k=%d failed: %s", k, e)
            result = {
                "k": k,
                "laplacian": laplacian_type,
                "status": "failed",
                "error": str(e),
                "runtime_seconds": round(elapsed, 3),
            }

        results.append(result)
        if result["status"] == "success":
            logger.info(
                "  k=%d: ARI=%.4f, NMI=%.4f, Silhouette=%s, sizes=%s (%.2fs)",
                k, result["ari"], result["nmi"],
                f"{result['silhouette']:.4f}" if result["silhouette"] is not None else "N/A",
                result["cluster_sizes"], elapsed,
            )

    return results


def select_primary_k(
    results: list[dict[str, Any]],
    k_primary: int | None = None,
) -> dict[str, Any]:
    """
    Select the primary experiment.

    Selection procedure (unsupervised — does NOT use ground-truth):
    1. If k_primary is specified, use that k.
    2. Otherwise, select the k with the highest valid Silhouette Score.
       Silhouette is a structural metric that does not require labels.
    """
    successful = [r for r in results if r["status"] == "success"]
    if not successful:
        raise RuntimeError("All experiments failed — cannot select primary k")

    if k_primary is not None:
        match = [r for r in successful if r["k"] == k_primary]
        if match:
            return match[0]
        logger.warning("Requested k=%d not found; falling back to auto-select", k_primary)

    # Unsupervised selection: best silhouette score
    with_silhouette = [r for r in successful if r["silhouette"] is not None]
    if with_silhouette:
        best = max(with_silhouette, key=lambda r: r["silhouette"])
        logger.info(
            "Auto-selected k=%d (best silhouette=%.4f) — "
            "selection based on structural metric, not ground-truth labels",
            best["k"], best["silhouette"],
        )
        return best

    # Fallback: first successful
    return successful[0]


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_spectral_embedding(
    embedding: np.ndarray,
    labels: np.ndarray,
    reviewer_ids: list[str],
    output_path: Path,
    k: int,
) -> None:
    """Scatter plot of the first two spectral embedding dimensions."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1] if embedding.shape[1] > 1 else np.zeros(len(embedding)),
        c=labels, cmap="tab10", alpha=0.6, s=15, edgecolors="none",
    )
    ax.set_xlabel("Spectral Dimension 1", fontsize=11)
    ax.set_ylabel("Spectral Dimension 2", fontsize=11)
    ax.set_title(f"Spectral Embedding (k={k}) — Colored by Predicted Cluster", fontsize=13)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Cluster ID")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_ground_truth_comparison(
    embedding: np.ndarray,
    labels: np.ndarray,
    reviewer_ids: list[str],
    gt_df: pd.DataFrame,
    output_path: Path,
    k: int,
) -> None:
    """Side-by-side: predicted clusters vs ground-truth categories."""
    gt_indexed = gt_df.set_index("reviewer_id")
    gt_cats = [gt_indexed.loc[rid, "ground_truth_category"] if rid in gt_indexed.index else "unknown"
               for rid in reviewer_ids]
    cat_names = sorted(set(gt_cats))
    cat_map = {c: i for i, c in enumerate(cat_names)}
    gt_numeric = np.array([cat_map[c] for c in gt_cats])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    dim2 = embedding[:, 1] if embedding.shape[1] > 1 else np.zeros(len(embedding))

    sc1 = ax1.scatter(embedding[:, 0], dim2, c=labels, cmap="tab10", alpha=0.6, s=15, edgecolors="none")
    ax1.set_title(f"Predicted Clusters (k={k})", fontsize=12)
    ax1.set_xlabel("Spectral Dim 1")
    ax1.set_ylabel("Spectral Dim 2")
    plt.colorbar(sc1, ax=ax1, label="Cluster ID")

    sc2 = ax2.scatter(embedding[:, 0], dim2, c=gt_numeric, cmap="Set1", alpha=0.6, s=15, edgecolors="none")
    ax2.set_title("Ground-Truth Categories (Evaluation Only)", fontsize=12)
    ax2.set_xlabel("Spectral Dim 1")
    ax2.set_ylabel("Spectral Dim 2")
    cbar2 = plt.colorbar(sc2, ax=ax2, label="Category")
    cbar2.set_ticks(range(len(cat_names)))
    cbar2.set_ticklabels(cat_names)

    fig.suptitle("Spectral Clustering vs Ground Truth\n(Ground truth used for evaluation only, not as model input)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_cluster_behavior(
    behavior_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Bar charts comparing behavioral features across clusters."""
    metrics = ["avg_rating", "mean_rating_deviation", "mean_abs_rating_deviation",
               "review_count", "frac_high_rating", "frac_low_rating"]
    available = [m for m in metrics if m in behavior_df.columns]
    n_metrics = len(available)
    if n_metrics == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    labels_display = {
        "avg_rating": "Avg Rating",
        "mean_rating_deviation": "Mean Rating Deviation\n(signed)",
        "mean_abs_rating_deviation": "Mean Abs Rating Deviation",
        "review_count": "Avg Review Count",
        "frac_high_rating": "Fraction 4-5 Star",
        "frac_low_rating": "Fraction 1-2 Star",
    }

    for i, metric in enumerate(available):
        ax = axes[i]
        clusters = behavior_df["cluster_id"].astype(str)
        values = behavior_df[metric]
        colors = plt.cm.tab10(np.arange(len(clusters)) % 10)
        ax.bar(clusters, values, color=colors, alpha=0.8, edgecolor="white")
        ax.set_xlabel("Cluster")
        ax.set_ylabel(labels_display.get(metric, metric))
        ax.set_title(labels_display.get(metric, metric), fontsize=11)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    for i in range(len(available), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Cluster Behavioral Profiles\n(Patterns, not fraud classifications)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_k_evaluation(
    results: list[dict[str, Any]],
    primary_k: int,
    output_path: Path,
) -> None:
    """Plot ARI, NMI, and Silhouette across different k values."""
    successful = [r for r in results if r["status"] == "success"]
    if not successful:
        return

    ks = [r["k"] for r in successful]
    aris = [r["ari"] for r in successful]
    nmis = [r["nmi"] for r in successful]
    sils = [r["silhouette"] if r["silhouette"] is not None else 0 for r in successful]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(ks, aris, "o-", label="ARI", color="#e74c3c", linewidth=2, markersize=8)
    ax1.plot(ks, nmis, "s-", label="NMI", color="#3498db", linewidth=2, markersize=8)
    ax1.axvline(x=primary_k, color="gray", linestyle="--", alpha=0.5, label=f"Selected k={primary_k}")
    ax1.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_title("Clustering Quality vs Ground Truth\n(Evaluation only)", fontsize=12)
    ax1.legend()
    ax1.set_xticks(ks)
    ax1.grid(alpha=0.3)

    ax2.plot(ks, sils, "D-", label="Silhouette", color="#2ecc71", linewidth=2, markersize=8)
    ax2.axvline(x=primary_k, color="gray", linestyle="--", alpha=0.5, label=f"Selected k={primary_k}")
    ax2.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax2.set_ylabel("Silhouette Score", fontsize=11)
    ax2.set_title("Structural Quality (Unsupervised)\n(Used for k selection)", fontsize=12)
    ax2.legend()
    ax2.set_xticks(ks)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------

def save_outputs(
    primary: dict[str, Any],
    all_results: list[dict[str, Any]],
    reviewer_ids: list[str],
    gt_df: pd.DataFrame,
    features_df: pd.DataFrame,
    contingency: pd.DataFrame,
    behavior_df: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Save all Module 3 outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    labels = primary["labels"]

    # 1. Cluster assignments
    assign_df = pd.DataFrame({
        "reviewer_id": reviewer_ids,
        "cluster_id": labels,
    })
    assign_path = output_dir / "reviewer_cluster_assignments.csv"
    assign_df.to_csv(assign_path, index=False)
    paths["cluster_assignments"] = assign_path

    # 2. Clustering metrics (all k values)
    metrics_rows = []
    for r in all_results:
        if r["status"] == "success":
            metrics_rows.append({
                "k": r["k"],
                "laplacian": r["laplacian"],
                "ari": r["ari"],
                "nmi": r["nmi"],
                "silhouette": r["silhouette"],
                "inertia": r.get("inertia"),
                "runtime_seconds": r["runtime_seconds"],
            })
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = output_dir / "clustering_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    paths["clustering_metrics"] = metrics_path

    # 3. Ground-truth comparison
    cont_path = output_dir / "cluster_ground_truth_comparison.csv"
    contingency.to_csv(cont_path)
    paths["ground_truth_comparison"] = cont_path

    # 4. Cluster behavior summary
    behavior_path = output_dir / "cluster_behavior_summary.csv"
    behavior_df.to_csv(behavior_path, index=False)
    paths["behavior_summary"] = behavior_path

    # 5. Experiment summary JSON
    summary = {
        "config": config,
        "primary_k": int(primary["k"]),
        "primary_metrics": {
            "ari": primary["ari"],
            "nmi": primary["nmi"],
            "silhouette": primary["silhouette"],
            "cluster_sizes": primary["cluster_sizes"],
        },
        "all_experiments": [
            {k: v for k, v in r.items()
             if k not in ("labels", "embedding", "eigenvalues")}
            for r in all_results
        ],
        "eigenvalues_primary": primary.get("eigenvalues", []),
    }
    summary_path = output_dir / "experiment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    paths["experiment_summary"] = summary_path

    # 6. Evaluation report
    report_text = format_evaluation_report(
        primary, all_results, contingency, behavior_df, config,
    )
    report_path = output_dir / "evaluation_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    paths["evaluation_report"] = report_path

    for label, p in paths.items():
        logger.info("Saved %s -> %s", label, p)

    return paths


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_module3(
    graph_dir: Path,
    features_dir: Path,
    output_dir: Path,
    k_min: int = 2,
    k_max: int = 8,
    k_primary: int | None = None,
    laplacian_type: str = "normalized",
    seed: int = 42,
) -> dict[str, Any]:
    """Execute the full Module 3 pipeline."""

    config = {
        "graph_dir": str(graph_dir),
        "features_dir": str(features_dir),
        "k_min": k_min,
        "k_max": k_max,
        "k_primary_requested": k_primary,
        "laplacian": laplacian_type,
        "seed": seed,
        "kmeans_n_init": 20,
    }

    # Step 1: Load inputs
    W, adj_ids = load_adjacency_matrix(graph_dir)
    mapping_ids = load_node_mapping(graph_dir)
    validate_ordering(adj_ids, mapping_ids)
    reviewer_ids = adj_ids

    gt_df = load_ground_truth(features_dir)
    features_df = load_reviewer_features(features_dir)

    # Leakage guard
    leaked = set(features_df.columns) & GROUND_TRUTH_COLUMNS
    if leaked:
        logger.warning("Removing leaked columns from features: %s", leaked)
        features_df = features_df.drop(columns=list(leaked), errors="ignore")

    # Step 2: Run experiments for k_min..k_max
    all_results = run_experiments(
        W, reviewer_ids, gt_df,
        k_min=k_min, k_max=k_max,
        laplacian_type=laplacian_type, seed=seed,
    )

    # Step 3: Select primary k
    primary = select_primary_k(all_results, k_primary)
    config["k_primary_selected"] = int(primary["k"])
    config["selection_method"] = (
        f"user-specified k={k_primary}" if k_primary is not None
        else "best silhouette score (unsupervised structural criterion)"
    )

    # Step 4: Build evaluation tables
    labels = primary["labels"]
    contingency = build_contingency_table(labels, reviewer_ids, gt_df)
    behavior_df = build_cluster_behavior_summary(labels, reviewer_ids, features_df)

    # Step 5: Save outputs
    paths = save_outputs(
        primary, all_results, reviewer_ids, gt_df, features_df,
        contingency, behavior_df, config, output_dir,
    )

    # Step 6: Visualizations
    fig_dir = output_dir.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    embedding = primary["embedding"]
    k = primary["k"]

    plot_spectral_embedding(embedding, labels, reviewer_ids,
                            fig_dir / "spectral_embedding.png", k)
    paths["fig_embedding"] = fig_dir / "spectral_embedding.png"

    plot_ground_truth_comparison(embedding, labels, reviewer_ids, gt_df,
                                fig_dir / "ground_truth_comparison.png", k)
    paths["fig_gt_comparison"] = fig_dir / "ground_truth_comparison.png"

    plot_cluster_behavior(behavior_df, fig_dir / "cluster_behavior.png")
    paths["fig_behavior"] = fig_dir / "cluster_behavior.png"

    plot_k_evaluation(all_results, k, fig_dir / "k_evaluation.png")
    paths["fig_k_evaluation"] = fig_dir / "k_evaluation.png"

    return {
        "primary": primary,
        "all_results": all_results,
        "contingency": contingency,
        "behavior": behavior_df,
        "config": config,
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
        description="Module 3: Spectral Clustering and Evaluation",
    )
    parser.add_argument("--graph-dir", type=Path, default=Path("outputs/graph"))
    parser.add_argument("--features-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/clustering"))
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--k-primary", type=int, default=None)
    parser.add_argument("--laplacian", type=str, default="normalized",
                        choices=["normalized", "unnormalized"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = run_module3(
        graph_dir=args.graph_dir,
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        k_min=args.k_min,
        k_max=args.k_max,
        k_primary=args.k_primary,
        laplacian_type=args.laplacian,
        seed=args.seed,
    )

    primary = result["primary"]
    config = result["config"]

    print("\n" + "=" * 70)
    print("MODULE 3 — EXECUTION COMPLETE")
    print("=" * 70)
    print(f"Primary k                   : {primary['k']}")
    print(f"Selection method            : {config['selection_method']}")
    print(f"Laplacian                   : {config['laplacian']}")
    print(f"ARI                         : {primary['ari']:.4f}")
    print(f"NMI                         : {primary['nmi']:.4f}")
    sil = primary['silhouette']
    print(f"Silhouette Score            : {f'{sil:.4f}' if sil is not None else 'N/A'}")
    print(f"Cluster sizes               : {primary['cluster_sizes']}")
    print(f"\nContingency table:")
    print(result["contingency"].to_string())
    print(f"\nOutput files:")
    for label, p in result["paths"].items():
        print(f"  {label}: {p}")
    print("=" * 70)


if __name__ == "__main__":
    main()
