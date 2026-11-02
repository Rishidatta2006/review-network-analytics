"""
similarity_graph.py — Module 2: Reviewer Similarity Network and Graph Construction.

Loads the scaled reviewer feature matrix from Module 1, computes cosine
similarity, constructs a weighted undirected graph via threshold or top-k
nearest-neighbour edge selection, and produces the adjacency matrix, degree
matrix, and graph Laplacians.

Run with:
    python -m src.similarity_graph [OPTIONS]

Options:
    --input-dir       Directory containing Module 1 outputs (default: outputs/)
    --output-dir      Directory for graph outputs (default: outputs/graph/)
    --strategy        Edge selection: 'threshold' or 'topk' (default: topk)
    --threshold       Similarity threshold for 'threshold' strategy (default: 0.3)
    --top-k           Number of nearest neighbours for 'topk' strategy (default: 15)
    --min-similarity  Minimum similarity to retain in top-k (default: 0.0)
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.feature_engineering import GROUND_TRUTH_COLUMNS
from src.graph_validation import validate_graph, GraphValidationReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_feature_matrix(input_dir: Path) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Load the scaled reviewer feature matrix from Module 1.

    Parameters
    ----------
    input_dir : Path
        Directory containing reviewer_features_scaled.csv.

    Returns
    -------
    tuple of (feature_array, reviewer_ids, feature_columns)
        feature_array : np.ndarray of shape (n_reviewers, n_features)
        reviewer_ids  : list of reviewer ID strings
        feature_columns : list of feature column names
    """
    scaled_path = input_dir / "reviewer_features_scaled.csv"
    if not scaled_path.exists():
        raise FileNotFoundError(
            f"Scaled feature matrix not found at {scaled_path}. "
            f"Run Module 1 first: python -m src.preprocessing"
        )

    df = pd.read_csv(scaled_path)

    if "reviewer_id" not in df.columns:
        raise ValueError("reviewer_features_scaled.csv must contain a 'reviewer_id' column")

    # Data leakage guard
    leaked = set(df.columns) & GROUND_TRUTH_COLUMNS
    if leaked:
        raise RuntimeError(
            f"DATA LEAKAGE: ground-truth columns found in feature matrix: {leaked}"
        )

    reviewer_ids = df["reviewer_id"].tolist()
    feature_cols = [c for c in df.columns if c != "reviewer_id"]
    feature_array = df[feature_cols].values.astype(np.float64)

    # Check for duplicates
    if len(reviewer_ids) != len(set(reviewer_ids)):
        raise ValueError("Duplicate reviewer IDs in feature matrix")

    # Check for NaN
    n_nan = np.isnan(feature_array).sum()
    if n_nan > 0:
        raise ValueError(f"Feature matrix contains {n_nan} NaN values")

    logger.info(
        "Loaded feature matrix: %d reviewers x %d features",
        len(reviewer_ids), len(feature_cols),
    )
    return feature_array, reviewer_ids, feature_cols


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def compute_cosine_similarity(
    X: np.ndarray,
    reviewer_ids: list[str],
) -> np.ndarray:
    """
    Compute the full pairwise cosine similarity matrix.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n, d).
    reviewer_ids : list[str]
        Reviewer IDs corresponding to rows.

    Returns
    -------
    np.ndarray
        Symmetric similarity matrix of shape (n, n) with diagonal set to 0.

    Notes
    -----
    Memory requirement: n^2 * 8 bytes (float64).
    For n=1000 this is ~8 MB, well within practical limits.
    For n>50000, consider blockwise computation or approximate methods.

    Zero-vector handling: sklearn's cosine_similarity returns 0 for pairs
    involving a zero vector (since ||0|| = 0, the cosine is undefined).
    We detect zero vectors, log a warning, and keep the 0 value — these
    reviewers will have no edges (isolated nodes) which is the correct
    structural interpretation.
    """
    n = X.shape[0]
    mem_mb = n * n * 8 / (1024 * 1024)
    logger.info(
        "Computing %d x %d cosine similarity matrix (est. %.1f MB)",
        n, n, mem_mb,
    )

    # Detect zero vectors
    norms = np.linalg.norm(X, axis=1)
    zero_mask = norms == 0
    n_zero = int(zero_mask.sum())
    if n_zero > 0:
        zero_ids = [reviewer_ids[i] for i in range(n) if zero_mask[i]]
        logger.warning(
            "%d zero-vector reviewer(s) detected: %s. "
            "These will become isolated nodes (cosine similarity = 0).",
            n_zero, zero_ids[:10],
        )

    # Compute cosine similarity (numerically stable via sklearn)
    S = cosine_similarity(X)

    # Enforce exact symmetry (handle floating-point asymmetry)
    S = (S + S.T) / 2.0

    # Clamp to [-1, 1] to handle numerical drift
    np.clip(S, -1.0, 1.0, out=S)

    # Set diagonal to zero (no self-loops)
    np.fill_diagonal(S, 0.0)

    # Verify finiteness
    if not np.all(np.isfinite(S)):
        raise RuntimeError("Cosine similarity matrix contains non-finite values")

    logger.info(
        "Similarity stats: min=%.4f, max=%.4f, mean=%.4f",
        S.min(), S.max(), S[np.triu_indices(n, k=1)].mean(),
    )

    return S


# ---------------------------------------------------------------------------
# Edge selection strategies
# ---------------------------------------------------------------------------

def threshold_graph(
    S: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Strategy A: Create edges where similarity >= threshold.

    Parameters
    ----------
    S : np.ndarray
        Full cosine similarity matrix with zero diagonal.
    threshold : float
        Minimum similarity for edge creation.

    Returns
    -------
    np.ndarray
        Weighted adjacency matrix W (symmetric, non-negative, zero diagonal).

    Trade-offs
    ----------
    - Can produce a disconnected or very sparse graph if threshold is too high.
    - Very low thresholds produce dense graphs that obscure network structure.
    - Default threshold of 0.3 retains moderately strong similarities while
      filtering noise from weakly-correlated reviewer pairs.
    """
    W = np.where(S >= threshold, S, 0.0)
    # Clamp negatives (cosine similarity can be negative for opposing vectors)
    W = np.maximum(W, 0.0)
    np.fill_diagonal(W, 0.0)
    return W


def topk_graph(
    S: np.ndarray,
    k: int,
    min_similarity: float = 0.0,
) -> np.ndarray:
    """
    Strategy B: Top-k nearest neighbours with union symmetrization.

    For each reviewer, the k most similar other reviewers are selected.
    An undirected edge is created if *either* reviewer selects the other
    (union rule). Edge weight is the original cosine similarity.

    Parameters
    ----------
    S : np.ndarray
        Full cosine similarity matrix with zero diagonal.
    k : int
        Number of nearest neighbours per reviewer.
    min_similarity : float
        Minimum similarity to retain even among top-k.

    Returns
    -------
    np.ndarray
        Symmetric weighted adjacency matrix W.

    Trade-offs
    ----------
    - Ensures every non-zero-vector reviewer has at least k connections,
      preventing excessive isolation.
    - May introduce weaker edges compared to a pure threshold approach.
    - Union symmetrization results in some nodes having more than k edges.
    - k=15 balances local structure preservation with sparsity for n=1000.
    """
    n = S.shape[0]
    W = np.zeros_like(S)

    for i in range(n):
        row = S[i].copy()
        row[i] = -np.inf  # exclude self
        # Select top-k indices
        topk_idx = np.argpartition(row, -k)[-k:]
        for j in topk_idx:
            if row[j] >= min_similarity and row[j] > 0:
                W[i, j] = S[i, j]

    # Symmetrize: union rule — edge if either reviewer selects the other
    W = np.maximum(W, W.T)

    # Ensure non-negative and zero diagonal
    W = np.maximum(W, 0.0)
    np.fill_diagonal(W, 0.0)

    return W


# ---------------------------------------------------------------------------
# Graph matrices
# ---------------------------------------------------------------------------

def compute_degree_matrix(W: np.ndarray) -> np.ndarray:
    """
    Compute the diagonal weighted degree matrix D.

    D_ii = sum_j W_ij
    """
    degrees = W.sum(axis=1)
    return np.diag(degrees)


def compute_unnormalized_laplacian(D: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Compute the unnormalized graph Laplacian L = D - W.

    Properties:
    - Symmetric (since both D and W are symmetric).
    - Positive semi-definite.
    - Each row sums to approximately zero.
    """
    return D - W


def compute_normalized_laplacian(D: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Compute the symmetric normalized Laplacian L_sym = I - D^{-1/2} W D^{-1/2}.

    Zero-degree nodes are handled safely: their corresponding rows/columns
    in the normalized Laplacian are set to 0 (the row of identity minus zeros).
    """
    n = W.shape[0]
    degrees = np.diag(D)

    # Safe inverse square root of degrees
    d_inv_sqrt = np.zeros(n)
    nonzero = degrees > 0
    d_inv_sqrt[nonzero] = 1.0 / np.sqrt(degrees[nonzero])

    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_sym = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt

    # Enforce symmetry
    L_sym = (L_sym + L_sym.T) / 2.0

    return L_sym


# ---------------------------------------------------------------------------
# NetworkX graph construction
# ---------------------------------------------------------------------------

def build_networkx_graph(
    W: np.ndarray,
    reviewer_ids: list[str],
) -> nx.Graph:
    """
    Construct an undirected weighted NetworkX graph from the adjacency matrix.

    Parameters
    ----------
    W : np.ndarray
        Symmetric weighted adjacency matrix.
    reviewer_ids : list[str]
        Reviewer IDs corresponding to matrix rows.

    Returns
    -------
    nx.Graph
        Undirected weighted graph with all reviewers as nodes.
    """
    G = nx.Graph()

    # Add all reviewers as nodes (including potential isolates)
    for idx, rid in enumerate(reviewer_ids):
        G.add_node(rid, matrix_index=idx)

    # Add weighted edges
    n = len(reviewer_ids)
    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] > 0:
                G.add_edge(reviewer_ids[i], reviewer_ids[j], weight=W[i, j])

    logger.info(
        "NetworkX graph: %d nodes, %d edges",
        G.number_of_nodes(), G.number_of_edges(),
    )
    return G


# ---------------------------------------------------------------------------
# Graph statistics
# ---------------------------------------------------------------------------

def compute_graph_statistics(
    G: nx.Graph,
    W: np.ndarray,
    strategy: str,
    strategy_params: dict[str, Any],
) -> dict[str, Any]:
    """Compute comprehensive graph statistics for the report."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    max_edges = n * (n - 1) // 2

    # Degree statistics
    degrees = [d for _, d in G.degree()]
    w_degrees = [d for _, d in G.degree(weight="weight")]

    # Connected components
    components = list(nx.connected_components(G))
    component_sizes = sorted([len(c) for c in components], reverse=True)
    isolated = [node for node in G.nodes() if G.degree(node) == 0]

    # Edge weight statistics
    edge_weights = [d["weight"] for _, _, d in G.edges(data=True)]

    stats = {
        "node_count": n,
        "edge_count": m,
        "max_possible_edges": max_edges,
        "edge_retention_ratio": m / max_edges if max_edges > 0 else 0,
        "graph_density": nx.density(G),
        "avg_degree": np.mean(degrees) if degrees else 0,
        "min_degree": min(degrees) if degrees else 0,
        "max_degree": max(degrees) if degrees else 0,
        "median_degree": float(np.median(degrees)) if degrees else 0,
        "avg_weighted_degree": np.mean(w_degrees) if w_degrees else 0,
        "min_edge_weight": min(edge_weights) if edge_weights else 0,
        "max_edge_weight": max(edge_weights) if edge_weights else 0,
        "mean_edge_weight": float(np.mean(edge_weights)) if edge_weights else 0,
        "connected_components": len(components),
        "largest_component_size": component_sizes[0] if component_sizes else 0,
        "component_size_distribution": component_sizes,
        "isolated_node_count": len(isolated),
        "isolated_nodes": sorted(isolated)[:20],
        "strategy": strategy,
        "strategy_params": strategy_params,
        "degree_distribution_summary": {
            "mean": float(np.mean(degrees)) if degrees else 0,
            "std": float(np.std(degrees)) if degrees else 0,
            "min": int(min(degrees)) if degrees else 0,
            "25%": float(np.percentile(degrees, 25)) if degrees else 0,
            "50%": float(np.percentile(degrees, 50)) if degrees else 0,
            "75%": float(np.percentile(degrees, 75)) if degrees else 0,
            "max": int(max(degrees)) if degrees else 0,
        },
    }
    return stats


def format_stats_text(stats: dict[str, Any], validation: GraphValidationReport) -> str:
    """Format graph statistics and validation results as human-readable text."""
    lines = [
        "=" * 70,
        "MODULE 2 — GRAPH CONSTRUCTION REPORT",
        "=" * 70, "",
        "— GRAPH STRUCTURE —",
        f"  Nodes (reviewers)         : {stats['node_count']}",
        f"  Edges                     : {stats['edge_count']}",
        f"  Max possible edges        : {stats['max_possible_edges']}",
        f"  Edge retention ratio      : {stats['edge_retention_ratio']:.4f}",
        f"  Graph density             : {stats['graph_density']:.6f}",
        "",
        "— DEGREE STATISTICS —",
        f"  Average degree            : {stats['avg_degree']:.2f}",
        f"  Min / Max degree          : {stats['min_degree']} / {stats['max_degree']}",
        f"  Median degree             : {stats['median_degree']:.1f}",
        f"  Average weighted degree   : {stats['avg_weighted_degree']:.4f}",
        "",
        "— EDGE WEIGHTS —",
        f"  Min edge weight           : {stats['min_edge_weight']:.4f}",
        f"  Max edge weight           : {stats['max_edge_weight']:.4f}",
        f"  Mean edge weight          : {stats['mean_edge_weight']:.4f}",
        "",
        "— CONNECTIVITY —",
        f"  Connected components      : {stats['connected_components']}",
        f"  Largest component size    : {stats['largest_component_size']}",
        f"  Isolated nodes            : {stats['isolated_node_count']}",
        f"  Component sizes           : {stats['component_size_distribution'][:10]}"
        + (" ..." if len(stats['component_size_distribution']) > 10 else ""),
        "",
        "— STRATEGY —",
        f"  Edge selection            : {stats['strategy']}",
        f"  Parameters                : {stats['strategy_params']}",
        "",
        "— VALIDATION RESULTS —",
    ]
    for result in validation.results:
        status = "PASS" if result["passed"] else "FAIL"
        lines.append(f"  [{status}] {result['check']}")
        if not result["passed"]:
            lines.append(f"          Detail: {result.get('detail', 'N/A')}")

    lines.extend([
        "",
        f"  Total: {validation.passed_count} passed, {validation.failed_count} failed",
        "",
        "=" * 70,
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_graph(
    G: nx.Graph,
    output_path: Path,
    max_nodes: int = 200,
    seed: int = 42,
) -> None:
    """
    Create and save a basic network visualization.

    If the graph has more than max_nodes, a connected subgraph is sampled
    for visualization. The full graph is never modified.

    This is a structural view, not a fraud classification.
    Nodes are NOT colored by ground-truth labels.
    """
    if G.number_of_nodes() <= max_nodes:
        subG = G
        title_suffix = f"(all {G.number_of_nodes()} nodes)"
    else:
        # Sample the largest connected component, truncated to max_nodes
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        largest = components[0]
        if len(largest) > max_nodes:
            # Take a BFS subgraph from the highest-degree node
            sub_nodes = sorted(largest, key=lambda n: G.degree(n), reverse=True)
            sub_nodes = sub_nodes[:max_nodes]
        else:
            sub_nodes = list(largest)
        subG = G.subgraph(sub_nodes).copy()
        title_suffix = f"(sampled {subG.number_of_nodes()} of {G.number_of_nodes()} nodes)"

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Layout
    pos = nx.spring_layout(subG, seed=seed, k=1.5 / np.sqrt(max(subG.number_of_nodes(), 1)))

    # Edge weights for width/alpha
    edges = subG.edges(data=True)
    if len(list(edges)) > 0:
        weights = [d.get("weight", 0.1) for _, _, d in subG.edges(data=True)]
        max_w = max(weights) if weights else 1
        edge_widths = [0.3 + 2.0 * w / max_w for w in weights]
        edge_alphas = [0.15 + 0.6 * w / max_w for w in weights]

        nx.draw_networkx_edges(
            subG, pos, ax=ax,
            width=edge_widths,
            alpha=0.3,
            edge_color="#6c757d",
        )

    # Node degree for size
    node_degrees = dict(subG.degree())
    max_deg = max(node_degrees.values()) if node_degrees else 1
    node_sizes = [30 + 200 * node_degrees[n] / max_deg for n in subG.nodes()]

    nx.draw_networkx_nodes(
        subG, pos, ax=ax,
        node_size=node_sizes,
        node_color="#4a90d9",
        alpha=0.7,
        edgecolors="#2c5282",
        linewidths=0.5,
    )

    ax.set_title(
        f"Reviewer Similarity Network {title_suffix}\n"
        f"(Structural view — not fraud classification)",
        fontsize=12, fontweight="bold",
    )
    ax.axis("off")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved visualization to %s", output_path)


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------

def save_graph_outputs(
    S: np.ndarray,
    W: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    L_sym: np.ndarray,
    reviewer_ids: list[str],
    G: nx.Graph,
    stats: dict[str, Any],
    validation: GraphValidationReport,
    output_dir: Path,
) -> dict[str, Path]:
    """Save all Module 2 outputs and return file paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    n = len(reviewer_ids)

    # Node mapping (stable ordering)
    mapping_df = pd.DataFrame({
        "matrix_index": range(n),
        "reviewer_id": reviewer_ids,
    })
    mapping_path = output_dir / "reviewer_node_mapping.csv"
    mapping_df.to_csv(mapping_path, index=False)
    paths["reviewer_node_mapping"] = mapping_path

    # Similarity matrix — save as sparse for efficiency documentation,
    # but use dense CSV since n=1000 is manageable (~8MB)
    sim_df = pd.DataFrame(S, index=reviewer_ids, columns=reviewer_ids)
    sim_path = output_dir / "reviewer_similarity_matrix.csv"
    sim_df.to_csv(sim_path)
    paths["reviewer_similarity_matrix"] = sim_path

    # Adjacency matrix — save sparse representation for large graphs
    # For n=1000, dense CSV is practical
    adj_df = pd.DataFrame(W, index=reviewer_ids, columns=reviewer_ids)
    adj_path = output_dir / "graph_adjacency_matrix.csv"
    adj_df.to_csv(adj_path)
    paths["graph_adjacency_matrix"] = adj_path

    # Degree matrix — save only the diagonal values (full matrix is diagonal)
    deg_df = pd.DataFrame({
        "reviewer_id": reviewer_ids,
        "weighted_degree": np.diag(D),
    })
    deg_path = output_dir / "graph_degree_matrix.csv"
    deg_df.to_csv(deg_path, index=False)
    paths["graph_degree_matrix"] = deg_path

    # Unnormalized Laplacian
    lap_df = pd.DataFrame(L, index=reviewer_ids, columns=reviewer_ids)
    lap_path = output_dir / "graph_laplacian.csv"
    lap_df.to_csv(lap_path)
    paths["graph_laplacian"] = lap_path

    # Normalized Laplacian — save as NumPy binary for space efficiency
    lsym_path = output_dir / "graph_laplacian_normalized.npy"
    np.save(lsym_path, L_sym)
    paths["graph_laplacian_normalized"] = lsym_path

    # GraphML with reviewer IDs and edge weights
    graphml_path = output_dir / "reviewer_network.graphml"
    nx.write_graphml(G, str(graphml_path))
    paths["reviewer_network_graphml"] = graphml_path

    # Statistics JSON
    stats_path = output_dir / "graph_statistics.json"
    stats_path.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    paths["graph_statistics"] = stats_path

    # Validation report text
    report_text = format_stats_text(stats, validation)
    report_path = output_dir / "graph_validation_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    paths["graph_validation_report"] = report_path

    for label, p in paths.items():
        logger.info("Saved %s -> %s", label, p)

    return paths


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_module2(
    input_dir: Path,
    output_dir: Path,
    strategy: str = "topk",
    threshold: float = 0.3,
    top_k: int = 15,
    min_similarity: float = 0.0,
) -> dict[str, Any]:
    """
    Execute the full Module 2 pipeline.

    Parameters
    ----------
    input_dir : Path
        Directory containing Module 1 outputs.
    output_dir : Path
        Directory for graph outputs.
    strategy : str
        'threshold' or 'topk'.
    threshold : float
        Similarity threshold for threshold strategy.
    top_k : int
        Number of neighbours for top-k strategy.
    min_similarity : float
        Minimum similarity for top-k edges.

    Returns
    -------
    dict with all pipeline outputs.
    """
    # Step 1: Load features
    X, reviewer_ids, feature_cols = load_feature_matrix(input_dir)

    # Step 2: Compute cosine similarity
    S = compute_cosine_similarity(X, reviewer_ids)

    # Step 3: Build weighted adjacency matrix
    strategy_params: dict[str, Any] = {"strategy": strategy}
    if strategy == "threshold":
        W = threshold_graph(S, threshold)
        strategy_params["threshold"] = threshold
        logger.info("Threshold strategy: tau=%.3f", threshold)
    elif strategy == "topk":
        W = topk_graph(S, top_k, min_similarity)
        strategy_params["top_k"] = top_k
        strategy_params["min_similarity"] = min_similarity
        logger.info("Top-k strategy: k=%d, min_similarity=%.3f", top_k, min_similarity)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'threshold' or 'topk'.")

    # Step 4: Compute graph matrices
    D = compute_degree_matrix(W)
    L = compute_unnormalized_laplacian(D, W)
    L_sym = compute_normalized_laplacian(D, W)

    # Step 5: Build NetworkX graph
    G = build_networkx_graph(W, reviewer_ids)

    # Step 6: Validate
    validation = validate_graph(G, W, D, L, reviewer_ids)

    # Step 7: Statistics
    stats = compute_graph_statistics(G, W, strategy, strategy_params)

    # Step 8: Save outputs
    paths = save_graph_outputs(
        S, W, D, L, L_sym, reviewer_ids, G, stats, validation, output_dir,
    )

    # Step 9: Visualization
    fig_dir = output_dir.parent / "figures"
    fig_path = fig_dir / "reviewer_network.png"
    visualize_graph(G, fig_path)
    paths["visualization"] = fig_path

    return {
        "S": S,
        "W": W,
        "D": D,
        "L": L,
        "L_sym": L_sym,
        "G": G,
        "reviewer_ids": reviewer_ids,
        "stats": stats,
        "validation": validation,
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
        description="Module 2: Reviewer Similarity Network and Graph Construction",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("outputs"),
        help="Directory containing Module 1 outputs (default: outputs/)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/graph"),
        help="Directory for graph outputs (default: outputs/graph/)",
    )
    parser.add_argument(
        "--strategy", type=str, default="topk", choices=["threshold", "topk"],
        help="Edge selection strategy (default: topk)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.3,
        help="Similarity threshold for 'threshold' strategy (default: 0.3)",
    )
    parser.add_argument(
        "--top-k", type=int, default=15,
        help="Number of nearest neighbours for 'topk' strategy (default: 15)",
    )
    parser.add_argument(
        "--min-similarity", type=float, default=0.0,
        help="Minimum similarity for top-k edges (default: 0.0)",
    )
    args = parser.parse_args()

    result = run_module2(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        strategy=args.strategy,
        threshold=args.threshold,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
    )

    # Summary
    stats = result["stats"]
    validation = result["validation"]

    print("\n" + "=" * 70)
    print("MODULE 2 — EXECUTION COMPLETE")
    print("=" * 70)
    print(f"Nodes (reviewers)           : {stats['node_count']}")
    print(f"Edges                       : {stats['edge_count']}")
    print(f"Graph density               : {stats['graph_density']:.6f}")
    print(f"Average degree              : {stats['avg_degree']:.2f}")
    print(f"Average weighted degree     : {stats['avg_weighted_degree']:.4f}")
    print(f"Connected components        : {stats['connected_components']}")
    print(f"Isolated nodes              : {stats['isolated_node_count']}")
    print(f"Edge weight range           : [{stats['min_edge_weight']:.4f}, {stats['max_edge_weight']:.4f}]")
    print(f"Strategy                    : {stats['strategy']}")
    print(f"Parameters                  : {stats['strategy_params']}")
    print(f"Validation                  : {validation.passed_count} passed, {validation.failed_count} failed")
    print("\nOutput files:")
    for label, p in result["paths"].items():
        print(f"  {label}: {p}")
    print("=" * 70)


if __name__ == "__main__":
    main()
