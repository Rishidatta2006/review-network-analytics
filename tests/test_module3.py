"""
test_module3.py — Comprehensive unit tests for Module 3: Spectral Clustering.

Run with:
    python -m pytest tests/test_module3.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.spectral_clustering import (
    compute_normalized_laplacian,
    compute_unnormalized_laplacian,
    load_adjacency_matrix,
    run_kmeans,
    spectral_embedding,
    validate_ordering,
    evaluate_clustering,
    run_module3,
)
from src.clustering_evaluation import (
    build_contingency_table,
    build_cluster_behavior_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _simple_adjacency() -> tuple[np.ndarray, list[str]]:
    """Two clear 2-cliques connected by a weak bridge."""
    W = np.array([
        [0, 0.9, 0.1, 0.0],
        [0.9, 0, 0.0, 0.1],
        [0.1, 0.0, 0, 0.9],
        [0.0, 0.1, 0.9, 0],
    ], dtype=np.float64)
    ids = ["R1", "R2", "R3", "R4"]
    return W, ids


def _simple_gt() -> pd.DataFrame:
    return pd.DataFrame({
        "reviewer_id": ["R1", "R2", "R3", "R4"],
        "ground_truth_category": ["normal", "normal", "manipulation", "manipulation"],
    })


def _simple_features() -> pd.DataFrame:
    return pd.DataFrame({
        "avg_rating": [3.5, 3.8, 4.9, 4.7],
        "mean_rating_deviation": [-0.1, 0.0, 1.2, 1.0],
        "mean_absolute_rating_deviation": [0.5, 0.4, 1.2, 1.0],
        "review_count": [5, 6, 20, 18],
        "unique_products_reviewed": [5, 6, 3, 3],
        "product_category_entropy": [1.5, 1.6, 0.3, 0.4],
        "verified_purchase_ratio": [0.8, 0.9, 0.2, 0.3],
        "reviews_per_day": [0.05, 0.04, 0.5, 0.45],
        "avg_inter_review_hours": [400, 500, 50, 55],
    }, index=["R1", "R2", "R3", "R4"])
    df.index.name = "reviewer_id"
    return df


# ---------------------------------------------------------------------------
# Test: Loading a valid graph
# ---------------------------------------------------------------------------

class TestLoadGraph:
    def test_load_valid_adjacency(self, tmp_path):
        W, ids = _simple_adjacency()
        df = pd.DataFrame(W, index=ids, columns=ids)
        df.to_csv(tmp_path / "graph_adjacency_matrix.csv")
        W_loaded, ids_loaded = load_adjacency_matrix(tmp_path)
        np.testing.assert_array_almost_equal(W_loaded, W)
        assert ids_loaded == ids


# ---------------------------------------------------------------------------
# Test: Invalid matrix dimensions
# ---------------------------------------------------------------------------

class TestInvalidDimensions:
    def test_non_square_matrix(self, tmp_path):
        df = pd.DataFrame(np.zeros((3, 4)), index=["R1", "R2", "R3"],
                          columns=["R1", "R2", "R3", "R4"])
        df.to_csv(tmp_path / "graph_adjacency_matrix.csv")
        with pytest.raises(ValueError, match="not square"):
            load_adjacency_matrix(tmp_path)


# ---------------------------------------------------------------------------
# Test: Asymmetric adjacency matrix
# ---------------------------------------------------------------------------

class TestAsymmetricMatrix:
    def test_asymmetric_rejected(self, tmp_path):
        W = np.array([[0, 0.5], [0.3, 0]])
        df = pd.DataFrame(W, index=["R1", "R2"], columns=["R1", "R2"])
        df.to_csv(tmp_path / "graph_adjacency_matrix.csv")
        with pytest.raises(ValueError, match="not symmetric"):
            load_adjacency_matrix(tmp_path)


# ---------------------------------------------------------------------------
# Test: Negative or non-finite edge weights
# ---------------------------------------------------------------------------

class TestInvalidWeights:
    def test_negative_weights(self, tmp_path):
        W = np.array([[0, -0.5], [-0.5, 0]])
        df = pd.DataFrame(W, index=["R1", "R2"], columns=["R1", "R2"])
        df.to_csv(tmp_path / "graph_adjacency_matrix.csv")
        with pytest.raises(ValueError, match="negative"):
            load_adjacency_matrix(tmp_path)

    def test_nan_weights(self, tmp_path):
        W = np.array([[0, np.nan], [np.nan, 0]])
        df = pd.DataFrame(W, index=["R1", "R2"], columns=["R1", "R2"])
        df.to_csv(tmp_path / "graph_adjacency_matrix.csv")
        with pytest.raises(ValueError, match="non-finite"):
            load_adjacency_matrix(tmp_path)


# ---------------------------------------------------------------------------
# Test: Incorrect reviewer ordering
# ---------------------------------------------------------------------------

class TestReviewerOrdering:
    def test_mismatched_ordering_detected(self):
        with pytest.raises(ValueError, match="mismatch"):
            validate_ordering(["R1", "R2", "R3"], ["R1", "R3", "R2"])

    def test_matching_ordering_passes(self):
        validate_ordering(["R1", "R2", "R3"], ["R1", "R2", "R3"])


# ---------------------------------------------------------------------------
# Test: Ground-truth leakage prevention
# ---------------------------------------------------------------------------

class TestLeakagePrevention:
    def test_gt_columns_not_in_adjacency(self, tmp_path):
        """Adjacency CSV with a ground-truth column name should not crash
        (the column would only be in the row index, not the values)."""
        W, ids = _simple_adjacency()
        df = pd.DataFrame(W, index=ids, columns=ids)
        df.to_csv(tmp_path / "graph_adjacency_matrix.csv")
        W_loaded, _ = load_adjacency_matrix(tmp_path)
        assert W_loaded.shape == (4, 4)


# ---------------------------------------------------------------------------
# Test: Reproducible clustering with fixed seed
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_fixed_seed_same_labels(self):
        W, ids = _simple_adjacency()
        emb1, _ = spectral_embedding(W, 2, "normalized")
        emb2, _ = spectral_embedding(W, 2, "normalized")
        np.testing.assert_array_equal(emb1, emb2)

        labels1, _ = run_kmeans(emb1, 2, seed=42)
        labels2, _ = run_kmeans(emb2, 2, seed=42)
        np.testing.assert_array_equal(labels1, labels2)


# ---------------------------------------------------------------------------
# Test: One predicted cluster assignment per reviewer
# ---------------------------------------------------------------------------

class TestOneAssignmentPerReviewer:
    def test_all_reviewers_assigned(self):
        W, ids = _simple_adjacency()
        emb, _ = spectral_embedding(W, 2, "normalized")
        labels, _ = run_kmeans(emb, 2, seed=42)
        assert len(labels) == len(ids)
        assert all(isinstance(int(l), int) for l in labels)


# ---------------------------------------------------------------------------
# Test: Correct reviewer-ID mapping
# ---------------------------------------------------------------------------

class TestReviewerMapping:
    def test_labels_match_reviewer_ids(self):
        W, ids = _simple_adjacency()
        emb, _ = spectral_embedding(W, 2, "normalized")
        labels, _ = run_kmeans(emb, 2, seed=42)
        # Two clear cliques: {R1,R2} and {R3,R4}
        assert labels[0] == labels[1]  # R1, R2 same cluster
        assert labels[2] == labels[3]  # R3, R4 same cluster
        assert labels[0] != labels[2]  # different clusters


# ---------------------------------------------------------------------------
# Test: Correct ARI and NMI on known example
# ---------------------------------------------------------------------------

class TestKnownMetrics:
    def test_perfect_agreement(self):
        labels = np.array([0, 0, 1, 1])
        gt_df = pd.DataFrame({
            "reviewer_id": ["R1", "R2", "R3", "R4"],
            "ground_truth_category": ["A", "A", "B", "B"],
        })
        ids = ["R1", "R2", "R3", "R4"]
        emb = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=float)
        result = evaluate_clustering(labels, ids, gt_df, emb)
        assert result["ari"] == pytest.approx(1.0)
        assert result["nmi"] == pytest.approx(1.0)

    def test_random_agreement(self):
        # Completely wrong assignment
        labels = np.array([0, 1, 0, 1])
        gt_df = pd.DataFrame({
            "reviewer_id": ["R1", "R2", "R3", "R4"],
            "ground_truth_category": ["A", "A", "B", "B"],
        })
        ids = ["R1", "R2", "R3", "R4"]
        emb = np.random.randn(4, 2)
        result = evaluate_clustering(labels, ids, gt_df, emb)
        # ARI should be negative or near zero for this mismatch
        assert result["ari"] < 0.5


# ---------------------------------------------------------------------------
# Test: Invalid cluster counts
# ---------------------------------------------------------------------------

class TestInvalidK:
    def test_k_one_no_crash(self):
        W, ids = _simple_adjacency()
        emb, _ = spectral_embedding(W, 1, "normalized")
        labels, _ = run_kmeans(emb, 1, seed=42)
        assert len(set(labels)) == 1

    def test_k_equals_n(self):
        W, ids = _simple_adjacency()
        emb, _ = spectral_embedding(W, 4, "normalized")
        labels, _ = run_kmeans(emb, 4, seed=42)
        assert len(labels) == 4


# ---------------------------------------------------------------------------
# Test: Silhouette Score edge cases
# ---------------------------------------------------------------------------

class TestSilhouetteEdgeCases:
    def test_single_cluster_no_silhouette(self):
        labels = np.array([0, 0, 0, 0])
        gt_df = _simple_gt()
        ids = ["R1", "R2", "R3", "R4"]
        emb = np.random.randn(4, 2)
        result = evaluate_clustering(labels, ids, gt_df, emb)
        assert result["silhouette"] is None

    def test_valid_silhouette(self):
        labels = np.array([0, 0, 1, 1])
        gt_df = _simple_gt()
        ids = ["R1", "R2", "R3", "R4"]
        emb = np.array([[0, 0], [0.1, 0], [5, 5], [5.1, 5]], dtype=float)
        result = evaluate_clustering(labels, ids, gt_df, emb)
        assert result["silhouette"] is not None
        assert -1 <= result["silhouette"] <= 1


# ---------------------------------------------------------------------------
# Test: Isolated nodes and disconnected graphs
# ---------------------------------------------------------------------------

class TestDisconnectedGraph:
    def test_isolated_node_preserved(self):
        # Node R3 is isolated
        W = np.array([
            [0, 0.9, 0.0],
            [0.9, 0, 0.0],
            [0.0, 0.0, 0],
        ], dtype=np.float64)
        ids = ["R1", "R2", "R3"]
        emb, _ = spectral_embedding(W, 2, "normalized")
        labels, _ = run_kmeans(emb, 2, seed=42)
        assert len(labels) == 3  # all reviewers get assignments

    def test_disconnected_components(self):
        # Two disconnected pairs
        W = np.array([
            [0, 0.9, 0.0, 0.0],
            [0.9, 0, 0.0, 0.0],
            [0.0, 0.0, 0, 0.9],
            [0.0, 0.0, 0.9, 0],
        ], dtype=np.float64)
        ids = ["R1", "R2", "R3", "R4"]
        emb, _ = spectral_embedding(W, 2, "normalized")
        labels, _ = run_kmeans(emb, 2, seed=42)
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]


# ---------------------------------------------------------------------------
# Test: Saving and reloading prediction files
# ---------------------------------------------------------------------------

class TestSaveReload:
    def test_cluster_assignments_roundtrip(self, tmp_path):
        W, ids = _simple_adjacency()
        gt_df = _simple_gt()
        features_df = _simple_features()

        # Create input files
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        adj_df = pd.DataFrame(W, index=ids, columns=ids)
        adj_df.to_csv(graph_dir / "graph_adjacency_matrix.csv")
        mapping_df = pd.DataFrame({"matrix_index": range(len(ids)), "reviewer_id": ids})
        mapping_df.to_csv(graph_dir / "reviewer_node_mapping.csv", index=False)

        features_dir = tmp_path / "features"
        features_dir.mkdir()
        gt_df.to_csv(features_dir / "reviewer_ground_truth.csv", index=False)
        features_df.to_csv(features_dir / "reviewer_features.csv")

        output_dir = tmp_path / "output"

        result = run_module3(
            graph_dir, features_dir, output_dir,
            k_min=2, k_max=2, laplacian_type="normalized", seed=42,
        )

        # Reload and verify
        loaded = pd.read_csv(output_dir / "reviewer_cluster_assignments.csv")
        assert set(loaded["reviewer_id"]) == set(ids)
        assert len(loaded) == len(ids)
        assert "cluster_id" in loaded.columns


# ---------------------------------------------------------------------------
# Test: Contingency table
# ---------------------------------------------------------------------------

class TestContingencyTable:
    def test_contingency_correct(self):
        labels = np.array([0, 0, 1, 1])
        ids = ["R1", "R2", "R3", "R4"]
        gt_df = _simple_gt()
        cont = build_contingency_table(labels, ids, gt_df)
        assert cont.shape[0] == 2  # 2 clusters
        assert cont.values.sum() == 4  # all reviewers accounted for


# ---------------------------------------------------------------------------
# Test: Cluster behavior summary
# ---------------------------------------------------------------------------

class TestBehaviorSummary:
    def test_summary_has_all_clusters(self):
        labels = np.array([0, 0, 1, 1])
        ids = ["R1", "R2", "R3", "R4"]
        features = _simple_features()
        behavior = build_cluster_behavior_summary(labels, ids, features)
        assert len(behavior) == 2
        assert set(behavior["cluster_id"]) == {0, 1}

    def test_summary_columns(self):
        labels = np.array([0, 0, 1, 1])
        ids = ["R1", "R2", "R3", "R4"]
        features = _simple_features()
        behavior = build_cluster_behavior_summary(labels, ids, features)
        assert "avg_rating" in behavior.columns
        assert "mean_rating_deviation" in behavior.columns
        assert "size" in behavior.columns


# ---------------------------------------------------------------------------
# Test: Laplacian properties
# ---------------------------------------------------------------------------

class TestLaplacianProperties:
    def test_unnormalized_rows_sum_zero(self):
        W, _ = _simple_adjacency()
        L = compute_unnormalized_laplacian(W)
        np.testing.assert_array_almost_equal(L.sum(axis=1), 0)

    def test_normalized_no_nan(self):
        W = np.array([[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0]], dtype=float)
        L = compute_normalized_laplacian(W)
        assert np.all(np.isfinite(L))
