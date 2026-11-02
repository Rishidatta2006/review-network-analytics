"""
test_module2.py — Comprehensive unit tests for Module 2.

Run with:
    python -m pytest tests/test_module2.py -v
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.similarity_graph import (
    compute_cosine_similarity,
    threshold_graph,
    topk_graph,
    compute_degree_matrix,
    compute_unnormalized_laplacian,
    compute_normalized_laplacian,
    build_networkx_graph,
    load_feature_matrix,
    run_module2,
)
from src.graph_validation import validate_graph


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _known_features() -> tuple[np.ndarray, list[str]]:
    """Small known feature matrix for reproducible tests."""
    X = np.array([
        [1.0, 0.0, 0.0],   # R1 — unit vector along dim 0
        [0.0, 1.0, 0.0],   # R2 — orthogonal to R1
        [1.0, 1.0, 0.0],   # R3 — 45° between R1 and R2
        [1.0, 0.0, 0.0],   # R4 — identical to R1
        [0.5, 0.5, 0.5],   # R5 — spread across all dims
    ])
    ids = ["R0001", "R0002", "R0003", "R0004", "R0005"]
    return X, ids


def _known_features_with_zero() -> tuple[np.ndarray, list[str]]:
    """Feature matrix with a zero vector."""
    X = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],   # zero vector
    ])
    ids = ["R0001", "R0002", "R0003"]
    return X, ids


# ---------------------------------------------------------------------------
# Test: Correct cosine similarities
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_known_values(self):
        X, ids = _known_features()
        S = compute_cosine_similarity(X, ids)

        # R1 and R4 are identical → cosine = 1.0 (but diagonal is 0)
        # Self-similarity is on diagonal (set to 0)
        assert S[0, 3] == pytest.approx(1.0, abs=1e-10)
        assert S[3, 0] == pytest.approx(1.0, abs=1e-10)

        # R1 and R2 are orthogonal → cosine = 0
        assert S[0, 1] == pytest.approx(0.0, abs=1e-10)

        # R1 and R3: cos = 1/sqrt(2) ≈ 0.707
        assert S[0, 2] == pytest.approx(1.0 / np.sqrt(2), abs=1e-6)

    def test_diagonal_is_zero(self):
        X, ids = _known_features()
        S = compute_cosine_similarity(X, ids)
        assert np.all(np.diag(S) == 0)


# ---------------------------------------------------------------------------
# Test: Symmetric similarity matrix
# ---------------------------------------------------------------------------

class TestSymmetry:
    def test_similarity_symmetric(self):
        X, ids = _known_features()
        S = compute_cosine_similarity(X, ids)
        np.testing.assert_array_almost_equal(S, S.T)

    def test_adjacency_symmetric_threshold(self):
        X, ids = _known_features()
        S = compute_cosine_similarity(X, ids)
        W = threshold_graph(S, 0.5)
        np.testing.assert_array_almost_equal(W, W.T)

    def test_adjacency_symmetric_topk(self):
        X, ids = _known_features()
        S = compute_cosine_similarity(X, ids)
        W = topk_graph(S, k=2)
        np.testing.assert_array_almost_equal(W, W.T)


# ---------------------------------------------------------------------------
# Test: Threshold-based edge selection
# ---------------------------------------------------------------------------

class TestThresholdGraph:
    def test_high_threshold_sparse(self):
        X, ids = _known_features()
        S = compute_cosine_similarity(X, ids)
        W = threshold_graph(S, 0.99)
        # Only R1-R4 should have similarity >= 0.99
        assert W[0, 3] > 0
        # R1-R2 orthogonal should have no edge
        assert W[0, 1] == 0

    def test_low_threshold_dense(self):
        X, ids = _known_features()
        S = compute_cosine_similarity(X, ids)
        W = threshold_graph(S, 0.01)
        # Most pairs should be connected (except orthogonal ones)
        edge_count = np.sum(W > 0) // 2
        assert edge_count > 0

    def test_negative_similarities_clamped(self):
        # Create vectors with negative cosine similarity
        X = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
        ids = ["R1", "R2", "R3"]
        S = compute_cosine_similarity(X, ids)
        W = threshold_graph(S, -0.5)
        # Negative similarities should be clamped to 0
        assert np.all(W >= 0)


# ---------------------------------------------------------------------------
# Test: Top-k graph construction
# ---------------------------------------------------------------------------

class TestTopKGraph:
    def test_topk_connectivity(self):
        X, ids = _known_features()
        S = compute_cosine_similarity(X, ids)
        W = topk_graph(S, k=2)
        # Every non-zero-vector node should have at least some edges
        for i in range(len(ids)):
            if np.linalg.norm(X[i]) > 0:
                assert np.sum(W[i] > 0) >= 1, f"Node {ids[i]} has no edges"

    def test_topk_min_similarity(self):
        X, ids = _known_features()
        S = compute_cosine_similarity(X, ids)
        W = topk_graph(S, k=2, min_similarity=0.9)
        # Only very similar pairs should be connected
        for i in range(len(ids)):
            for j in range(len(ids)):
                if W[i, j] > 0:
                    assert S[i, j] >= 0.9


# ---------------------------------------------------------------------------
# Test: Symmetrization behavior
# ---------------------------------------------------------------------------

class TestSymmetrization:
    def test_union_symmetrization(self):
        """Top-k with union rule: if A selects B or B selects A, edge exists."""
        X, ids = _known_features()
        S = compute_cosine_similarity(X, ids)
        W = topk_graph(S, k=1)
        # W should be symmetric despite k=1 being asymmetric
        np.testing.assert_array_almost_equal(W, W.T)


# ---------------------------------------------------------------------------
# Test: No self-loops
# ---------------------------------------------------------------------------

class TestNoSelfLoops:
    def test_no_self_loops_threshold(self):
        X, ids = _known_features()
        S = compute_cosine_similarity(X, ids)
        W = threshold_graph(S, 0.1)
        assert np.all(np.diag(W) == 0)

    def test_no_self_loops_topk(self):
        X, ids = _known_features()
        S = compute_cosine_similarity(X, ids)
        W = topk_graph(S, k=3)
        assert np.all(np.diag(W) == 0)

    def test_no_self_loops_networkx(self):
        X, ids = _known_features()
        S = compute_cosine_similarity(X, ids)
        W = topk_graph(S, k=2)
        G = build_networkx_graph(W, ids)
        assert len(list(nx.selfloop_edges(G))) == 0


# ---------------------------------------------------------------------------
# Test: Correct weighted degree values
# ---------------------------------------------------------------------------

class TestDegreeMatrix:
    def test_degree_equals_row_sum(self):
        X, ids = _known_features()
        S = compute_cosine_similarity(X, ids)
        W = topk_graph(S, k=2)
        D = compute_degree_matrix(W)
        row_sums = W.sum(axis=1)
        np.testing.assert_array_almost_equal(np.diag(D), row_sums)

    def test_degree_is_diagonal(self):
        X, ids = _known_features()
        S = compute_cosine_similarity(X, ids)
        W = topk_graph(S, k=2)
        D = compute_degree_matrix(W)
        D_off = D.copy()
        np.fill_diagonal(D_off, 0)
        assert np.all(D_off == 0)


# ---------------------------------------------------------------------------
# Test: Correct Laplacian calculation
# ---------------------------------------------------------------------------

class TestLaplacian:
    def test_unnormalized_equals_D_minus_W(self):
        X, ids = _known_features()
        S = compute_cosine_similarity(X, ids)
        W = topk_graph(S, k=2)
        D = compute_degree_matrix(W)
        L = compute_unnormalized_laplacian(D, W)
        np.testing.assert_array_almost_equal(L, D - W)

    def test_laplacian_symmetric(self):
        X, ids = _known_features()
        S = compute_cosine_similarity(X, ids)
        W = topk_graph(S, k=2)
        D = compute_degree_matrix(W)
        L = compute_unnormalized_laplacian(D, W)
        np.testing.assert_array_almost_equal(L, L.T)

    def test_laplacian_rows_sum_zero(self):
        X, ids = _known_features()
        S = compute_cosine_similarity(X, ids)
        W = topk_graph(S, k=2)
        D = compute_degree_matrix(W)
        L = compute_unnormalized_laplacian(D, W)
        row_sums = L.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, 0, decimal=10)

    def test_normalized_laplacian_no_nan(self):
        X, ids = _known_features_with_zero()
        S = compute_cosine_similarity(X, ids)
        W = topk_graph(S, k=1)
        D = compute_degree_matrix(W)
        L_sym = compute_normalized_laplacian(D, W)
        assert np.all(np.isfinite(L_sym))


# ---------------------------------------------------------------------------
# Test: Zero-vector handling
# ---------------------------------------------------------------------------

class TestZeroVectorHandling:
    def test_zero_vector_isolated(self):
        X, ids = _known_features_with_zero()
        S = compute_cosine_similarity(X, ids)
        # Zero vector should have all-zero similarities
        assert np.all(S[2, :] == 0)
        assert np.all(S[:, 2] == 0)

    def test_zero_vector_node_exists(self):
        X, ids = _known_features_with_zero()
        S = compute_cosine_similarity(X, ids)
        W = topk_graph(S, k=1)
        G = build_networkx_graph(W, ids)
        assert "R0003" in G.nodes()
        assert G.degree("R0003") == 0


# ---------------------------------------------------------------------------
# Test: Isolated nodes
# ---------------------------------------------------------------------------

class TestIsolatedNodes:
    def test_isolated_nodes_present_in_graph(self):
        X, ids = _known_features_with_zero()
        S = compute_cosine_similarity(X, ids)
        W = topk_graph(S, k=1)
        G = build_networkx_graph(W, ids)
        # R0003 (zero vector) should be isolated but present
        assert "R0003" in G.nodes()
        assert G.degree("R0003") == 0


# ---------------------------------------------------------------------------
# Test: Correct reviewer-to-matrix index mapping
# ---------------------------------------------------------------------------

class TestNodeMapping:
    def test_matrix_index_attribute(self):
        X, ids = _known_features()
        S = compute_cosine_similarity(X, ids)
        W = topk_graph(S, k=2)
        G = build_networkx_graph(W, ids)
        for idx, rid in enumerate(ids):
            assert G.nodes[rid]["matrix_index"] == idx


# ---------------------------------------------------------------------------
# Test: Invalid inputs and missing columns
# ---------------------------------------------------------------------------

class TestInvalidInputs:
    def test_missing_feature_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_feature_matrix(tmp_path)

    def test_missing_reviewer_id_column(self, tmp_path):
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        df.to_csv(tmp_path / "reviewer_features_scaled.csv", index=False)
        with pytest.raises(ValueError, match="reviewer_id"):
            load_feature_matrix(tmp_path)

    def test_ground_truth_leakage_detected(self, tmp_path):
        df = pd.DataFrame({
            "reviewer_id": ["R1", "R2"],
            "feat1": [1.0, 2.0],
            "campaign_id": ["C1", "C2"],
        })
        df.to_csv(tmp_path / "reviewer_features_scaled.csv", index=False)
        with pytest.raises(RuntimeError, match="LEAKAGE"):
            load_feature_matrix(tmp_path)


# ---------------------------------------------------------------------------
# Test: Consistent outputs across repeated runs
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_identical_runs(self):
        X, ids = _known_features()
        S1 = compute_cosine_similarity(X, ids)
        S2 = compute_cosine_similarity(X, ids)
        np.testing.assert_array_equal(S1, S2)

        W1 = topk_graph(S1, k=2)
        W2 = topk_graph(S2, k=2)
        np.testing.assert_array_equal(W1, W2)

    def test_full_pipeline_reproducible(self, tmp_path):
        # Create minimal feature CSV
        df = pd.DataFrame({
            "reviewer_id": ["R1", "R2", "R3", "R4"],
            "f1": [1.0, 0.0, 1.0, 0.5],
            "f2": [0.0, 1.0, 1.0, 0.5],
        })
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        df.to_csv(input_dir / "reviewer_features_scaled.csv", index=False)

        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"

        run_module2(input_dir, out1, strategy="topk", top_k=2)
        run_module2(input_dir, out2, strategy="topk", top_k=2)

        adj1 = pd.read_csv(out1 / "graph_adjacency_matrix.csv", index_col=0)
        adj2 = pd.read_csv(out2 / "graph_adjacency_matrix.csv", index_col=0)
        pd.testing.assert_frame_equal(adj1, adj2)


# ---------------------------------------------------------------------------
# Test: Full validation suite
# ---------------------------------------------------------------------------

class TestGraphValidation:
    def test_all_checks_pass(self):
        X, ids = _known_features()
        S = compute_cosine_similarity(X, ids)
        W = topk_graph(S, k=2)
        D = compute_degree_matrix(W)
        L = compute_unnormalized_laplacian(D, W)
        G = build_networkx_graph(W, ids)

        report = validate_graph(G, W, D, L, ids)
        assert report.all_passed, (
            f"Validation failures: "
            + ", ".join(r["check"] for r in report.results if not r["passed"])
        )
