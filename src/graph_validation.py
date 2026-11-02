"""
graph_validation.py — Automated validation checks for the reviewer similarity graph.

Validates mathematical properties of the adjacency matrix, degree matrix,
graph Laplacian, and NetworkX graph structure.
"""
from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class GraphValidationReport:
    """Accumulates validation check results."""

    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r["passed"])

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r["passed"])

    @property
    def all_passed(self) -> bool:
        return self.failed_count == 0

    def add_check(self, check: str, passed: bool, detail: str = "") -> None:
        status = "PASS" if passed else "FAIL"
        if passed:
            logger.info("[%s] %s", status, check)
        else:
            logger.error("[%s] %s — %s", status, check, detail)
        self.results.append({
            "check": check,
            "passed": passed,
            "detail": detail,
        })


# Floating-point tolerance
ATOL = 1e-10


def validate_graph(
    G: nx.Graph,
    W: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    reviewer_ids: list[str],
) -> GraphValidationReport:
    """
    Run all graph validation checks.

    Parameters
    ----------
    G : nx.Graph
        The constructed reviewer graph.
    W : np.ndarray
        Weighted adjacency matrix.
    D : np.ndarray
        Diagonal degree matrix.
    L : np.ndarray
        Unnormalized graph Laplacian.
    reviewer_ids : list[str]
        Ordered reviewer IDs matching matrix rows.

    Returns
    -------
    GraphValidationReport
    """
    report = GraphValidationReport()
    n = len(reviewer_ids)

    # --- Check 1: Every reviewer appears as a graph node ---
    graph_nodes = set(G.nodes())
    all_present = all(rid in graph_nodes for rid in reviewer_ids)
    report.add_check(
        "1. Every reviewer appears as a graph node",
        all_present,
        f"Missing: {set(reviewer_ids) - graph_nodes}" if not all_present else "",
    )

    # --- Check 2: Reviewer IDs are unique ---
    unique = len(reviewer_ids) == len(set(reviewer_ids))
    report.add_check(
        "2. Reviewer IDs are unique",
        unique,
        f"Count: {len(reviewer_ids)}, Unique: {len(set(reviewer_ids))}"
        if not unique else "",
    )

    # --- Check 3: No self-loops ---
    self_loops = list(nx.selfloop_edges(G))
    report.add_check(
        "3. No self-loops in graph",
        len(self_loops) == 0,
        f"Self-loops found: {self_loops[:5]}" if self_loops else "",
    )

    # --- Check 4: Edge weights are finite and non-negative ---
    edge_weights = [d.get("weight", 0) for _, _, d in G.edges(data=True)]
    all_finite = all(np.isfinite(w) for w in edge_weights)
    all_nonneg = all(w >= 0 for w in edge_weights)
    report.add_check(
        "4. Edge weights are finite and non-negative",
        all_finite and all_nonneg,
        f"Finite: {all_finite}, Non-negative: {all_nonneg}"
        if not (all_finite and all_nonneg) else "",
    )

    # --- Check 5: Adjacency matrix is symmetric ---
    sym_diff = np.max(np.abs(W - W.T))
    report.add_check(
        "5. Adjacency matrix is symmetric",
        sym_diff < ATOL,
        f"Max asymmetry: {sym_diff}" if sym_diff >= ATOL else "",
    )

    # --- Check 6: Adjacency diagonal is zero ---
    diag_max = np.max(np.abs(np.diag(W)))
    report.add_check(
        "6. Adjacency matrix diagonal is zero",
        diag_max < ATOL,
        f"Max diagonal: {diag_max}" if diag_max >= ATOL else "",
    )

    # --- Check 7: Degree matrix is diagonal ---
    D_off_diag = D.copy()
    np.fill_diagonal(D_off_diag, 0)
    off_diag_max = np.max(np.abs(D_off_diag))
    report.add_check(
        "7. Degree matrix is diagonal",
        off_diag_max < ATOL,
        f"Max off-diagonal: {off_diag_max}" if off_diag_max >= ATOL else "",
    )

    # --- Check 8: Each degree equals the row sum of adjacency ---
    row_sums = W.sum(axis=1)
    diag_vals = np.diag(D)
    degree_match = np.allclose(row_sums, diag_vals, atol=ATOL)
    report.add_check(
        "8. Degree values equal adjacency row sums",
        degree_match,
        f"Max diff: {np.max(np.abs(row_sums - diag_vals))}"
        if not degree_match else "",
    )

    # --- Check 9: Laplacian equals D - W ---
    L_expected = D - W
    lap_match = np.allclose(L, L_expected, atol=ATOL)
    report.add_check(
        "9. Laplacian equals D - W",
        lap_match,
        f"Max diff: {np.max(np.abs(L - L_expected))}"
        if not lap_match else "",
    )

    # --- Check 10: Laplacian is symmetric ---
    lap_sym_diff = np.max(np.abs(L - L.T))
    report.add_check(
        "10. Laplacian is symmetric",
        lap_sym_diff < ATOL,
        f"Max asymmetry: {lap_sym_diff}" if lap_sym_diff >= ATOL else "",
    )

    # --- Check 11: Every Laplacian row sums to approximately zero ---
    row_sums_L = L.sum(axis=1)
    max_row_sum = np.max(np.abs(row_sums_L))
    report.add_check(
        "11. Laplacian rows sum to approximately zero",
        max_row_sum < ATOL,
        f"Max row sum: {max_row_sum}" if max_row_sum >= ATOL else "",
    )

    # --- Check 12: Graph node IDs map correctly to matrix row indices ---
    mapping_ok = True
    for idx, rid in enumerate(reviewer_ids):
        if rid not in G.nodes():
            mapping_ok = False
            break
        stored_idx = G.nodes[rid].get("matrix_index", None)
        if stored_idx != idx:
            mapping_ok = False
            break
    report.add_check(
        "12. Node IDs map correctly to matrix row indices",
        mapping_ok,
    )

    # --- Check 13: Connected components and isolated nodes reported ---
    components = list(nx.connected_components(G))
    isolated = [node for node in G.nodes() if G.degree(node) == 0]
    report.add_check(
        f"13. Connectivity: {len(components)} component(s), "
        f"{len(isolated)} isolated node(s)",
        True,  # informational
        f"Largest component: {max(len(c) for c in components)} nodes"
        if components else "No components",
    )

    # --- Check 14: Node count matches feature matrix ---
    node_match = G.number_of_nodes() == n
    report.add_check(
        "14. Graph node count matches feature matrix reviewer count",
        node_match,
        f"Graph: {G.number_of_nodes()}, Features: {n}"
        if not node_match else "",
    )

    # --- Check 15: No ground-truth labels in graph attributes ---
    gt_attrs = {"ground_truth_category", "campaign_id", "campaign_direction", "is_coordinated"}
    leaked_node_attrs = set()
    for _, data in G.nodes(data=True):
        leaked_node_attrs.update(set(data.keys()) & gt_attrs)
    leaked_edge_attrs = set()
    for _, _, data in G.edges(data=True):
        leaked_edge_attrs.update(set(data.keys()) & gt_attrs)
    no_leakage = len(leaked_node_attrs) == 0 and len(leaked_edge_attrs) == 0
    report.add_check(
        "15. No ground-truth labels in graph attributes",
        no_leakage,
        f"Node attrs: {leaked_node_attrs}, Edge attrs: {leaked_edge_attrs}"
        if not no_leakage else "",
    )

    return report
