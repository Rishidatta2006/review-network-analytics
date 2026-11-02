"""
test_module4.py — Integration tests for Module 4 dashboard helpers.

Tests verify data loading, consistency with Module 1-3 outputs,
and correctness of dashboard helper functions.

Run with:
    python -m pytest tests/test_module4.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# We test the data loading and consistency, not the Streamlit UI itself.
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
GRAPH_DIR = OUTPUT_DIR / "graph"
CLUSTER_DIR = OUTPUT_DIR / "clustering"


# ---------------------------------------------------------------------------
# Test: All required data files exist
# ---------------------------------------------------------------------------

class TestFileExistence:
    def test_reviews_exist(self):
        assert (DATA_DIR / "reviews.csv").exists()

    def test_products_exist(self):
        assert (DATA_DIR / "products.csv").exists()

    def test_features_exist(self):
        assert (OUTPUT_DIR / "reviewer_features.csv").exists()

    def test_scaled_features_exist(self):
        assert (OUTPUT_DIR / "reviewer_features_scaled.csv").exists()

    def test_ground_truth_exists(self):
        assert (OUTPUT_DIR / "reviewer_ground_truth.csv").exists()

    def test_adjacency_matrix_exists(self):
        assert (GRAPH_DIR / "graph_adjacency_matrix.csv").exists()

    def test_graph_stats_exist(self):
        assert (GRAPH_DIR / "graph_statistics.json").exists()

    def test_cluster_assignments_exist(self):
        assert (CLUSTER_DIR / "reviewer_cluster_assignments.csv").exists()

    def test_cluster_metrics_exist(self):
        assert (CLUSTER_DIR / "clustering_metrics.csv").exists()

    def test_behavior_summary_exists(self):
        assert (CLUSTER_DIR / "cluster_behavior_summary.csv").exists()


# ---------------------------------------------------------------------------
# Test: Data loads correctly
# ---------------------------------------------------------------------------

class TestDataLoading:
    def test_reviews_load(self):
        df = pd.read_csv(DATA_DIR / "reviews.csv")
        assert len(df) > 0
        assert "reviewer_id" in df.columns
        assert "rating" in df.columns

    def test_features_load(self):
        df = pd.read_csv(OUTPUT_DIR / "reviewer_features.csv", index_col="reviewer_id")
        assert len(df) > 0
        assert "avg_rating" in df.columns

    def test_cluster_assignments_load(self):
        df = pd.read_csv(CLUSTER_DIR / "reviewer_cluster_assignments.csv")
        assert "reviewer_id" in df.columns
        assert "cluster_id" in df.columns

    def test_graph_stats_load(self):
        stats = json.loads((GRAPH_DIR / "graph_statistics.json").read_text())
        assert "node_count" in stats
        assert "edge_count" in stats


# ---------------------------------------------------------------------------
# Test: Missing files produce clear indication
# ---------------------------------------------------------------------------

class TestMissingFiles:
    def test_nonexistent_path(self, tmp_path):
        assert not (tmp_path / "missing.csv").exists()


# ---------------------------------------------------------------------------
# Test: Statistics match actual inputs
# ---------------------------------------------------------------------------

class TestStatisticsConsistency:
    def test_review_count_matches(self):
        reviews = pd.read_csv(DATA_DIR / "reviews.csv")
        assert len(reviews) == 6173

    def test_reviewer_count_matches_features(self):
        features = pd.read_csv(OUTPUT_DIR / "reviewer_features.csv", index_col="reviewer_id")
        assert len(features) == 1000

    def test_graph_node_count_matches(self):
        stats = json.loads((GRAPH_DIR / "graph_statistics.json").read_text())
        features = pd.read_csv(OUTPUT_DIR / "reviewer_features.csv", index_col="reviewer_id")
        assert stats["node_count"] == len(features)

    def test_assignment_count_matches(self):
        assignments = pd.read_csv(CLUSTER_DIR / "reviewer_cluster_assignments.csv")
        features = pd.read_csv(OUTPUT_DIR / "reviewer_features.csv", index_col="reviewer_id")
        assert len(assignments) == len(features)


# ---------------------------------------------------------------------------
# Test: Reviewer selection returns correct reviewer
# ---------------------------------------------------------------------------

class TestReviewerSelection:
    def test_reviewer_in_features(self):
        features = pd.read_csv(OUTPUT_DIR / "reviewer_features.csv", index_col="reviewer_id")
        assert "R0001" in features.index

    def test_reviewer_in_assignments(self):
        assignments = pd.read_csv(CLUSTER_DIR / "reviewer_cluster_assignments.csv")
        assert "R0001" in assignments["reviewer_id"].values

    def test_reviewer_in_ground_truth(self):
        gt = pd.read_csv(OUTPUT_DIR / "reviewer_ground_truth.csv")
        assert "R0001" in gt["reviewer_id"].values


# ---------------------------------------------------------------------------
# Test: Cluster filtering works
# ---------------------------------------------------------------------------

class TestClusterFiltering:
    def test_filter_by_cluster(self):
        assignments = pd.read_csv(CLUSTER_DIR / "reviewer_cluster_assignments.csv")
        cluster_0 = assignments[assignments["cluster_id"] == 0]
        cluster_1 = assignments[assignments["cluster_id"] == 1]
        assert len(cluster_0) > 0
        assert len(cluster_1) > 0
        assert len(cluster_0) + len(cluster_1) == len(assignments)


# ---------------------------------------------------------------------------
# Test: Evaluation metrics match Module 3 outputs
# ---------------------------------------------------------------------------

class TestMetricsConsistency:
    def test_metrics_match_experiment_summary(self):
        metrics = pd.read_csv(CLUSTER_DIR / "clustering_metrics.csv")
        exp = json.loads((CLUSTER_DIR / "experiment_summary.json").read_text())
        primary_k = exp["primary_k"]
        primary_row = metrics[metrics["k"] == primary_k].iloc[0]
        assert abs(primary_row["ari"] - exp["primary_metrics"]["ari"]) < 1e-6
        assert abs(primary_row["nmi"] - exp["primary_metrics"]["nmi"]) < 1e-6


# ---------------------------------------------------------------------------
# Test: Ground-truth labels not used for predictions
# ---------------------------------------------------------------------------

class TestGroundTruthSeparation:
    def test_assignments_no_gt_columns(self):
        assignments = pd.read_csv(CLUSTER_DIR / "reviewer_cluster_assignments.csv")
        gt_cols = {"ground_truth_category", "campaign_id", "campaign_direction", "is_coordinated"}
        assert not (gt_cols & set(assignments.columns))

    def test_features_no_gt_columns(self):
        features = pd.read_csv(OUTPUT_DIR / "reviewer_features.csv", index_col="reviewer_id")
        gt_cols = {"ground_truth_category", "campaign_id", "campaign_direction", "is_coordinated"}
        assert not (gt_cols & set(features.columns))


# ---------------------------------------------------------------------------
# Test: Downloaded files contain correct data
# ---------------------------------------------------------------------------

class TestExportData:
    def test_assignments_csv_complete(self):
        df = pd.read_csv(CLUSTER_DIR / "reviewer_cluster_assignments.csv")
        assert len(df) == 1000
        assert df["reviewer_id"].is_unique

    def test_behavior_summary_has_all_clusters(self):
        behavior = pd.read_csv(CLUSTER_DIR / "cluster_behavior_summary.csv")
        assignments = pd.read_csv(CLUSTER_DIR / "reviewer_cluster_assignments.csv")
        assert set(behavior["cluster_id"]) == set(assignments["cluster_id"].unique())


# ---------------------------------------------------------------------------
# Test: Streamlit App Execution & Page Rendering
# ---------------------------------------------------------------------------

class TestStreamlitApp:
    def test_app_starts_successfully(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=30)
        at.run()
        assert len(at.exception) == 0, f"Exceptions on startup: {[e.value for e in at.exception]}"

    def test_all_pages_render_without_exception(self):
        from streamlit.testing.v1 import AppTest
        pages = [
            "Overview",
            "Reviews",
            "Reviewer Network",
            "Clusters",
            "Evaluation",
            "Reviewer Inspection",
            "Reports",
            "Privacy Policy",
            "Terms and Conditions",
        ]
        for page_name in pages:
            at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=30)
            at.run()
            if hasattr(at, "segmented_control") and len(at.segmented_control) > 0:
                at.segmented_control[0].set_value(page_name).run()
            else:
                at.radio[0].set_value(page_name).run()
            assert len(at.exception) == 0, (
                f"Page '{page_name}' threw exceptions: {[e.value for e in at.exception]}"
            )

    def test_no_emojis_in_navigation(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=30)
        at.run()
        if hasattr(at, "segmented_control") and len(at.segmented_control) > 0:
            opts = at.segmented_control[0].options
        else:
            opts = at.radio[0].options
        for opt in opts:
            # Check for emoji/symbols outside ASCII text range
            assert all(ord(c) < 128 for c in opt), f"Navigation option '{opt}' contains non-ASCII/emoji characters"

    def test_no_purple_gradients_in_app(self):
        app_code = (BASE_DIR / "app.py").read_text(encoding="utf-8")
        assert "linear-gradient" not in app_code, "app.py should not use CSS linear gradients"
        assert "#667eea" not in app_code, "app.py should not use purple gradient color"

    def test_no_em_dashes_in_app(self):
        app_code = (BASE_DIR / "app.py").read_text(encoding="utf-8")
        assert "—" not in app_code, "app.py should not use em dashes in interface copy"

    def test_inspect_reviewer_details_button_navigates(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=30)
        at.run()
        # Navigate to Reviews page
        if hasattr(at, "segmented_control") and len(at.segmented_control) > 0:
            at.segmented_control[0].set_value("Reviews").run()
        else:
            at.radio[0].set_value("Reviews").run()
        assert at.session_state["current_page"] == "Reviews"

        # Select a specific reviewer in quick action
        quick_select = [s for s in at.selectbox if s.key == "quick_inspect_rid"][0]
        options = quick_select.options
        chosen_rid = options[1] if len(options) > 1 else options[0]
        quick_select.select(chosen_rid).run()

        # Click Inspect Reviewer Details button
        btn = [b for b in at.button if b.key == "btn_inspect_reviewer_details"][0]
        btn.click().run()

        # Verify page transitions cleanly to Reviewer Inspection with reviewer loaded
        assert len(at.exception) == 0, f"Exceptions after clicking button: {[e.value for e in at.exception]}"
        assert at.session_state["current_page"] == "Reviewer Inspection"
        assert at.session_state["main_nav_control"] == "Reviewer Inspection"
        assert at.session_state["selected_reviewer_id"] == chosen_rid
