"""
app.py: Professional Review Network Analytics Platform.

Detection of Coordinated Fake Product Review Networks Using Spectral Clustering.
Enterprise research and investigation interface.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths and Environment
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
GRAPH_DIR = OUTPUT_DIR / "graph"
CLUSTER_DIR = OUTPUT_DIR / "clustering"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

# ---------------------------------------------------------------------------
# Cached Data Loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_reviews() -> pd.DataFrame | None:
    p = DATA_DIR / "reviews.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, parse_dates=["timestamp"])


@st.cache_data(show_spinner=False)
def load_products() -> pd.DataFrame | None:
    p = DATA_DIR / "products.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_ground_truth() -> pd.DataFrame | None:
    p = OUTPUT_DIR / "reviewer_ground_truth.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_features() -> pd.DataFrame | None:
    p = OUTPUT_DIR / "reviewer_features.csv"
    if not p.exists():
        return None
    try:
        return pd.read_csv(p, index_col="reviewer_id")
    except (ValueError, KeyError):
        df = pd.read_csv(p, index_col=0)
        df.index.name = "reviewer_id"
        return df


@st.cache_data(show_spinner=False)
def load_cluster_assignments() -> pd.DataFrame | None:
    p = CLUSTER_DIR / "reviewer_cluster_assignments.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_cluster_metrics() -> pd.DataFrame | None:
    p = CLUSTER_DIR / "clustering_metrics.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_behavior_summary() -> pd.DataFrame | None:
    p = CLUSTER_DIR / "cluster_behavior_summary.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_contingency() -> pd.DataFrame | None:
    p = CLUSTER_DIR / "cluster_ground_truth_comparison.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, index_col=0)


@st.cache_data(show_spinner=False)
def load_experiment_summary() -> dict | None:
    p = CLUSTER_DIR / "experiment_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_graph_stats() -> dict | None:
    p = GRAPH_DIR / "graph_statistics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_validation_report() -> str | None:
    p = REPORTS_DIR / "module1_validation_report.txt"
    return p.read_text(encoding="utf-8") if p.exists() else None


@st.cache_data(show_spinner=False)
def load_graph_report() -> str | None:
    p = GRAPH_DIR / "graph_validation_report.txt"
    return p.read_text(encoding="utf-8") if p.exists() else None


@st.cache_data(show_spinner=False)
def load_evaluation_report() -> str | None:
    p = CLUSTER_DIR / "evaluation_report.txt"
    return p.read_text(encoding="utf-8") if p.exists() else None


@st.cache_data(show_spinner=False)
def load_degree_matrix() -> pd.DataFrame | None:
    p = GRAPH_DIR / "graph_degree_matrix.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_adjacency_matrix() -> pd.DataFrame | None:
    p = GRAPH_DIR / "graph_adjacency_matrix.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, index_col=0)


# ---------------------------------------------------------------------------
# Streamlit Page Setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Review Network Analytics",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Enterprise CSS Design System
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #182230;
        background-color: #F5F7FA;
    }

    .block-container {
        padding-top: 4.5rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1400px !important;
    }

    /* Top Application Header */
    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #FFFFFF;
        border: 1px solid #DCE3EA;
        border-radius: 6px;
        padding: 14px 20px;
        margin-top: 0 !important;
        margin-bottom: 14px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
        flex-wrap: wrap;
        gap: 12px;
        box-sizing: border-box;
    }
    .app-header-left {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .app-logo-mark {
        background: #245B8F;
        color: #FFFFFF;
        font-weight: 700;
        font-size: 13px;
        padding: 6px 10px;
        border-radius: 5px;
        letter-spacing: 0.5px;
        font-family: 'JetBrains Mono', monospace;
    }
    .app-title {
        font-size: 16px;
        font-weight: 700;
        color: #182230;
        margin: 0;
        line-height: 1.2;
    }
    .app-subtitle {
        font-size: 12px;
        color: #596579;
        margin: 0;
        line-height: 1.2;
    }
    .app-header-right {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Status Badges */
    .status-badge {
        font-size: 11px;
        font-weight: 600;
        padding: 4px 9px;
        border-radius: 4px;
        letter-spacing: 0.3px;
        text-transform: uppercase;
        font-family: 'JetBrains Mono', monospace;
    }
    .status-badge-neutral {
        background: #F1F5F9;
        color: #475569;
        border: 1px solid #CBD5E1;
    }
    .status-badge-success {
        background: #ECFDF5;
        color: #237A57;
        border: 1px solid #A7F3D0;
    }
    .status-badge-warning {
        background: #FFFBEB;
        color: #A66B16;
        border: 1px solid #FDE68A;
    }
    .status-badge-error {
        background: #FEF2F2;
        color: #B33A3A;
        border: 1px solid #FECACA;
    }

    /* Native Metric Cards Styling */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #DCE3EA !important;
        border-radius: 6px !important;
        padding: 14px 16px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
    }
    div[data-testid="stMetric"] label[data-testid="stMetricLabel"] {
        font-size: 11px !important;
        font-weight: 600 !important;
        color: #596579 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        margin-bottom: 4px !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 22px !important;
        font-weight: 700 !important;
        color: #182230 !important;
        font-family: 'JetBrains Mono', monospace !important;
        line-height: 1.2 !important;
    }

    /* Container Borders */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border: 1px solid #DCE3EA !important;
        border-radius: 6px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
    }

    /* Segmented Control Navigation Overrides */
    div[data-testid="stSegmentedControl"] {
        margin-bottom: 12px !important;
    }
    div[data-testid="stSegmentedControl"] button {
        font-size: 13px !important;
        font-weight: 500 !important;
        border-radius: 5px !important;
        padding: 6px 14px !important;
        color: #596579 !important;
        background-color: #FFFFFF !important;
        border: 1px solid #DCE3EA !important;
        transition: all 0.15s ease-in-out !important;
    }
    div[data-testid="stSegmentedControl"] button:hover {
        color: #19466F !important;
        border-color: #245B8F !important;
        background-color: #F8FAFC !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-checked="true"],
    div[data-testid="stSegmentedControl"] button[data-checked="true"],
    div[data-testid="stSegmentedControl"] [aria-checked="true"],
    div[data-testid="stSegmentedControl"] [data-baseweb="button-group"] button[aria-checked="true"],
    div[data-testid="stSegmentedControl"] [data-baseweb="button-group"] [aria-checked="true"] {
        background-color: #245B8F !important;
        color: #FFFFFF !important;
        border-color: #245B8F !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 2px rgba(36, 91, 143, 0.2) !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-checked="true"] p,
    div[data-testid="stSegmentedControl"] [aria-checked="true"] p {
        color: #FFFFFF !important;
    }

    /* Tag and Multi-select Overrides (No Bright Red Chips) */
    [data-baseweb="tag"], span[data-baseweb="tag"], div[data-baseweb="tag"] {
        background-color: #EAF2FA !important;
        color: #19466F !important;
        border: 1px solid #C3D9EE !important;
        border-radius: 4px !important;
    }
    [data-baseweb="tag"] span, span[data-baseweb="tag"] span, div[data-baseweb="tag"] span {
        color: #19466F !important;
        font-weight: 500 !important;
    }
    [data-baseweb="tag"] svg, span[data-baseweb="tag"] svg, div[data-baseweb="tag"] svg {
        fill: #19466F !important;
    }

    /* Tabs Styling Overrides */
    button[data-baseweb="tab"] {
        font-size: 13px !important;
        font-weight: 500 !important;
        color: #596579 !important;
        padding: 8px 16px !important;
        border-bottom: 2px solid transparent !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #245B8F !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #245B8F !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #245B8F !important;
        background-color: #FFFFFF !important;
    }

    /* Notice Banners */
    .notice-box {
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 13px;
        line-height: 1.5;
        margin-bottom: 16px;
    }
    .notice-info {
        background: #F8FAFC;
        border: 1px solid #CBD5E1;
        color: #334155;
    }
    .notice-warning {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        color: #A66B16;
    }

    /* Breadcrumbs */
    .breadcrumb-bar {
        font-size: 12px;
        color: #596579;
        margin-bottom: 10px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Footer */
    .app-footer {
        border-top: 1px solid #DCE3EA;
        margin-top: 36px;
        padding-top: 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 12px;
        color: #596579;
    }

    /* Monospace formatting */
    .code-id {
        font-family: 'JetBrains Mono', monospace;
        background: #F1F5F9;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 12px;
        color: #182230;
        border: 1px solid #E2E8F0;
    }

    /* Responsive Header and Container Adjustments */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 5rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        .app-header {
            flex-direction: column;
            align-items: flex-start;
            padding: 12px 14px;
            gap: 10px;
        }
        .app-header-right {
            width: 100%;
            justify-content: flex-start;
            flex-wrap: wrap;
        }
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Global App State
# ---------------------------------------------------------------------------
PAGES = [
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

if "current_page" not in st.session_state:
    st.session_state.current_page = "Overview"
if "main_nav_control" not in st.session_state:
    st.session_state.main_nav_control = st.session_state.current_page
if "selected_reviewer_id" not in st.session_state:
    st.session_state.selected_reviewer_id = "R0001"

# Handle pending programmatic page navigation before widget instantiation
if "pending_page" in st.session_state and st.session_state.pending_page:
    target_page = st.session_state.pop("pending_page")
    if target_page in PAGES:
        st.session_state.current_page = target_page
        st.session_state.main_nav_control = target_page

# ---------------------------------------------------------------------------
# Visual Helper: Enterprise Plotly Layout
# ---------------------------------------------------------------------------
def apply_enterprise_theme(fig: go.Figure, height: int = 360, title: str = "") -> go.Figure:
    """Apply restrained enterprise styling to a Plotly figure."""
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=14, color="#182230", family="Inter, sans-serif")))
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, -apple-system, sans-serif", size=12, color="#182230"),
        margin=dict(l=45, r=30, t=45 if title else 25, b=45),
        height=height,
        xaxis=dict(
            gridcolor="#F1F5F9",
            linecolor="#DCE3EA",
            tickcolor="#DCE3EA",
            zerolinecolor="#E2E8F0",
        ),
        yaxis=dict(
            gridcolor="#F1F5F9",
            linecolor="#DCE3EA",
            tickcolor="#DCE3EA",
            zerolinecolor="#E2E8F0",
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#DCE3EA",
            borderwidth=1,
            font=dict(size=11),
        ),
    )
    return fig


def render_metric_panel(label: str, value: str | int | float, subtext: str = ""):
    """Render a clean metric card using native st.metric."""
    st.metric(label=label, value=str(value), help=subtext if subtext else None)


# ---------------------------------------------------------------------------
# Global Application Header
# ---------------------------------------------------------------------------
m1_exists = (OUTPUT_DIR / "reviewer_features.csv").exists()
m2_exists = (GRAPH_DIR / "graph_adjacency_matrix.csv").exists()
m3_exists = (CLUSTER_DIR / "reviewer_cluster_assignments.csv").exists()
pipeline_fully_verified = m1_exists and m2_exists and m3_exists

status_badge_html = (
    '<span class="status-badge status-badge-success">Pipeline Verified</span>'
    if pipeline_fully_verified
    else '<span class="status-badge status-badge-warning">Partial Verification</span>'
)

st.markdown(
    f"""<div class="app-header">
<div class="app-header-left">
<div class="app-logo-mark">RNA</div>
<div>
<div class="app-title">Review Network Analytics</div>
<div class="app-subtitle">Research and Investigation Platform</div>
</div>
</div>
<div class="app-header-right">
<span class="status-badge status-badge-neutral">Synthetic Research Dataset</span>
{status_badge_html}
</div>
</div>""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Primary Navigation
# ---------------------------------------------------------------------------
default_idx = PAGES.index(st.session_state.current_page) if st.session_state.current_page in PAGES else 0

if hasattr(st, "segmented_control"):
    selected_nav = st.segmented_control(
        "Navigation",
        PAGES,
        label_visibility="collapsed",
        key="main_nav_control",
    )
    if not selected_nav:
        selected_nav = st.session_state.current_page
else:
    selected_nav = st.radio(
        "Navigation",
        PAGES,
        horizontal=True,
        label_visibility="collapsed",
        key="main_nav_control",
    )

if selected_nav != st.session_state.current_page:
    st.session_state.current_page = selected_nav

current_page = st.session_state.current_page

# Context breadcrumb
st.markdown(f'<div class="breadcrumb-bar">Application / {current_page}</div>', unsafe_allow_html=True)


# ===========================================================================
# PAGE: OVERVIEW
# ===========================================================================
if current_page == "Overview":
    st.subheader("Overview")
    st.markdown(
        "Review network structure, clustering results, and rating manipulation patterns in the current dataset."
    )

    st.markdown(
        '<div class="notice-box notice-info">'
        "<strong>Research Context:</strong> This platform performs unsupervised Spectral Clustering "
        "on reviewer behavioral similarity graphs. Ground-truth labels describe synthetic simulation "
        "parameters and were strictly excluded from graph construction and clustering."
        "</div>",
        unsafe_allow_html=True,
    )

    reviews = load_reviews()
    graph_stats = load_graph_stats()
    assignments = load_cluster_assignments()
    exp = load_experiment_summary()

    # KPI summary grid using 2 rows of 3 columns to ensure ample width and zero label truncation
    kpi_row1 = st.columns(3)
    with kpi_row1[0]:
        val = f"{len(reviews):,}" if reviews is not None else "Not available"
        render_metric_panel("Total Reviews", val, "Synthetic transaction records")
    with kpi_row1[1]:
        val = f"{reviews['reviewer_id'].nunique():,}" if reviews is not None else "Not available"
        render_metric_panel("Unique Reviewers", val, "Graph node entities")
    with kpi_row1[2]:
        val = f"{reviews['product_id'].nunique():,}" if reviews is not None else "Not available"
        render_metric_panel("Unique Products", val, "Catalog item inventory")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    kpi_row2 = st.columns(3)
    with kpi_row2[0]:
        val = str(assignments["cluster_id"].nunique()) if assignments is not None else "Not available"
        render_metric_panel("Predicted Clusters", val, "Unsupervised primary k selection")
    with kpi_row2[1]:
        if graph_stats:
            val = f"{graph_stats.get('node_count', 0):,} / {graph_stats.get('edge_count', 0):,}"
        else:
            val = "Not available"
        render_metric_panel("Network Nodes / Edges", val, "Similarity graph topology")
    with kpi_row2[2]:
        val = str(graph_stats.get("isolated_node_count", 0)) if graph_stats else "Not available"
        render_metric_panel("Isolated Nodes", val, "Zero-degree disconnected entities")

    # Pipeline Verification Status
    st.markdown("---")
    st.markdown("### Pipeline Verification Status")

    pcols = st.columns(3)
    with pcols[0]:
        with st.container(border=True):
            st.markdown("**Module 1: Preprocessing & Features**")
            st.caption("13 reviewer-level behavioral features engineered and standardized.")
            if m1_exists:
                st.markdown('<span class="status-badge status-badge-success">Verified: reviewer_features.csv</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-badge status-badge-warning">Missing output</span>', unsafe_allow_html=True)

    with pcols[1]:
        with st.container(border=True):
            st.markdown("**Module 2: Similarity Network**")
            st.caption("Cosine similarity network with top-15 union symmetrization.")
            if m2_exists:
                st.markdown('<span class="status-badge status-badge-success">Verified: graph_adjacency_matrix.csv</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-badge status-badge-warning">Missing output</span>', unsafe_allow_html=True)

    with pcols[2]:
        with st.container(border=True):
            st.markdown("**Module 3: Spectral Clustering**")
            st.caption("Normalized symmetric Laplacian with primary k=2 selected.")
            if m3_exists:
                st.markdown('<span class="status-badge status-badge-success">Verified: reviewer_cluster_assignments.csv</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-badge status-badge-warning">Missing output</span>', unsafe_allow_html=True)

    # Core Analytical Visualizations
    st.markdown("---")
    st.markdown("### Analytical Distributions")

    c1, c2 = st.columns(2)
    with c1:
        if reviews is not None:
            fig = px.histogram(
                reviews,
                x="rating",
                nbins=5,
                color_discrete_sequence=["#245B8F"],
            )
            fig.update_layout(bargap=0.15, xaxis_title="Rating (Stars)", yaxis_title="Review Count")
            apply_enterprise_theme(fig, height=320, title="Rating Distribution Across Dataset")
            st.plotly_chart(fig, width="stretch")
    with c2:
        behavior = load_behavior_summary()
        if behavior is not None:
            fig = px.bar(
                behavior,
                x="cluster_id",
                y="size",
                text="size",
                color="cluster_id",
                color_discrete_sequence=["#245B8F", "#596579"],
            )
            fig.update_layout(
                xaxis_title="Predicted Cluster",
                yaxis_title="Reviewer Count",
                showlegend=False,
            )
            fig.update_traces(textposition="outside")
            apply_enterprise_theme(fig, height=320, title="Predicted Cluster Size Breakdown")
            st.plotly_chart(fig, width="stretch")

    # Reviewer Volume Distribution & Degree Distribution Row
    c3, c4 = st.columns(2)
    with c3:
        if reviews is not None:
            user_counts = reviews.groupby("reviewer_id").size().reset_index(name="review_count")
            fig_vol = px.histogram(
                user_counts,
                x="review_count",
                nbins=25,
                color_discrete_sequence=["#245B8F"],
            )
            fig_vol.update_layout(bargap=0.1, xaxis_title="Reviews per Reviewer", yaxis_title="Reviewer Frequency")
            apply_enterprise_theme(fig_vol, height=320, title="Reviewer Activity Volume Distribution")
            st.plotly_chart(fig_vol, width="stretch")
    with c4:
        degrees_df = load_degree_matrix()
        if degrees_df is not None:
            fig_deg_ov = px.histogram(
                degrees_df,
                x="weighted_degree",
                nbins=30,
                color_discrete_sequence=["#237A57"],
            )
            fig_deg_ov.update_layout(bargap=0.1, xaxis_title="Weighted Node Degree", yaxis_title="Reviewer Count")
            apply_enterprise_theme(fig_deg_ov, height=320, title="Similarity Graph Degree Distribution")
            st.plotly_chart(fig_deg_ov, width="stretch")

    # Architecture Overview
    st.markdown("---")
    st.markdown("### Architecture Pipeline")
    st.code(
        """
Module 1: Data Validation & Feature Engineering
    Output: reviewer_features.csv (1,000 reviewers x 13 behavioral features)

Module 2: Similarity Network & Graph Construction
    Output: graph_adjacency_matrix.csv (1,000 x 1,000 top-15 cosine similarity graph)

Module 3: Spectral Clustering & Evaluation
    Output: reviewer_cluster_assignments.csv (Unsupervised normalized Laplacian clustering)

Module 4: Enterprise Analytics Platform
    Interface: Interactive investigation workspace and reporting
        """.strip(),
        language="text",
    )


# ===========================================================================
# PAGE: REVIEWS
# ===========================================================================
elif current_page == "Reviews":
    st.subheader("Reviews Explorer")
    st.markdown("Search, filter, and inspect individual review records across the synthetic catalog.")

    reviews = load_reviews()
    if reviews is None:
        st.error("reviews.csv was not found in data directory. Run Module 1 first.")
        st.stop()

    # Filter section inside a bordered container
    with st.container(border=True):
        st.markdown("**Filter Review Records**")
        fcol1, fcol2, fcol3, fcol4 = st.columns(4)
        with fcol1:
            reviewer_search = st.text_input("Reviewer ID contains", placeholder="e.g. R0001", key="filter_rid")
        with fcol2:
            product_search = st.text_input("Product ID contains", placeholder="e.g. P0001", key="filter_pid")
        with fcol3:
            all_ratings = sorted(reviews["rating"].unique())
            selected_ratings = st.multiselect("Rating", all_ratings, default=all_ratings, key="filter_ratings")
        with fcol4:
            vp_options = ["All", "Verified Only", "Unverified Only"]
            selected_vp = st.selectbox("Verified Purchase", vp_options, index=0, key="filter_vp")

        fcol5, fcol6 = st.columns([3, 1])
        with fcol5:
            if "product_category" in reviews.columns:
                categories = sorted(reviews["product_category"].dropna().unique())
                selected_categories = st.multiselect("Product Category", categories, default=categories, key="filter_cats")
            else:
                selected_categories = []
        with fcol6:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            reset_filters = st.button("Clear Filters", width="stretch")

    # Filter application
    filtered = reviews.copy()
    if not reset_filters:
        if reviewer_search.strip():
            filtered = filtered[filtered["reviewer_id"].str.contains(reviewer_search.strip(), case=False, na=False)]
        if product_search.strip():
            filtered = filtered[filtered["product_id"].str.contains(product_search.strip(), case=False, na=False)]
        if selected_ratings:
            filtered = filtered[filtered["rating"].isin(selected_ratings)]
        if selected_vp == "Verified Only" and "verified_purchase" in filtered.columns:
            filtered = filtered[filtered["verified_purchase"] == True]
        elif selected_vp == "Unverified Only" and "verified_purchase" in filtered.columns:
            filtered = filtered[filtered["verified_purchase"] == False]
        if selected_categories and "product_category" in filtered.columns:
            filtered = filtered[filtered["product_category"].isin(selected_categories)]

    # Filter summary
    st.markdown(
        f'<div class="notice-box notice-info" style="margin-bottom: 12px;">'
        f"Displaying <strong>{len(filtered):,}</strong> matching records of {len(reviews):,} total reviews."
        f"</div>",
        unsafe_allow_html=True,
    )

    if filtered.empty:
        st.warning("No review records match the active filter criteria. Adjust or clear filters.")
    else:
        disp_cols = [c for c in ["review_id", "reviewer_id", "product_id", "rating", "timestamp", "product_category", "verified_purchase", "helpful_votes", "review_text"] if c in filtered.columns]
        st.dataframe(
            filtered[disp_cols].head(300),
            width="stretch",
            height=440,
        )

        st.markdown("---")
        st.markdown("### Reviewer Quick Action")
        avail_rids = sorted(filtered["reviewer_id"].unique())
        qcol1, qcol2 = st.columns([3, 1])
        with qcol1:
            inspect_rid = st.selectbox("Select reviewer from filtered results to inspect", avail_rids, key="quick_inspect_rid")
        with qcol2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("Inspect Reviewer Details", width="stretch", key="btn_inspect_reviewer_details"):
                st.session_state.selected_reviewer_id = inspect_rid
                st.session_state.pending_page = "Reviewer Inspection"
                st.rerun()


# ===========================================================================
# PAGE: REVIEWER NETWORK
# ===========================================================================
elif current_page == "Reviewer Network":
    st.subheader("Reviewer Similarity Network")
    st.markdown("Structural analysis and local neighbourhood inspection of the behavioral similarity graph.")

    graph_stats = load_graph_stats()
    adj_df = load_adjacency_matrix()
    assignments = load_cluster_assignments()

    if graph_stats is None or adj_df is None:
        st.error("Graph artifacts not found. Run Module 2 first: python -m src.similarity_graph")
        st.stop()

    g1, g2, g3, g4 = st.columns(4)
    with g1:
        render_metric_panel("Total Nodes", f"{graph_stats.get('node_count', 0):,}", "Reviewer entities")
    with g2:
        render_metric_panel("Total Edges", f"{graph_stats.get('edge_count', 0):,}", "Cosine similarity pairs")
    with g3:
        render_metric_panel("Graph Density", f"{graph_stats.get('graph_density', 0):.4f}", "Edge capacity ratio")
    with g4:
        render_metric_panel("Connected Components", str(graph_stats.get("connected_components", 0)), "Global connectivity")

    st.markdown(
        '<div class="notice-box notice-info" style="margin-top: 14px;">'
        "<strong>Graph Formulation:</strong> Each node represents a reviewer. Edges represent behavioral "
        f"cosine similarity above the top-{graph_stats.get('strategy_params', {}).get('top_k', 15)} "
        "nearest neighbor threshold with union symmetrization. Edge proximity reflects shared behavioral "
        "feature patterns, not platform friendship or verified collusion."
        "</div>",
        unsafe_allow_html=True,
    )

    tab_ego, tab_full, tab_degree = st.tabs([
        "Local Ego Network Inspection",
        "Full Network Structural Overview",
        "Degree Distribution",
    ])

    with tab_ego:
        st.markdown("#### Reviewer Local Neighborhood")
        st.caption("Select a reviewer to render their connected sub-network and examine neighbor similarity weights.")

        all_rids = sorted(adj_df.index.tolist())
        target_rid = st.selectbox(
            "Select Target Reviewer",
            all_rids,
            index=all_rids.index(st.session_state.selected_reviewer_id) if st.session_state.selected_reviewer_id in all_rids else 0,
        )
        st.session_state.selected_reviewer_id = target_rid

        row_weights = adj_df.loc[target_rid]
        nbrs = row_weights[row_weights > 0].sort_values(ascending=False)
        st.markdown(f"**Target Reviewer:** `{target_rid}` | Connected Neighbors: **{len(nbrs)}**")

        if len(nbrs) > 0:
            top_k_nbrs = list(nbrs.head(15).items())
            cluster_map = dict(zip(assignments["reviewer_id"], assignments["cluster_id"])) if assignments is not None else {}

            G = nx.Graph()
            G.add_node(target_rid, role="target", cluster=cluster_map.get(target_rid, 0))
            for peer, w in top_k_nbrs:
                G.add_node(peer, role="peer", cluster=cluster_map.get(peer, 0))
                G.add_edge(target_rid, peer, weight=float(w))

            for i, (n1, _) in enumerate(top_k_nbrs):
                for n2, _ in top_k_nbrs[i+1:]:
                    if n1 in adj_df.index and n2 in adj_df.columns:
                        w_inter = float(adj_df.loc[n1, n2])
                        if w_inter > 0:
                            G.add_edge(n1, n2, weight=w_inter)

            pos = nx.spring_layout(G, seed=42, k=0.5)

            edge_x, edge_y = [], []
            for u, v in G.edges():
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color="#CBD5E1"),
                hoverinfo="none",
                mode="lines",
            )

            node_x, node_y, node_colors, node_texts, node_sizes = [], [], [], [], []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                c_id = cluster_map.get(node, 0)
                if node == target_rid:
                    node_colors.append("#245B8F")
                    node_sizes.append(22)
                    node_texts.append(f"Target: {node}<br>Cluster: {c_id}<br>Neighbors: {len(G[node])}")
                else:
                    node_colors.append("#237A57" if c_id == 0 else "#A66B16")
                    node_sizes.append(14)
                    node_texts.append(f"Peer: {node}<br>Cluster: {c_id}<br>Neighbors: {len(G[node])}")

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode="markers+text",
                hoverinfo="text",
                text=[n if n == target_rid else "" for n in G.nodes()],
                textposition="top center",
                textfont=dict(size=11, color="#182230", family="JetBrains Mono, monospace"),
                hovertext=node_texts,
                marker=dict(
                    color=node_colors,
                    size=node_sizes,
                    line=dict(width=1.5, color="#FFFFFF"),
                ),
            )

            fig_ego = go.Figure(data=[edge_trace, node_trace])
            fig_ego.update_layout(
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            )
            apply_enterprise_theme(fig_ego, height=440, title=f"Local Ego Network for {target_rid} (Top Neighbors)")
            st.plotly_chart(fig_ego, width="stretch")

            nbr_table_data = []
            for peer, w in top_k_nbrs:
                nbr_table_data.append({
                    "Neighbor Reviewer ID": peer,
                    "Cosine Similarity Weight": f"{w:.4f}",
                    "Predicted Cluster": cluster_map.get(peer, "Unknown"),
                })
            st.markdown("##### Top Neighbor Peer Similarity Weights")
            st.dataframe(pd.DataFrame(nbr_table_data), width="stretch")
        else:
            st.info(f"Reviewer {target_rid} has no connected neighbors in the similarity network.")

    with tab_full:
        st.markdown("#### Global Similarity Network Topology")
        st.caption("Visual representation of all 1,000 reviewers organized by behavioral feature proximity.")
        net_img = FIGURES_DIR / "reviewer_network.png"
        if net_img.exists():
            st.image(
                str(net_img),
                caption="Global reviewer similarity network. Node proximity reflects behavioral feature similarity; not an official fraud classification.",
                width="stretch",
            )
        else:
            st.warning("Network image not found in outputs/figures directory.")

    with tab_degree:
        st.markdown("#### Node Weighted Degree Distribution")
        degrees_df = load_degree_matrix()
        if degrees_df is not None:
            fig_deg = px.histogram(
                degrees_df,
                x="weighted_degree",
                nbins=35,
                color_discrete_sequence=["#245B8F"],
            )
            fig_deg.update_layout(xaxis_title="Weighted Degree", yaxis_title="Reviewer Count")
            apply_enterprise_theme(fig_deg, height=340, title="Weighted Degree Histogram")
            st.plotly_chart(fig_deg, width="stretch")

    st.markdown("---")
    with st.expander("Review Module 2 Graph Validation Report"):
        rep = load_graph_report()
        st.code(rep if rep else "Report unavailable.", language="text")


# ===========================================================================
# PAGE: CLUSTERS
# ===========================================================================
elif current_page == "Clusters":
    st.subheader("Spectral Clustering Analysis")
    st.markdown("Unsupervised reviewer partition discovery and behavioral divergence investigation.")

    assignments = load_cluster_assignments()
    behavior = load_behavior_summary()
    exp = load_experiment_summary()
    features = load_features()

    if assignments is None or behavior is None:
        st.error("Clustering outputs not found. Run Module 3 first: python -m src.spectral_clustering")
        st.stop()

    st.markdown(
        '<div class="notice-box notice-warning">'
        "<strong>Cautionary Interpretation:</strong> Cluster identifiers (0, 1) are arbitrary integer labels "
        "assigned by K-Means in the spectral embedding space. Cluster membership indicates behavioral similarity, "
        "not confirmed fraud. Positive and negative rating manipulation patterns must be analyzed separately."
        "</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_panel("Primary k Selection", str(exp.get("primary_k", 2)) if exp else "2", "Unsupervised criterion")
    with c2:
        c0_size = len(assignments[assignments["cluster_id"] == 0])
        render_metric_panel("Cluster 0 Size", f"{c0_size} ({c0_size/10:.1f}%)", "Predicted partition")
    with c3:
        c1_size = len(assignments[assignments["cluster_id"] == 1])
        render_metric_panel("Cluster 1 Size", f"{c1_size} ({c1_size/10:.1f}%)", "Predicted partition")
    with c4:
        sil = exp.get("primary_metrics", {}).get("silhouette", 0) if exp else 0
        render_metric_panel("Silhouette Score", f"{sil:.4f}", "Spectral embedding separation")

    st.markdown("---")

    ctab1, ctab2, ctab3, ctab4 = st.tabs([
        "Behavioral Comparison",
        "Manipulation Analysis",
        "Spectral Embedding",
        "Cluster Member Explorer",
    ])

    with ctab1:
        st.markdown("#### Behavioral Profile Comparison")
        st.caption("Averaged behavioral feature profiles by cluster partition.")
        disp_behavior = behavior.copy()
        if "cluster_id" in disp_behavior.columns:
            disp_behavior["cluster_id"] = disp_behavior["cluster_id"].apply(lambda x: f"Cluster {x}")
        st.dataframe(disp_behavior, width="stretch")

        if "mean_rating_deviation" in behavior.columns:
            bcol1, bcol2 = st.columns(2)
            with bcol1:
                fig_dev = px.bar(
                    behavior,
                    x="cluster_id",
                    y="mean_rating_deviation",
                    text="mean_rating_deviation",
                    color="cluster_id",
                    color_discrete_sequence=["#237A57", "#A66B16"],
                )
                fig_dev.update_layout(xaxis_title="Cluster", yaxis_title="Signed Rating Deviation", showlegend=False)
                fig_dev.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                apply_enterprise_theme(fig_dev, height=320, title="Mean Signed Rating Deviation")
                st.plotly_chart(fig_dev, width="stretch")
            with bcol2:
                fig_cnt = px.bar(
                    behavior,
                    x="cluster_id",
                    y="review_count",
                    text="review_count",
                    color="cluster_id",
                    color_discrete_sequence=["#237A57", "#A66B16"],
                )
                fig_cnt.update_layout(xaxis_title="Cluster", yaxis_title="Average Reviews per User", showlegend=False)
                fig_cnt.update_traces(texttemplate="%{text:.1f}", textposition="outside")
                apply_enterprise_theme(fig_cnt, height=320, title="Average Review Volume")
                st.plotly_chart(fig_cnt, width="stretch")

    with ctab2:
        st.markdown("#### Positive and Negative Manipulation Investigation")
        st.caption("Disaggregated analysis of rating deviations, review volume, and verified purchase ratios.")

        if features is not None:
            merged = assignments.merge(features.reset_index(), on="reviewer_id", how="left")

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                fig_box1 = px.box(
                    merged,
                    x="cluster_id",
                    y="mean_rating_deviation",
                    color="cluster_id",
                    color_discrete_sequence=["#237A57", "#A66B16"],
                )
                fig_box1.update_layout(xaxis_title="Cluster ID", yaxis_title="Signed Deviation", showlegend=False)
                apply_enterprise_theme(fig_box1, height=320, title="Signed Rating Deviation Distribution")
                st.plotly_chart(fig_box1, width="stretch")
            with col_m2:
                fig_box2 = px.box(
                    merged,
                    x="cluster_id",
                    y="mean_absolute_rating_deviation",
                    color="cluster_id",
                    color_discrete_sequence=["#237A57", "#A66B16"],
                )
                fig_box2.update_layout(xaxis_title="Cluster ID", yaxis_title="Absolute Deviation", showlegend=False)
                apply_enterprise_theme(fig_box2, height=320, title="Absolute Rating Deviation Distribution")
                st.plotly_chart(fig_box2, width="stretch")

            col_m3, col_m4 = st.columns(2)
            with col_m3:
                fig_vp = px.box(
                    merged,
                    x="cluster_id",
                    y="verified_purchase_ratio",
                    color="cluster_id",
                    color_discrete_sequence=["#237A57", "#A66B16"],
                )
                fig_vp.update_layout(xaxis_title="Cluster ID", yaxis_title="Verified Purchase Ratio", showlegend=False)
                apply_enterprise_theme(fig_vp, height=320, title="Verified Purchase Proportion")
                st.plotly_chart(fig_vp, width="stretch")
            with col_m4:
                fig_cnt = px.box(
                    merged,
                    x="cluster_id",
                    y="review_count",
                    color="cluster_id",
                    color_discrete_sequence=["#237A57", "#A66B16"],
                )
                fig_cnt.update_layout(xaxis_title="Cluster ID", yaxis_title="Total Review Count", showlegend=False)
                apply_enterprise_theme(fig_cnt, height=320, title="User Activity Volume")
                st.plotly_chart(fig_cnt, width="stretch")

    with ctab3:
        st.markdown("#### Spectral Embedding Representation")
        st.caption("2D projection of reviewers in graph Laplacian spectral space.")
        emb_img = FIGURES_DIR / "spectral_embedding.png"
        if emb_img.exists():
            st.image(str(emb_img), caption="Reviewer coordinates in spectral embedding space colored by predicted cluster.", width="stretch")
        else:
            st.info("Spectral embedding visualization not available in outputs/figures.")

    with ctab4:
        st.markdown("#### Cluster Member Exploration")
        target_cluster = st.selectbox("Select Cluster to Inspect", sorted(assignments["cluster_id"].unique()))
        members = assignments[assignments["cluster_id"] == target_cluster]
        if features is not None:
            members_full = members.merge(features.reset_index(), on="reviewer_id", how="left")
            st.dataframe(members_full.head(200), width="stretch")
        else:
            st.dataframe(members.head(200), width="stretch")


# ===========================================================================
# PAGE: EVALUATION
# ===========================================================================
elif current_page == "Evaluation":
    st.subheader("Clustering Evaluation")
    st.markdown("Experimental evaluation against synthetic ground-truth benchmark categories.")

    st.markdown(
        '<div class="notice-box notice-info">'
        "<strong>Ground-Truth Isolation Policy:</strong> Ground-truth labels describe synthetically generated "
        "categories and campaigns. They were strictly quarantined from feature engineering, similarity graph "
        "construction, and spectral clustering. Metrics evaluate partition agreement against synthetic benchmark "
        "parameters, not real-world fraud detection accuracy."
        "</div>",
        unsafe_allow_html=True,
    )

    metrics_df = load_cluster_metrics()
    contingency = load_contingency()
    exp = load_experiment_summary()

    if metrics_df is None or contingency is None:
        st.error("Evaluation outputs not found. Run Module 3 first: python -m src.spectral_clustering")
        st.stop()

    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        with st.container(border=True):
            st.markdown("**Adjusted Rand Index (ARI)**")
            st.caption("Measures pair-wise partition agreement adjusted for chance. Range: [-1, 1]. Baseline random = 0.0.")
    with ec2:
        with st.container(border=True):
            st.markdown("**Normalized Mutual Info (NMI)**")
            st.caption("Information-theoretic partition agreement. Range: [0, 1]. Independent = 0.0.")
    with ec3:
        with st.container(border=True):
            st.markdown("**Silhouette Score**")
            st.caption("Unsupervised structural separation in spectral space. Range: [-1, 1]. Unsupervised selection criterion.")

    st.markdown("---")

    if exp:
        pm = exp.get("primary_metrics", {})
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            render_metric_panel("Primary k", str(exp.get("primary_k", 2)), "Selected cluster count")
        with mc2:
            render_metric_panel("Primary ARI", f"{pm.get('ari', 0):.4f}", "Synthetic partition agreement")
        with mc3:
            render_metric_panel("Primary NMI", f"{pm.get('nmi', 0):.4f}", "Information overlap")
        with mc4:
            render_metric_panel("Primary Silhouette", f"{pm.get('silhouette', 0):.4f}", "Spectral compactness")

    st.markdown("---")
    st.markdown("### Contingency Matrix (Predicted Cluster vs Synthetic Category)")

    ch1, ch2 = st.columns([1, 1])
    with ch1:
        st.dataframe(contingency, width="stretch")
    with ch2:
        fig_heat = px.imshow(
            contingency.values,
            labels=dict(x="Synthetic Category", y="Predicted Cluster", color="Count"),
            x=list(contingency.columns),
            y=[f"Cluster {i}" for i in contingency.index],
            color_continuous_scale="Blues",
            text_auto=True,
        )
        apply_enterprise_theme(fig_heat, height=280, title="Contingency Heatmap")
        st.plotly_chart(fig_heat, width="stretch")

    st.markdown("---")
    st.markdown("### Multi-k Parameter Sweep")
    st.dataframe(metrics_df, width="stretch")

    keval_img = FIGURES_DIR / "k_evaluation.png"
    if keval_img.exists():
        st.image(str(keval_img), caption="Evaluation metrics progression across cluster counts k=2 through k=8.", width="stretch")

    gt_img = FIGURES_DIR / "ground_truth_comparison.png"
    if gt_img.exists():
        st.markdown("---")
        st.markdown("### Spectral Space Partition Comparison")
        st.image(str(gt_img), caption="Side-by-side comparison: Predicted clusters (left) vs Synthetic ground truth (right).", width="stretch")


# ===========================================================================
# PAGE: REVIEWER INSPECTION
# ===========================================================================
elif current_page == "Reviewer Inspection":
    st.subheader("Reviewer Inspection")
    st.markdown("Comprehensive behavioral fingerprint, network neighborhood, and historical activity for an individual reviewer.")

    reviews = load_reviews()
    features = load_features()
    assignments = load_cluster_assignments()
    degrees_df = load_degree_matrix()
    adj_df = load_adjacency_matrix()
    gt = load_ground_truth()

    if features is None or reviews is None:
        st.error("Feature and review datasets not found. Run Modules 1-3 first.")
        st.stop()

    all_rids = sorted(features.index.tolist())
    target_rid = st.selectbox(
        "Select Reviewer ID to Inspect",
        all_rids,
        index=all_rids.index(st.session_state.selected_reviewer_id) if st.session_state.selected_reviewer_id in all_rids else 0,
    )
    st.session_state.selected_reviewer_id = target_rid

    c_match = assignments[assignments["reviewer_id"] == target_rid] if assignments is not None else None
    c_id = int(c_match.iloc[0]["cluster_id"]) if (c_match is not None and not c_match.empty) else "N/A"

    deg_match = degrees_df[degrees_df["reviewer_id"] == target_rid] if degrees_df is not None else None
    deg_val = f"{deg_match.iloc[0]['weighted_degree']:.4f}" if (deg_match is not None and not deg_match.empty) else "N/A"

    user_reviews = reviews[reviews["reviewer_id"] == target_rid]

    with st.container(border=True):
        st.markdown(
            f"**Reviewer:** `{target_rid}` &nbsp;|&nbsp; "
            f"**Predicted Cluster:** `Cluster {c_id}` &nbsp;|&nbsp; "
            f"**Weighted Degree:** `{deg_val}` &nbsp;|&nbsp; "
            f"**Total Reviews:** `{len(user_reviews)}`"
        )

    col_f1, col_f2 = st.columns([1, 1])

    with col_f1:
        st.markdown("#### Behavioral Feature Fingerprint")
        f_row = features.loc[target_rid]
        f_data = []
        for col_name, val in f_row.items():
            f_data.append({
                "Feature Name": col_name,
                "Standardized Value": f"{val:.4f}" if isinstance(val, (float, np.floating)) else str(val),
            })
        st.dataframe(pd.DataFrame(f_data), width="stretch", height=360)

    with col_f2:
        st.markdown("#### Similarity Network Neighborhood")
        if adj_df is not None and target_rid in adj_df.index:
            row_w = adj_df.loc[target_rid]
            nbr_nonzero = row_w[row_w > 0].sort_values(ascending=False).head(10)
            if not nbr_nonzero.empty:
                nbr_table = []
                for p_id, w in nbr_nonzero.items():
                    p_c = assignments[assignments["reviewer_id"] == p_id]["cluster_id"].values[0] if assignments is not None else "N/A"
                    nbr_table.append({
                        "Peer Reviewer ID": p_id,
                        "Similarity Weight": f"{w:.4f}",
                        "Predicted Cluster": f"Cluster {p_c}",
                    })
                st.dataframe(pd.DataFrame(nbr_table), width="stretch", height=360)
            else:
                st.info("No connected neighbors in similarity network.")
        else:
            st.info("Adjacency data not available.")

    st.markdown("---")
    st.markdown("#### Review History Timeline")
    if not user_reviews.empty:
        r_cols = [c for c in ["review_id", "product_id", "rating", "timestamp", "product_category", "verified_purchase", "helpful_votes", "review_text"] if c in user_reviews.columns]
        st.dataframe(user_reviews[r_cols], width="stretch")
    else:
        st.info("No reviews recorded for this user.")

    st.markdown("---")
    with st.container(border=True):
        st.markdown("**Synthetic Ground Truth Reference (Evaluation Only)**")
        st.caption(
            "This label was generated during synthetic dataset simulation and was strictly quarantined "
            "from the unsupervised clustering pipeline."
        )

        if gt is not None:
            gt_user = gt[gt["reviewer_id"] == target_rid]
            if not gt_user.empty:
                g_row = gt_user.iloc[0]
                g_cat = g_row.get("ground_truth_category", "Unknown")
                g_camp = g_row.get("campaign_id", "None")
                g_dir = g_row.get("campaign_direction", "None")
                st.markdown(f"Category: **`{g_cat}`** &nbsp;|&nbsp; Campaign ID: **`{g_camp}`** &nbsp;|&nbsp; Direction: **`{g_dir}`**")
            else:
                st.caption("No ground truth entry for this reviewer.")


# ===========================================================================
# PAGE: REPORTS
# ===========================================================================
elif current_page == "Reports":
    st.subheader("Reports & Data Exports")
    st.markdown("Download validated analytical artifacts or execute pipeline stages.")

    st.markdown("### Generated Data Artifacts")

    files_to_download = [
        ("reviewer_cluster_assignments.csv", CLUSTER_DIR / "reviewer_cluster_assignments.csv", "Reviewer cluster partition assignments", "text/csv"),
        ("cluster_behavior_summary.csv", CLUSTER_DIR / "cluster_behavior_summary.csv", "Cluster aggregated behavioral statistics", "text/csv"),
        ("clustering_metrics.csv", CLUSTER_DIR / "clustering_metrics.csv", "Multi-k evaluation metrics sweep", "text/csv"),
        ("graph_statistics.json", GRAPH_DIR / "graph_statistics.json", "Graph structural statistics and density", "application/json"),
        ("module1_validation_report.txt", REPORTS_DIR / "module1_validation_report.txt", "Data integrity validation report", "text/plain"),
        ("graph_validation_report.txt", GRAPH_DIR / "graph_validation_report.txt", "Graph property verification report", "text/plain"),
        ("evaluation_report.txt", CLUSTER_DIR / "evaluation_report.txt", "Academic clustering evaluation report", "text/plain"),
    ]

    for fname, path, desc, mime in files_to_download:
        c_desc, c_btn = st.columns([3, 1])
        with c_desc:
            st.markdown(f"**`{fname}`** - {desc}")
        with c_btn:
            if path.exists():
                st.download_button(
                    label=f"Download {fname.split('.')[-1].upper()}",
                    data=path.read_bytes(),
                    file_name=fname,
                    mime=mime,
                    key=f"dl_{fname}",
                    width="stretch",
                )
            else:
                st.button(label="Unavailable", disabled=True, key=f"dl_dis_{fname}", width="stretch")

    st.markdown("---")
    st.markdown("### Pipeline Execution Panel")
    st.markdown(
        "Execute project pipeline modules directly from the environment. "
        "Subprocesses run the validated command-line modules in sequence."
    )

    if st.button("Execute Full Pipeline (Modules 1 - 3)", type="primary"):
        with st.status("Executing End-to-End Pipeline...", expanded=True) as status:
            st.write("Running Module 1: Preprocessing & Feature Engineering...")
            r1 = subprocess.run(
                [sys.executable, "-m", "src.preprocessing", "--data-dir", "data", "--output-dir", "outputs"],
                capture_output=True, text=True, cwd=str(BASE_DIR),
            )
            if r1.returncode != 0:
                st.error(f"Module 1 Failed: {r1.stderr}")
                status.update(label="Pipeline execution failed", state="error")
                st.stop()
            st.write("Module 1 completed.")

            st.write("Running Module 2: Similarity Graph Construction...")
            r2 = subprocess.run(
                [sys.executable, "-m", "src.similarity_graph", "--input-dir", "outputs", "--output-dir", "outputs/graph"],
                capture_output=True, text=True, cwd=str(BASE_DIR),
            )
            if r2.returncode != 0:
                st.error(f"Module 2 Failed: {r2.stderr}")
                status.update(label="Pipeline execution failed", state="error")
                st.stop()
            st.write("Module 2 completed.")

            st.write("Running Module 3: Spectral Clustering & Evaluation...")
            r3 = subprocess.run(
                [sys.executable, "-m", "src.spectral_clustering", "--graph-dir", "outputs/graph", "--features-dir", "outputs", "--output-dir", "outputs/clustering"],
                capture_output=True, text=True, cwd=str(BASE_DIR),
            )
            if r3.returncode != 0:
                st.error(f"Module 3 Failed: {r3.stderr}")
                status.update(label="Pipeline execution failed", state="error")
                st.stop()
            st.write("Module 3 completed.")

            status.update(label="Pipeline execution completed successfully", state="complete")
            st.cache_data.clear()
            st.rerun()


# ===========================================================================
# PAGE: PRIVACY POLICY
# ===========================================================================
elif current_page == "Privacy Policy":
    st.subheader("Privacy Policy")
    st.markdown("Transparency, data handling practices, and local execution boundaries.")

    with st.container(border=True):
        st.markdown("#### 1. Scope and Research Context")
        st.write(
            "This software is an academic research prototype developed for analyzing coordinated review networks "
            "using graph spectral algorithms. The application operates solely as a local analytical tool."
        )

    with st.container(border=True):
        st.markdown("#### 2. Synthetic Data Processing")
        st.write(
            "The dataset bundled with this application consists entirely of synthetically generated product "
            "reviews, simulated reviewer entities, and artificial rating campaigns. No authentic consumer personal "
            "identifiable information (PII), payment credentials, or real-world merchant accounts are processed, "
            "stored, or inferred."
        )

    with st.container(border=True):
        st.markdown("#### 3. Local Execution and Absence of Remote Telemetry")
        st.write(
            "All mathematical computations, graph operations, and dashboard renderings occur strictly within the "
            "user's local workstation environment. The application does not transmit data over the network, does not "
            "communicate with third-party tracking services, and collects zero telemetry or analytics cookies."
        )

    with st.container(border=True):
        st.markdown("#### 4. Data Lifecycle and Local Storage")
        st.write(
            "Generated analytical artifacts (such as feature matrices, adjacency graphs, and cluster assignments) "
            "are written exclusively to the local file system in the outputs/ directory. Users may purge "
            "or re-initialize these files at any time by removing the respective directories."
        )

    with st.container(border=True):
        st.markdown("#### 5. Contact and Academic Governance")
        st.write(
            "For questions regarding research methodology, algorithmic specifications, or reproducibility, "
            "please refer to the project documentation repository."
        )


# ===========================================================================
# PAGE: TERMS AND CONDITIONS
# ===========================================================================
elif current_page == "Terms and Conditions":
    st.subheader("Terms and Conditions")
    st.markdown("Research scope, analytical limitations, and responsible interpretation guidelines.")

    with st.container(border=True):
        st.markdown("#### 1. Permitted Use and Academic Scope")
        st.write(
            "This application is provided exclusively for educational demonstrations, algorithmic research, "
            "and academic benchmarking of unsupervised graph clustering techniques. It is not licensed or "
            "certified for commercial fraud detection or automated moderation enforcement."
        )

    with st.container(border=True):
        st.markdown("#### 2. Synthetic Dataset Limitations")
        st.write(
            "All review content, user IDs, and product identifiers are generated using algorithmic simulation models. "
            "Evaluation metrics (such as ARI and NMI) measure partition agreement against synthetic generation parameters "
            "rather than real-world detection efficacy."
        )

    with st.container(border=True):
        st.markdown("#### 3. Unsupervised Clustering Interpretation")
        st.write(
            "Spectral Clustering identifies mathematical graph partitions based on feature similarity. Membership "
            "in a particular cluster does not constitute proof of fraud, malice, or coordination. False positives "
            "and non-malicious anomalies are inherent to unsupervised graph clustering."
        )

    with st.container(border=True):
        st.markdown("#### 4. User Responsibilities for External Data")
        st.write(
            "Users who replace the default synthetic dataset with proprietary or empirical data assume sole responsibility "
            "for compliance with applicable data protection laws, terms of service of source platforms, and institutional "
            "ethics guidelines."
        )

    with st.container(border=True):
        st.markdown("#### 5. Non-Warranty Disclaimer")
        st.write(
            "The software and analytical outputs are provided on an 'as-is' basis without warranties of any kind, "
            "express or implied. The project authors disclaim liability for any operational or business decisions "
            "based on this research prototype."
        )


# ---------------------------------------------------------------------------
# Global Application Footer
# ---------------------------------------------------------------------------
st.markdown(
    """<div class="app-footer">
<div><strong>Review Network Analytics</strong> | Research and Investigation Platform</div>
<div>v1.0.0-research | Synthetic benchmark dataset</div>
<div>Strictly local execution | Academic demonstration</div>
</div>""",
    unsafe_allow_html=True,
)
