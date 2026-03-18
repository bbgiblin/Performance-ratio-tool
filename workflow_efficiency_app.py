#!/usr/bin/env python3
"""
Workflow Efficiency Index Tool - Web UI
Interactive dashboard with multi-level aggregation and hybrid goal setting
Now with optional quality coefficient adjustment.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Workflow Efficiency Index",
    page_icon="peccy.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# COLUMN MAPPING & REGION DEFINITIONS
# ---------------------------------------------------------------------------

COLUMN_MAPPING = {
    "Column-1:Transformation Type": "workflow",
    "Column-2:Locale": "locale",
    "Column-3:Site": "site",
    "Select Date Part": "date_part",
    "Average Handle Time(In Secs)": "aht_median",
    "Processed Units": "processed_units",
    "Column-4:Ops Manager": "ops_manager",
    "Column-5:Team Manager": "team_manager",
    "Column-6:DA": "da_name",
    "Column-7:Demand Category": "demand_category",
    "Column-8:Customer": "customer",
}

# Region mapping
REGION_MAPPING = {
    "BOS": "AMER",
    "SJO": "AMER",
    "AMS": "EMEA",
    "CBG": "EMEA",
    "GDN": "EMEA",
    "LHR": "EMEA",
    "AMM": "APAJ",
    "HYD": "APAJ",
    "KIX": "APAJ",
    "MAA": "APAJ",
}

# Employee-based aggregation levels
EMPLOYEE_AGGREGATION_LEVELS = {
    "Network": "network",
    "Region": "region",
    "Site": "site",
    "Ops Manager": "ops_manager",
    "Team Manager": "team_manager",
    "Data Associate": "da_name",
}

# Work type aggregation levels
WORK_TYPE_AGGREGATION_LEVELS = {
    "Network": "network",
    "Demand Category": "demand_category",
    "Customer": "customer",
    "Workflow": "workflow",
    "Workflow+Locale": "workflow_key",
}

# ---------------------------------------------------------------------------
# QUALITY COLUMN DETECTION
# ---------------------------------------------------------------------------

# The quality CSV has 5 leading columns whose order may change.
# We identify them by prefix pattern and map to canonical names.

QUALITY_CANONICAL = {
    "workflow_name": None,      # Col1 - Workflow Name
    "customer": None,           # Col2 - Customer
    "skill_type": None,         # Col3 - SkillType
    "locale": None,             # Col4 - Locale
    "site": None,               # Col5 - Site
}

# Fixed-name columns (always present after the first 5)
QUALITY_FIXED_COLUMNS = {
    "Time Period": "time_period",
    "Tasks audited (A)": "tasks_audited",
    "Tasks Meeting Quality / Pass Count (B)": "pass_count",
    "Fail Count (C)": "fail_count",
    "Quality % Pre Appeals (D=B/(B+C))": "quality_pre_appeals",
    "# Appeals (Task Appealed (E+F+G))": "num_appeals",
    "Accepted Appeals (E)": "accepted_appeals",
    "Rejected Appeals (F)": "rejected_appeals",
    "Pending Appeals (G)": "pending_appeals",
    "Quality Post Appeal Process (C=(B+E)/(B+C))": "quality_post_appeals",
    "Quality % Goals": "quality_goal",
    "Appeals % (E+F+G)/(B+C)": "appeals_pct",
    "Appeals Accepted %  (E/ (E+F+G))": "appeals_accepted_pct",
    "Quality % Status": "quality_status",
}


def _detect_quality_leading_columns(columns):
    """
    Detect the first 5 columns (Col1–Col5) regardless of order.
    Returns a dict mapping canonical name -> actual column name.
    """
    mapping = {}
    col_pattern_map = {
        "Workflow Name": "workflow_name",
        "Customer": "customer",
        "SkillType": "skill_type",
        "Locale": "locale",
        "Site": "site",
    }
    for col in columns:
        col_stripped = col.strip()
        for pattern, canonical in col_pattern_map.items():
            if pattern.lower() in col_stripped.lower() and canonical not in mapping:
                mapping[canonical] = col_stripped
                break
    return mapping


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

@st.cache_data
def load_and_clean_data(uploaded_file):
    """Load and clean the uploaded CSV."""
    df = pd.read_csv(uploaded_file)

    # Rename columns
    df = df.rename(columns=COLUMN_MAPPING)
    df.columns = df.columns.str.strip()

    # Convert to strings and clean required columns
    for col in ["site", "workflow", "locale"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df = df[df[col].notna() & (df[col] != "") & (df[col] != "nan")]

    # Add region column based on site
    df["region"] = df["site"].map(REGION_MAPPING)
    df["region"] = df["region"].fillna("Unknown")

    # Add network column (all data belongs to "Network")
    df["network"] = "Network"

    # Handle optional hierarchical columns (employee-based)
    for col in ["ops_manager", "team_manager", "da_name"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(["nan", ""], pd.NA)
        else:
            df[col] = pd.NA

    # Handle optional work type columns
    for col in ["demand_category", "customer"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(["nan", ""], pd.NA)
        else:
            df[col] = pd.NA

    # Convert numeric columns
    df["aht_median"] = pd.to_numeric(df["aht_median"], errors="coerce")
    df["processed_units"] = pd.to_numeric(df["processed_units"], errors="coerce")
    df = df.dropna(subset=["aht_median", "processed_units"])
    df = df[df["processed_units"] > 0].copy()
    df = df[df["aht_median"] > 0].copy()

    # Handle date_part format - WEEKS ONLY
    df["date_part"] = df["date_part"].astype(str)

    if df["date_part"].str.contains("Week", case=False).any():
        df["year"] = df["date_part"].str.extract(r"(\d{4})").astype(int)
        df["week_num"] = df["date_part"].str.extract(r"Week\s*(\d+)", expand=False).astype(int)
        df["period_sort"] = df["year"] * 100 + df["week_num"]
        df["period_label"] = df["year"].astype(str) + "-W" + df["week_num"].astype(str).str.zfill(2)
    else:
        df["period_sort"] = df["date_part"].astype(int)
        df["period_label"] = "W" + df["period_sort"].astype(str).str.zfill(2)

    df["workflow_key"] = df["workflow"] + " | " + df["locale"]

    return df


@st.cache_data
def load_expected_aht(uploaded_file):
    """Load expected AHT goals from CSV (handles various formats)."""
    edf = pd.read_csv(uploaded_file)

    if len(edf.columns) >= 3:
        edf = edf.iloc[:, :3].copy()
        edf.columns = ["workflow", "locale", "expected_aht"]
    else:
        edf.columns = edf.columns.str.strip().str.lower().str.replace(" ", "_")

    edf["workflow"] = edf["workflow"].astype(str).str.strip()
    edf["locale"] = edf["locale"].astype(str).str.strip()
    edf["expected_aht"] = pd.to_numeric(edf["expected_aht"], errors="coerce")

    edf = edf.dropna(subset=["expected_aht"])
    edf = edf[edf["expected_aht"] > 0]

    edf["workflow_key"] = edf["workflow"] + " | " + edf["locale"]

    return edf[["workflow_key", "expected_aht"]].drop_duplicates()


@st.cache_data
def load_quality_data(uploaded_file):
    """
    Load quality CSV.  The first 5 columns may appear in any order;
    the remaining columns have fixed names.
    Returns a cleaned DataFrame with canonical column names plus:
      - workflow_key  (workflow_name + " | " + locale)
      - period_sort / period_label  (parsed from time_period)
    """
    qdf = pd.read_csv(uploaded_file)
    qdf.columns = qdf.columns.str.strip()

    # --- detect the 5 leading columns ---
    leading_map = _detect_quality_leading_columns(qdf.columns)

    rename_map = {}
    for canonical, actual in leading_map.items():
        rename_map[actual] = canonical

    # --- fixed columns ---
    for original, canonical in QUALITY_FIXED_COLUMNS.items():
        # find the column (may have slight whitespace differences)
        for col in qdf.columns:
            if col.strip() == original.strip():
                rename_map[col] = canonical
                break

    qdf = qdf.rename(columns=rename_map)

    # Clean string columns
    for col in ["workflow_name", "locale", "site", "customer", "skill_type"]:
        if col in qdf.columns:
            qdf[col] = qdf[col].astype(str).str.strip()

    # Numeric columns
    for col in [
        "tasks_audited", "pass_count", "fail_count",
        "quality_pre_appeals", "num_appeals",
        "accepted_appeals", "rejected_appeals", "pending_appeals",
        "quality_post_appeals", "quality_goal",
        "appeals_pct", "appeals_accepted_pct",
    ]:
        if col in qdf.columns:
            qdf[col] = pd.to_numeric(qdf[col], errors="coerce")

    # Build workflow_key
    qdf["workflow_key"] = qdf["workflow_name"] + " | " + qdf["locale"]

    # Parse time_period  (e.g. "2026 Week 10")
    if "time_period" in qdf.columns:
        qdf["time_period"] = qdf["time_period"].astype(str).str.strip()
        qdf["q_year"] = qdf["time_period"].str.extract(r"(\d{4})").astype(float)
        qdf["q_week"] = qdf["time_period"].str.extract(r"Week\s*(\d+)", expand=False).astype(float)
        qdf["period_sort"] = (qdf["q_year"] * 100 + qdf["q_week"]).astype("Int64")
        qdf["period_label"] = (
            qdf["q_year"].astype("Int64").astype(str)
            + "-W"
            + qdf["q_week"].astype("Int64").astype(str).str.zfill(2)
        )

    # Add region
    if "site" in qdf.columns:
        qdf["region"] = qdf["site"].map(REGION_MAPPING).fillna("Unknown")

    qdf["network"] = "Network"

    return qdf


@st.cache_data
def compute_default_expected_aht(df, use_median=False):
    """Compute expected AHT (network average) from FULL dataset."""
    if use_median:
        expected = (
            df.groupby("workflow_key")["aht_median"]
            .median()
            .reset_index()
        )
    else:
        def weighted_mean(group):
            return np.average(group["aht_median"], weights=group["processed_units"])

        expected = (
            df.groupby("workflow_key")
            .apply(weighted_mean, include_groups=False)
            .reset_index()
        )

    expected.columns = ["workflow_key", "expected_aht"]
    return expected


# ---------------------------------------------------------------------------
# QUALITY COEFFICIENT COMPUTATION
# ---------------------------------------------------------------------------

def compute_quality_coefficients(
    quality_df,
    agg_level,
    sensitivity_k=3.0,
    min_audits=1,
):
    """
    Compute a quality coefficient per (entity, workflow_key, period_sort).

    Quality Coefficient = (Q_actual / Q_goal) ^ k

    Parameters
    ----------
    quality_df : pd.DataFrame
        Cleaned quality data (from load_quality_data).
    agg_level : str
        The column in quality_df to aggregate by.  For employee-level
        aggregation the quality data is at site level, so we map:
          - network / region / site  -> use site-level quality directly
            (for region/network we aggregate across sites)
          - ops_manager / team_manager / da_name -> not available in quality
            data, so we fall back to site-level quality.
    sensitivity_k : float
        Exponent controlling how aggressively deviations are penalised/rewarded.
    min_audits : int
        Minimum number of audits required for the coefficient to be applied.
        1 = apply if any audit exists; 6 = require 6+ audits.

    Returns
    -------
    pd.DataFrame with columns:
        [merge_key, workflow_key, period_sort, quality_coeff,
         q_actual, q_goal, tasks_audited_total]
        where merge_key matches the entity column used in the main data.
    """
    if quality_df is None or quality_df.empty:
        return pd.DataFrame()

    qdf = quality_df.copy()

    # Determine the grouping column in quality data.
    # Quality data only has site-level granularity, so for levels below site
    # we return site-level coefficients and let the caller join on site.
    # For region/network we aggregate UP from site.
    if agg_level in ("network", "region", "site"):
        q_group_col = agg_level
    else:
        # For ops_manager, team_manager, da_name — quality is at site level.
        # We'll return site-level coefficients; the caller joins via site.
        q_group_col = "site"

    # Ensure the group column exists
    if q_group_col not in qdf.columns:
        return pd.DataFrame()

    # Filter rows with valid quality data
    required = ["quality_post_appeals", "quality_goal", "tasks_audited"]
    for c in required:
        if c not in qdf.columns:
            return pd.DataFrame()

    qdf = qdf.dropna(subset=required)
    qdf = qdf[qdf["quality_goal"] > 0].copy()

    # Aggregate: sum pass_count, fail_count, accepted_appeals at the
    # (group, workflow_key, period_sort) level, then recompute quality.
    agg_dict = {
        "pass_count": "sum",
        "fail_count": "sum",
        "accepted_appeals": "sum",
        "tasks_audited": "sum",
        "quality_goal": "first",  # goal is the same for a workflow+locale
    }
    # Only include columns that exist
    agg_dict = {k: v for k, v in agg_dict.items() if k in qdf.columns}

    grouped = (
        qdf.groupby([q_group_col, "workflow_key", "period_sort"])
        .agg(agg_dict)
        .reset_index()
    )

    # Recompute quality post appeals from aggregated counts
    total = grouped["pass_count"] + grouped["fail_count"]
    grouped["q_actual"] = np.where(
        total > 0,
        (grouped["pass_count"] + grouped.get("accepted_appeals", 0)) / total,
        np.nan,
    )
    grouped["q_goal"] = grouped["quality_goal"]

    # Apply minimum audit threshold
    grouped = grouped[grouped["tasks_audited"] >= min_audits].copy()
    grouped = grouped.dropna(subset=["q_actual"])

    # Compute coefficient: (Q_actual / Q_goal) ^ k
    ratio = grouped["q_actual"] / grouped["q_goal"]
    # Clamp ratio to avoid extreme values (e.g. cap at 2x boost / 0.1x floor)
    ratio = ratio.clip(lower=0.1, upper=2.0)
    grouped["quality_coeff"] = ratio ** sensitivity_k

    # Rename group column to a standard merge key
    grouped = grouped.rename(columns={q_group_col: "q_merge_key"})

    result = grouped[
        ["q_merge_key", "workflow_key", "period_sort",
         "quality_coeff", "q_actual", "q_goal", "tasks_audited"]
    ].copy()
    result["q_agg_level"] = q_group_col  # remember what level this is at

    return result


def _resolve_quality_join_column(agg_level):
    """
    Return the column name in the main AHT data that should be used to
    join quality coefficients.  Quality data is at site level, so for
    sub-site aggregation levels we join on 'site'.
    """
    if agg_level in ("network", "region", "site"):
        return agg_level
    else:
        return "site"


# ---------------------------------------------------------------------------
# HELPER: PRIORITISED WORKFLOW/SITE LIST
# ---------------------------------------------------------------------------

def get_prioritised_workflows(df_full, agg_level, selected_entities, all_workflows):
    """
    Return workflow list with workflows done by the selected entities first,
    then all remaining workflows after.  Both sub-lists are sorted alphabetically.
    """
    if not selected_entities:
        return sorted(all_workflows)

    entity_workflows = set(
        df_full[df_full[agg_level].isin(selected_entities)]["workflow_key"].unique()
    )
    done = sorted([w for w in all_workflows if w in entity_workflows])
    rest = sorted([w for w in all_workflows if w not in entity_workflows])
    return done + rest


def get_prioritised_sites(df_full, agg_level, selected_entities, all_sites):
    """
    Return site list with sites that have the selected entities first,
    then all remaining sites after. Both sub-lists are sorted alphabetically.
    """
    if not selected_entities:
        return sorted(all_sites)

    entity_sites = set(
        df_full[df_full[agg_level].isin(selected_entities)]["site"].unique()
    )
    done = sorted([s for s in all_sites if s in entity_sites])
    rest = sorted([s for s in all_sites if s not in entity_sites])
    return done + rest


# ---------------------------------------------------------------------------
# HELPER: WEEK RANGE SELECTOR
# ---------------------------------------------------------------------------

def week_range_selector(all_weeks_sorted, key_prefix):
    """
    Render a start/end week range selector using select_slider.
    Returns a list of selected week labels, or None if 'All' is chosen.
    """
    week_filter_mode = st.radio(
        "Weeks:",
        options=["All", "Range"],
        key=f"{key_prefix}_week_mode",
        horizontal=True
    )

    if week_filter_mode == "Range":
        if len(all_weeks_sorted) < 2:
            st.caption("Not enough weeks for a range.")
            return None

        start_week, end_week = st.select_slider(
            "Select week range:",
            options=all_weeks_sorted,
            value=(all_weeks_sorted[0], all_weeks_sorted[-1]),
            key=f"{key_prefix}_week_slider",
        )

        start_idx = all_weeks_sorted.index(start_week)
        end_idx = all_weeks_sorted.index(end_week)
        selected_weeks = all_weeks_sorted[start_idx : end_idx + 1]

        st.caption(f"**{start_week}** → **{end_week}**  ({len(selected_weeks)} week{'s' if len(selected_weeks) != 1 else ''})")
        return selected_weeks
    else:
        return None


# ---------------------------------------------------------------------------
# CALCULATION
# ---------------------------------------------------------------------------

def calculate_performance_ratios(
    df, expected_aht_df, agg_level,
    use_hybrid=True, full_df_for_defaults=None, use_median=False,
    quality_coefficients=None,
):
    """Calculate Rs at the specified aggregation level, optionally adjusted by quality."""

    if use_hybrid:
        if full_df_for_defaults is not None:
            default_expected = compute_default_expected_aht(full_df_for_defaults, use_median)
        else:
            default_expected = compute_default_expected_aht(df, use_median)

        merged = df.merge(expected_aht_df, on="workflow_key", how="left", suffixes=("", "_goal"))
        merged = merged.merge(default_expected, on="workflow_key", how="left", suffixes=("", "_default"))
        merged["expected_aht"] = merged["expected_aht"].fillna(merged["expected_aht_default"])

        if "expected_aht_default" in merged.columns:
            merged = merged.drop(columns=["expected_aht_default"])
    else:
        merged = df.merge(expected_aht_df, on="workflow_key", how="left")

    merged = merged.dropna(subset=["expected_aht"])

    group_col = agg_level

    if group_col in merged.columns:
        merged = merged.dropna(subset=[group_col])
        merged = merged[merged[group_col] != ""]
    else:
        st.error(f"Column '{group_col}' not found in data.")
        return pd.DataFrame()

    # --- Merge quality coefficients ---
    if quality_coefficients is not None and not quality_coefficients.empty:
        q_join_col = _resolve_quality_join_column(agg_level)
        qc = quality_coefficients.copy()

        # Join on the appropriate column + workflow_key + period_sort
        merged = merged.merge(
            qc[["q_merge_key", "workflow_key", "period_sort", "quality_coeff"]],
            left_on=[q_join_col, "workflow_key", "period_sort"],
            right_on=["q_merge_key", "workflow_key", "period_sort"],
            how="left",
        )
        # Fill missing coefficients with 1.0 (no adjustment)
        merged["quality_coeff"] = merged["quality_coeff"].fillna(1.0)
        if "q_merge_key" in merged.columns:
            merged = merged.drop(columns=["q_merge_key"])
    else:
        merged["quality_coeff"] = 1.0

    group_period_totals = (
        merged.groupby([group_col, "period_sort"])["processed_units"]
        .sum()
        .reset_index()
        .rename(columns={"processed_units": "total_units"})
    )

    merged = merged.merge(group_period_totals, on=[group_col, "period_sort"])
    merged["pw_s"] = merged["processed_units"] / merged["total_units"]
    merged["rs_contribution"] = (
        (merged["expected_aht"] / merged["aht_median"])
        * merged["quality_coeff"]
        * merged["pw_s"]
    )

    result = (
        merged.groupby([group_col, "period_sort", "period_label"])
        .agg(
            performance_ratio=("rs_contribution", "sum"),
            total_units=("total_units", "first"),
            num_workflows=("workflow_key", "nunique"),
            avg_quality_coeff=("quality_coeff", "mean"),
        )
        .reset_index()
    )

    result["performance_ratio_pct"] = result["performance_ratio"] * 100
    result = result.rename(columns={group_col: "entity"})

    return result


def calculate_workflow_detail(
    df, expected_aht_df, agg_level,
    use_hybrid=True, full_df_for_defaults=None, use_median=False,
    quality_coefficients=None,
):
    """Calculate workflow-level detail, optionally adjusted by quality."""

    if use_hybrid:
        if full_df_for_defaults is not None:
            default_expected = compute_default_expected_aht(full_df_for_defaults, use_median)
        else:
            default_expected = compute_default_expected_aht(df, use_median)

        merged = df.merge(expected_aht_df, on="workflow_key", how="left", suffixes=("", "_goal"))
        merged = merged.merge(default_expected, on="workflow_key", how="left", suffixes=("", "_default"))
        merged["expected_aht"] = merged["expected_aht"].fillna(merged["expected_aht_default"])

        if "expected_aht_default" in merged.columns:
            merged = merged.drop(columns=["expected_aht_default"])
    else:
        merged = df.merge(expected_aht_df, on="workflow_key", how="left")

    merged = merged.dropna(subset=["expected_aht"])

    group_col = agg_level

    if group_col in merged.columns:
        merged = merged.dropna(subset=[group_col])
        merged = merged[merged[group_col] != ""]
    else:
        return pd.DataFrame()

    # --- Merge quality coefficients ---
    if quality_coefficients is not None and not quality_coefficients.empty:
        q_join_col = _resolve_quality_join_column(agg_level)
        qc = quality_coefficients.copy()

        merged = merged.merge(
            qc[["q_merge_key", "workflow_key", "period_sort", "quality_coeff",
                "q_actual", "q_goal", "tasks_audited"]],
            left_on=[q_join_col, "workflow_key", "period_sort"],
            right_on=["q_merge_key", "workflow_key", "period_sort"],
            how="left",
        )
        merged["quality_coeff"] = merged["quality_coeff"].fillna(1.0)
        if "q_merge_key" in merged.columns:
            merged = merged.drop(columns=["q_merge_key"])
    else:
        merged["quality_coeff"] = 1.0

    group_period_totals = (
        merged.groupby([group_col, "period_sort"])["processed_units"]
        .sum()
        .reset_index()
        .rename(columns={"processed_units": "total_units"})
    )

    merged = merged.merge(group_period_totals, on=[group_col, "period_sort"])
    merged["pw_s"] = merged["processed_units"] / merged["total_units"]
    merged["efficiency"] = merged["expected_aht"] / merged["aht_median"]
    merged["rs_contribution"] = merged["efficiency"] * merged["quality_coeff"] * merged["pw_s"]
    merged = merged.rename(columns={group_col: "entity"})

    return merged


# ---------------------------------------------------------------------------
# VISUALIZATION (PLOTLY - INTERACTIVE)
# ---------------------------------------------------------------------------

def plot_performance_ratios(result, title_suffix=""):
    """Plot interactive Rs% over time with hover details."""
    fig = go.Figure()

    entities = sorted(result["entity"].unique())
    colors = px.colors.qualitative.T10
    if len(entities) > len(colors):
        colors = px.colors.qualitative.Alphabet

    has_quality = "avg_quality_coeff" in result.columns and (result["avg_quality_coeff"] != 1.0).any()

    for i, entity in enumerate(entities):
        entity_data = result[result["entity"] == entity].sort_values("period_sort")
        color = colors[i % len(colors)]

        if has_quality:
            hover_template = (
                "<b>%{fullData.name}</b><br>"
                "Period: %{x}<br>"
                "Rs: %{y:.1f}%<br>"
                "Units: %{customdata[0]:,.0f}<br>"
                "Workflows: %{customdata[1]}<br>"
                "Avg Quality Coeff: %{customdata[2]:.3f}"
                "<extra></extra>"
            )
            custom_cols = ["total_units", "num_workflows", "avg_quality_coeff"]
        else:
            hover_template = (
                "<b>%{fullData.name}</b><br>"
                "Period: %{x}<br>"
                "Rs: %{y:.1f}%<br>"
                "Units: %{customdata[0]:,.0f}<br>"
                "Workflows: %{customdata[1]}"
                "<extra></extra>"
            )
            custom_cols = ["total_units", "num_workflows"]

        fig.add_trace(go.Scatter(
            x=entity_data["period_label"],
            y=entity_data["performance_ratio_pct"],
            mode="lines+markers",
            name=entity,
            line=dict(color=color, width=2.5),
            marker=dict(color=color, size=8),
            hovertemplate=hover_template,
            customdata=entity_data[custom_cols].values,
        ))

    # Expected line
    fig.add_hline(
        y=100,
        line_dash="dash",
        line_color="#E53935",
        line_width=2,
        annotation_text="Expected (100%)",
        annotation_position="top left",
        annotation_font_color="#E53935",
    )

    title = "Workflow Efficiency Index — Performance Over Time"
    if title_suffix:
        title += f" ({title_suffix})"

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#333")),
        xaxis_title="Week",
        yaxis_title="Performance Ratio (Rs %)",
        yaxis_ticksuffix="%",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.35,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        template="plotly_white",
        height=550,
        margin=dict(b=120),
    )

    # Use categorical x-axis to keep weeks consecutive
    fig.update_xaxes(
        type='category',
        categoryorder='array',
        categoryarray=sorted(result["period_label"].unique(),
                           key=lambda x: result[result["period_label"]==x]["period_sort"].iloc[0]),
        tickangle=-45
    )

    return fig


def plot_unit_volume(result, title_suffix=""):
    """Plot interactive volume bar chart with hover details."""
    fig = go.Figure()

    entities = sorted(result["entity"].unique())
    colors = px.colors.qualitative.T10
    if len(entities) > len(colors):
        colors = px.colors.qualitative.Alphabet

    for i, entity in enumerate(entities):
        entity_data = result[result["entity"] == entity].sort_values("period_sort")
        color = colors[i % len(colors)]

        fig.add_trace(go.Bar(
            x=entity_data["period_label"],
            y=entity_data["total_units"],
            name=entity,
            marker_color=color,
            opacity=0.8,
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Period: %{x}<br>"
                "Units: %{y:,.0f}<br>"
                "Rs: %{customdata[0]:.1f}%<br>"
                "Workflows: %{customdata[1]}"
                "<extra></extra>"
            ),
            customdata=entity_data[["performance_ratio_pct", "num_workflows"]].values,
        ))

    title = "Weekly Task Volume"
    if title_suffix:
        title += f" ({title_suffix})"

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#333")),
        xaxis_title="Week",
        yaxis_title="Total Units Processed",
        barmode="group",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.35,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        template="plotly_white",
        height=450,
        margin=dict(b=120),
    )

    # Use categorical x-axis to keep weeks consecutive
    fig.update_xaxes(
        type='category',
        categoryorder='array',
        categoryarray=sorted(result["period_label"].unique(),
                           key=lambda x: result[result["period_label"]==x]["period_sort"].iloc[0]),
        tickangle=-45
    )

    return fig


def plot_comparison(result_1, result_2, selected_entities_1, selected_entities_2,
                    agg_level_1_label, agg_level_2_label):
    """Plot interactive cross-level comparison chart."""
    fig = go.Figure()

    colors_1 = px.colors.qualitative.Set1
    colors_2 = px.colors.qualitative.Set2

    # Combine results for period mapping
    all_results = pd.concat([result_1, result_2])

    # Plot Group 1
    for i, entity in enumerate(selected_entities_1):
        entity_data = result_1[result_1["entity"] == entity].sort_values("period_sort")
        if entity_data.empty:
            continue
        color = colors_1[i % len(colors_1)]

        fig.add_trace(go.Scatter(
            x=entity_data["period_label"],
            y=entity_data["performance_ratio_pct"],
            mode="lines+markers",
            name=f"{entity} ({agg_level_1_label})",
            line=dict(color=color, width=2.5),
            marker=dict(color=color, size=8, symbol="circle"),
            legendgroup="group1",
            legendgrouptitle_text=f"Group 1 — {agg_level_1_label}",
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Period: %{x}<br>"
                "Rs: %{y:.1f}%<br>"
                "Units: %{customdata[0]:,.0f}<br>"
                "Workflows: %{customdata[1]}"
                "<extra></extra>"
            ),
            customdata=entity_data[["total_units", "num_workflows"]].values,
        ))

    # Plot Group 2
    for i, entity in enumerate(selected_entities_2):
        entity_data = result_2[result_2["entity"] == entity].sort_values("period_sort")
        if entity_data.empty:
            continue
        color = colors_2[i % len(colors_2)]

        fig.add_trace(go.Scatter(
            x=entity_data["period_label"],
            y=entity_data["performance_ratio_pct"],
            mode="lines+markers",
            name=f"{entity} ({agg_level_2_label})",
            line=dict(color=color, width=2, dash="dash"),
            marker=dict(color=color, size=7, symbol="square"),
            legendgroup="group2",
            legendgrouptitle_text=f"Group 2 — {agg_level_2_label}",
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Period: %{x}<br>"
                "Rs: %{y:.1f}%<br>"
                "Units: %{customdata[0]:,.0f}<br>"
                "Workflows: %{customdata[1]}"
                "<extra></extra>"
            ),
            customdata=entity_data[["total_units", "num_workflows"]].values,
        ))

    # Expected line
    fig.add_hline(
        y=100,
        line_dash="dot",
        line_color="#E53935",
        line_width=2,
        annotation_text="Expected (100%)",
        annotation_position="top left",
        annotation_font_color="#E53935",
    )

    fig.update_layout(
        title=dict(text="Cross-Level Performance Comparison", font=dict(size=16, color="#333")),
        xaxis_title="Week",
        yaxis_title="Performance Ratio (Rs %)",
        yaxis_ticksuffix="%",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.5,
            xanchor="center",
            x=0.5,
            font=dict(size=9),
            groupclick="toggleitem",
            tracegroupgap=30,
        ),
        template="plotly_white",
        height=600,
        margin=dict(b=160),
    )

    # Use categorical x-axis to keep weeks consecutive
    fig.update_xaxes(
        type='category',
        categoryorder='array',
        categoryarray=sorted(all_results["period_label"].unique(),
                           key=lambda x: all_results[all_results["period_label"]==x]["period_sort"].iloc[0]),
        tickangle=-45
    )

    return fig


# ---------------------------------------------------------------------------
# STREAMLIT APP
# ---------------------------------------------------------------------------

def main():
    st.title("📊 Workflow Efficiency Index Tool")
    st.markdown("**Complexity-Adjusted AHT Benchmarking**")

    # Formula explanation
    with st.expander("ℹ️ How is the Performance Ratio (Rs) calculated?", expanded=False):
        st.markdown("""
        ### Formula

        **Rs = Σ (Ew / Aw,s) × Qw,s × Pw,s**

        ### Terms

        - **Rs** = Performance Ratio for entity s (expressed as %)
        - **Ew** = Expected/goal AHT for workflow w (in seconds)
        - **Aw,s** = Actual AHT for entity s on workflow w (in seconds)
        - **Qw,s** = Quality coefficient for entity s on workflow w (default 1.0 if no quality data)
        - **Pw,s** = Proportion of entity s's total output allocated to workflow w
        - **Σ** = Sum across all workflows

        ### Quality Coefficient

        When quality data is uploaded, each workflow+locale pairing gets a quality adjustment:

        **Qw,s = (Q_actual / Q_goal) ^ k**

        - **Q_actual** = Quality % Post Appeals for that workflow+locale+entity+week
        - **Q_goal** = Quality % Goal for that workflow+locale
        - **k** = Sensitivity exponent (configurable; default 3)

        | Quality vs Goal | Coefficient (k=3) | Effect |
        |---|---|---|
        | 4% above goal | ~1.13 | +13% boost |
        | At goal | 1.00 | No change |
        | 4% below goal | ~0.88 | -12% penalty |
        | 8% below goal | ~0.77 | -23% penalty |

        Two audit threshold modes are available:
        - **Mode 1 (≥1 audit):** Coefficient applied whenever any audit exists
        - **Mode 2 (≥6 audits):** Coefficient only applied with 6+ audits (more statistically reliable)

        ### Interpretation

        - **Rs = 100%** → Entity performs at expected speed (and quality)
        - **Rs > 100%** → Entity is faster than expected (and/or higher quality)
        - **Rs < 100%** → Entity is slower than expected (and/or lower quality)

        ### Example

        If a site processes two workflows:
        - Workflow A: Expected 60s, Actual 50s, Quality Coeff 1.05, 70% of volume → (60/50) × 1.05 × 0.7 = 0.882
        - Workflow B: Expected 30s, Actual 40s, Quality Coeff 0.90, 30% of volume → (30/40) × 0.90 × 0.3 = 0.2025
        - **Rs = 0.882 + 0.2025 = 1.0845 = 108.5%**

        ---

        ### Expected AHT Calculation Methods

        **Weighted Mean (Default):**
        - High-volume entities have more influence on the baseline
        - Reflects "typical task experience" across the network
        - Lines often trend above 100% (this is expected)
        - **Use for:** Operational metrics, capacity planning, customer-facing reports

        **Median:**
        - All entities weighted equally regardless of volume
        - Lines center around 100%
        - Easier to explain for peer comparisons
        - **Use for:** Coaching conversations, fairness assessments, peer benchmarking

        💡 **Tip:** If you upload a goals file, the baseline uses those targets instead of network average.
        """)

    # File uploads
    st.markdown("### 📁 Upload Data")
    col_f1, col_f2, col_f3 = st.columns([1, 1, 1])

    with col_f1:
        data_file = st.file_uploader(
            "Upload Data CSV",
            type=["csv"],
            help="Your workflow data export"
        )

    with col_f2:
        expected_file = st.file_uploader(
            "Upload Expected AHT CSV (Optional)",
            type=["csv"],
            help="First 3 columns: Workflow, Locale, Expected AHT"
        )

    with col_f3:
        quality_file = st.file_uploader(
            "Upload Quality CSV (Optional)",
            type=["csv"],
            help="Quality metrics per workflow+locale+site+week"
        )

    if data_file is None:
        st.info("📂 Upload your data CSV to get started")
        with st.expander("📋 Expected CSV Format", expanded=False):
            st.markdown("""
            **Required columns:**
            - `Column-1:Transformation Type`, `Column-2:Locale`, `Column-3:Site`
            - `Select Date Part`, `Average Handle Time(In Secs)`, `Processed Units`

            **Optional (Employee Groups):** `Column-4:Ops Manager`, `Column-5:Team Manager`, `Column-6:DA`

            **Optional (Work Type):** `Column-7:Demand Category`, `Column-8:Customer`

            **Regions auto-assigned:** AMER (BOS, SJO) | EMEA (AMS, CBG, GDN, LHR) | APAJ (AMM, HYD, KIX, MAA)

            **Quality CSV:** First 5 columns (Workflow Name, Customer, SkillType, Locale, Site) in any order,
            followed by fixed columns (Time Period, Tasks audited, Pass Count, Fail Count, etc.)
            """)
        return

    # Load data
    with st.spinner("Loading data..."):
        df_full = load_and_clean_data(data_file)

    st.success(f"✅ Loaded {len(df_full):,} rows | {df_full['period_sort'].nunique()} weeks | {df_full['workflow_key'].nunique()} workflows")

    # Load expected AHT
    if expected_file:
        expected_aht_df = load_expected_aht(expected_file)
        st.success(f"✅ Loaded {len(expected_aht_df)} workflow goals")
    else:
        expected_aht_df = pd.DataFrame(columns=["workflow_key", "expected_aht"])

    # Load quality data
    quality_df = None
    if quality_file:
        with st.spinner("Loading quality data..."):
            quality_df = load_quality_data(quality_file)
        n_wf = quality_df["workflow_key"].nunique() if quality_df is not None else 0
        n_weeks = quality_df["period_sort"].nunique() if quality_df is not None else 0
        st.success(f"✅ Loaded quality data: {len(quality_df):,} rows | {n_wf} workflow+locale pairs | {n_weeks} weeks")

    st.markdown("---")

    # ========================================================================
    # QUALITY SETTINGS (shown only when quality data is uploaded)
    # ========================================================================

    quality_enabled = False
    quality_coefficients_cache = {}  # keyed by (agg_level, min_audits, sensitivity_k)

    if quality_df is not None:
        with st.expander("🎯 Quality Coefficient Settings", expanded=True):
            q_col1, q_col2, q_col3 = st.columns(3)

            with q_col1:
                quality_enabled = st.toggle(
                    "Enable Quality Coefficient",
                    value=True,
                    help="Apply quality adjustment to performance ratios"
                )

            with q_col2:
                audit_mode = st.radio(
                    "Minimum audit threshold:",
                    options=["≥1 audit (any data)", "≥6 audits (reliable)"],
                    key="quality_audit_mode",
                    horizontal=True,
                    help="Mode 1: apply coefficient if any audit exists. Mode 2: require 6+ audits."
                )
                min_audits = 1 if "≥1" in audit_mode else 6

            with q_col3:
                sensitivity_k = st.slider(
                    "Sensitivity exponent (k):",
                    min_value=1.0,
                    max_value=10.0,
                    value=3.0,
                    step=0.5,
                    help="Higher k = more aggressive reward/penalty for quality deviations"
                )

            # Show example impact table
            if quality_enabled:
                st.markdown("**Example impact** (assuming goal = 96%):")
                example_goal = 0.96
                example_actuals = [1.00, 0.98, 0.96, 0.92, 0.88, 0.80]
                example_data = []
                for qa in example_actuals:
                    ratio = qa / example_goal
                    coeff = ratio ** sensitivity_k
                    example_data.append({
                        "Actual Quality": f"{qa:.0%}",
                        "Ratio (Q/Goal)": f"{ratio:.3f}",
                        f"Coefficient (k={sensitivity_k})": f"{coeff:.3f}",
                        "Effect on Rs": f"{(coeff - 1) * 100:+.1f}%",
                    })
                st.dataframe(pd.DataFrame(example_data), use_container_width=True, hide_index=True)

    # ========================================================================
    # MODE SWITCH
    # ========================================================================

    st.markdown("### 🎯 Analysis Mode")

    analysis_mode = st.radio(
        "Select aggregation type:",
        options=["👥 Employee Groups", "📦 Work Type"],
        horizontal=True,
        help="Employee Groups: Region, Site, Ops Manager, Team Manager, DAs | Work Type: Demand Category, Customer, Workflow, Workflow+Locale"
    )

    is_work_type_mode = (analysis_mode == "📦 Work Type")

    st.markdown("---")

    # Build sorted week list once (used by both tabs)
    all_weeks_sorted = sorted(df_full["period_label"].unique(),
                             key=lambda x: df_full[df_full["period_label"]==x]["period_sort"].iloc[0])

    # Helper to get quality coefficients (with caching within session)
    def get_quality_coefficients(agg_level):
        if not quality_enabled or quality_df is None:
            return None
        cache_key = (agg_level, min_audits, sensitivity_k)
        if cache_key not in quality_coefficients_cache:
            quality_coefficients_cache[cache_key] = compute_quality_coefficients(
                quality_df, agg_level,
                sensitivity_k=sensitivity_k,
                min_audits=min_audits,
            )
        return quality_coefficients_cache[cache_key]

    # Tabs for Basic vs Advanced mode
    tab1, tab2 = st.tabs(["📈 Basic Analysis", "🔬 Advanced Comparison"])

    # ========================================================================
    # TAB 1: BASIC ANALYSIS
    # ========================================================================

    with tab1:
        col1, col2 = st.columns([1, 3])

        with col1:
            st.markdown("### ⚙️ Settings")

            # Aggregation level (based on mode)
            if is_work_type_mode:
                available_levels = {label: col for label, col in WORK_TYPE_AGGREGATION_LEVELS.items()
                                  if col == "network" or (col in df_full.columns and df_full[col].notna().any())}
            else:
                available_levels = {label: col for label, col in EMPLOYEE_AGGREGATION_LEVELS.items()
                                  if col == "network" or (col in df_full.columns and df_full[col].notna().any())}

            agg_level_label = st.selectbox(
                "Group by:",
                options=list(available_levels.keys()),
                key="basic_agg_level"
            )
            agg_level = available_levels[agg_level_label]

            # Entity filter (skip for Network level)
            if agg_level == "network":
                selected_entities = ["Network"]
                st.info("🌐 Viewing entire network performance")
            else:
                all_entities = sorted([str(x) for x in df_full[agg_level].dropna().unique()
                                     if str(x) != "nan" and str(x) != ""])

                # Validate session state against current options
                if f"basic_selected_{agg_level}" in st.session_state:
                    valid_defaults = [e for e in st.session_state[f"basic_selected_{agg_level}"] if e in all_entities]
                    if not valid_defaults:
                        valid_defaults = all_entities
                    st.session_state[f"basic_selected_{agg_level}"] = valid_defaults
                else:
                    st.session_state[f"basic_selected_{agg_level}"] = all_entities

                selected_entities = st.multiselect(
                    f"Select {agg_level_label}:",
                    options=all_entities,
                    default=st.session_state[f"basic_selected_{agg_level}"],
                    key=f"basic_entity_select"
                )

            # Baseline settings
            st.markdown("---")

            # For Network level, baseline scope doesn't make sense
            if agg_level == "network":
                use_full_dataset = True
                st.caption("ℹ️ Network level always uses full dataset")
            else:
                baseline_mode = st.radio(
                    "Baseline scope:",
                    options=["Network", "Relative"],
                    help="Network: compare to all entities. Relative: compare to selected only.",
                    key="basic_baseline"
                )
                use_full_dataset = (baseline_mode == "Network")

            baseline_calc = st.radio(
                "Baseline calculation:",
                options=["Weighted Mean", "Median"],
                help="Weighted Mean: volume matters. Median: all entities equal.",
                key="basic_calc_method"
            )
            use_median = (baseline_calc == "Median")

            st.markdown("---")
            st.markdown("**Week Filter**")

            selected_weeks = week_range_selector(all_weeks_sorted, key_prefix="basic")

            st.markdown("---")

            # Filter section depends on mode
            if is_work_type_mode:
                st.markdown("**Site Filter**")

                all_sites = sorted(df_full["site"].unique())

                site_filter_mode = st.radio(
                    "Sites:",
                    options=["All", "Specific"],
                    key="basic_site_mode",
                    horizontal=True
                )

                if site_filter_mode == "Specific":
                    # Prioritise sites that have the selected entities
                    if agg_level != "network":
                        prioritised_sites = get_prioritised_sites(
                            df_full, agg_level, selected_entities, all_sites
                        )

                        entity_site_set = set(
                            df_full[df_full[agg_level].isin(selected_entities)]["site"].unique()
                        ) if selected_entities else set()
                        n_entity = len([s for s in prioritised_sites if s in entity_site_set])

                        if selected_entities and n_entity < len(prioritised_sites):
                            st.caption(
                                f"ℹ️ First **{n_entity}** sites have the selected "
                                f"{agg_level_label.lower()}(s); remaining "
                                f"**{len(prioritised_sites) - n_entity}** are the rest."
                            )
                    else:
                        prioritised_sites = all_sites

                    if "basic_selected_sites" not in st.session_state:
                        st.session_state.basic_selected_sites = prioritised_sites

                    selected_sites = st.multiselect(
                        "Select sites:",
                        options=prioritised_sites,
                        default=st.session_state.basic_selected_sites,
                        key="basic_site_select"
                    )

                    if selected_sites:
                        st.session_state.basic_selected_sites = selected_sites
                else:
                    selected_sites = None
            else:
                st.markdown("**Workflow Filter**")

                all_workflows = sorted(df_full["workflow_key"].unique())

                workflow_filter_mode = st.radio(
                    "Workflows:",
                    options=["All", "Specific"],
                    key="basic_workflow_mode",
                    horizontal=True
                )

                if workflow_filter_mode == "Specific":
                    # Prioritise workflows done by the selected entities
                    if agg_level != "network":
                        prioritised_workflows = get_prioritised_workflows(
                            df_full, agg_level, selected_entities, all_workflows
                        )

                        entity_wf_set = set(
                            df_full[df_full[agg_level].isin(selected_entities)]["workflow_key"].unique()
                        ) if selected_entities else set()
                        n_entity = len([w for w in prioritised_workflows if w in entity_wf_set])

                        if selected_entities and n_entity < len(prioritised_workflows):
                            st.caption(
                                f"ℹ️ First **{n_entity}** workflows are ones done by the "
                                f"selected {agg_level_label.lower()}(s); remaining "
                                f"**{len(prioritised_workflows) - n_entity}** are the rest."
                            )
                    else:
                        prioritised_workflows = all_workflows

                    if "basic_selected_workflows" not in st.session_state:
                        st.session_state.basic_selected_workflows = prioritised_workflows

                    selected_workflows = st.multiselect(
                        "Select workflows:",
                        options=prioritised_workflows,
                        default=st.session_state.basic_selected_workflows[:10] if len(st.session_state.basic_selected_workflows) > 10 else st.session_state.basic_selected_workflows,
                        key="basic_workflow_select"
                    )

                    if selected_workflows:
                        st.session_state.basic_selected_workflows = selected_workflows
                else:
                    selected_workflows = None

            if agg_level != "network":
                if st.button("Apply Filters", key="basic_apply", use_container_width=True, type="primary"):
                    st.session_state[f"basic_selected_{agg_level}"] = selected_entities
                    st.rerun()

        with col2:
            if agg_level != "network" and not selected_entities:
                st.warning("⚠️ Select at least one entity")
                return

            # Filter by entities
            if agg_level == "network":
                df_filtered = df_full.copy()
            else:
                df_filtered = df_full[df_full[agg_level].isin(selected_entities)].copy()

            # Filter by weeks (for display only, not for baseline calculation)
            if selected_weeks is not None:
                df_display = df_filtered[df_filtered["period_label"].isin(selected_weeks)].copy()
                st.info(f"📅 Filtered to {len(selected_weeks)} week{'s' if len(selected_weeks) != 1 else ''}: **{selected_weeks[0]}** → **{selected_weeks[-1]}**")
            else:
                df_display = df_filtered.copy()

            # Filter by workflows or sites depending on mode
            if is_work_type_mode:
                if site_filter_mode == "Specific" and selected_sites:
                    df_filtered = df_filtered[df_filtered["site"].isin(selected_sites)].copy()
                    df_display = df_display[df_display["site"].isin(selected_sites)].copy()
                    st.info(f"🏢 Filtered to {len(selected_sites)} sites")
            else:
                if workflow_filter_mode == "Specific" and selected_workflows:
                    df_filtered = df_filtered[df_filtered["workflow_key"].isin(selected_workflows)].copy()
                    df_display = df_display[df_display["workflow_key"].isin(selected_workflows)].copy()
                    st.info(f"🧩 Filtered to {len(selected_workflows)} workflows")

            # Baseline always uses full dataset (or selected entities if Relative mode)
            # but NEVER filtered by weeks
            baseline_df = df_full if use_full_dataset else df_filtered

            use_hybrid = True

            # Get quality coefficients for this aggregation level
            qc = get_quality_coefficients(agg_level)

            # Calculate using display data, but baseline from full data
            result = calculate_performance_ratios(
                df_display, expected_aht_df, agg_level, use_hybrid, baseline_df, use_median,
                quality_coefficients=qc,
            )
            detail = calculate_workflow_detail(
                df_display, expected_aht_df, agg_level, use_hybrid, baseline_df, use_median,
                quality_coefficients=qc,
            )

            if result.empty:
                st.error("No data to display")
                return

            # Metrics
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                if agg_level == "network":
                    st.metric("Level", "Network")
                else:
                    st.metric("Entities", result["entity"].nunique())
            with col_b:
                st.metric("Weeks", result["period_sort"].nunique())
            with col_c:
                st.metric("Avg Rs%", f"{result['performance_ratio_pct'].mean():.1f}%")
            with col_d:
                st.metric("Total Units", f"{result['total_units'].sum():,.0f}")

            # Quality indicator
            if quality_enabled and qc is not None and not qc.empty:
                avg_coeff = result["avg_quality_coeff"].mean()
                st.caption(f"🎯 Quality coefficient active | Avg coefficient: **{avg_coeff:.3f}** | "
                          f"Sensitivity: k={sensitivity_k} | Min audits: {min_audits}")

            # Interactive Charts
            st.markdown("#### Performance Ratio Over Time")
            fig1 = plot_performance_ratios(result)
            st.plotly_chart(fig1, use_container_width=True)

            st.markdown("#### Task Volume")
            fig2 = plot_unit_volume(result)
            st.plotly_chart(fig2, use_container_width=True)

            # Latest period table
            latest_period = result["period_sort"].max()
            latest_label = result[result["period_sort"] == latest_period]["period_label"].iloc[0]
            latest_data = result[result["period_sort"] == latest_period].sort_values("performance_ratio_pct", ascending=False)

            st.markdown(f"#### Latest Period Results ({latest_label})")

            display_cols = ["entity", "performance_ratio_pct", "total_units", "num_workflows"]
            rename_cols = {
                "entity": agg_level_label,
                "performance_ratio_pct": "Rs %",
                "total_units": "Units",
                "num_workflows": "Workflows"
            }
            if quality_enabled and "avg_quality_coeff" in latest_data.columns:
                display_cols.append("avg_quality_coeff")
                rename_cols["avg_quality_coeff"] = "Avg Quality Coeff"

            st.dataframe(
                latest_data[display_cols].rename(columns=rename_cols),
                use_container_width=True
            )

            # Download
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                csv1 = result.to_csv(index=False)
                st.download_button(
                    "💾 Performance Ratios CSV",
                    csv1,
                    "performance_ratios.csv",
                    "text/csv",
                    use_container_width=True
                )
            with col_dl2:
                csv2 = detail.to_csv(index=False)
                st.download_button(
                    "💾 Workflow Detail CSV",
                    csv2,
                    "workflow_detail.csv",
                    "text/csv",
                    use_container_width=True
                )

    # ========================================================================
    # TAB 2: ADVANCED COMPARISON
    # ========================================================================

    with tab2:
        st.markdown("### 🔬 Cross-Level Comparison")
        st.info("Compare entities from different aggregation levels (e.g., EMEA region vs. individual sites)")

        col1, col2 = st.columns(2)

        # ---- Comparison Group 1 ----
        with col1:
            st.markdown("#### 🔵 Group 1")

            if is_work_type_mode:
                available_levels_1 = {label: col for label, col in WORK_TYPE_AGGREGATION_LEVELS.items()
                                    if col == "network" or (col in df_full.columns and df_full[col].notna().any())}
            else:
                available_levels_1 = {label: col for label, col in EMPLOYEE_AGGREGATION_LEVELS.items()
                                    if col == "network" or (col in df_full.columns and df_full[col].notna().any())}

            agg_level_1_label = st.selectbox(
                "Aggregation level:",
                options=list(available_levels_1.keys()),
                key="adv_agg_1"
            )
            agg_level_1 = available_levels_1[agg_level_1_label]

            if agg_level_1 == "network":
                selected_entities_1 = ["Network"]
                st.info("🌐 Network level selected")
            else:
                all_entities_1 = sorted([str(x) for x in df_full[agg_level_1].dropna().unique()
                                       if str(x) != "nan" and str(x) != ""])

                selected_entities_1 = st.multiselect(
                    f"Select {agg_level_1_label}:",
                    options=all_entities_1,
                    default=all_entities_1[:3] if len(all_entities_1) >= 3 else all_entities_1,
                    key="adv_entities_1"
                )

        # ---- Comparison Group 2 ----
        with col2:
            st.markdown("#### 🟢 Group 2")

            if is_work_type_mode:
                available_levels_2 = {label: col for label, col in WORK_TYPE_AGGREGATION_LEVELS.items()
                                    if col == "network" or (col in df_full.columns and df_full[col].notna().any())}
            else:
                available_levels_2 = {label: col for label, col in EMPLOYEE_AGGREGATION_LEVELS.items()
                                    if col == "network" or (col in df_full.columns and df_full[col].notna().any())}

            agg_level_2_label = st.selectbox(
                "Aggregation level:",
                options=list(available_levels_2.keys()),
                key="adv_agg_2"
            )
            agg_level_2 = available_levels_2[agg_level_2_label]

            if agg_level_2 == "network":
                selected_entities_2 = ["Network"]
                st.info("🌐 Network level selected")
            else:
                all_entities_2 = sorted([str(x) for x in df_full[agg_level_2].dropna().unique()
                                       if str(x) != "nan" and str(x) != ""])

                selected_entities_2 = st.multiselect(
                    f"Select {agg_level_2_label}:",
                    options=all_entities_2,
                    default=all_entities_2[:3] if len(all_entities_2) >= 3 else all_entities_2,
                    key="adv_entities_2"
                )

        # Shared settings
        st.markdown("---")
        col_set1, col_set2, col_set3, col_set4 = st.columns(4)

        with col_set1:
            baseline_mode_adv = st.radio(
                "Baseline scope:",
                options=["Network", "Relative"],
                key="adv_baseline"
            )
            use_full_dataset_adv = (baseline_mode_adv == "Network")

        with col_set2:
            baseline_calc_adv = st.radio(
                "Baseline calculation:",
                options=["Weighted Mean", "Median"],
                key="adv_calc_method"
            )
            use_median_adv = (baseline_calc_adv == "Median")

        with col_set3:
            st.markdown("**Week Filter**")
            selected_weeks_adv = week_range_selector(all_weeks_sorted, key_prefix="adv")

        with col_set4:
            if is_work_type_mode:
                st.markdown("**Site Filter**")

                all_sites_adv = sorted(df_full["site"].unique())

                site_filter_mode_adv = st.radio(
                    "Sites:",
                    options=["All", "Specific"],
                    key="adv_site_mode",
                    horizontal=True
                )
            else:
                st.markdown("**Workflow Filter**")

                all_workflows_adv = sorted(df_full["workflow_key"].unique())

                workflow_filter_mode_adv = st.radio(
                    "Workflows:",
                    options=["All", "Specific"],
                    key="adv_workflow_mode",
                    horizontal=True
                )

        # Filter selection for advanced (if specific is chosen)
        if is_work_type_mode:
            if site_filter_mode_adv == "Specific":
                # Prioritise sites that have Group 1's selected entities
                if agg_level_1 != "network":
                    prioritised_sites_adv = get_prioritised_sites(
                        df_full, agg_level_1, selected_entities_1, all_sites_adv
                    )

                    entity_site_set_adv = set(
                        df_full[df_full[agg_level_1].isin(selected_entities_1)]["site"].unique()
                    ) if selected_entities_1 else set()
                    n_entity_adv = len([s for s in prioritised_sites_adv if s in entity_site_set_adv])

                    if selected_entities_1 and n_entity_adv < len(prioritised_sites_adv):
                        st.caption(
                            f"ℹ️ First **{n_entity_adv}** sites have "
                            f"Group 1's selected {agg_level_1_label.lower()}(s); remaining "
                            f"**{len(prioritised_sites_adv) - n_entity_adv}** are the rest."
                        )
                else:
                    prioritised_sites_adv = all_sites_adv

                if "adv_selected_sites" not in st.session_state:
                    st.session_state.adv_selected_sites = prioritised_sites_adv

                selected_sites_adv = st.multiselect(
                    "Select sites:",
                    options=prioritised_sites_adv,
                    default=st.session_state.adv_selected_sites,
                    key="adv_site_select"
                )

                if selected_sites_adv:
                    st.session_state.adv_selected_sites = selected_sites_adv
            else:
                selected_sites_adv = None
        else:
            if workflow_filter_mode_adv == "Specific":
                # Prioritise workflows done by Group 1's selected entities
                if agg_level_1 != "network":
                    prioritised_workflows_adv = get_prioritised_workflows(
                        df_full, agg_level_1, selected_entities_1, all_workflows_adv
                    )

                    entity_wf_set_adv = set(
                        df_full[df_full[agg_level_1].isin(selected_entities_1)]["workflow_key"].unique()
                    ) if selected_entities_1 else set()
                    n_entity_adv = len([w for w in prioritised_workflows_adv if w in entity_wf_set_adv])

                    if selected_entities_1 and n_entity_adv < len(prioritised_workflows_adv):
                        st.caption(
                            f"ℹ️ First **{n_entity_adv}** workflows are ones done by "
                            f"Group 1's selected {agg_level_1_label.lower()}(s); remaining "
                            f"**{len(prioritised_workflows_adv) - n_entity_adv}** are the rest."
                        )
                else:
                    prioritised_workflows_adv = all_workflows_adv

                if "adv_selected_workflows" not in st.session_state:
                    st.session_state.adv_selected_workflows = prioritised_workflows_adv

                selected_workflows_adv = st.multiselect(
                    "Select workflows:",
                    options=prioritised_workflows_adv,
                    default=st.session_state.adv_selected_workflows[:10] if len(st.session_state.adv_selected_workflows) > 10 else st.session_state.adv_selected_workflows,
                    key="adv_workflow_select"
                )

                if selected_workflows_adv:
                    st.session_state.adv_selected_workflows = selected_workflows_adv
            else:
                selected_workflows_adv = None

        st.markdown("---")

        if st.button("🔍 Calculate Comparison", use_container_width=True, type="primary"):
            st.session_state.adv_calculate = True

        # Calculate and display
        if st.session_state.get("adv_calculate", False):
            if not selected_entities_1 or not selected_entities_2:
                st.warning("⚠️ Select entities in both groups")
                return

            # Get quality coefficients for each group's aggregation level
            qc_1 = get_quality_coefficients(agg_level_1)
            qc_2 = get_quality_coefficients(agg_level_2)

            # Calculate for Group 1
            if agg_level_1 == "network":
                df_filtered_1 = df_full.copy()
            else:
                df_filtered_1 = df_full[df_full[agg_level_1].isin(selected_entities_1)].copy()

            # Apply week filter for display
            if selected_weeks_adv is not None:
                df_display_1 = df_filtered_1[df_filtered_1["period_label"].isin(selected_weeks_adv)].copy()
            else:
                df_display_1 = df_filtered_1.copy()

            # Apply workflow or site filter
            if is_work_type_mode:
                if site_filter_mode_adv == "Specific" and selected_sites_adv:
                    df_filtered_1 = df_filtered_1[df_filtered_1["site"].isin(selected_sites_adv)].copy()
                    df_display_1 = df_display_1[df_display_1["site"].isin(selected_sites_adv)].copy()
            else:
                if workflow_filter_mode_adv == "Specific" and selected_workflows_adv:
                    df_filtered_1 = df_filtered_1[df_filtered_1["workflow_key"].isin(selected_workflows_adv)].copy()
                    df_display_1 = df_display_1[df_display_1["workflow_key"].isin(selected_workflows_adv)].copy()

            baseline_df_1 = df_full if use_full_dataset_adv else df_filtered_1
            result_1 = calculate_performance_ratios(
                df_display_1, expected_aht_df, agg_level_1, True, baseline_df_1, use_median_adv,
                quality_coefficients=qc_1,
            )

            # Calculate for Group 2
            if agg_level_2 == "network":
                df_filtered_2 = df_full.copy()
            else:
                df_filtered_2 = df_full[df_full[agg_level_2].isin(selected_entities_2)].copy()

            # Apply week filter for display
            if selected_weeks_adv is not None:
                df_display_2 = df_filtered_2[df_filtered_2["period_label"].isin(selected_weeks_adv)].copy()
            else:
                df_display_2 = df_filtered_2.copy()

            # Apply workflow or site filter
            if is_work_type_mode:
                if site_filter_mode_adv == "Specific" and selected_sites_adv:
                    df_filtered_2 = df_filtered_2[df_filtered_2["site"].isin(selected_sites_adv)].copy()
                    df_display_2 = df_display_2[df_display_2["site"].isin(selected_sites_adv)].copy()
            else:
                if workflow_filter_mode_adv == "Specific" and selected_workflows_adv:
                    df_filtered_2 = df_filtered_2[df_filtered_2["workflow_key"].isin(selected_workflows_adv)].copy()
                    df_display_2 = df_display_2[df_display_2["workflow_key"].isin(selected_workflows_adv)].copy()

            baseline_df_2 = df_full if use_full_dataset_adv else df_filtered_2
            result_2 = calculate_performance_ratios(
                df_display_2, expected_aht_df, agg_level_2, True, baseline_df_2, use_median_adv,
                quality_coefficients=qc_2,
            )

            if result_1.empty or result_2.empty:
                st.error("No data to display for one or both groups")
                return

            # Show filter info
            filter_info = []
            if selected_weeks_adv is not None:
                filter_info.append(f"{len(selected_weeks_adv)} weeks ({selected_weeks_adv[0]} → {selected_weeks_adv[-1]})")
            if is_work_type_mode:
                if site_filter_mode_adv == "Specific" and selected_sites_adv:
                    filter_info.append(f"{len(selected_sites_adv)} sites")
            else:
                if workflow_filter_mode_adv == "Specific" and selected_workflows_adv:
                    filter_info.append(f"{len(selected_workflows_adv)} workflows")

            if filter_info:
                st.info(f"📅 Filtered to {' and '.join(filter_info)}")

            if quality_enabled:
                st.caption(f"🎯 Quality coefficient active | Sensitivity: k={sensitivity_k} | Min audits: {min_audits}")

            # Display
            st.markdown("### 📊 Comparison Results")

            # Side-by-side metrics
            col_m1, col_m2 = st.columns(2)

            with col_m1:
                st.markdown(f"**Group 1: {agg_level_1_label}**")
                col_m1a, col_m1b = st.columns(2)
                with col_m1a:
                    st.metric("Avg Rs%", f"{result_1['performance_ratio_pct'].mean():.1f}%")
                with col_m1b:
                    st.metric("Total Units", f"{result_1['total_units'].sum():,.0f}")

            with col_m2:
                st.markdown(f"**Group 2: {agg_level_2_label}**")
                col_m2a, col_m2b = st.columns(2)
                with col_m2a:
                    st.metric("Avg Rs%", f"{result_2['performance_ratio_pct'].mean():.1f}%")
                with col_m2b:
                    st.metric("Total Units", f"{result_2['total_units'].sum():,.0f}")

            # Interactive comparison chart
            st.markdown("#### Performance Comparison")
            fig_combined = plot_comparison(
                result_1, result_2,
                selected_entities_1, selected_entities_2,
                agg_level_1_label, agg_level_2_label
            )
            st.plotly_chart(fig_combined, use_container_width=True)

            # Combined table
            st.markdown("#### Comparison Summary (Latest Period)")

            result_1_display = result_1.copy()
            result_1_display["group"] = f"Group 1 ({agg_level_1_label})"

            result_2_display = result_2.copy()
            result_2_display["group"] = f"Group 2 ({agg_level_2_label})"

            result_combined = pd.concat([result_1_display, result_2_display], ignore_index=True)

            latest_period = result_combined["period_sort"].max()
            latest_combined = result_combined[result_combined["period_sort"] == latest_period].sort_values("performance_ratio_pct", ascending=False)

            display_cols_adv = ["group", "entity", "performance_ratio_pct", "total_units", "num_workflows"]
            rename_cols_adv = {
                "group": "Group",
                "entity": "Entity",
                "performance_ratio_pct": "Rs %",
                "total_units": "Units",
                "num_workflows": "Workflows"
            }
            if quality_enabled and "avg_quality_coeff" in latest_combined.columns:
                display_cols_adv.append("avg_quality_coeff")
                rename_cols_adv["avg_quality_coeff"] = "Avg Quality Coeff"

            st.dataframe(
                latest_combined[display_cols_adv].rename(columns=rename_cols_adv),
                use_container_width=True
            )

            csv_combined = result_combined.to_csv(index=False)
            st.download_button(
                "💾 Download Comparison Results",
                csv_combined,
                "cross_level_comparison.csv",
                "text/csv",
                use_container_width=True
            )


if __name__ == "__main__":
    main()
