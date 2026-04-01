#!/usr/bin/env python3
"""
Workflow Efficiency Index Tool - Web UI
Interactive dashboard with multi-level aggregation and hybrid goal setting.
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

def main():
    st.title("🎯 Workflow Efficiency Index Tool")

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

EMPLOYEE_AGGREGATION_LEVELS = {
    "Network": "network",
    "Region": "region",
    "Site": "site",
    "Ops Manager": "ops_manager",
    "Team Manager": "team_manager",
    "Data Associate": "da_name",
}

EMPLOYEE_HIERARCHY_CHAIN = ["network", "region", "site", "ops_manager", "team_manager", "da_name"]
EMPLOYEE_HIERARCHY_LABELS = {
    "network": "Network",
    "region": "Region",
    "site": "Site",
    "ops_manager": "Ops Manager",
    "team_manager": "Team Manager",
    "da_name": "Data Associate",
}

WORK_TYPE_AGGREGATION_LEVELS = {
    "Network": "network",
    "Demand Category": "demand_category",
    "Customer": "customer",
    "Workflow": "workflow",
    "Workflow+Locale": "workflow_key",
}

WORK_TYPE_HIERARCHY_CHAIN = ["network", "demand_category", "customer", "workflow", "workflow_key"]
WORK_TYPE_HIERARCHY_LABELS = {
    "network": "Network",
    "demand_category": "Demand Category",
    "customer": "Customer",
    "workflow": "Workflow",
    "workflow_key": "Workflow+Locale",
}

# ---------------------------------------------------------------------------
# QUALITY COLUMN DETECTION
# ---------------------------------------------------------------------------

QUALITY_CANONICAL = {
    "workflow_name": None,
    "customer": None,
    "skill_type": None,
    "locale": None,
    "site": None,
}

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
# HIERARCHICAL DRILL-DOWN HELPERS
# ---------------------------------------------------------------------------

def get_hierarchy_chain(is_work_type_mode):
    if is_work_type_mode:
        return WORK_TYPE_HIERARCHY_CHAIN, WORK_TYPE_HIERARCHY_LABELS
    else:
        return EMPLOYEE_HIERARCHY_CHAIN, EMPLOYEE_HIERARCHY_LABELS


def get_parent_levels(agg_level, is_work_type_mode):
    chain, _ = get_hierarchy_chain(is_work_type_mode)
    if agg_level not in chain:
        return []
    idx = chain.index(agg_level)
    return [level for level in chain[1:idx] if level != agg_level]


def get_children_for_parent(df, parent_level, parent_values, child_level):
    if parent_level == "network" or not parent_values:
        children = df[child_level].dropna().unique()
    else:
        mask = df[parent_level].isin(parent_values)
        children = df.loc[mask, child_level].dropna().unique()
    
    return sorted([str(c) for c in children if str(c) != "nan" and str(c) != ""])


def render_hierarchy_selector(df, agg_level, is_work_type_mode, key_prefix="basic"):
    chain, labels = get_hierarchy_chain(is_work_type_mode)
    parent_levels = get_parent_levels(agg_level, is_work_type_mode)
    
    available_parents = [
        p for p in parent_levels
        if p in df.columns and df[p].notna().any()
    ]
    
    mask = pd.Series(True, index=df.index)
    parent_selections = {}
    
    if available_parents:
        st.markdown("#### 🏗️ Hierarchy Filter")
        
        for parent_level in available_parents:
            parent_label = labels.get(parent_level, parent_level)
            
            available_values = sorted([
                str(x) for x in df.loc[mask, parent_level].dropna().unique()
                if str(x) != "nan" and str(x) != ""
            ])
            
            if not available_values:
                continue
            
            ss_key = f"{key_prefix}_hier_{parent_level}"
            col_toggle, col_select = st.columns([1, 3])
            
            with col_toggle:
                use_all = st.checkbox(
                    f"All {parent_label}s",
                    value=True,
                    key=f"{ss_key}_all"
                )
            
            with col_select:
                if use_all:
                    selected_parents = available_values
                    st.caption(f"✓ All {len(available_values)} {parent_label}(s)")
                else:
                    selected_parents = st.multiselect(
                        f"Select {parent_label}:",
                        options=available_values,
                        default=available_values[:3] if len(available_values) > 3 else available_values,
                        key=f"{ss_key}_select",
                        label_visibility="collapsed"
                    )
            
            parent_selections[parent_level] = selected_parents
            
            if selected_parents and not use_all:
                mask = mask & df[parent_level].isin(selected_parents)
        
        st.markdown("---")
    
    all_entities = sorted([
        str(x) for x in df.loc[mask, agg_level].dropna().unique()
        if str(x) != "nan" and str(x) != ""
    ])
    
    total_entities = sorted([
        str(x) for x in df[agg_level].dropna().unique()
        if str(x) != "nan" and str(x) != ""
    ])
    
    if len(all_entities) < len(total_entities):
        st.caption(f"🔍 Showing **{len(all_entities)}** of {len(total_entities)} {labels.get(agg_level, agg_level)}(s) based on hierarchy filter")
    
    col_all, col_none = st.columns(2)
    with col_all:
        if st.button("✅ Select All", key=f"{key_prefix}_select_all", use_container_width=True):
            st.session_state[f"{key_prefix}_entity_selection"] = all_entities
    with col_none:
        if st.button("❌ Clear All", key=f"{key_prefix}_clear_all", use_container_width=True):
            st.session_state[f"{key_prefix}_entity_selection"] = []
    
    prev_selection_key = f"{key_prefix}_entity_selection"
    if prev_selection_key in st.session_state:
        default_selection = [e for e in st.session_state[prev_selection_key] if e in all_entities]
        if not default_selection:
            default_selection = all_entities
    else:
        default_selection = all_entities
    
    selected_entities = st.multiselect(
        f"Select {labels.get(agg_level, agg_level)}:",
        options=all_entities,
        default=default_selection,
        key=f"{key_prefix}_entity_multiselect"
    )
    
    st.session_state[prev_selection_key] = selected_entities
    
    if parent_selections and selected_entities:
        with st.expander("📊 Selection Summary", expanded=False):
            for parent_level, parent_values in parent_selections.items():
                parent_label = labels.get(parent_level, parent_level)
                if len(parent_values) <= 5:
                    st.caption(f"**{parent_label}:** {', '.join(parent_values)}")
                else:
                    st.caption(f"**{parent_label}:** {len(parent_values)} selected")
            
            st.caption(f"**{labels.get(agg_level, agg_level)}:** {len(selected_entities)} selected")
            
            if available_parents and selected_entities:
                st.markdown("**Hierarchy Tree:**")
                tree_df = df.loc[
                    mask & df[agg_level].isin(selected_entities),
                    [p for p in available_parents if p in df.columns] + [agg_level]
                ].drop_duplicates()
                
                if len(available_parents) >= 1 and len(tree_df) <= 100:
                    tree_display = []
                    first_parent = available_parents[0]
                    for parent_val in sorted(tree_df[first_parent].dropna().unique()):
                        tree_display.append(f"📁 **{parent_val}**")
                        
                        if len(available_parents) >= 2:
                            second_parent = available_parents[1]
                            sub_df = tree_df[tree_df[first_parent] == parent_val]
                            for sub_val in sorted(sub_df[second_parent].dropna().unique()):
                                tree_display.append(f"&nbsp;&nbsp;&nbsp;&nbsp;📂 {sub_val}")
                                
                                child_df = sub_df[sub_df[second_parent] == sub_val]
                                for child_val in sorted(child_df[agg_level].dropna().unique()):
                                    if str(child_val) in selected_entities:
                                        tree_display.append(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;👤 {child_val}")
                        else:
                            sub_df = tree_df[tree_df[first_parent] == parent_val]
                            for child_val in sorted(sub_df[agg_level].dropna().unique()):
                                if str(child_val) in selected_entities:
                                    tree_display.append(f"&nbsp;&nbsp;&nbsp;&nbsp;👤 {child_val}")
                    
                    st.markdown("<br>".join(tree_display), unsafe_allow_html=True)
    
    return selected_entities


def render_hierarchy_selector_advanced(df, agg_level, is_work_type_mode, key_prefix="adv"):
    chain, labels = get_hierarchy_chain(is_work_type_mode)
    parent_levels = get_parent_levels(agg_level, is_work_type_mode)
    
    available_parents = [
        p for p in parent_levels
        if p in df.columns and df[p].notna().any()
    ]
    
    mask = pd.Series(True, index=df.index)
    
    if available_parents:
        st.caption("🏗️ **Hierarchy filter:**")
        
        for parent_level in available_parents:
            parent_label = labels.get(parent_level, parent_level)
            
            available_values = sorted([
                str(x) for x in df.loc[mask, parent_level].dropna().unique()
                if str(x) != "nan" and str(x) != ""
            ])
            
            if not available_values:
                continue
            
            use_all = st.checkbox(
                f"All {parent_label}s",
                value=True,
                key=f"{key_prefix}_hier_{parent_level}_all"
            )
            
            if not use_all:
                selected_parents = st.multiselect(
                    f"{parent_label}:",
                    options=available_values,
                    default=available_values[:3] if len(available_values) > 3 else available_values,
                    key=f"{key_prefix}_hier_{parent_level}_select"
                )
                if selected_parents:
                    mask = mask & df[parent_level].isin(selected_parents)
    
    all_entities = sorted([
        str(x) for x in df.loc[mask, agg_level].dropna().unique()
        if str(x) != "nan" and str(x) != ""
    ])
    
    selected_entities = st.multiselect(
        f"Select {labels.get(agg_level, agg_level)}:",
        options=all_entities,
        default=all_entities[:3] if len(all_entities) >= 3 else all_entities,
        key=f"{key_prefix}_entities"
    )
    
    return selected_entities


# ---------------------------------------------------------------------------
# DATA VALIDATION FUNCTIONS
# ---------------------------------------------------------------------------

def validate_data_quality(df):
    issues = []
    stats = {}
    
    required_cols = ['workflow', 'locale', 'site', 'aht_median', 'processed_units', 'date_part']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        issues.append(("error", f"❌ Missing required columns: {', '.join(missing)}"))
        return issues, stats
    
    if (df['aht_median'] <= 0).any():
        count = (df['aht_median'] <= 0).sum()
        issues.append(("warning", f"⚠️ {count:,} rows with invalid AHT (≤0) → these will be filtered"))
    
    if (df['processed_units'] <= 0).any():
        count = (df['processed_units'] <= 0).sum()
        issues.append(("warning", f"⚠️ {count:,} rows with invalid volume (≤0) → these will be filtered"))
    
    weeks = df['period_sort'].nunique() if 'period_sort' in df.columns else 0
    stats['weeks'] = weeks
    if weeks < 4:
        issues.append(("warning", f"⚠️ Limited time coverage: only {weeks} week(s) of data"))
    elif weeks >= 12:
        issues.append(("success", f"✅ Good time coverage: {weeks} weeks of data"))
    
    q99 = df['aht_median'].quantile(0.99)
    q01 = df['aht_median'].quantile(0.01)
    outliers_high = (df['aht_median'] > q99 * 3).sum()
    outliers_low = (df['aht_median'] < q01 / 3).sum()
    
    if outliers_high > 0:
        issues.append(("info", f"ℹ️ {outliers_high:,} rows with very high AHT (>3x P99: {q99:.0f}s)"))
    if outliers_low > 0:
        issues.append(("info", f"ℹ️ {outliers_low:,} rows with very low AHT (<P01/3: {q01:.0f}s)"))
    
    if 'site' in df.columns:
        site_weeks = df.groupby('site')['period_sort'].nunique()
        incomplete_sites = site_weeks[site_weeks < weeks * 0.5].index.tolist()
        if incomplete_sites:
            issues.append(("warning", f"⚠️ {len(incomplete_sites)} site(s) with <50% week coverage: {', '.join(incomplete_sites[:3])}{'...' if len(incomplete_sites) > 3 else ''}"))
    
    workflow_counts = df['workflow_key'].value_counts() if 'workflow_key' in df.columns else pd.Series()
    stats['workflows'] = len(workflow_counts)
    stats['avg_records_per_workflow'] = workflow_counts.mean() if len(workflow_counts) > 0 else 0
    
    low_volume_workflows = (workflow_counts < 10).sum()
    if low_volume_workflows > 0:
        issues.append(("info", f"ℹ️ {low_volume_workflows} workflow(s) with <10 records"))
    
    optional_cols = ['ops_manager', 'team_manager', 'da_name', 'demand_category', 'customer']
    missing_optional = []
    for col in optional_cols:
        if col in df.columns:
            missing_pct = (df[col].isna() | (df[col] == '')).sum() / len(df) * 100
            if missing_pct > 50:
                missing_optional.append(f"{col} ({missing_pct:.0f}% missing)")
    
    if missing_optional:
        issues.append(("info", f"ℹ️ High missing rate in: {', '.join(missing_optional)}"))
    
    stats['total_rows'] = len(df)
    stats['total_volume'] = df['processed_units'].sum()
    stats['avg_aht'] = df['aht_median'].mean()
    stats['median_aht'] = df['aht_median'].median()
    stats['sites'] = df['site'].nunique() if 'site' in df.columns else 0
    
    if not issues:
        issues.append(("success", "✅ All data quality checks passed!"))
    
    return issues, stats


def display_data_quality_report(issues, stats):
    st.markdown("### 📊 Data Quality Report")
    
    errors = [msg for severity, msg in issues if severity == "error"]
    warnings = [msg for severity, msg in issues if severity == "warning"]
    infos = [msg for severity, msg in issues if severity == "info"]
    successes = [msg for severity, msg in issues if severity == "success"]
    
    if errors:
        for msg in errors:
            st.error(msg)
    
    if warnings:
        for msg in warnings:
            st.warning(msg)
    
    if successes:
        for msg in successes:
            st.success(msg)
    
    if infos:
        with st.expander("ℹ️ Additional Information", expanded=False):
            for msg in infos:
                st.info(msg)
    
    if stats:
        st.markdown("#### 📈 Dataset Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{stats.get('total_rows', 0):,}")
            st.metric("Workflows", f"{stats.get('workflows', 0):,}")
        
        with col2:
            st.metric("Time Coverage", f"{stats.get('weeks', 0)} weeks")
            st.metric("Sites", f"{stats.get('sites', 0)}")
        
        with col3:
            st.metric("Total Volume", f"{stats.get('total_volume', 0):,.0f}")
            st.metric("Avg AHT", f"{stats.get('avg_aht', 0):.0f}s")
        
        with col4:
            st.metric("Median AHT", f"{stats.get('median_aht', 0):.0f}s")
            st.metric("Avg Records/Workflow", f"{stats.get('avg_records_per_workflow', 0):.0f}")


# ---------------------------------------------------------------------------
# EXECUTIVE SUMMARY FUNCTIONS
# ---------------------------------------------------------------------------

def show_executive_summary(df_full, result, quality_enabled=False):
    st.markdown("### 📊 Executive Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        avg_rs = result['performance_ratio_pct'].mean()
        delta = avg_rs - 100
        st.metric(
            "Network Rs%",
            f"{avg_rs:.1f}%",
            delta=f"{delta:+.1f}%",
            delta_color="normal"
        )
    
    with col2:
        latest_period = result['period_sort'].max()
        latest_result = result[result['period_sort'] == latest_period]
        if not latest_result.empty:
            top_performer = latest_result.loc[latest_result['performance_ratio_pct'].idxmax(), 'entity']
            top_rs = latest_result['performance_ratio_pct'].max()
            st.metric("Top Performer", top_performer, f"{top_rs:.1f}%")
        else:
            st.metric("Top Performer", "N/A")
    
    with col3:
        total_volume = result['total_units'].sum()
        st.metric("Total Volume", f"{total_volume:,.0f}")
    
    with col4:
        workflow_count = df_full['workflow_key'].nunique()
        st.metric("Active Workflows", workflow_count)
    
    with col5:
        week_range = f"{result['period_sort'].nunique()} weeks"
        date_range = f"{df_full['period_label'].min()} to {df_full['period_label'].max()}"
        st.metric("Date Range", week_range, date_range)
    
    if quality_enabled and 'avg_quality_coeff' in result.columns:
        avg_coeff = result['avg_quality_coeff'].mean()
        if avg_coeff != 1.0:
            st.info(f"🎯 Quality adjustment active | Network avg coefficient: **{avg_coeff:.3f}**")
    
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("#### 🎯 Performance Distribution (Latest Period)")
        latest_result = result[result['period_sort'] == latest_period].copy()
        
        if not latest_result.empty:
            # Define fixed order: Excellent first (top), then Above Target, Below Target, Needs Attention last
            CATEGORY_ORDER = [
                "🟢 Excellent (≥110%)",
                "✅ Above Target (100-110%)",
                "⚠️ Below Target (90-100%)",
                "🔴 Needs Attention (<90%)",
            ]
            CATEGORY_COLORS = {
                "🟢 Excellent (≥110%)":       "#4CAF50",   # green
                "✅ Above Target (100-110%)":  "#8BC34A",   # pale green
                "⚠️ Below Target (90-100%)":   "#FFC107",   # yellow
                "🔴 Needs Attention (<90%)":   "#F44336",   # red
            }

            def categorize_performance(rs):
                if rs >= 110:
                    return "🟢 Excellent (≥110%)"
                elif rs >= 100:
                    return "✅ Above Target (100-110%)"
                elif rs >= 90:
                    return "⚠️ Below Target (90-100%)"
                else:
                    return "🔴 Needs Attention (<90%)"
            
            latest_result['category'] = latest_result['performance_ratio_pct'].apply(categorize_performance)
            
            # Build counts in the fixed order, keeping only categories that exist
            ordered_labels = []
            ordered_values = []
            ordered_colors = []
            category_counts = latest_result['category'].value_counts()
            
            for cat in CATEGORY_ORDER:
                if cat in category_counts.index:
                    ordered_labels.append(cat)
                    ordered_values.append(category_counts[cat])
                    ordered_colors.append(CATEGORY_COLORS[cat])
            
            fig = go.Figure(data=[go.Pie(
                labels=ordered_labels,
                values=ordered_values,
                hole=0.4,
                marker=dict(colors=ordered_colors),
                sort=False,           # preserve the explicit order
                direction='clockwise' # Excellent starts at top, goes clockwise
            )])
            fig.update_layout(
                showlegend=True,
                legend=dict(traceorder='normal'),  # keep legend in the same order
                height=300,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col_b:
        st.markdown("#### 📦 Volume Distribution (Latest Period)")
        
        if not latest_result.empty:
            top_5 = latest_result.nlargest(5, 'total_units')
            
            fig = go.Figure(data=[go.Bar(
                x=top_5['total_units'],
                y=top_5['entity'],
                orientation='h',
                marker=dict(
                    color=top_5['performance_ratio_pct'],
                    colorscale='RdYlGn',
                    cmin=80,
                    cmax=120,
                    colorbar=dict(title="Rs%")
                ),
                text=top_5['performance_ratio_pct'].apply(lambda x: f"{x:.1f}%"),
                textposition='auto',
            )])
            
            fig.update_layout(
                xaxis_title="Units Processed",
                yaxis_title="",
                height=300,
                margin=dict(l=20, r=20, t=20, b=40),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### 💡 Key Insights")
    insights = generate_insights(result, df_full, quality_enabled)
    
    if insights:
        cols = st.columns(len(insights))
        for idx, insight in enumerate(insights):
            with cols[idx]:
                insight_type = insight['type']
                if insight_type == 'success':
                    st.success(insight['message'])
                elif insight_type == 'warning':
                    st.warning(insight['message'])
                elif insight_type == 'info':
                    st.info(insight['message'])
    else:
        st.info("No significant insights detected")


def generate_insights(result, df_full, quality_enabled):
    insights = []
    
    avg_rs = result['performance_ratio_pct'].mean()
    if avg_rs > 105:
        insights.append({
            'type': 'success',
            'message': f"🚀 Network performing **{avg_rs - 100:.1f}%** above target"
        })
    elif avg_rs < 95:
        insights.append({
            'type': 'warning',
            'message': f"⚠️ Network performing **{100 - avg_rs:.1f}%** below target"
        })
    else:
        insights.append({
            'type': 'info',
            'message': f"✓ Network performing near target (**{avg_rs:.1f}%**)"
        })
    
    entity_volatility = result.groupby('entity')['performance_ratio_pct'].std()
    avg_volatility = entity_volatility.mean()
    
    if avg_volatility > 15:
        insights.append({
            'type': 'warning',
            'message': f"📊 High performance volatility detected (avg σ={avg_volatility:.1f}%)"
        })
    elif avg_volatility < 5:
        insights.append({
            'type': 'success',
            'message': f"📊 Stable performance across periods (avg σ={avg_volatility:.1f}%)"
        })
    
    latest_period = result['period_sort'].max()
    if result['period_sort'].nunique() >= 4:
        recent_4_weeks = result[result['period_sort'] >= latest_period - 3]
        older_4_weeks = result[result['period_sort'] < latest_period - 3]
        
        if not recent_4_weeks.empty and not older_4_weeks.empty:
            recent_avg = recent_4_weeks['performance_ratio_pct'].mean()
            older_avg = older_4_weeks['performance_ratio_pct'].mean()
            change = recent_avg - older_avg
            
            if abs(change) > 3:
                trend_type = 'success' if change > 0 else 'warning'
                direction = 'improving' if change > 0 else 'declining'
                insights.append({
                    'type': trend_type,
                    'message': f"📈 Performance {direction} (**{change:+.1f}%** vs. earlier period)"
                })
    
    if quality_enabled and 'avg_quality_coeff' in result.columns:
        avg_coeff = result['avg_quality_coeff'].mean()
        if avg_coeff < 0.95:
            insights.append({
                'type': 'warning',
                'message': f"🎯 Quality penalties detected (avg coeff: **{avg_coeff:.3f}**)"
            })
        elif avg_coeff > 1.05:
            insights.append({
                'type': 'success',
                'message': f"🎯 Quality bonuses active (avg coeff: **{avg_coeff:.3f}**)"
            })
    
    return insights[:4]

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

@st.cache_data
def load_and_clean_data(uploaded_file):
    df = pd.read_csv(uploaded_file)

    df = df.rename(columns=COLUMN_MAPPING)
    df.columns = df.columns.str.strip()

    for col in ["site", "workflow", "locale"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df = df[df[col].notna() & (df[col] != "") & (df[col] != "nan")]

    df["region"] = df["site"].map(REGION_MAPPING)
    df["region"] = df["region"].fillna("Unknown")

    df["network"] = "Network"

    for col in ["ops_manager", "team_manager", "da_name"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(["nan", ""], pd.NA)
        else:
            df[col] = pd.NA

    for col in ["demand_category", "customer"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(["nan", ""], pd.NA)
        else:
            df[col] = pd.NA

    df["aht_median"] = pd.to_numeric(df["aht_median"], errors="coerce")
    df["processed_units"] = pd.to_numeric(df["processed_units"], errors="coerce")
    df = df.dropna(subset=["aht_median", "processed_units"])
    df = df[df["processed_units"] > 0].copy()
    df = df[df["aht_median"] > 0].copy()

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
    qdf = pd.read_csv(uploaded_file)
    qdf.columns = qdf.columns.str.strip()

    leading_map = _detect_quality_leading_columns(qdf.columns)

    rename_map = {}
    for canonical, actual in leading_map.items():
        rename_map[actual] = canonical

    for original, canonical in QUALITY_FIXED_COLUMNS.items():
        for col in qdf.columns:
            if col.strip() == original.strip():
                rename_map[col] = canonical
                break

    qdf = qdf.rename(columns=rename_map)

    for col in ["workflow_name", "locale", "site", "customer", "skill_type"]:
        if col in qdf.columns:
            qdf[col] = qdf[col].astype(str).str.strip()

    for col in [
        "tasks_audited", "pass_count", "fail_count",
        "quality_pre_appeals", "num_appeals",
        "accepted_appeals", "rejected_appeals", "pending_appeals",
        "quality_post_appeals", "quality_goal",
        "appeals_pct", "appeals_accepted_pct",
    ]:
        if col in qdf.columns:
            qdf[col] = pd.to_numeric(qdf[col], errors="coerce")

    qdf["workflow_key"] = qdf["workflow_name"] + " | " + qdf["locale"]

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

    if "site" in qdf.columns:
        qdf["region"] = qdf["site"].map(REGION_MAPPING).fillna("Unknown")

    qdf["network"] = "Network"

    return qdf


@st.cache_data
def compute_default_expected_aht(df, use_median=False):
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
    if quality_df is None or quality_df.empty:
        return pd.DataFrame()

    qdf = quality_df.copy()

    if agg_level in ("network", "region", "site"):
        q_group_col = agg_level
    else:
        q_group_col = "site"

    if q_group_col not in qdf.columns:
        return pd.DataFrame()

    required = ["quality_post_appeals", "quality_goal", "tasks_audited"]
    for c in required:
        if c not in qdf.columns:
            return pd.DataFrame()

    qdf = qdf.dropna(subset=required)
    qdf = qdf[qdf["quality_goal"] > 0].copy()

    agg_dict = {
        "pass_count": "sum",
        "fail_count": "sum",
        "accepted_appeals": "sum",
        "tasks_audited": "sum",
        "quality_goal": "first",
    }
    agg_dict = {k: v for k, v in agg_dict.items() if k in qdf.columns}

    grouped = (
        qdf.groupby([q_group_col, "workflow_key", "period_sort"])
        .agg(agg_dict)
        .reset_index()
    )

    total = grouped["pass_count"] + grouped["fail_count"]
    grouped["q_actual"] = np.where(
        total > 0,
        (grouped["pass_count"] + grouped.get("accepted_appeals", 0)) / total,
        np.nan,
    )
    grouped["q_goal"] = grouped["quality_goal"]

    grouped = grouped[grouped["tasks_audited"] >= min_audits].copy()
    grouped = grouped.dropna(subset=["q_actual"])

    ratio = grouped["q_actual"] / grouped["q_goal"]
    ratio = ratio.clip(lower=0.1, upper=2.0)
    grouped["quality_coeff"] = ratio ** sensitivity_k

    grouped = grouped.rename(columns={q_group_col: "q_merge_key"})

    result = grouped[
        ["q_merge_key", "workflow_key", "period_sort",
         "quality_coeff", "q_actual", "q_goal", "tasks_audited"]
    ].copy()
    result["q_agg_level"] = q_group_col

    return result


def _resolve_quality_join_column(agg_level):
    if agg_level in ("network", "region", "site"):
        return agg_level
    else:
        return "site"


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def get_prioritised_workflows(df_full, agg_level, selected_entities, all_workflows):
    if not selected_entities:
        return sorted(all_workflows)

    entity_workflows = set(
        df_full[df_full[agg_level].isin(selected_entities)]["workflow_key"].unique()
    )
    done = sorted([w for w in all_workflows if w in entity_workflows])
    rest = sorted([w for w in all_workflows if w not in entity_workflows])
    return done + rest


def get_prioritised_sites(df_full, agg_level, selected_entities, all_sites):
    if not selected_entities:
        return sorted(all_sites)

    entity_sites = set(
        df_full[df_full[agg_level].isin(selected_entities)]["site"].unique()
    )
    done = sorted([s for s in all_sites if s in entity_sites])
    rest = sorted([s for s in all_sites if s not in entity_sites])
    return done + rest


def week_range_selector(all_weeks_sorted, key_prefix):
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

    if quality_coefficients is not None and not quality_coefficients.empty:
        q_join_col = _resolve_quality_join_column(agg_level)
        qc = quality_coefficients.copy()

        merged = merged.merge(
            qc[["q_merge_key", "workflow_key", "period_sort", "quality_coeff"]],
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
# VISUALIZATION
# ---------------------------------------------------------------------------

def plot_performance_ratios(result, title_suffix=""):
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

    fig.add_hline(
        y=100,
        line_dash="dash",
        line_color="#E53935",
        line_width=2,
        annotation_text="Expected (100%)",
        annotation_position="top left",
        annotation_font_color="#E53935",
    )

    title = "Workflow Efficiency Index – Performance Over Time"
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

    fig.update_xaxes(
        type='category',
        categoryorder='array',
        categoryarray=sorted(result["period_label"].unique(),
                           key=lambda x: result[result["period_label"]==x]["period_sort"].iloc[0]),
        tickangle=-45
    )

    return fig


def plot_unit_volume(result, title_suffix=""):
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
    fig = go.Figure()

    colors_1 = px.colors.qualitative.Set1
    colors_2 = px.colors.qualitative.Set2

    all_results = pd.concat([result_1, result_2])

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
            legendgrouptitle_text=f"Group 1 – {agg_level_1_label}",
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
            legendgrouptitle_text=f"Group 2 – {agg_level_2_label}",
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

    fig.update_xaxes(
        type='category',
        categoryorder='array',
        categoryarray=sorted(all_results["period_label"].unique(),
                           key=lambda x: all_results[all_results["period_label"]==x]["period_sort"].iloc[0]),
        tickangle=-45
    )

    return fig


# ---------------------------------------------------------------------------
# EXPLANATION SECTION
# ---------------------------------------------------------------------------

def show_explanation_section():
    """Display a comprehensive explanation of all tool sections and terminology."""
    st.markdown("## 📚 Explanation of Sections & Terminology")
    st.markdown("---")

    st.markdown("### 🔢 Core Formula")
    st.markdown("""
    **Rs = Σ (Ew / Aw,s) × Qw,s × Pw,s**

    | Symbol | Meaning |
    |--------|---------|
    | **Rs** | Performance Ratio – the final efficiency score for an entity in a given period |
    | **Ew** | Expected AHT – the target average handle time for a workflow (from uploaded goals or computed network average) |
    | **Aw,s** | Actual AHT – the observed average handle time for a workflow performed by entity *s* |
    | **Qw,s** | Quality Coefficient – an optional multiplier derived from audit pass rates vs. quality goals |
    | **Pw,s** | Volume Proportion – the share of total units that a particular workflow represents for entity *s* in that period |

    A score of **100%** means the entity is performing exactly at the expected level.
    Scores **above 100%** indicate faster-than-expected processing (higher efficiency).
    Scores **below 100%** indicate slower-than-expected processing.
    """)

    st.markdown("---")
    st.markdown("### 📁 Data Sources (Sidebar)")
    st.markdown("""
    | Upload | Purpose |
    |--------|---------|
    | **Data CSV** *(required)* | The primary dataset containing workflow AHT, volume, site, employee, and date information. |
    | **Expected AHT CSV** *(optional)* | Custom per-workflow target AHT values. When not provided, the tool computes network-wide averages as the baseline. |
    | **Quality CSV** *(optional)* | Audit and quality metrics per workflow+locale+site+week. Enables the quality coefficient adjustment. |
    """)

    st.markdown("---")
    st.markdown("### 🎯 Quality Settings (Sidebar)")
    st.markdown("""
    These controls appear when a Quality CSV is uploaded:

    | Setting | Description |
    |---------|-------------|
    | **Enable Quality Coefficient** | Toggles whether quality data adjusts the Rs score. When off, Qw,s = 1.0 for all rows. |
    | **Minimum Audit Threshold** | *≥1 audit*: apply the coefficient whenever any audit exists. *≥6 audits*: require at least 6 audited tasks before applying – rows below the threshold default to Qw,s = 1.0. |
    | **Sensitivity (k)** | Controls how aggressively quality deviations affect the score. The coefficient is calculated as *(actual quality / goal quality)^k*. Higher k amplifies both bonuses and penalties. |
    """)

    st.markdown("---")
    st.markdown("### 📊 Data Quality Report")
    st.markdown("""
    Automatically generated when data is loaded. Checks include:

    - **Missing columns** – verifies all required fields are present.
    - **Invalid values** – flags rows with AHT ≤ 0 or volume ≤ 0.
    - **Time coverage** – reports how many weeks of data are available.
    - **AHT outliers** – identifies extreme values that may skew results.
    - **Site completeness** – warns if any site has data for less than half the available weeks.
    - **Duplicate records** – detects repeated workflow+site+period combinations.
    - **Low-volume workflows** – notes workflows with very few data points.
    - **Missing hierarchy data** – reports high missing rates in optional columns (managers, demand category, etc.).
    """)

    st.markdown("---")
    st.markdown("### 📊 Executive Summary")
    st.markdown("""
    A high-level overview displayed at the top of the Dashboard tab:

    | Metric | Description |
    |--------|-------------|
    | **Network Rs%** | The average performance ratio across all entities and periods in the current view. |
    | **Top Performer** | The entity with the highest Rs% in the most recent period. |
    | **Total Volume** | Sum of all processed units across the selected data. |
    | **Active Workflows** | Count of distinct workflow+locale combinations in the data. |
    | **Date Range** | Number of weeks and calendar range covered. |
    | **Performance Distribution** | Pie chart categorising entities into Excellent (≥110%), Above Target (100–110%), Below Target (90–100%), and Needs Attention (<90%). |
    | **Volume Distribution** | Horizontal bar chart of the top 5 entities by volume, colour-coded by Rs%. |
    | **Key Insights** | Auto-generated observations about network performance level, trend direction, volatility, and quality impact. |
    """)

    st.markdown("---")
    st.markdown("### ⚙️ Dashboard Settings (Left Panel)")
    st.markdown("""
    | Setting | Description |
    |---------|-------------|
    | **Analysis Mode** | *Employee Groups*: aggregate by Region → Site → Ops Manager → Team Manager → Data Associate. *Work Type*: aggregate by Demand Category → Customer → Workflow → Workflow+Locale. |
    | **Group by** | Selects the aggregation level – each entity at this level gets its own Rs% line. |
    | **Hierarchy Filter** | When viewing granular levels, parent-level dropdowns appear so you can narrow the entity list (e.g., pick a Region first, then see only Sites within it). |
    | **Baseline Scope** | *Network*: expected AHT is computed from the entire dataset. *Relative*: expected AHT is computed only from the selected entities' data. |
    | **Baseline Calculation** | *Weighted Mean*: volume-weighted average AHT per workflow (high-volume periods count more). *Median*: simple median AHT per workflow (resistant to outliers). |
    | **Week Filter** | Restrict the analysis to a specific range of weeks. |
    | **Workflow / Site Filter** | Limit which workflows (in Employee mode) or sites (in Work Type mode) are included in the calculation. |
    """)

    st.markdown("---")
    st.markdown("### 📈 Performance Trends Chart")
    st.markdown("""
    An interactive line chart showing Rs% over time for each selected entity.
    The red dashed line at 100% represents the expected performance level.
    Hover over data points to see exact Rs%, unit volume, workflow count, and quality coefficient.
    """)

    st.markdown("---")
    st.markdown("### 📊 Volume Analysis Chart")
    st.markdown("""
    A grouped bar chart showing the total units processed per entity per week.
    Useful for understanding whether performance changes correlate with volume shifts.
    Hover for Rs% and workflow count alongside volume figures.
    """)

    st.markdown("---")
    st.markdown("### 📋 Latest Period Results Table")
    st.markdown("""
    A sortable table showing each entity's Rs%, total units, workflow count, and (if enabled) average quality coefficient for the most recent week.
    Entities are ranked from highest to lowest Rs%.
    """)

    st.markdown("---")
    st.markdown("### 🔬 Advanced Comparison Tab")
    st.markdown("""
    Allows side-by-side comparison of two groups of entities, potentially at different aggregation levels.

    | Element | Description |
    |---------|-------------|
    | **Group 1 / Group 2** | Each group has its own aggregation level selector and entity picker with hierarchy filtering. |
    | **Shared Settings** | Baseline scope, calculation method, week filter, and workflow/site filter apply to both groups. |
    | **Comparison Chart** | Overlays both groups on a single chart – Group 1 as solid lines, Group 2 as dashed lines. |
    | **Comparison Summary** | Table combining both groups' latest-period results for direct comparison. |
    """)

    st.markdown("---")
    st.markdown("### 💾 Downloads")
    st.markdown("""
    | Download | Contents |
    |----------|----------|
    | **Performance Ratios** | One row per entity per period: Rs%, total units, workflow count, quality coefficient. |
    | **Workflow Detail** | Granular row-level data showing each workflow's contribution to the Rs% calculation, including efficiency ratio, volume weight, and quality coefficient. |
    | **Comparison Results** | Combined output from both groups in the Advanced Comparison tab. |
    """)


# ---------------------------------------------------------------------------
# STREAMLIT APP
# ---------------------------------------------------------------------------

def main():
    st.title("🎯 Workflow Efficiency Index Tool")
    
    # ---------------------------------------------------------------------------
    # SIDEBAR
    # ---------------------------------------------------------------------------
    
    with st.sidebar:
        try:
            st.image("peccy.png", width=120)
        except:
            pass

        st.markdown("""
        <div style='font-size: 0.85em; color: #666; padding: 10px 0; border-bottom: 1px solid #ddd; margin-bottom: 15px;'>
        <strong>Tool Creator:</strong><br>
        Benjamin Giblin (bgiblin@)<br>
        AGI-DS Quality Team<br>
        CBG10
        </div>
        """, unsafe_allow_html=True)
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")
        
        with st.expander("📁 Data Sources", expanded=True):
            # Data CSV with hyperlink
            st.markdown(
                "**[Data CSV](https://us-east-1.quicksight.aws.amazon.com/sn/account/ads-quicksight-de/dashboards/4ee31a3c-0a67-4d34-9fc4-75ab3b1a88df_dashboard_id)** (Required)",
                unsafe_allow_html=True
            )
            data_file = st.file_uploader(
                "Upload Data CSV",
                type=["csv"],
                key="data_csv_uploader",
                label_visibility="collapsed"
            )
            
            st.markdown("---")
            
            expected_file = st.file_uploader(
                "**Expected AHT CSV** (Optional)",
                type=["csv"],
            )
            
            # Quality CSV with hyperlink
            st.markdown(
                "**[Quality CSV](https://us-east-1.quicksight.aws.amazon.com/sn/account/ads-quicksight-de/dashboards/7f34b9a0-7844-4177-9bec-1b4f7bee5b6d_dashboard_id)** (Optional)",
                unsafe_allow_html=True
            )
            quality_file = st.file_uploader(
                "Upload Quality CSV",
                type=["csv"],
                key="quality_csv_uploader",
                label_visibility="collapsed"
            )
        
        if quality_file:
            with st.expander("🎯 Quality Settings", expanded=False):
                quality_enabled = st.toggle(
                    "Enable Quality Coefficient",
                    value=True,
                )
                
                if quality_enabled:
                    audit_mode = st.radio(
                        "Minimum audit threshold:",
                        options=["≥1 audit", "≥6 audits"],
                        key="quality_audit_mode",
                    )
                    min_audits = 1 if "≥1" in audit_mode else 6
                    
                    sensitivity_k = st.slider(
                        "Sensitivity (k):",
                        min_value=1.0,
                        max_value=10.0,
                        value=3.0,
                        step=0.5,
                    )
        else:
            quality_enabled = False
            min_audits = 1
            sensitivity_k = 3.0
        
        st.markdown("---")
        st.caption("v2.2")

    # ---------------------------------------------------------------------------
    # MAIN CONTENT AREA
    # ---------------------------------------------------------------------------
    
    if data_file is None:
        st.info("📤 Upload your data CSV in the sidebar to get started")
        
        # Show explanation section when no data is loaded
        show_explanation_section()
        return
    
    # Load data
    with st.spinner("Loading and validating data..."):
        df_full = load_and_clean_data(data_file)
    
    # Data Quality Report
    with st.expander("📊 Data Quality Report", expanded=True):
        issues, stats = validate_data_quality(df_full)
        display_data_quality_report(issues, stats)
    
    has_errors = any(severity == "error" for severity, _ in issues)
    if has_errors:
        st.error("❌ Critical data quality issues detected. Please fix the data and re-upload.")
        return
    
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
    
    # Quality coefficients cache
    quality_coefficients_cache = {}
    
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
    
    # Mode selection
    analysis_mode = st.radio(
        "Select aggregation type:",
        options=["👥 Employee Groups", "📦 Work Type"],
        horizontal=True,
    )
    
    is_work_type_mode = (analysis_mode == "📦 Work Type")
    
    st.markdown("---")
    
    # Build sorted week list
    all_weeks_sorted = sorted(df_full["period_label"].unique(),
                             key=lambda x: df_full[df_full["period_label"]==x]["period_sort"].iloc[0])
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔬 Advanced Comparison", "📚 Explanation"])
    
    # ========================================================================
    # TAB 1: DASHBOARD
    # ========================================================================
    
    with tab1:
        col_left, col_right = st.columns([1, 3])
        
        with col_left:
            st.markdown("### ⚙️ Settings")
            
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
            
            if agg_level == "network":
                selected_entities = ["Network"]
                st.info("🌐 Viewing entire network")
            else:
                selected_entities = render_hierarchy_selector(
                    df_full, agg_level, is_work_type_mode, key_prefix="basic"
                )
            
            st.markdown("---")
            
            if agg_level == "network":
                use_full_dataset = True
            else:
                baseline_mode = st.radio(
                    "Baseline scope:",
                    options=["Network", "Relative"],
                    key="basic_baseline"
                )
                use_full_dataset = (baseline_mode == "Network")
            
            baseline_calc = st.radio(
                "Baseline calculation:",
                options=["Weighted Mean", "Median"],
                key="basic_calc_method"
            )
            use_median = (baseline_calc == "Median")
            
            st.markdown("---")
            st.markdown("**Week Filter**")
            selected_weeks = week_range_selector(all_weeks_sorted, key_prefix="basic")
            
            st.markdown("---")
            
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
                    if agg_level != "network":
                        prioritised_sites = get_prioritised_sites(
                            df_full, agg_level, selected_entities, all_sites
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
                    if agg_level != "network":
                        prioritised_workflows = get_prioritised_workflows(
                            df_full, agg_level, selected_entities, all_workflows
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
        
        with col_right:
            if agg_level != "network" and not selected_entities:
                st.warning("⚠️ Select at least one entity")
                return
            
            if agg_level == "network":
                df_filtered = df_full.copy()
            else:
                df_filtered = df_full[df_full[agg_level].isin(selected_entities)].copy()
            
            if selected_weeks is not None:
                df_display = df_filtered[df_filtered["period_label"].isin(selected_weeks)].copy()
            else:
                df_display = df_filtered.copy()
            
            if is_work_type_mode:
                if site_filter_mode == "Specific" and selected_sites:
                    df_filtered = df_filtered[df_filtered["site"].isin(selected_sites)].copy()
                    df_display = df_display[df_display["site"].isin(selected_sites)].copy()
            else:
                if workflow_filter_mode == "Specific" and selected_workflows:
                    df_filtered = df_filtered[df_filtered["workflow_key"].isin(selected_workflows)].copy()
                    df_display = df_display[df_display["workflow_key"].isin(selected_workflows)].copy()
            
            baseline_df = df_full if use_full_dataset else df_filtered
            
            use_hybrid = True
            qc = get_quality_coefficients(agg_level)
            
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
            
            show_executive_summary(df_display, result, quality_enabled)
            
            st.markdown("---")
            
            st.markdown("### 📈 Performance Trends")
            fig1 = plot_performance_ratios(result)
            st.plotly_chart(fig1, use_container_width=True)
            
            st.markdown("### 📊 Volume Analysis")
            fig2 = plot_unit_volume(result)
            st.plotly_chart(fig2, use_container_width=True)
            
            latest_period = result["period_sort"].max()
            latest_label = result[result["period_sort"] == latest_period]["period_label"].iloc[0]
            latest_data = result[result["period_sort"] == latest_period].sort_values("performance_ratio_pct", ascending=False)
            
            st.markdown(f"### 📋 Latest Period Results ({latest_label})")
            
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
                use_container_width=True,
                hide_index=True
            )
            
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                csv1 = result.to_csv(index=False)
                st.download_button(
                    "💾 Download Performance Ratios",
                    csv1,
                    "performance_ratios.csv",
                    "text/csv",
                    use_container_width=True
                )
            with col_dl2:
                csv2 = detail.to_csv(index=False)
                st.download_button(
                    "💾 Download Workflow Detail",
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
        
        col1, col2 = st.columns(2)
        
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
                st.info("🌐 Network level")
            else:
                selected_entities_1 = render_hierarchy_selector_advanced(
                    df_full, agg_level_1, is_work_type_mode, key_prefix="adv_g1"
                )
        
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
                st.info("🌐 Network level")
            else:
                selected_entities_2 = render_hierarchy_selector_advanced(
                    df_full, agg_level_2, is_work_type_mode, key_prefix="adv_g2"
                )
        
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
        
        if is_work_type_mode:
            if site_filter_mode_adv == "Specific":
                if agg_level_1 != "network":
                    prioritised_sites_adv = get_prioritised_sites(
                        df_full, agg_level_1, selected_entities_1, all_sites_adv
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
                if agg_level_1 != "network":
                    prioritised_workflows_adv = get_prioritised_workflows(
                        df_full, agg_level_1, selected_entities_1, all_workflows_adv
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
        
        if st.button("🔬 Calculate Comparison", use_container_width=True, type="primary"):
            st.session_state.adv_calculate = True
        
        if st.session_state.get("adv_calculate", False):
            if not selected_entities_1 or not selected_entities_2:
                st.warning("⚠️ Select entities in both groups")
                return
            
            qc_1 = get_quality_coefficients(agg_level_1)
            qc_2 = get_quality_coefficients(agg_level_2)
            
            if agg_level_1 == "network":
                df_filtered_1 = df_full.copy()
            else:
                df_filtered_1 = df_full[df_full[agg_level_1].isin(selected_entities_1)].copy()
            
            if selected_weeks_adv is not None:
                df_display_1 = df_filtered_1[df_filtered_1["period_label"].isin(selected_weeks_adv)].copy()
            else:
                df_display_1 = df_filtered_1.copy()
            
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
            
            if agg_level_2 == "network":
                df_filtered_2 = df_full.copy()
            else:
                df_filtered_2 = df_full[df_full[agg_level_2].isin(selected_entities_2)].copy()
            
            if selected_weeks_adv is not None:
                df_display_2 = df_filtered_2[df_filtered_2["period_label"].isin(selected_weeks_adv)].copy()
            else:
                df_display_2 = df_filtered_2.copy()
            
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
            
            st.markdown("### 📊 Comparison Results")
            
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
            
            fig_combined = plot_comparison(
                result_1, result_2,
                selected_entities_1, selected_entities_2,
                agg_level_1_label, agg_level_2_label
            )
            st.plotly_chart(fig_combined, use_container_width=True)
            
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
                use_container_width=True,
                hide_index=True
            )
            
            csv_combined = result_combined.to_csv(index=False)
            st.download_button(
                "💾 Download Comparison Results",
                csv_combined,
                "cross_level_comparison.csv",
                "text/csv",
                use_container_width=True
            )
    
    # ========================================================================
    # TAB 3: EXPLANATION
    # ========================================================================
    
    with tab3:
        show_explanation_section()


if __name__ == "__main__":
    main()
