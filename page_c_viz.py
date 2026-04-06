"""
Page C: Visualization Builder
Dynamic chart builder from the working dataset.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")


# ─── Colour palette ──────────────────────────────────────────────────────────
COLORS = ['#34d399', '#34d399', '#fbbf24', '#a7f3d0', '#f59e0b', '#ec4899', '#10b981', '#6366f1']
BG = '#112218'
BG2 = '#080d10'
TEXT = '#e6edf3'
TEXT2 = '#6abf8a'
BORDER = '#1c3528'


def style_ax(ax, title=""):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT2, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    if title:
        ax.set_title(title, color=TEXT, fontsize=11, pad=10)
    ax.xaxis.label.set_color(TEXT2)
    ax.yaxis.label.set_color(TEXT2)


def style_fig(fig):
    fig.patch.set_facecolor(BG2)


# ─── Chart functions ─────────────────────────────────────────────────────────

def plot_histogram(df, x_col, color_col=None, bins=30):
    fig, ax = plt.subplots(figsize=(10, 5))
    style_fig(fig)
    style_ax(ax, f"Histogram — {x_col}")
    if color_col and color_col in df.columns:
        groups = df[color_col].dropna().unique()[:8]
        for i, g in enumerate(groups):
            subset = df[df[color_col] == g][x_col].dropna()
            ax.hist(subset, bins=bins, alpha=0.6, label=str(g), color=COLORS[i % len(COLORS)], edgecolor='none')
        ax.legend(fontsize=8, labelcolor=TEXT2, framealpha=0.2)
    else:
        ax.hist(df[x_col].dropna(), bins=bins, color=COLORS[0], alpha=0.8, edgecolor='none')
    ax.set_xlabel(x_col)
    ax.set_ylabel("Count")
    plt.tight_layout()
    return fig


def plot_boxplot(df, y_col, x_col=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    style_fig(fig)
    style_ax(ax, f"Box Plot — {y_col}")
    if x_col and x_col in df.columns:
        groups = df[x_col].dropna().value_counts().head(8).index.tolist()
        data = [df[df[x_col] == g][y_col].dropna().values for g in groups]
        bp = ax.boxplot(data, patch_artist=True, labels=[str(g) for g in groups],
                        medianprops=dict(color='#34d399', linewidth=2),
                        whiskerprops=dict(color=TEXT2), capprops=dict(color=TEXT2),
                        flierprops=dict(marker='o', color=COLORS[2], alpha=0.4, markersize=3))
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(COLORS[i % len(COLORS)])
            patch.set_alpha(0.4)
        ax.tick_params(axis='x', rotation=30)
    else:
        bp = ax.boxplot(df[y_col].dropna(), patch_artist=True,
                        medianprops=dict(color='#34d399', linewidth=2),
                        whiskerprops=dict(color=TEXT2), capprops=dict(color=TEXT2),
                        flierprops=dict(marker='o', color=COLORS[2], alpha=0.4, markersize=4))
        bp['boxes'][0].set_facecolor(COLORS[0])
        bp['boxes'][0].set_alpha(0.5)
    ax.set_ylabel(y_col)
    plt.tight_layout()
    return fig


def plot_scatter(df, x_col, y_col, color_col=None, sample_n=2000):
    fig, ax = plt.subplots(figsize=(10, 6))
    style_fig(fig)
    style_ax(ax, f"Scatter — {x_col} vs {y_col}")
    plot_df = df[[x_col, y_col] + ([color_col] if color_col else [])].dropna().sample(min(sample_n, len(df)), random_state=42)
    if color_col and color_col in df.columns:
        groups = plot_df[color_col].unique()[:8]
        for i, g in enumerate(groups):
            sub = plot_df[plot_df[color_col] == g]
            ax.scatter(sub[x_col], sub[y_col], alpha=0.5, s=20, color=COLORS[i % len(COLORS)], label=str(g))
        ax.legend(fontsize=8, labelcolor=TEXT2, framealpha=0.2)
    else:
        ax.scatter(plot_df[x_col], plot_df[y_col], alpha=0.4, s=15, color=COLORS[0])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.tight_layout()
    return fig


def plot_line(df, x_col, y_col, color_col=None, agg_func="mean"):
    fig, ax = plt.subplots(figsize=(12, 5))
    style_fig(fig)
    style_ax(ax, f"Line Chart — {y_col} over {x_col}")
    try:
        plot_df = df[[x_col, y_col] + ([color_col] if color_col else [])].dropna().copy()
        plot_df[x_col] = pd.to_datetime(plot_df[x_col], errors='coerce')
        plot_df = plot_df.dropna(subset=[x_col])
        agg_map = {"mean": "mean", "sum": "sum", "count": "count", "median": "median"}
        fn = agg_map.get(agg_func, "mean")
        if color_col and color_col in df.columns:
            groups = plot_df[color_col].dropna().unique()[:6]
            for i, g in enumerate(groups):
                sub = plot_df[plot_df[color_col] == g].groupby(x_col)[y_col].agg(fn).reset_index()
                ax.plot(sub[x_col], sub[y_col], color=COLORS[i % len(COLORS)], linewidth=1.5, label=str(g), marker='o', markersize=3)
            ax.legend(fontsize=8, labelcolor=TEXT2, framealpha=0.2)
        else:
            agg_df = plot_df.groupby(x_col)[y_col].agg(fn).reset_index()
            ax.plot(agg_df[x_col], agg_df[y_col], color=COLORS[0], linewidth=2, marker='o', markersize=3)
            ax.fill_between(agg_df[x_col], agg_df[y_col], alpha=0.1, color=COLORS[0])
        ax.set_xlabel(x_col)
        ax.set_ylabel(f"{agg_func}({y_col})")
        plt.xticks(rotation=30)
    except Exception as e:
        ax.text(0.5, 0.5, f"Could not plot: {e}", transform=ax.transAxes, color=TEXT2, ha='center')
    plt.tight_layout()
    return fig


def plot_bar(df, x_col, y_col, color_col=None, agg_func="mean", top_n=15):
    fig, ax = plt.subplots(figsize=(12, 5))
    style_fig(fig)
    style_ax(ax, f"Bar Chart — {agg_func}({y_col}) by {x_col}")
    agg_map = {"mean": "mean", "sum": "sum", "count": "count", "median": "median"}
    fn = agg_map.get(agg_func, "mean")
    try:
        if color_col and color_col in df.columns:
            plot_df = df.groupby([x_col, color_col])[y_col].agg(fn).reset_index()
            groups = plot_df[color_col].unique()[:6]
            x_vals = plot_df[x_col].value_counts().head(top_n).index.tolist()
            x_pos = np.arange(len(x_vals))
            width = 0.8 / len(groups)
            for i, g in enumerate(groups):
                sub = plot_df[plot_df[color_col] == g].set_index(x_col).reindex(x_vals)[y_col].fillna(0)
                ax.bar(x_pos + i * width, sub.values, width=width, color=COLORS[i % len(COLORS)], alpha=0.8, label=str(g))
            ax.set_xticks(x_pos + width * len(groups) / 2)
            ax.set_xticklabels(x_vals, rotation=35, ha='right', color=TEXT2)
            ax.legend(fontsize=8, labelcolor=TEXT2, framealpha=0.2)
        else:
            plot_df = df.groupby(x_col)[y_col].agg(fn).reset_index().nlargest(top_n, y_col)
            bars = ax.bar(plot_df[x_col].astype(str), plot_df[y_col],
                         color=[COLORS[i % len(COLORS)] for i in range(len(plot_df))], alpha=0.85, edgecolor='none')
            ax.tick_params(axis='x', rotation=35)
        ax.set_xlabel(x_col)
        ax.set_ylabel(f"{agg_func}({y_col})")
    except Exception as e:
        ax.text(0.5, 0.5, f"Could not plot: {e}", transform=ax.transAxes, color=TEXT2, ha='center')
    plt.tight_layout()
    return fig


def plot_heatmap(df, num_cols):
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(max(6, len(num_cols)), max(5, len(num_cols) * 0.8)))
    style_fig(fig)
    style_ax(ax, "Correlation Matrix")
    import matplotlib.colors as mcolors
    cmap = plt.cm.RdYlGn
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=45, ha='right', color=TEXT2, fontsize=9)
    ax.set_yticklabels(num_cols, color=TEXT2, fontsize=9)
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            val = corr.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='white', fontsize=8, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


# ─── Main render ────────────────────────────────────────────────────────────

def render():
    st.markdown('<div class="step-badge">PAGE C</div>', unsafe_allow_html=True)
    st.title("Visualization Builder")
    st.markdown("<p style='color:#6abf8a;margin-top:-0.5rem;margin-bottom:1.5rem;'>Build charts dynamically from your cleaned dataset. All charts use the current working copy.</p>", unsafe_allow_html=True)

    if st.session_state.working_df is None:
        st.warning("⚠️ No dataset loaded. Go to **Upload & Overview** first.")
        return

    df = st.session_state.working_df.copy()

    # ── Column groups ─────────────────────────────────────────────────────
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    dt_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    all_cols = df.columns.tolist()

    # ── Sidebar-style controls ─────────────────────────────────────────────
    with st.expander("🎛 Chart Configuration", expanded=True):
        cfg_col1, cfg_col2, cfg_col3 = st.columns(3)

        with cfg_col1:
            chart_type = st.selectbox("Chart type", [
                "Histogram",
                "Box Plot",
                "Scatter Plot",
                "Line Chart (Time Series)",
                "Bar Chart (Grouped)",
                "Heatmap / Correlation Matrix",
            ], key="chart_type")

        with cfg_col2:
            if chart_type == "Histogram":
                x_col = st.selectbox("Column (X)", num_cols or all_cols, key="viz_x")
                y_col = None
                bins = st.slider("Bins", 5, 100, 30, key="viz_bins")
            elif chart_type == "Box Plot":
                y_col = st.selectbox("Numeric column (Y)", num_cols or all_cols, key="viz_y")
                x_col = st.selectbox("Group by (X, optional)", ["— none —"] + cat_cols, key="viz_x")
                x_col = None if x_col == "— none —" else x_col
                bins = None
            elif chart_type == "Scatter Plot":
                x_col = st.selectbox("X axis", num_cols or all_cols, key="viz_x")
                y_col = st.selectbox("Y axis", [c for c in (num_cols or all_cols) if c != x_col] or all_cols, key="viz_y")
                bins = None
            elif chart_type == "Line Chart (Time Series)":
                x_col = st.selectbox("X axis (datetime or numeric)", dt_cols + num_cols + all_cols, key="viz_x")
                y_col = st.selectbox("Y axis (numeric)", num_cols or all_cols, key="viz_y")
                bins = None
            elif chart_type == "Bar Chart (Grouped)":
                x_col = st.selectbox("X axis (categories)", cat_cols + all_cols, key="viz_x")
                y_col = st.selectbox("Y axis (numeric)", num_cols or all_cols, key="viz_y")
                bins = None
            elif chart_type == "Heatmap / Correlation Matrix":
                x_col = None
                y_col = None
                bins = None

        with cfg_col3:
            if chart_type not in ("Heatmap / Correlation Matrix",):
                color_col = st.selectbox("Color / Group by (optional)", ["— none —"] + cat_cols + num_cols, key="viz_color")
                color_col = None if color_col == "— none —" else color_col
            else:
                color_col = None
            if chart_type in ("Bar Chart (Grouped)",):
                top_n = st.slider("Top N categories", 3, 30, 12, key="viz_topn")
            else:
                top_n = 15
            if chart_type in ("Line Chart (Time Series)", "Bar Chart (Grouped)"):
                agg_func = st.selectbox("Aggregation", ["mean", "sum", "count", "median"], key="viz_agg")
            else:
                agg_func = "mean"

        # Heatmap column selection
        if chart_type == "Heatmap / Correlation Matrix":
            heatmap_cols = st.multiselect("Columns (numeric only)", num_cols, default=num_cols[:8], key="viz_hm_cols")

    # ── Filters ──────────────────────────────────────────────────────────────
    with st.expander("🔽 Filters (optional)", expanded=False):
        filter_df = df.copy()
        fc1, fc2 = st.columns(2)
        with fc1:
            if cat_cols:
                filt_cat_col = st.selectbox("Filter by category", ["— none —"] + cat_cols, key="filt_cat")
                if filt_cat_col != "— none —":
                    unique_vals = df[filt_cat_col].dropna().unique().tolist()
                    filt_cat_vals = st.multiselect("Allowed values", unique_vals, default=unique_vals, key="filt_cat_vals")
                    filter_df = filter_df[filter_df[filt_cat_col].isin(filt_cat_vals)]
        with fc2:
            if num_cols:
                filt_num_col = st.selectbox("Filter by numeric range", ["— none —"] + num_cols, key="filt_num")
                if filt_num_col != "— none —":
                    col_min = float(df[filt_num_col].min())
                    col_max = float(df[filt_num_col].max())
                    filt_range = st.slider("Range", col_min, col_max, (col_min, col_max), key="filt_range")
                    filter_df = filter_df[filter_df[filt_num_col].between(filt_range[0], filt_range[1])]
        st.caption(f"Filtered dataset: {len(filter_df):,} rows (of {len(df):,} total)")

    # ── Render chart ──────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    render_btn = st.button("📊 Render Chart", type="primary", key="render_chart")

    if render_btn:
        try:
            if chart_type == "Histogram":
                fig = plot_histogram(filter_df, x_col, color_col, bins or 30)
            elif chart_type == "Box Plot":
                fig = plot_boxplot(filter_df, y_col, x_col)
            elif chart_type == "Scatter Plot":
                fig = plot_scatter(filter_df, x_col, y_col, color_col)
            elif chart_type == "Line Chart (Time Series)":
                fig = plot_line(filter_df, x_col, y_col, color_col, agg_func)
            elif chart_type == "Bar Chart (Grouped)":
                fig = plot_bar(filter_df, x_col, y_col, color_col, agg_func, top_n)
            elif chart_type == "Heatmap / Correlation Matrix":
                if not heatmap_cols or len(heatmap_cols) < 2:
                    st.warning("Please select at least 2 numeric columns for the heatmap.")
                    return
                fig = plot_heatmap(filter_df, heatmap_cols)

            st.pyplot(fig)
            plt.close()

            # Download
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches='tight', facecolor=BG2)
            buf.seek(0)
            st.download_button("⬇ Download Chart (PNG)", buf, "chart.png", "image/png", key="dl_chart")

        except Exception as e:
            st.error(f"❌ Chart error: {e}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())

    # ── Quick stats panel ─────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📋 Column Quick Stats", expanded=False):
        qs_col = st.selectbox("Select column", all_cols, key="qs_col")
        if qs_col:
            col_data = filter_df[qs_col]
            if pd.api.types.is_numeric_dtype(col_data):
                st.dataframe(col_data.describe().to_frame().T.round(4), use_container_width=True)
            else:
                vc = col_data.value_counts().reset_index()
                vc.columns = ["Value", "Count"]
                vc["Percent"] = (vc["Count"] / len(filter_df) * 100).round(2)
                st.dataframe(vc.head(20), use_container_width=True, hide_index=True)

    # Navigation
    st.markdown("<br>", unsafe_allow_html=True)
    col_nav1, col_nav2 = st.columns([4, 1])
    with col_nav2:
        if st.button("→ Go to Export", type="primary", use_container_width=True):
            st.session_state.current_page = "Export & Report"
            st.rerun()
