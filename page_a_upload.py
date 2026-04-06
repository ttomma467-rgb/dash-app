"""
Page A: Upload & Overview
Profile the uploaded dataset and display key stats.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
from datetime import datetime


# ─── Helpers ────────────────────────────────────────────────────────────────

@st.cache_data
def load_csv(file_bytes, filename):
    return pd.read_csv(io.BytesIO(file_bytes))

@st.cache_data
def load_excel(file_bytes, filename):
    return pd.read_excel(io.BytesIO(file_bytes))

@st.cache_data
def load_json(file_bytes, filename):
    df = pd.read_json(io.BytesIO(file_bytes))
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df


def profile_dataset(df):
    """Compute profiling stats for the dataset."""
    profile = {}
    profile["shape"] = df.shape
    profile["dtypes"] = df.dtypes
    profile["missing"] = df.isnull().sum()
    profile["missing_pct"] = (df.isnull().sum() / len(df) * 100).round(2)
    profile["duplicates"] = df.duplicated().sum()
    profile["numeric_cols"] = df.select_dtypes(include="number").columns.tolist()
    profile["cat_cols"] = df.select_dtypes(include=["object", "category"]).columns.tolist()
    profile["dt_cols"] = df.select_dtypes(include=["datetime64"]).columns.tolist()
    return profile


def dtype_badge(dtype):
    dtype_str = str(dtype)
    if "int" in dtype_str or "float" in dtype_str:
        color, label = "#34d399", "NUM"
    elif "datetime" in dtype_str:
        color, label = "#a7f3d0", "DATE"
    elif "object" in dtype_str or "category" in dtype_str:
        color, label = "#fbbf24", "CAT"
    elif "bool" in dtype_str:
        color, label = "#34d399", "BOOL"
    else:
        color, label = "#6abf8a", "OTHER"
    return f"<span style='background:rgba({','.join(str(int(color.lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.15);color:{color};font-family:Space Mono,monospace;font-size:0.65rem;padding:2px 6px;border-radius:3px;border:1px solid {color}40;'>{label}</span>"


# ─── Main render ────────────────────────────────────────────────────────────

def render():
    st.markdown('<div class="step-badge">PAGE A</div>', unsafe_allow_html=True)
    st.title("Upload & Overview")
    st.markdown("<p style='color:#6abf8a;margin-top:-0.5rem;margin-bottom:1.5rem;'>Load your dataset and explore its structure before cleaning.</p>", unsafe_allow_html=True)

    # ── Upload section ───────────────────────────────────────────────────────
    col_up, col_sample = st.columns([3, 1])

    with col_up:
        uploaded = st.file_uploader(
            "Drop your dataset here",
            type=["csv", "xlsx", "xls", "json"],
            help="Supported: CSV, Excel (.xlsx/.xls), JSON"
        )

    with col_sample:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:Space Mono,monospace;font-size:0.7rem;color:#6abf8a;margin-bottom:6px;'>Or try a sample:</div>", unsafe_allow_html=True)
        sample_choice = st.selectbox("", ["— select —", "E-Commerce Sales", "HR Employee Data"], label_visibility="collapsed")
        load_sample = st.button("Load Sample", use_container_width=True)

    # Load data from upload or sample
    df_loaded = None
    fname = None

    if uploaded is not None:
        fname = uploaded.name
        try:
            raw = uploaded.read()
            ext = fname.split(".")[-1].lower()
            if ext == "csv":
                df_loaded = load_csv(raw, fname)
            elif ext in ("xlsx", "xls"):
                df_loaded = load_excel(raw, fname)
            elif ext == "json":
                df_loaded = load_json(raw, fname)
            st.session_state.filetype = ext
        except Exception as e:
            st.error(f"❌ Could not load file: {e}")

    elif load_sample and sample_choice != "— select —":
        try:
            path_map = {
                "E-Commerce Sales": "sample_data/ecommerce_sales.csv",
                "HR Employee Data": "sample_data/hr_employee_data.csv",
            }
            fname = path_map[sample_choice]
            df_loaded = pd.read_csv(fname)
            st.session_state.filetype = "csv"
        except Exception as e:
            st.error(f"❌ Could not load sample: {e}")

    # Commit to session state
    if df_loaded is not None:
        st.session_state.original_df = df_loaded.copy()
        st.session_state.working_df = df_loaded.copy()
        st.session_state.filename = fname.split("/")[-1] if "/" in str(fname) else fname
        st.session_state.transformation_log = []
        st.session_state.upload_ts = datetime.now().isoformat()
        st.success(f"✓ Loaded **{st.session_state.filename}** — {df_loaded.shape[0]:,} rows × {df_loaded.shape[1]} columns")

    # ── Overview ─────────────────────────────────────────────────────────────
    if st.session_state.working_df is None:
        st.markdown("""
        <div style='border:2px dashed #30363d;border-radius:12px;padding:60px;text-align:center;margin-top:2rem;'>
            <div style='font-size:3rem;margin-bottom:1rem;'>📂</div>
            <div style='font-family:Space Mono,monospace;color:#6abf8a;font-size:0.9rem;'>Upload a CSV, Excel, or JSON file to get started</div>
            <div style='color:#6abf8a;font-size:0.8rem;margin-top:0.5rem;'>Supports datasets with 1,000+ rows and mixed column types</div>
        </div>
        """, unsafe_allow_html=True)
        return

    df = st.session_state.working_df
    profile = profile_dataset(df)

    # ── KPI row ──────────────────────────────────────────────────────────────
    st.markdown("### Dataset Snapshot")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Rows", f"{df.shape[0]:,}")
    k2.metric("Columns", f"{df.shape[1]}")
    k3.metric("Missing Values", f"{profile['missing'].sum():,}")
    k4.metric("Duplicate Rows", f"{profile['duplicates']:,}")
    k5.metric("Numeric Cols", f"{len(profile['numeric_cols'])}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Column Inspector", "📊 Summary Stats", "🕳️ Missing Values", "👯 Duplicates"])

    with tab1:
        st.markdown("#### Column Inspector")
        rows = []
        for col in df.columns:
            dtype = df[col].dtype
            missing = df[col].isnull().sum()
            missing_pct = missing / len(df) * 100
            n_unique = df[col].nunique()
            sample_vals = ", ".join([str(v) for v in df[col].dropna().unique()[:3]])
            rows.append({
                "Column": col,
                "Type": str(dtype),
                "Missing": missing,
                "Missing %": f"{missing_pct:.1f}%",
                "Unique": n_unique,
                "Sample Values": sample_vals,
            })
        col_df = pd.DataFrame(rows)
        st.dataframe(col_df, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("#### Numeric Summary")
        num_df = df.select_dtypes(include="number")
        if not num_df.empty:
            desc = num_df.describe().T.round(3)
            desc.index.name = "Column"
            desc = desc.reset_index()
            st.dataframe(desc, use_container_width=True, hide_index=True)
        else:
            st.info("No numeric columns found.")

        st.markdown("#### Categorical Summary")
        cat_df = df.select_dtypes(include=["object", "category"])
        if not cat_df.empty:
            cat_rows = []
            for col in cat_df.columns:
                top_val = df[col].value_counts().index[0] if df[col].value_counts().shape[0] > 0 else "—"
                top_cnt = df[col].value_counts().iloc[0] if df[col].value_counts().shape[0] > 0 else 0
                cat_rows.append({
                    "Column": col,
                    "Unique Values": df[col].nunique(),
                    "Top Value": str(top_val),
                    "Top Count": top_cnt,
                    "Top %": f"{top_cnt/len(df)*100:.1f}%",
                })
            st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No categorical columns found.")

    with tab3:
        st.markdown("#### Missing Values by Column")
        missing_data = pd.DataFrame({
            "Column": df.columns,
            "Missing Count": profile["missing"].values,
            "Missing %": profile["missing_pct"].values,
        }).query("`Missing Count` > 0").sort_values("Missing %", ascending=False)

        if missing_data.empty:
            st.success("✅ No missing values detected!")
        else:
            # Bar chart
            fig, ax = plt.subplots(figsize=(10, max(3, len(missing_data) * 0.5)))
            fig.patch.set_facecolor('#112218')
            ax.set_facecolor('#112218')
            bars = ax.barh(missing_data["Column"], missing_data["Missing %"],
                          color='#34d399', alpha=0.8, edgecolor='none')
            for bar, val in zip(bars, missing_data["Missing %"]):
                ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                       f'{val:.1f}%', va='center', color='#6abf8a', fontsize=9)
            ax.set_xlabel("Missing %", color='#6abf8a')
            ax.tick_params(colors='#6abf8a')
            ax.spines[:].set_color('#1c3528')
            ax.set_xlim(0, max(missing_data["Missing %"]) * 1.2)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.dataframe(missing_data, use_container_width=True, hide_index=True)

    with tab4:
        st.markdown("#### Duplicate Row Analysis")
        n_dups = profile["duplicates"]
        if n_dups == 0:
            st.success("✅ No duplicate rows detected!")
        else:
            st.warning(f"⚠️ Found **{n_dups:,}** duplicate rows ({n_dups/len(df)*100:.1f}% of data)")
            dup_rows = df[df.duplicated(keep=False)]
            st.markdown(f"Showing duplicate rows (total {len(dup_rows):,} affected rows):")
            st.dataframe(dup_rows.head(50), use_container_width=True, hide_index=True)

    # ── Data preview ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🔍 Preview Working Dataset (first 100 rows)", expanded=False):
        st.dataframe(df.head(100), use_container_width=True)

    # ── Missing value heatmap ─────────────────────────────────────────────────
    if profile["missing"].sum() > 0:
        with st.expander("🗺️ Missingness Heatmap (sample of 200 rows)", expanded=False):
            sample = df.sample(min(200, len(df)), random_state=42)
            fig, ax = plt.subplots(figsize=(12, 4))
            fig.patch.set_facecolor('#112218')
            ax.set_facecolor('#112218')
            miss_matrix = sample.isnull().T.astype(int)
            ax.imshow(miss_matrix, aspect='auto', cmap='RdYlGn_r', interpolation='none', vmin=0, vmax=1)
            ax.set_yticks(range(len(df.columns)))
            ax.set_yticklabels(df.columns, color='#6abf8a', fontsize=8)
            ax.set_xlabel("Row index (sample)", color='#6abf8a')
            ax.tick_params(colors='#6abf8a')
            for spine in ax.spines.values():
                spine.set_color('#1c3528')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    col_nav1, col_nav2 = st.columns([4, 1])
    with col_nav2:
        if st.button("→ Go to Cleaning Studio", type="primary", use_container_width=True):
            st.session_state.current_page = "Cleaning & Prep Studio"
            st.rerun()
