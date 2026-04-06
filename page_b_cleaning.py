"""
Page B: Cleaning & Preparation Studio
All transformations act on the working copy of the dataset.
Personal touch: live Before/After preview before committing any change.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# ─── Palette ────────────────────────────────────────────────────────────────
CYAN   = "#00d4ff"
GREEN  = "#39ff14"
ORANGE = "#ff6b35"
BG     = "#1c2128"
BG2    = "#0d1117"
TEXT2  = "#8b949e"
BORDER = "#30363d"

# ─── Utility ────────────────────────────────────────────────────────────────

def log_step(operation: str, params: dict, affected_cols: list):
    entry = {
        "step": len(st.session_state.transformation_log) + 1,
        "operation": operation,
        "params": params,
        "affected_columns": affected_cols,
        "timestamp": datetime.now().isoformat(),
        "rows_before": len(st.session_state.working_df),
    }
    st.session_state.transformation_log.append(entry)


def guard():
    if st.session_state.working_df is None:
        st.warning("No dataset loaded. Go to Upload & Overview first.")
        return True
    return False


def _stat_chip(label, before, after):
    delta = after - before
    if delta < 0:
        delta_color, arrow = GREEN, "▼"
    elif delta > 0:
        delta_color, arrow = ORANGE, "▲"
    else:
        delta_color, arrow = TEXT2, "="
    return f"""
      <div style='min-width:110px;'>
        <div style='font-size:0.65rem;color:{TEXT2};text-transform:uppercase;
                    letter-spacing:0.1em;margin-bottom:2px;'>{label}</div>
        <div style='font-size:1.15rem;font-family:Space Mono,monospace;color:{CYAN};'>{after:,}</div>
        <div style='font-size:0.75rem;color:{delta_color};'>
          {arrow} {abs(delta):,} (was {before:,})
        </div>
      </div>"""


def preview_panel(df_before: pd.DataFrame, df_after: pd.DataFrame,
                  affected_cols: list, label: str = ""):
    """
    Rich Before/After comparison panel shown BEFORE user commits a transform.
    Shows: row/col/missing deltas + per-column stat diff + side-by-side sample rows.
    """
    miss_before = int(df_before.isnull().sum().sum())
    miss_after  = int(df_after.isnull().sum().sum())

    st.markdown(f"""
    <div style='background:{BG};border:1px solid {BORDER};border-radius:10px;
                padding:14px 18px;margin:12px 0 8px 0;'>
      <div style='font-family:Space Mono,monospace;font-size:0.68rem;
                  color:{TEXT2};text-transform:uppercase;letter-spacing:0.12em;
                  margin-bottom:12px;'>
        ⚡ Live Preview &nbsp;·&nbsp;
        <span style='color:{CYAN};'>{label}</span>
      </div>
      <div style='display:flex;gap:36px;flex-wrap:wrap;'>
        {_stat_chip("Rows",    df_before.shape[0], df_after.shape[0])}
        {_stat_chip("Columns", df_before.shape[1], df_after.shape[1])}
        {_stat_chip("Missing", miss_before,        miss_after)}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Per-column stats diff
    common = [c for c in affected_cols
              if c in df_before.columns and c in df_after.columns]
    if common:
        rows = []
        for col in common:
            b = df_before[col]
            a = df_after[col]
            is_num = pd.api.types.is_numeric_dtype(b) and pd.api.types.is_numeric_dtype(a)
            row = {
                "Column":         col,
                "Before dtype":   str(b.dtype),
                "After dtype":    str(a.dtype),
                "Before missing": int(b.isnull().sum()),
                "After missing":  int(a.isnull().sum()),
            }
            if is_num:
                row["Before mean"] = round(float(b.mean()), 4)
                row["After mean"]  = round(float(a.mean()), 4)
            else:
                row["Before top"] = str(b.mode().iloc[0]) if not b.mode().empty else "—"
                row["After top"]  = str(a.mode().iloc[0]) if not a.mode().empty else "—"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Side-by-side sample rows
    c1, c2 = st.columns(2)
    sample_b = [c for c in (affected_cols + list(df_before.columns)) if c in df_before.columns][:6]
    sample_a = [c for c in (affected_cols + list(df_after.columns))  if c in df_after.columns][:6]
    with c1:
        st.markdown(f"<div style='font-family:Space Mono,monospace;font-size:0.68rem;"
                    f"color:{ORANGE};margin-bottom:4px;'>◀ BEFORE (5 rows)</div>",
                    unsafe_allow_html=True)
        st.dataframe(df_before[sample_b].head(5), use_container_width=True, hide_index=True)
    with c2:
        st.markdown(f"<div style='font-family:Space Mono,monospace;font-size:0.68rem;"
                    f"color:{GREEN};margin-bottom:4px;'>AFTER ▶ (5 rows)</div>",
                    unsafe_allow_html=True)
        st.dataframe(df_after[sample_a].head(5), use_container_width=True, hide_index=True)


def simulate_mv(df, action, target_cols, extra):
    """Simulate missing value action on a copy."""
    wdf = df.copy()
    for col in target_cols:
        if col not in wdf.columns:
            continue
        if action == "Drop rows with missing":
            wdf = wdf.dropna(subset=[col])
        elif action == "Drop column if missing% above threshold":
            pct = wdf[col].isnull().mean() * 100
            if pct >= extra.get("threshold", 50):
                wdf = wdf.drop(columns=[col])
        elif action == "Fill → Constant value":
            val = extra.get("constant", "0")
            try:
                val = float(val) if wdf[col].dtype.kind in "iufcb" else val
            except Exception:
                pass
            wdf[col] = wdf[col].fillna(val)
        elif action == "Fill → Mean (numeric)":
            if wdf[col].dtype.kind in "iufcb":
                wdf[col] = wdf[col].fillna(wdf[col].mean())
        elif action == "Fill → Median (numeric)":
            if wdf[col].dtype.kind in "iufcb":
                wdf[col] = wdf[col].fillna(wdf[col].median())
        elif action == "Fill → Mode / Most frequent":
            mode_val = wdf[col].mode()
            if not mode_val.empty:
                wdf[col] = wdf[col].fillna(mode_val.iloc[0])
        elif action == "Fill → Forward fill":
            wdf[col] = wdf[col].ffill()
        elif action == "Fill → Backward fill":
            wdf[col] = wdf[col].bfill()
    return wdf


# ─── Main render ────────────────────────────────────────────────────────────

def render():
    st.markdown('<div class="step-badge">PAGE B</div>', unsafe_allow_html=True)
    st.title("Cleaning & Prep Studio")
    st.markdown(
        f"<p style='color:{TEXT2};margin-top:-0.5rem;margin-bottom:1rem;'>"
        f"All operations apply to a <b>working copy</b>. The original is always preserved. "
        f"<span style='color:{CYAN};font-weight:600;'>"
        f"Every action shows a live Before/After preview before you commit.</span></p>",
        unsafe_allow_html=True,
    )

    if guard():
        return

    df = st.session_state.working_df

    # ── Controls row ──────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([3, 1, 1])
    with ctrl1:
        st.markdown(
            f"<div style='color:{TEXT2};font-size:0.85rem;padding-top:8px;'>"
            f"{len(st.session_state.transformation_log)} transformation(s) applied · "
            f"{df.shape[0]:,} rows × {df.shape[1]} cols</div>",
            unsafe_allow_html=True,
        )
    with ctrl2:
        if st.button("↩ Undo Last", use_container_width=True):
            if st.session_state.transformation_log:
                st.session_state.transformation_log.pop()
                st.session_state.working_df = st.session_state.original_df.copy()
                st.success("Last step removed. Working data reset to original.")
                st.rerun()
            else:
                st.info("Nothing to undo.")
    with ctrl3:
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.working_df = st.session_state.original_df.copy()
            st.session_state.transformation_log = []
            st.rerun()

    st.divider()

    tabs = st.tabs([
        "🕳 Missing Values",
        "👯 Duplicates",
        "🔠 Data Types",
        "🏷 Categorical",
        "📐 Numeric",
        "📏 Normalization",
        "🔧 Column Ops",
        "✅ Validation",
        "📜 Log",
    ])

    # ═══ TAB 0 — MISSING VALUES ══════════════════════════════════════════════
    with tabs[0]:
        st.markdown("### Missing Value Handler")
        df = st.session_state.working_df
        missing     = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        miss_df = pd.DataFrame({
            "Column":    df.columns,
            "Missing":   missing.values,
            "Missing %": missing_pct.values,
        }).query("Missing > 0")

        if miss_df.empty:
            st.success("No missing values in the working dataset!")
        else:
            st.dataframe(miss_df, use_container_width=True, hide_index=True)

        cols_with_missing = miss_df["Column"].tolist() if not miss_df.empty else []
        mv_col, mv_action = st.columns([2, 2])
        with mv_col:
            target_cols = st.multiselect(
                "Target columns", df.columns.tolist(),
                default=cols_with_missing[:3] if cols_with_missing else [],
                key="mv_cols",
            )
        with mv_action:
            action = st.selectbox("Action", [
                "Drop rows with missing",
                "Drop column if missing% above threshold",
                "Fill → Constant value",
                "Fill → Mean (numeric)",
                "Fill → Median (numeric)",
                "Fill → Mode / Most frequent",
                "Fill → Forward fill",
                "Fill → Backward fill",
            ], key="mv_action")

        extra = {}
        if action == "Drop column if missing% above threshold":
            extra["threshold"] = st.slider("Threshold (%)", 0, 100, 50, key="mv_threshold")
        if action == "Fill → Constant value":
            extra["constant"] = st.text_input("Constant value", value="0", key="mv_const")

        if target_cols:
            try:
                df_preview = simulate_mv(df, action, target_cols, extra)
                preview_panel(df, df_preview, target_cols, action)
            except Exception as e:
                st.caption(f"Preview unavailable: {e}")

            if st.button("✓ Confirm & Apply", key="mv_apply", type="primary"):
                try:
                    wdf = simulate_mv(df, action, target_cols, extra)
                    affected = [c for c in target_cols if c in df.columns]
                    st.session_state.working_df = wdf
                    log_step(f"Missing Values: {action}", {**extra, "columns": target_cols}, affected)
                    st.success(f"Applied to {len(affected)} column(s).")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    # ═══ TAB 1 — DUPLICATES ══════════════════════════════════════════════════
    with tabs[1]:
        st.markdown("### Duplicate Handler")
        df = st.session_state.working_df
        dup_c1, dup_c2 = st.columns([2, 2])
        with dup_c1:
            dup_subset = st.multiselect("Detect by (empty = all cols)", df.columns.tolist(), key="dup_subset")
        with dup_c2:
            dup_keep = st.selectbox("Keep", ["first", "last", "none (drop all)"], key="dup_keep")

        subset_arg = dup_subset if dup_subset else None
        keep_arg   = False if dup_keep == "none (drop all)" else dup_keep
        n_dups = int(df.duplicated(subset=subset_arg, keep=keep_arg).sum())
        st.markdown(f"**Detected:** {n_dups:,} duplicate rows")

        if n_dups > 0:
            df_preview = df.drop_duplicates(subset=subset_arg, keep=keep_arg)
            preview_panel(df, df_preview, dup_subset or df.columns.tolist(),
                          f"Remove duplicates (keep={dup_keep})")
            with st.expander("View duplicate groups (up to 50 rows)"):
                st.dataframe(df[df.duplicated(subset=subset_arg, keep=False)].head(50),
                             use_container_width=True)
            if st.button("✓ Confirm & Remove", type="primary", key="dup_remove"):
                try:
                    removed = len(df) - len(df_preview)
                    st.session_state.working_df = df_preview
                    log_step("Remove Duplicates", {"subset": subset_arg or "all", "keep": dup_keep},
                             subset_arg or df.columns.tolist())
                    st.success(f"Removed {removed:,} rows.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.success("No duplicates with current settings.")

    # ═══ TAB 2 — DATA TYPES ══════════════════════════════════════════════════
    with tabs[2]:
        st.markdown("### Data Type Conversion")
        df = st.session_state.working_df
        st.dataframe(pd.DataFrame({"Column": df.columns, "Type": df.dtypes.astype(str).values,
                                   "Nulls": df.isnull().sum().values}),
                     use_container_width=True, hide_index=True)

        dt_c1, dt_c2 = st.columns([2, 2])
        with dt_c1:
            dt_target = st.selectbox("Column", df.columns.tolist(), key="dt_col")
        with dt_c2:
            dt_type = st.selectbox("Convert to", [
                "numeric (float)", "integer", "string / categorical", "datetime", "boolean",
            ], key="dt_type")

        dt_fmt = ""
        if dt_type == "datetime":
            dt_fmt = st.text_input("Format (blank = auto)", placeholder="%Y-%m-%d", key="dt_fmt")
        if dt_type == "numeric (float)":
            st.caption("Strips commas and currency symbols ($€£) before converting.")

        try:
            wdf_prev = df.copy()
            col = dt_target
            if dt_type == "numeric (float)":
                wdf_prev[col] = pd.to_numeric(
                    wdf_prev[col].astype(str).str.replace(r'[$€£,\s]', '', regex=True), errors='coerce')
            elif dt_type == "integer":
                wdf_prev[col] = pd.to_numeric(wdf_prev[col], errors='coerce').astype("Int64")
            elif dt_type == "string / categorical":
                wdf_prev[col] = wdf_prev[col].astype(str)
            elif dt_type == "datetime":
                wdf_prev[col] = pd.to_datetime(wdf_prev[col], format=dt_fmt or None,
                                               errors='coerce', infer_datetime_format=True)
            elif dt_type == "boolean":
                wdf_prev[col] = wdf_prev[col].astype(bool)
            preview_panel(df, wdf_prev, [col], f"Type → {dt_type}")
        except Exception as e:
            st.caption(f"Preview unavailable: {e}")
            wdf_prev = None

        if st.button("✓ Confirm & Convert", type="primary", key="dt_apply") and wdf_prev is not None:
            st.session_state.working_df = wdf_prev
            log_step(f"Type Conversion: {dt_target} → {dt_type}", {"format": dt_fmt}, [dt_target])
            st.success(f"Converted **{dt_target}** to {dt_type}.")
            st.rerun()

    # ═══ TAB 3 — CATEGORICAL ═════════════════════════════════════════════════
    with tabs[3]:
        st.markdown("### Categorical Data Tools")
        df = st.session_state.working_df
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        if not cat_cols:
            st.info("No categorical columns in current dataset.")
        else:
            cat_sub1, cat_sub2, cat_sub3 = st.tabs(["Standardize", "Value Mapping", "Encoding"])

            with cat_sub1:
                std_cols = st.multiselect("Columns", cat_cols, key="std_cols")
                std_ops  = st.multiselect("Operations", [
                    "Trim whitespace", "Lowercase", "Uppercase", "Title Case",
                    "Group rare values into 'Other'",
                ], default=["Trim whitespace", "Lowercase"], key="std_ops")
                rare_thresh = 1.0
                if "Group rare values into 'Other'" in std_ops:
                    rare_thresh = st.slider("Rare threshold (%)", 0.1, 10.0, 1.0, 0.1, key="rare_thresh")

                if std_cols:
                    try:
                        wdf_prev = df.copy()
                        for col in std_cols:
                            if col not in wdf_prev.columns: continue
                            if "Trim whitespace" in std_ops: wdf_prev[col] = wdf_prev[col].astype(str).str.strip()
                            if "Lowercase"      in std_ops: wdf_prev[col] = wdf_prev[col].str.lower()
                            if "Uppercase"      in std_ops: wdf_prev[col] = wdf_prev[col].str.upper()
                            if "Title Case"     in std_ops: wdf_prev[col] = wdf_prev[col].str.title()
                            if "Group rare values into 'Other'" in std_ops:
                                freq = wdf_prev[col].value_counts(normalize=True)
                                rare = freq[freq < rare_thresh / 100].index
                                wdf_prev[col] = wdf_prev[col].apply(lambda x: "Other" if x in rare else x)
                        preview_panel(df, wdf_prev, std_cols, "Standardization")
                    except Exception as e:
                        st.caption(f"Preview unavailable: {e}")
                        wdf_prev = None

                    if st.button("✓ Confirm & Apply", type="primary", key="std_apply") and wdf_prev is not None:
                        st.session_state.working_df = wdf_prev
                        log_step("Categorical Standardization", {"ops": std_ops, "rare_thresh": rare_thresh}, std_cols)
                        st.success(f"Standardized {len(std_cols)} column(s).")
                        st.rerun()

            with cat_sub2:
                map_col = st.selectbox("Column", cat_cols, key="map_col")
                if map_col:
                    unique_vals = df[map_col].dropna().unique().tolist()[:30]
                    st.caption(f"Unique values: `{'`, `'.join([str(v) for v in unique_vals])}`")
                    n_rows = st.number_input("Mapping rows", 1, 20, min(5, len(unique_vals)), key="map_n")
                    mapping_data = {}
                    for i in range(int(n_rows)):
                        c1, c2 = st.columns(2)
                        old_v = c1.text_input(f"From [{i+1}]", key=f"map_from_{i}",
                                              value=str(unique_vals[i]) if i < len(unique_vals) else "")
                        new_v = c2.text_input(f"To [{i+1}]",   key=f"map_to_{i}",
                                              value=str(unique_vals[i]) if i < len(unique_vals) else "")
                        if old_v: mapping_data[old_v] = new_v

                    unmatched = st.selectbox("Unmatched values", ["Keep unchanged", "Set to 'Other'"], key="map_unmatched")
                    try:
                        wdf_prev = df.copy()
                        if unmatched == "Set to 'Other'":
                            wdf_prev[map_col] = wdf_prev[map_col].map(mapping_data).fillna("Other")
                        else:
                            wdf_prev[map_col] = wdf_prev[map_col].map(lambda x: mapping_data.get(str(x), x))
                        preview_panel(df, wdf_prev, [map_col], "Value Mapping")
                    except Exception as e:
                        st.caption(f"Preview unavailable: {e}")
                        wdf_prev = None

                    if st.button("✓ Confirm & Map", type="primary", key="map_apply") and wdf_prev is not None:
                        st.session_state.working_df = wdf_prev
                        log_step("Value Mapping", {"column": map_col, "mapping": mapping_data}, [map_col])
                        st.success(f"Applied {len(mapping_data)} mappings.")
                        st.rerun()

            with cat_sub3:
                enc_cols       = st.multiselect("Columns to encode", cat_cols, key="enc_cols")
                enc_drop_first = st.checkbox("Drop first column", value=False, key="enc_drop")
                if enc_cols:
                    try:
                        wdf_prev = pd.get_dummies(df.copy(), columns=enc_cols,
                                                  drop_first=enc_drop_first, dtype=int)
                        preview_panel(df, wdf_prev, enc_cols, "One-Hot Encoding")
                    except Exception as e:
                        st.caption(f"Preview unavailable: {e}")
                        wdf_prev = None

                    if st.button("✓ Confirm & Encode", type="primary", key="enc_apply") and wdf_prev is not None:
                        st.session_state.working_df = wdf_prev
                        log_step("One-Hot Encoding", {"columns": enc_cols, "drop_first": enc_drop_first}, enc_cols)
                        st.success(f"Encoded. Shape now: {wdf_prev.shape}")
                        st.rerun()

    # ═══ TAB 4 — NUMERIC ═════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown("### Numeric Cleaning & Outlier Handling")
        df = st.session_state.working_df
        num_cols = df.select_dtypes(include="number").columns.tolist()

        if not num_cols:
            st.info("No numeric columns found.")
        else:
            out_col    = st.selectbox("Column", num_cols, key="out_col")
            out_method = st.selectbox("Detection method", [
                "IQR (1.5×)", "IQR (3.0×)", "Z-score (2σ)", "Z-score (3σ)"
            ], key="out_method")

            col_data = df[out_col].dropna()
            if "IQR" in out_method:
                mult = 1.5 if "1.5" in out_method else 3.0
                Q1, Q3 = col_data.quantile(0.25), col_data.quantile(0.75)
                lower, upper = Q1 - mult * (Q3 - Q1), Q3 + mult * (Q3 - Q1)
            else:
                sigma = 2 if "2σ" in out_method else 3
                lower = col_data.mean() - sigma * col_data.std()
                upper = col_data.mean() + sigma * col_data.std()

            n_out = int(((df[out_col] < lower) | (df[out_col] > upper)).sum())
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Outliers",    f"{n_out:,}")
            m2.metric("Lower bound", f"{lower:.3f}")
            m3.metric("Upper bound", f"{upper:.3f}")
            m4.metric("% of data",   f"{n_out/max(len(df),1)*100:.1f}%")

            out_action = st.selectbox("Action", [
                "Cap / Winsorize at bounds", "Remove outlier rows", "Do nothing (inspect only)"
            ], key="out_action")

            wdf_prev = None
            if out_action != "Do nothing (inspect only)" and n_out > 0:
                try:
                    wdf_prev = df.copy()
                    if out_action == "Cap / Winsorize at bounds":
                        wdf_prev[out_col] = wdf_prev[out_col].clip(lower=lower, upper=upper)
                    else:
                        wdf_prev = wdf_prev[~((wdf_prev[out_col] < lower) | (wdf_prev[out_col] > upper))]
                    preview_panel(df, wdf_prev, [out_col], out_action)
                except Exception as e:
                    st.caption(f"Preview unavailable: {e}")

            with st.expander("📈 Distribution Preview"):
                fig, axes = plt.subplots(1, 2, figsize=(12, 3))
                fig.patch.set_facecolor(BG2)
                for ax in axes:
                    ax.set_facecolor(BG); ax.tick_params(colors=TEXT2)
                    for sp in ax.spines.values(): sp.set_color(BORDER)
                axes[0].hist(col_data, bins=40, color=CYAN, alpha=0.7, edgecolor='none')
                axes[0].axvline(lower, color=ORANGE, linestyle='--', lw=1.2, label=f'Lower {lower:.1f}')
                axes[0].axvline(upper, color=ORANGE, linestyle='--', lw=1.2, label=f'Upper {upper:.1f}')
                axes[0].legend(fontsize=8, labelcolor=TEXT2)
                axes[0].set_title(f"{out_col} — Histogram", color='#e6edf3', fontsize=10)
                bp = axes[1].boxplot(col_data, patch_artist=True,
                                     medianprops=dict(color=GREEN, linewidth=2),
                                     whiskerprops=dict(color=TEXT2), capprops=dict(color=TEXT2),
                                     flierprops=dict(marker='o', color=ORANGE, alpha=0.4, markersize=4))
                bp['boxes'][0].set_facecolor(CYAN); bp['boxes'][0].set_alpha(0.3)
                axes[1].set_title(f"{out_col} — Boxplot", color='#e6edf3', fontsize=10)
                plt.tight_layout(); st.pyplot(fig); plt.close()

            if st.button("✓ Confirm & Apply", type="primary", key="out_apply") and wdf_prev is not None:
                st.session_state.working_df = wdf_prev
                log_step(f"Outlier: {out_action}",
                         {"column": out_col, "method": out_method, "lower": lower, "upper": upper},
                         [out_col])
                st.success(f"Applied: {out_action}.")
                st.rerun()

    # ═══ TAB 5 — NORMALIZATION ═══════════════════════════════════════════════
    with tabs[5]:
        st.markdown("### Normalization & Scaling")
        df = st.session_state.working_df
        num_cols = df.select_dtypes(include="number").columns.tolist()

        if not num_cols:
            st.info("No numeric columns available.")
        else:
            scale_cols   = st.multiselect("Columns to scale", num_cols, key="scale_cols")
            scale_method = st.selectbox("Method", [
                "Min-Max (0–1)", "Z-Score Standardization",
                "Robust Scaler (median/IQR)", "Log Transform (log1p)",
            ], key="scale_method")

            wdf_prev = None
            if scale_cols:
                try:
                    wdf_prev = df.copy()
                    for col in scale_cols:
                        s = wdf_prev[col].dropna()
                        if scale_method == "Min-Max (0–1)":
                            mn, mx = s.min(), s.max()
                            wdf_prev[col] = (wdf_prev[col] - mn) / (mx - mn) if mx != mn else 0
                        elif scale_method == "Z-Score Standardization":
                            mean, std = s.mean(), s.std()
                            wdf_prev[col] = (wdf_prev[col] - mean) / std if std != 0 else 0
                        elif scale_method == "Robust Scaler (median/IQR)":
                            med = s.median()
                            iqr = s.quantile(0.75) - s.quantile(0.25)
                            wdf_prev[col] = (wdf_prev[col] - med) / iqr if iqr != 0 else 0
                        elif scale_method == "Log Transform (log1p)":
                            wdf_prev[col] = np.log1p(wdf_prev[col].clip(lower=0))
                    preview_panel(df, wdf_prev, scale_cols, scale_method)
                except Exception as e:
                    st.caption(f"Preview unavailable: {e}")
                    wdf_prev = None

            if st.button("✓ Confirm & Scale", type="primary", key="scale_apply") and wdf_prev is not None:
                st.session_state.working_df = wdf_prev
                log_step(f"Scaling: {scale_method}", {"columns": scale_cols}, scale_cols)
                st.success(f"Scaled {len(scale_cols)} column(s).")
                st.rerun()

    # ═══ TAB 6 — COLUMN OPS ══════════════════════════════════════════════════
    with tabs[6]:
        st.markdown("### Column Operations")
        df = st.session_state.working_df
        col_sub1, col_sub2, col_sub3 = st.tabs(["Rename / Drop", "Create Column", "Binning"])

        with col_sub1:
            st.markdown("#### Rename Column")
            rn_c1, rn_c2 = st.columns(2)
            rn_src = rn_c1.selectbox("Column to rename", df.columns.tolist(), key="rn_src")
            rn_new = rn_c2.text_input("New name", value=rn_src or "", key="rn_new")
            if rn_src and rn_new and rn_new != rn_src:
                wdf_prev = df.rename(columns={rn_src: rn_new})
                preview_panel(df, wdf_prev, [rn_src], f"Rename '{rn_src}' → '{rn_new}'")
                if st.button("✓ Confirm Rename", type="primary", key="rn_apply"):
                    st.session_state.working_df = wdf_prev
                    log_step("Rename Column", {"from": rn_src, "to": rn_new}, [rn_src])
                    st.success(f"Renamed `{rn_src}` → `{rn_new}`.")
                    st.rerun()

            st.markdown("#### Drop Columns")
            drop_cols = st.multiselect("Columns to drop", df.columns.tolist(), key="drop_cols")
            if drop_cols:
                wdf_prev = df.drop(columns=drop_cols)
                preview_panel(df, wdf_prev, drop_cols, f"Drop {drop_cols}")
                if st.button("✓ Confirm Drop", type="primary", key="drop_apply"):
                    st.session_state.working_df = wdf_prev
                    log_step("Drop Columns", {"columns": drop_cols}, drop_cols)
                    st.success(f"Dropped {len(drop_cols)} column(s).")
                    st.rerun()

        with col_sub2:
            st.markdown("#### Create Derived Column")
            new_col_name = st.text_input("New column name", placeholder="revenue_log", key="new_col_name")
            st.caption("Use column names directly. Functions: `log`, `sqrt`, `abs`. "
                       "E.g. `quantity * unit_price`, `log(unit_price + 1)`")
            formula  = st.text_area("Formula", placeholder="quantity * unit_price", key="formula", height=80)
            wdf_prev = None
            if new_col_name and formula:
                try:
                    wdf_prev = df.copy()
                    env = {c: wdf_prev[c] for c in wdf_prev.columns}
                    env.update({"log": np.log1p, "sqrt": np.sqrt, "abs": np.abs,
                                "round": np.round, "np": np, "pd": pd})
                    wdf_prev[new_col_name] = eval(formula, {"__builtins__": {}}, env)
                    preview_panel(df, wdf_prev, [new_col_name], f"Create '{new_col_name}'")
                except Exception as e:
                    st.warning(f"Formula preview error: {e}")
                    wdf_prev = None

            if st.button("✓ Confirm Create", type="primary", key="new_col_apply") and wdf_prev is not None:
                st.session_state.working_df = wdf_prev
                log_step("Create Column", {"name": new_col_name, "formula": formula}, [new_col_name])
                st.success(f"Created column `{new_col_name}`.")
                st.rerun()

        with col_sub3:
            df = st.session_state.working_df
            num_cols_b = df.select_dtypes(include="number").columns.tolist()
            bin_col    = st.selectbox("Column to bin", num_cols_b, key="bin_col")
            bin_name   = st.text_input("New column name", value=f"{bin_col}_bin" if bin_col else "", key="bin_name")
            bin_method = st.radio("Method", ["Equal-width", "Quantile"], horizontal=True, key="bin_method")
            n_bins     = st.slider("Bins", 2, 20, 5, key="n_bins")
            wdf_prev   = None
            if bin_col and bin_name:
                try:
                    wdf_prev = df.copy()
                    if bin_method == "Equal-width":
                        wdf_prev[bin_name] = pd.cut(wdf_prev[bin_col], bins=n_bins, labels=False).astype("Int64").astype(str)
                    else:
                        wdf_prev[bin_name] = pd.qcut(wdf_prev[bin_col], q=n_bins, labels=False, duplicates='drop').astype("Int64").astype(str)
                    preview_panel(df, wdf_prev, [bin_col, bin_name], f"Bin '{bin_col}' → '{bin_name}'")
                except Exception as e:
                    st.caption(f"Preview unavailable: {e}")
                    wdf_prev = None

            if st.button("✓ Confirm Binning", type="primary", key="bin_apply") and wdf_prev is not None:
                st.session_state.working_df = wdf_prev
                log_step(f"Binning: {bin_method}", {"column": bin_col, "n_bins": n_bins, "new_col": bin_name}, [bin_col])
                st.success(f"Created `{bin_name}`.")
                st.rerun()

    # ═══ TAB 7 — VALIDATION ══════════════════════════════════════════════════
    with tabs[7]:
        st.markdown("### Data Validation Rules")
        df = st.session_state.working_df
        val_type = st.selectbox("Rule type", [
            "Numeric range check (min/max)", "Allowed categories list", "Non-null constraint",
        ], key="val_type")
        val_col    = st.selectbox("Column", df.columns.tolist(), key="val_col")
        val_params = {}

        if val_type == "Numeric range check (min/max)":
            v1, v2 = st.columns(2)
            val_params["min"] = v1.number_input("Min", value=float(df[val_col].min()) if pd.api.types.is_numeric_dtype(df[val_col]) else 0.0, key="val_min")
            val_params["max"] = v2.number_input("Max", value=float(df[val_col].max()) if pd.api.types.is_numeric_dtype(df[val_col]) else 100.0, key="val_max")
        elif val_type == "Allowed categories list":
            unique_vals        = df[val_col].dropna().unique().tolist()
            val_params["allowed"] = st.multiselect("Allowed values", unique_vals, default=unique_vals[:5], key="val_allowed")
        else:
            st.info(f"Will flag rows where `{val_col}` is null.")

        if st.button("Check Violations", type="primary", key="val_check"):
            try:
                if val_type == "Numeric range check (min/max)":
                    mask = (df[val_col] < val_params["min"]) | (df[val_col] > val_params["max"])
                elif val_type == "Allowed categories list":
                    mask = ~df[val_col].isin(val_params["allowed"]) & df[val_col].notna()
                else:
                    mask = df[val_col].isnull()
                violations = df[mask]
                n_viol = len(violations)
                if n_viol == 0:
                    st.success("No violations found!")
                else:
                    st.warning(f"{n_viol:,} violations ({n_viol/len(df)*100:.1f}%)")
                    st.dataframe(violations.head(100), use_container_width=True)
                    st.download_button("⬇ Download Violations CSV",
                                       violations.to_csv(index=False).encode(),
                                       "violations.csv", "text/csv", key="dl_viol")
                    rule = {"type": val_type, "column": val_col, **val_params, "violations": n_viol}
                    if rule not in st.session_state.validation_rules:
                        st.session_state.validation_rules.append(rule)
            except Exception as e:
                st.error(f"Validation error: {e}")

    # ═══ TAB 8 — LOG ═════════════════════════════════════════════════════════
    with tabs[8]:
        st.markdown("### Transformation Log")
        if not st.session_state.transformation_log:
            st.info("No transformations applied yet.")
        else:
            for step in reversed(st.session_state.transformation_log):
                with st.expander(f"Step {step['step']}: {step['operation']} — {step['timestamp'][:19]}"):
                    st.json({
                        "operation":        step["operation"],
                        "parameters":       step["params"],
                        "affected_columns": step["affected_columns"],
                        "timestamp":        step["timestamp"],
                        "rows_before":      step["rows_before"],
                    })

    st.markdown("<br>", unsafe_allow_html=True)
    _, nav_col = st.columns([4, 1])
    with nav_col:
        if st.button("→ Visualization", type="primary", use_container_width=True):
            st.session_state.current_page = "Visualization Builder"
            st.rerun()
