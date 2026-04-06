"""
Page D: Export & Report
Export cleaned dataset, transformation report, and JSON recipe.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import io
from datetime import datetime


def render():
    st.markdown('<div class="step-badge">PAGE D</div>', unsafe_allow_html=True)
    st.title("Export & Report")
    st.markdown("<p style='color:#8b949e;margin-top:-0.5rem;margin-bottom:1.5rem;'>Download your cleaned dataset and a full record of every transformation applied.</p>", unsafe_allow_html=True)

    if st.session_state.working_df is None:
        st.warning("⚠️ No dataset loaded. Go to **Upload & Overview** first.")
        return

    df_orig = st.session_state.original_df
    df_work = st.session_state.working_df
    log = st.session_state.transformation_log
    fname = st.session_state.filename or "dataset"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Summary cards ─────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Original rows", f"{df_orig.shape[0]:,}")
    m2.metric("Cleaned rows", f"{df_work.shape[0]:,}")
    m3.metric("Original cols", f"{df_orig.shape[1]}")
    m4.metric("Cleaned cols", f"{df_work.shape[1]}")

    delta_rows = df_work.shape[0] - df_orig.shape[0]
    delta_cols = df_work.shape[1] - df_orig.shape[1]
    st.markdown(f"<div style='color:#8b949e;font-size:0.85rem;margin-top:-0.5rem;'>Row change: <span style='color:{'#39ff14' if delta_rows == 0 else '#ff6b35'};'>{delta_rows:+,}</span> · Column change: <span style='color:{'#39ff14' if delta_cols == 0 else '#a855f7'};'>{delta_cols:+,}</span> · {len(log)} transformation(s)</div>", unsafe_allow_html=True)

    st.divider()

    # ── Export tabs ───────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📁 Dataset", "📄 Transformation Report", "🔁 JSON Recipe"])

    # ─── Dataset export ───────────────────────────────────────────────────────
    with tab1:
        st.markdown("### Export Cleaned Dataset")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.dataframe(df_work.head(20), use_container_width=True)
            st.caption(f"Preview of first 20 rows. Full export: {df_work.shape[0]:,} rows × {df_work.shape[1]} columns.")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)

            # CSV export
            csv_bytes = df_work.to_csv(index=False).encode()
            st.download_button(
                "⬇ Download CSV",
                csv_bytes,
                f"{fname.rsplit('.',1)[0]}_cleaned_{ts}.csv",
                "text/csv",
                use_container_width=True,
                key="dl_csv",
                type="primary",
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # Excel export
            excel_buf = io.BytesIO()
            try:
                with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                    df_work.to_excel(writer, index=False, sheet_name="Cleaned Data")
                    if df_orig is not None:
                        df_orig.to_excel(writer, index=False, sheet_name="Original Data")
                excel_buf.seek(0)
                st.download_button(
                    "⬇ Download Excel",
                    excel_buf,
                    f"{fname.rsplit('.',1)[0]}_cleaned_{ts}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="dl_xlsx",
                )
            except Exception as e:
                st.warning(f"Excel export unavailable: {e}. Install openpyxl.")

    # ─── Transformation report ────────────────────────────────────────────────
    with tab2:
        st.markdown("### Transformation Report")

        if not log:
            st.info("No transformations have been applied yet.")
        else:
            # Build human-readable report
            report_lines = [
                "=" * 60,
                "  DATAWRANGLER PRO — TRANSFORMATION REPORT",
                "=" * 60,
                f"  Generated:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"  Source file:    {fname}",
                f"  Upload time:    {st.session_state.get('upload_ts', 'N/A')[:19]}",
                f"  Original shape: {df_orig.shape[0]:,} rows × {df_orig.shape[1]} cols",
                f"  Final shape:    {df_work.shape[0]:,} rows × {df_work.shape[1]} cols",
                f"  Steps applied:  {len(log)}",
                "",
                "─" * 60,
                "  STEPS",
                "─" * 60,
            ]

            for step in log:
                report_lines += [
                    "",
                    f"  [{step['step']:02d}] {step['operation']}",
                    f"       Timestamp:  {step['timestamp'][:19]}",
                    f"       Parameters: {json.dumps(step['params'], default=str)}",
                    f"       Affected:   {', '.join(str(c) for c in step['affected_columns'])}",
                    f"       Rows before:{step['rows_before']:,}",
                ]

            report_lines += [
                "",
                "─" * 60,
                "  VALIDATION RULES",
                "─" * 60,
            ]

            if st.session_state.validation_rules:
                for rule in st.session_state.validation_rules:
                    report_lines.append(f"  · {rule}")
            else:
                report_lines.append("  (no validation rules defined)")

            report_lines += [
                "",
                "─" * 60,
                "  COLUMN SUMMARY (final)",
                "─" * 60,
            ]
            for col in df_work.columns:
                miss = df_work[col].isnull().sum()
                report_lines.append(f"  {col:<30} {str(df_work[col].dtype):<12} missing: {miss}")

            report_lines += ["", "=" * 60]

            report_text = "\n".join(report_lines)
            st.code(report_text, language="text")

            st.download_button(
                "⬇ Download Report (.txt)",
                report_text.encode(),
                f"transformation_report_{ts}.txt",
                "text/plain",
                key="dl_report",
                type="primary",
            )

    # ─── JSON Recipe ──────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### JSON Recipe (Reproducible Pipeline)")
        st.markdown("<p style='color:#8b949e;font-size:0.85rem;'>This recipe records every transformation in a machine-readable format. It can be used to replay the exact same pipeline on new data.</p>", unsafe_allow_html=True)

        recipe = {
            "datawrangler_version": "1.0",
            "created_at": datetime.now().isoformat(),
            "source_file": fname,
            "original_shape": {"rows": df_orig.shape[0], "cols": df_orig.shape[1]},
            "final_shape": {"rows": df_work.shape[0], "cols": df_work.shape[1]},
            "steps": [
                {
                    "step": s["step"],
                    "operation": s["operation"],
                    "params": s["params"],
                    "affected_columns": s["affected_columns"],
                    "timestamp": s["timestamp"],
                }
                for s in log
            ],
            "validation_rules": st.session_state.validation_rules,
        }

        recipe_json = json.dumps(recipe, indent=2, default=str)
        st.code(recipe_json, language="json")

        st.download_button(
            "⬇ Download Recipe (.json)",
            recipe_json.encode(),
            f"pipeline_recipe_{ts}.json",
            "application/json",
            key="dl_recipe",
            type="primary",
        )

        # Python script snippet
        st.markdown("### Python Script Snippet")
        st.markdown("<p style='color:#8b949e;font-size:0.85rem;'>A Python/pandas script that approximately replays this pipeline.</p>", unsafe_allow_html=True)

        script_lines = [
            "import pandas as pd",
            "import numpy as np",
            "",
            f"# Generated by DataWrangler Pro — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"df = pd.read_csv('{fname}')",
            "",
            "# ── Transformations ──────────────────────────────────────",
        ]

        for s in log:
            op = s["operation"]
            params = s["params"]
            cols = s["affected_columns"]
            script_lines.append(f"\n# Step {s['step']}: {op}")
            if "Missing Values" in op:
                action = params.get("fill_action", op)
                if "Drop rows" in op:
                    script_lines.append(f"df = df.dropna(subset={cols})")
                elif "Constant" in op:
                    for c in cols:
                        script_lines.append(f"df['{c}'] = df['{c}'].fillna({repr(params.get('constant', 0))})")
                elif "Mean" in op:
                    for c in cols:
                        script_lines.append(f"df['{c}'] = df['{c}'].fillna(df['{c}'].mean())")
                elif "Median" in op:
                    for c in cols:
                        script_lines.append(f"df['{c}'] = df['{c}'].fillna(df['{c}'].median())")
                elif "Forward" in op:
                    for c in cols:
                        script_lines.append(f"df['{c}'] = df['{c}'].ffill()")
                elif "Backward" in op:
                    for c in cols:
                        script_lines.append(f"df['{c}'] = df['{c}'].bfill()")
            elif "Duplicate" in op:
                subset = params.get("subset")
                keep = params.get("keep", "first")
                subset_str = f"subset={subset}" if subset and subset != "all" else ""
                script_lines.append(f"df = df.drop_duplicates({subset_str}, keep='{keep}')")
            elif "Type Conversion" in op:
                for c in cols:
                    script_lines.append(f"df['{c}'] = pd.to_numeric(df['{c}'], errors='coerce')  # adjust as needed")
            elif "Scaling" in op or "Normalization" in op:
                method = params.get("method", op)
                for c in cols:
                    if "Min-Max" in method:
                        script_lines.append(f"df['{c}'] = (df['{c}'] - df['{c}'].min()) / (df['{c}'].max() - df['{c}'].min())")
                    elif "Z-Score" in method:
                        script_lines.append(f"df['{c}'] = (df['{c}'] - df['{c}'].mean()) / df['{c}'].std()")
            elif "Rename" in op:
                script_lines.append(f"df = df.rename(columns={{{repr(params.get('from', ''))}: {repr(params.get('to', ''))}}})")
            elif "Drop Columns" in op:
                script_lines.append(f"df = df.drop(columns={cols})")
            elif "Create Column" in op:
                script_lines.append(f"# df['{cols[0]}'] = {params.get('formula', '...')}  # review formula")
            elif "Binning" in op:
                col = params.get("column")
                n = params.get("n_bins", 5)
                script_lines.append(f"df['{params.get('new_col', col+'_bin')}'] = pd.cut(df['{col}'], bins={n}, labels=False)")
            elif "Encoding" in op:
                script_lines.append(f"df = pd.get_dummies(df, columns={cols}, drop_first={params.get('drop_first', False)}, dtype=int)")
            else:
                script_lines.append(f"# {op}: params={params}")

        script_lines += [
            "",
            "# ── Export ───────────────────────────────────────────────",
            f"df.to_csv('cleaned_output.csv', index=False)",
            "print('Done. Shape:', df.shape)",
        ]

        script_text = "\n".join(script_lines)
        st.code(script_text, language="python")
        st.download_button(
            "⬇ Download Python Script (.py)",
            script_text.encode(),
            f"pipeline_script_{ts}.py",
            "text/x-python",
            key="dl_script",
        )
