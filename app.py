"""
AI-Assisted Data Wrangler & Visualizer
Main application entry point — single-page app with custom sidebar navigation.
No pages/ folder used so Streamlit does not create automatic tab navigation.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataWrangler Pro",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary: #0d1117;
    --bg-secondary: #161b22;
    --bg-card: #1c2128;
    --accent-cyan: #00d4ff;
    --accent-green: #39ff14;
    --accent-orange: #ff6b35;
    --accent-purple: #a855f7;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --border: #30363d;
}

.stApp {
    background: var(--bg-primary);
    color: var(--text-primary);
    font-family: 'DM Sans', sans-serif;
}

section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    color: var(--text-primary) !important;
}

div[data-testid="metric-container"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    border-left: 3px solid var(--accent-cyan);
}
div[data-testid="metric-container"] label {
    color: var(--text-secondary) !important;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--accent-cyan) !important;
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
}

.stButton > button {
    background: transparent;
    border: 1px solid var(--accent-cyan);
    color: var(--accent-cyan) !important;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
    border-radius: 6px;
    padding: 0.4rem 1rem;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: var(--accent-cyan);
    color: var(--bg-primary) !important;
    box-shadow: 0 0 20px rgba(0, 212, 255, 0.4);
}
.stButton > button[kind="primary"] {
    background: var(--accent-cyan);
    color: var(--bg-primary) !important;
}

.stSelectbox > div, .stMultiSelect > div {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
}

.stDataFrame { border: 1px solid var(--border); border-radius: 8px; }

.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--text-secondary) !important;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
    border-bottom: 2px solid transparent;
    padding: 0.6rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: var(--accent-cyan) !important;
    border-bottom: 2px solid var(--accent-cyan) !important;
}

.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
}

.stAlert { border-radius: 8px; border-left: 3px solid; font-family: 'DM Sans', sans-serif; }

code {
    font-family: 'Space Mono', monospace !important;
    background: var(--bg-card) !important;
    color: var(--accent-green) !important;
    padding: 2px 6px;
    border-radius: 4px;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-secondary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-cyan); }

.brand-header {
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--accent-cyan);
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem;
}
.brand-sub {
    font-size: 0.7rem;
    color: var(--text-secondary);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}
.step-badge {
    display: inline-block;
    background: rgba(0, 212, 255, 0.1);
    border: 1px solid var(--accent-cyan);
    color: var(--accent-cyan);
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    padding: 2px 8px;
    border-radius: 3px;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.dot-green  { background: var(--accent-green);  box-shadow: 0 0 6px var(--accent-green); }
.dot-orange { background: var(--accent-orange); box-shadow: 0 0 6px var(--accent-orange); }
</style>
""", unsafe_allow_html=True)

# ─── Session state ────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "original_df": None,
        "working_df": None,
        "filename": None,
        "filetype": None,
        "transformation_log": [],
        "validation_rules": [],
        "current_page": "Upload & Overview",
        "upload_ts": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="brand-header">⚗️ DataWrangler</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Pro Studio v1.0</div>', unsafe_allow_html=True)

    if st.session_state.working_df is not None:
        df = st.session_state.working_df
        st.markdown(f"""
        <div style='background:#1c2128;border:1px solid #30363d;border-radius:8px;
                    padding:12px;margin-bottom:16px;'>
            <div style='font-family:Space Mono,monospace;font-size:0.7rem;color:#8b949e;
                        text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;'>
                <span class='status-dot dot-green'></span>Dataset Active
            </div>
            <div style='font-size:0.8rem;color:#e6edf3;'>{st.session_state.filename}</div>
            <div style='font-size:0.75rem;color:#8b949e;margin-top:4px;'>
                {df.shape[0]:,} rows × {df.shape[1]} cols
            </div>
            <div style='font-size:0.75rem;color:#8b949e;'>
                {len(st.session_state.transformation_log)} transformations
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:#1c2128;border:1px solid #30363d;border-radius:8px;
                    padding:12px;margin-bottom:16px;'>
            <div style='font-family:Space Mono,monospace;font-size:0.7rem;color:#8b949e;
                        text-transform:uppercase;letter-spacing:0.1em;'>
                <span class='status-dot dot-orange'></span>No Dataset
            </div>
            <div style='font-size:0.75rem;color:#8b949e;margin-top:4px;'>
                Upload a file to begin
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div style="font-family:Space Mono,monospace;font-size:0.65rem;color:#8b949e;'
        'text-transform:uppercase;letter-spacing:0.15em;margin-bottom:8px;">Navigation</div>',
        unsafe_allow_html=True,
    )

    pages = {
        "Upload & Overview":    "📁",
        "Cleaning & Prep Studio": "🧹",
        "Visualization Builder": "📊",
        "Export & Report":      "📤",
    }
    for page_name, icon in pages.items():
        is_active = st.session_state.current_page == page_name
        if st.button(
            f"{icon}  {page_name}",
            key=f"nav_{page_name}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.current_page = page_name
            st.rerun()

    st.divider()

    if st.session_state.transformation_log:
        st.markdown(
            '<div style="font-family:Space Mono,monospace;font-size:0.65rem;color:#8b949e;'
            'text-transform:uppercase;letter-spacing:0.15em;margin-bottom:8px;">Recent Steps</div>',
            unsafe_allow_html=True,
        )
        for step in st.session_state.transformation_log[-3:]:
            st.markdown(
                f"<div style='font-size:0.72rem;color:#8b949e;padding:2px 0;'>"
                f"· {step['operation']}</div>",
                unsafe_allow_html=True,
            )

    st.divider()
    if st.button("🔄  Reset Session", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ─── Route to modules (NOT pages/ — avoids Streamlit auto-navigation) ─────────
page = st.session_state.current_page

if page == "Upload & Overview":
    from modules.page_a_upload import render
    render()
elif page == "Cleaning & Prep Studio":
    from modules.page_b_cleaning import render
    render()
elif page == "Visualization Builder":
    from modules.page_c_viz import render
    render()
elif page == "Export & Report":
    from modules.page_d_export import render
    render()
