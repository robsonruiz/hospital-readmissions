import sys
from pathlib import Path

import streamlit as st


DASHBOARD_DIR = Path(__file__).resolve().parent

if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))


required_files = [
    "loader.py",
    "filters.py",
    "charts.py",
    "metrics.py",
    "interpretability.py",
]

missing_files = [
    filename
    for filename in required_files
    if not (DASHBOARD_DIR / filename).exists()
]

if missing_files:
    raise FileNotFoundError(
        "Missing dashboard files in "
        f"{DASHBOARD_DIR}: "
        + ", ".join(missing_files)
    )


from charts import (
    plot_discharge_reason,
    plot_discharge_reason_distribution,
    plot_heatmap,
    plot_hospital_rate,
    plot_icd_codes,
    plot_los_distribution,
    plot_los_vs_readmission,
    plot_monthly_trend,
    plot_previous_hospitalizations,
    plot_quarterly_trend,
    plot_specialty_rate,
)

from filters import apply_filters
from loader import load_data
from metrics import show_metrics
from interpretability import show_interpretability


st.set_page_config(
    page_title="Unplanned Hospital Readmissions",
    page_icon="▦",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    /* =========================================================
       Global
       ========================================================= */

    :root {
        --bg: #f6f8fb;
        --surface: #ffffff;
        --surface-soft: #f9fafc;
        --border: #e6e9ef;
        --text: #151922;
        --muted: #6f7785;
        --accent: #3157d5;
        --accent-soft: #edf2ff;
        --success: #2f9e62;
        --danger: #d9534f;
        --shadow: 0 2px 10px rgba(22, 34, 51, 0.045);
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    [data-testid="stAppViewContainer"] {
        background: var(--bg);
    }

    [data-testid="stHeader"] {
        background: rgba(246, 248, 251, 0.92);
    }

    .block-container {
        max-width: 1500px;
        padding-top: 4.4rem;
        padding-bottom: 3rem;
    }

    /* =========================================================
       Sidebar
       ========================================================= */

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.4rem;
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: var(--text);
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 11px;
        padding: 6px 8px 22px 8px;
        margin-bottom: 10px;
    }

    .sidebar-brand-mark {
        width: 36px;
        height: 36px;
        border-radius: 11px;
        background: var(--accent);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 17px;
        font-weight: 800;
        box-shadow: 0 5px 14px rgba(49, 87, 213, 0.22);
    }

    .sidebar-brand-title {
        font-size: 15px;
        font-weight: 750;
        line-height: 1.15;
    }

    .sidebar-brand-subtitle {
        color: var(--muted);
        font-size: 11px;
        margin-top: 3px;
    }

    .sidebar-section {
        color: #9299a6;
        font-size: 10px;
        font-weight: 750;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 15px 10px 7px;
    }


    /* =========================================================
       Sidebar navigation control
       ========================================================= */

    [data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 4px;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label {
        border-radius: 9px;
        padding: 8px 10px !important;
        margin: 0 !important;
        color: #747c89;
        transition: background 0.15s ease, color 0.15s ease;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background: #f5f7fb;
        color: #3157d5;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background: #edf2ff;
        color: #3157d5;
        font-weight: 700;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
        display: none;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label p {
        font-size: 12px !important;
        margin: 0 !important;
    }

    /* =========================================================
       Header
       ========================================================= */

    .page-kicker {
        color: var(--accent);
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    .page-title {
        font-size: 2.05rem;
        line-height: 1.08;
        font-weight: 760;
        letter-spacing: -0.035em;
        color: var(--text);
        margin: 0;
    }

    .page-subtitle {
        color: var(--muted);
        font-size: 0.93rem;
        margin-top: 9px;
        margin-bottom: 1.4rem;
    }

    .header-badge {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 7px 11px;
        border: 1px solid #dfe5f2;
        background: #ffffff;
        border-radius: 999px;
        color: #5d6675;
        font-size: 11px;
        font-weight: 650;
        box-shadow: var(--shadow);
    }

    .header-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--success);
    }

    /* =========================================================
       Navigation tabs
       ========================================================= */

    button[data-baseweb="tab"] {
        font-size: 13px !important;
        font-weight: 650 !important;
        color: #737b88 !important;
        border-radius: 9px !important;
        padding: 9px 15px !important;
        margin-right: 5px !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--accent) !important;
        background: var(--accent-soft) !important;
    }

    div[data-baseweb="tab-list"] {
        gap: 3px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1.6rem;
    }

    div[data-baseweb="tab-highlight"] {
        display: none;
    }

    /* =========================================================
       Metric cards
       ========================================================= */

    div[data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: var(--shadow);
        min-height: 122px;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted) !important;
    }

    div[data-testid="stMetricLabel"] p {
        font-size: 0.76rem !important;
        font-weight: 650 !important;
        letter-spacing: 0.01em;
    }

    div[data-testid="stMetricValue"] {
        color: var(--text) !important;
        font-size: 1.95rem !important;
        font-weight: 760 !important;
        letter-spacing: -0.025em;
        margin-top: 4px;
    }

    div[data-testid="stMetricDelta"] {
        font-size: 0.74rem !important;
        margin-top: 5px;
    }

    /* =========================================================
       Chart / content cards
       ========================================================= */

    .section-heading {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        margin: 1.7rem 0 0.75rem;
    }

    .section-title {
        font-size: 1.05rem;
        font-weight: 720;
        letter-spacing: -0.015em;
        color: var(--text);
    }

    .section-description {
        color: var(--muted);
        font-size: 0.78rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 14px;
        box-shadow: var(--shadow);
        overflow: hidden;
        padding: 0 !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background: #ffffff;
        border-radius: 14px;
        padding: 0 !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stPlotlyChart"] {
        background: #ffffff;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* =========================================================
       Buttons / controls
       ========================================================= */

    div[data-testid="stButton"] button {
        border-radius: 9px;
        border: 1px solid var(--border);
        font-weight: 650;
    }

    div[data-testid="stSelectbox"] > div,
    div[data-testid="stMultiSelect"] > div {
        border-radius: 9px;
    }

    /* =========================================================
       Dataframes
       ========================================================= */

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    /* =========================================================
       Footer
       ========================================================= */

    .dashboard-footer {
        color: #9aa1ad;
        font-size: 0.72rem;
        text-align: center;
        padding-top: 2rem;
    }

    @media (max-width: 900px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .page-title {
            font-size: 1.65rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def show_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-brand-mark">R</div>
                <div>
                    <div class="sidebar-brand-title">Readmission Analytics</div>
                    <div class="sidebar-brand-subtitle">Hospital performance</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="sidebar-section">Navigation</div>',
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigation",
            [
                "Analytics dashboard",
                "Model interpretability",
            ],
            key="dashboard_page",
            label_visibility="collapsed",
        )

        st.markdown(
            '<div class="sidebar-section">Scope</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            "Unplanned readmissions occurring "
            "2–30 days after discharge."
        )

    return page


def show_page_header() -> None:
    left, right = st.columns([7, 2])

    with left:
        st.markdown(
            '<div class="page-kicker">Clinical analytics</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="page-title">Unplanned Hospital Readmissions</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="page-subtitle">'
            "Monitor readmission patterns, hospital performance, "
            "patient history, and temporal trends."
            "</div>",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div style="text-align:right; padding-top:8px;">
                <span class="header-badge">
                    <span class="header-dot"></span>
                    Analytics ready
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def show_section_header(title: str, description: str = "") -> None:
    description_html = (
        f'<span class="section-description">{description}</span>'
        if description
        else ""
    )

    st.markdown(
        f"""
        <div class="section-heading">
            <span class="section-title">{title}</span>
            {description_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_dashboard() -> None:
    show_page_header()

    df = load_data()
    df, parameters = apply_filters(df)

    min_episodes = parameters["min_episodes"]
    top_n = parameters["top_n"]

    show_metrics(df)

    show_section_header(
        "Readmission trends",
        "Monthly and quarterly evolution",
    )

    c1, c2 = st.columns(2, gap="medium")

    with c1:
        with st.container(border=True):
            plot_monthly_trend(df)

    with c2:
        with st.container(border=True):
            plot_quarterly_trend(df)

    show_section_header(
        "Hospital and specialty performance",
        "Comparison across care settings",
    )

    c3, c4 = st.columns(2, gap="medium")

    with c3:
        with st.container(border=True):
            plot_specialty_rate(
                df,
                min_episodes=min_episodes,
                top_n=top_n,
            )

    with c4:
        with st.container(border=True):
            plot_hospital_rate(
                df,
                min_episodes=min_episodes,
                top_n=top_n,
            )

    show_section_header(
        "Patient and episode profile",
        "Length of stay and previous utilization",
    )

    c5, c6 = st.columns(2, gap="medium")

    with c5:
        with st.container(border=True):
            plot_los_distribution(df)

    with c6:
        with st.container(border=True):
            plot_los_vs_readmission(df)

    show_section_header(
        "Clinical and discharge patterns",
        "Diagnosis, discharge reason, and prior hospitalizations",
    )

    c7, c8 = st.columns(2, gap="medium")

    with c7:
        with st.container(border=True):
            plot_icd_codes(
                df,
                top_n=top_n,
            )

    with c8:
        with st.container(border=True):
            plot_discharge_reason(
                df,
                min_episodes=min_episodes,
            )

    c9, c10 = st.columns(2, gap="medium")

    with c9:
        with st.container(border=True):
            plot_discharge_reason_distribution(df)

    with c10:
        with st.container(border=True):
            plot_previous_hospitalizations(df)

    show_section_header(
        "Hospital × specialty",
        "Readmission rate across hospital and specialty combinations",
    )

    with st.container(border=True):
        plot_heatmap(
            df,
            min_episodes=min_episodes,
        )

    st.markdown(
        """
        <div class="dashboard-footer">
            Unplanned Hospital Readmissions Analytics
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    page = show_sidebar()

    if page == "Analytics dashboard":
        show_dashboard()
    else:
        show_interpretability()


if __name__ == "__main__":
    main()
