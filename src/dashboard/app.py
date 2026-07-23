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


st.set_page_config(
    page_title=(
        "Unplanned Hospital Readmissions Dashboard"
    ),
    layout="wide",
)


def main() -> None:
    st.title(
        "Unplanned Hospital Readmissions Dashboard"
    )

    st.markdown(
        "Analysis of unplanned hospital readmissions "
        "occurring between 2 and 30 days after discharge."
    )

    df = load_data()
    df, parameters = apply_filters(df)

    min_episodes = parameters["min_episodes"]
    top_n = parameters["top_n"]

    show_metrics(df)

    c1, c2 = st.columns(2)

    with c1:
        plot_monthly_trend(df)

    with c2:
        plot_quarterly_trend(df)

    c3, c4 = st.columns(2)

    with c3:
        plot_specialty_rate(
            df,
            min_episodes=min_episodes,
            top_n=top_n,
        )

    with c4:
        plot_hospital_rate(
            df,
            min_episodes=min_episodes,
            top_n=top_n,
        )

    c5, c6 = st.columns(2)

    with c5:
        plot_los_distribution(df)

    with c6:
        plot_los_vs_readmission(df)

    c7, c8 = st.columns(2)

    with c7:
        plot_icd_codes(
            df,
            top_n=top_n,
        )

    with c8:
        plot_discharge_reason(
            df,
            min_episodes=min_episodes,
        )

    c9, c10 = st.columns(2)

    with c9:
        plot_discharge_reason_distribution(df)

    with c10:
        plot_previous_hospitalizations(df)

    plot_heatmap(
        df,
        min_episodes=min_episodes,
    )


if __name__ == "__main__":
    main()