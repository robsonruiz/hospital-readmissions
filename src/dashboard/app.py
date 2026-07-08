import streamlit as st

from loader import load_data
from filters import apply_filters
from charts import (
    show_kpi_cards,
    plot_monthly_trend,
    plot_quarterly_trend,
    plot_specialty_rate,
    plot_hospital_rate,
    plot_los_distribution,
    plot_los_vs_readmission,
    plot_icd_codes,
    plot_discharge_reason,
    plot_heatmap,
    plot_previous_hospitalizations
)


st.set_page_config(
    page_title="Hospital Readmissions Dashboard",
    layout="wide"
)


def main():

    st.title("Hospital Readmissions Dashboard")

    st.markdown(
        "Analysis of hospital readmissions within 30 days."
    )

    df = load_data()

    df, parameters = apply_filters(df)

    min_episodes = parameters["min_episodes"]
    top_n = parameters["top_n"]

    show_kpi_cards(df)

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
            top_n=top_n
        )

    with c4:
        plot_hospital_rate(
            df,
            min_episodes=min_episodes,
            top_n=top_n
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
            top_n=top_n
        )

    with c8:
        plot_discharge_reason(
            df,
            min_episodes=min_episodes,
            top_n=top_n
        )

    c9, c10 = st.columns(2)

    with c9:
        plot_previous_hospitalizations(df)

    plot_heatmap(
        df,
        min_episodes=min_episodes
    )

if __name__ == "__main__":
    main()