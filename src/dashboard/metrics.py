import streamlit as st
import pandas as pd


def show_metrics(df: pd.DataFrame):
    """
    Display dashboard KPIs.
    """

    total_episodes = len(df)

    readmissions = int(
        df["readmitted_30d_clean"].sum()
    )

    readmission_rate = (
        readmissions / total_episodes * 100
        if total_episodes > 0
        else 0
    )

    average_los = (
        df["length_of_stay_days"]
        .dropna()
        .mean()
    )

    average_previous_hospitalizations = (
        df["previous_hospitalizations"]
        .mean()
    )

    average_previous_readmissions = (
        df["previous_readmissions"]
        .mean()
    )

    average_last365 = (
        df["admissions_last_365d"]
        .mean()
    )

    deaths = int(
        df["motivoalta"]
        .fillna("")
        .eq("Óbito")
        .sum()
    )

    transfers = int(
        df["motivoalta"]
        .fillna("")
        .eq("Transferência")
        .sum()
    )

    eligible_df = df[
        df["eligible_for_readmission_model"] == 1
    ].copy()

    eligible_episodes = len(eligible_df)

    eligible_readmissions = int(
        eligible_df["readmitted_30d_clean"]
        .fillna(0)
        .sum()
    )

    eligible_readmission_rate = (
        eligible_readmissions
        / eligible_episodes
        * 100
        if eligible_episodes > 0
        else 0
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Hospitalization Episodes",
            f"{total_episodes:,}"
        )

    with c2:
        st.metric(
            "Readmissions",
            f"{readmissions:,}"
        )

    with c3:
        st.metric(
            "Readmission Rate",
            f"{readmission_rate:.2f}%"
        )

    c4, c5, c6 = st.columns(3)

    with c4:
        st.metric(
            "Average Length of Stay",
            f"{average_los:.1f} days"
            if pd.notna(average_los)
            else "-"
        )

    with c5:
        st.metric(
            "Avg Previous Hospitalizations",
            f"{average_previous_hospitalizations:.2f}"
        )

    with c6:
        st.metric(
            "Avg Admissions (Last 365 Days)",
            f"{average_last365:.2f}"
        )

    c7, c8 = st.columns(2)

    with c7:
        st.metric(
            "Avg Previous Readmissions",
            f"{average_previous_readmissions:.2f}"
        )

    with c8:
        st.metric(
            "Patients Readmitted",
            f"{df.loc[df['readmitted_30d_clean'] == 1].shape[0]:,}"
        )

    k4, k5, k6 = st.columns(3)

    with k4:
        st.metric(
            "Deaths",
            f"{deaths:,}"
        )

    with k5:
        st.metric(
            "Transfers",
            f"{transfers:,}"
        )

    with k6:
        st.metric(
            "Eligible Readmission Rate",
            f"{eligible_readmission_rate:.2f}%",
            help=(
                f"{eligible_readmissions:,} readmissions among "
                f"{eligible_episodes:,} eligible episodes."
            )
        )

    st.divider()