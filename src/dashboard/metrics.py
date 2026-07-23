import pandas as pd
import streamlit as st


TARGET_COLUMN = "unplanned_readmitted_30d"


def show_metrics(df: pd.DataFrame) -> None:
    """Display dashboard KPIs for eligible episodes."""

    if TARGET_COLUMN not in df.columns:
        st.error(
            f"Target column not found: {TARGET_COLUMN}"
        )
        return

    total_episodes = len(df)
    unplanned_readmissions = int(
        df[TARGET_COLUMN].sum()
    )

    readmission_rate = (
        unplanned_readmissions
        / total_episodes
        * 100
        if total_episodes
        else 0.0
    )

    unique_patients = (
        df["identificador"].nunique()
        if "identificador" in df.columns
        else 0
    )

    readmitted_patients = (
        df.loc[
            df[TARGET_COLUMN] == 1,
            "identificador",
        ].nunique()
        if "identificador" in df.columns
        else 0
    )

    average_los = (
        df["length_of_stay_days"].mean()
        if "length_of_stay_days" in df.columns
        else float("nan")
    )

    average_previous_hospitalizations = (
        df["previous_hospitalizations"].mean()
        if "previous_hospitalizations" in df.columns
        else float("nan")
    )

    average_previous_unplanned = (
        df["previous_unplanned_readmissions"].mean()
        if "previous_unplanned_readmissions"
        in df.columns
        else float("nan")
    )

    average_last365 = (
        df["admissions_last_365d"].mean()
        if "admissions_last_365d" in df.columns
        else float("nan")
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Hospitalization Episodes",
        f"{total_episodes:,}",
    )
    c2.metric(
        "Unplanned Readmissions",
        f"{unplanned_readmissions:,}",
    )
    c3.metric(
        "Unplanned Readmission Rate",
        f"{readmission_rate:.2f}%",
    )

    c4, c5, c6 = st.columns(3)

    c4.metric(
        "Unique Patients",
        f"{unique_patients:,}",
    )
    c5.metric(
        "Patients with Unplanned Readmission",
        f"{readmitted_patients:,}",
    )
    c6.metric(
        "Average Length of Stay",
        (
            f"{average_los:.1f} days"
            if pd.notna(average_los)
            else "-"
        ),
    )

    c7, c8, c9 = st.columns(3)

    c7.metric(
        "Avg Previous Hospitalizations",
        (
            f"{average_previous_hospitalizations:.2f}"
            if pd.notna(
                average_previous_hospitalizations
            )
            else "-"
        ),
    )
    c8.metric(
        "Avg Previous Unplanned Readmissions",
        (
            f"{average_previous_unplanned:.2f}"
            if pd.notna(
                average_previous_unplanned
            )
            else "-"
        ),
    )
    c9.metric(
        "Avg Admissions (Last 365 Days)",
        (
            f"{average_last365:.2f}"
            if pd.notna(average_last365)
            else "-"
        ),
    )

    st.divider()