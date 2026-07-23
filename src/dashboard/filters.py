import pandas as pd
import streamlit as st


def apply_filters(df: pd.DataFrame):
    """
    Creates the dashboard sidebar filters and
    returns the filtered dataframe.
    """

    st.sidebar.title("Filters")

    # =====================================
    # Patient Filters
    # =====================================

    st.sidebar.subheader("Patient")

    sex = st.sidebar.multiselect(
        "Sex",
        sorted(df["sexo"].dropna().unique())
    )

    request = st.sidebar.multiselect(
        "Municipality of Request",
        sorted(df["municipiosolicitante"].dropna().unique())
    )

    hospital_city = st.sidebar.multiselect(
        "Hospital Municipality",
        sorted(df["municipioexecutante"].dropna().unique())
    )

    # =====================================
    # Hospital Filters
    # =====================================

    st.sidebar.subheader("Hospital")

    hospital = st.sidebar.multiselect(
        "Hospital",
        sorted(df["executante"].dropna().unique())
    )

    specialty = st.sidebar.multiselect(
        "Specialty",
        sorted(df["especialidade"].dropna().unique())
    )

    bed = st.sidebar.multiselect(
        "Bed Type",
        sorted(df["tipoleito"].dropna().unique())
    )

    admission_type = st.sidebar.multiselect(
        "Admission Type",
        sorted(df["carater"].dropna().unique())
    )

    discharge_reason = st.sidebar.multiselect(
        "Discharge Reason",
        sorted(df["motivoalta"].dropna().unique())
    )

    # =====================================
    # Time Filters
    # =====================================

    st.sidebar.subheader("Time")

    years = sorted(df["admission_year"].dropna().unique())

    selected_years = st.sidebar.multiselect(
        "Admission Year",
        years,
        default=years
    )

    quarters = sorted(df["admission_quarter"].dropna().unique())

    selected_quarters = st.sidebar.multiselect(
        "Quarter",
        quarters,
        default=quarters
    )

    weekdays = sorted(df["admission_weekday"].dropna().unique())

    selected_weekdays = st.sidebar.multiselect(
        "Weekday",
        weekdays,
        default=weekdays
    )

    # =====================================
    # Clinical Filters
    # =====================================

    st.sidebar.subheader("Clinical")

    readmitted_only = st.sidebar.checkbox(
        "Only Readmissions"
    )

    missing_discharge = st.sidebar.checkbox(
        "Missing Discharge Only"
    )

    # =====================================
    # Analysis Parameters
    # =====================================

    st.sidebar.subheader("Analysis")

    min_episodes = st.sidebar.slider(
        "Minimum Episodes",
        0,
        1000,
        100,
        25
    )

    top_n = st.sidebar.slider(
        "Top Categories",
        5,
        30,
        20
    )

    max_los = int(
        df["length_of_stay_days"]
        .fillna(0)
        .max()
    )

    los_limit = st.sidebar.slider(
        "Maximum Length of Stay (days)",
        1,
        max(1, max_los),
        min(60, max(1, max_los))
    )

    previous_hosp = st.sidebar.slider(
        "Maximum Previous Hospitalizations",
        0,
        int(df["previous_hospitalizations"].max()),
        int(df["previous_hospitalizations"].max())
    )

    previous_readm = st.sidebar.slider(
        "Maximum Previous Readmissions",
        0,
        int(df["previous_unplanned_readmissions"].max()),
        int(df["previous_unplanned_readmissions"].max())
    )

    admissions_last365 = st.sidebar.slider(
        "Maximum Admissions (Last 365 Days)",
        0,
        int(df["admissions_last_365d"].max()),
        int(df["admissions_last_365d"].max())
    )

    # =====================================
    # Apply Filters
    # =====================================

    if sex:
        df = df[df["sexo"].isin(sex)]

    if request:
        df = df[df["municipiosolicitante"].isin(request)]

    if hospital_city:
        df = df[df["municipioexecutante"].isin(hospital_city)]

    if hospital:
        df = df[df["executante"].isin(hospital)]

    if specialty:
        df = df[df["especialidade"].isin(specialty)]

    if bed:
        df = df[df["tipoleito"].isin(bed)]

    if admission_type:
        df = df[df["carater"].isin(admission_type)]

    if discharge_reason:
        df = df[df["motivoalta"].isin(discharge_reason)]

    if selected_years:
        df = df[df["admission_year"].isin(selected_years)]

    if selected_quarters:
        df = df[df["admission_quarter"].isin(selected_quarters)]

    if selected_weekdays:
        df = df[df["admission_weekday"].isin(selected_weekdays)]

    if readmitted_only:
        df = df[df["readmitted_30d_clean"] == 1]

    if missing_discharge:
        df = df[df["missing_discharge"] == 1]

    df = df[
        df["length_of_stay_days"].fillna(0)
        <= los_limit
    ]

    df = df[
        df["previous_hospitalizations"]
        <= previous_hosp
    ]

    df = df[
        df["previous_unplanned_readmissions"]
        <= previous_readm
    ]

    df = df[
        df["admissions_last_365d"]
        <= admissions_last365
    ]

    parameters = {
        "min_episodes": min_episodes,
        "top_n": top_n
    }

    return df, parameters