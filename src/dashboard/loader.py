import duckdb
import pandas as pd
import streamlit as st


DATABASE_PATH = "data/hospital_readmissions.duckdb"


@st.cache_data
def load_data():
    """
    Load analytics dataset from DuckDB.
    """

    conn = duckdb.connect(DATABASE_PATH)

    df = conn.execute("""
        SELECT *
        FROM ml_features
    """).fetchdf()

    conn.close()

    return clean_dataframe(df)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic data cleaning and type conversion.
    """

    # -----------------------------
    # Length of stay
    # -----------------------------

    if "length_of_stay_days" in df.columns:

        if df["length_of_stay_days"].dtype == object:

            df["length_of_stay_days"] = (
                df["length_of_stay_days"]
                .astype(str)
                .str.replace(",", ".", regex=False)
            )

        df["length_of_stay_days"] = pd.to_numeric(
            df["length_of_stay_days"],
            errors="coerce"
        )

    # -----------------------------
    # Previous hospitalizations
    # -----------------------------

    if "previous_hospitalizations" in df.columns:

        df["previous_hospitalizations"] = pd.to_numeric(
            df["previous_hospitalizations"],
            errors="coerce"
        ).fillna(0)

    # -----------------------------
    # Previous readmissions
    # -----------------------------

    if "previous_readmissions" in df.columns:

        df["previous_readmissions"] = pd.to_numeric(
            df["previous_readmissions"],
            errors="coerce"
        ).fillna(0)

    # -----------------------------
    # Admissions last 365 days
    # -----------------------------

    if "admissions_last_365d" in df.columns:

        df["admissions_last_365d"] = pd.to_numeric(
            df["admissions_last_365d"],
            errors="coerce"
        ).fillna(0)

    # -----------------------------
    # Readmission target
    # -----------------------------

    if "readmitted_30d_clean" in df.columns:

        df["readmitted_30d_clean"] = (
            df["readmitted_30d_clean"]
            .fillna(0)
            .astype(int)
        )

    # -----------------------------
    # Datetime columns
    # -----------------------------

    datetime_columns = [
        "admission_ts",
        "discharge_ts",
        "previous_discharge_ts",
        "admission_month"
    ]

    for column in datetime_columns:

        if column in df.columns:

            df[column] = pd.to_datetime(
                df[column],
                errors="coerce"
            )

    return df


def get_overall_readmission_rate(df: pd.DataFrame) -> float:
    """
    Overall readmission rate (%).
    """

    if len(df) == 0:
        return 0

    return (
        df["readmitted_30d_clean"].mean() * 100
    )


def get_kpis(df: pd.DataFrame) -> dict:
    """
    Compute dashboard KPIs.
    """

    total = len(df)

    readmissions = int(
        df["readmitted_30d_clean"].sum()
    )

    rate = (
        readmissions / total * 100
        if total > 0
        else 0
    )

    avg_los = (
        df["length_of_stay_days"]
        .mean()
    )

    avg_previous = (
        df["previous_hospitalizations"]
        .mean()
    )

    avg_last365 = (
        df["admissions_last_365d"]
        .mean()
    )

    return {

        "total_episodes": total,

        "readmissions": readmissions,

        "rate": rate,

        "avg_los": avg_los,

        "avg_previous_hospitalizations": avg_previous,

        "avg_admissions_last365": avg_last365

    }