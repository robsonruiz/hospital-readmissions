import duckdb
import pandas as pd
import streamlit as st


DATABASE_PATH = "data/hospital_readmissions.duckdb"
TARGET_COLUMN = "unplanned_readmitted_30d"


@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load the analytics dataset from DuckDB.
    """

    with duckdb.connect(
        DATABASE_PATH,
        read_only=True,
    ) as conn:
        df = conn.execute("""
            SELECT *
            FROM ml_features
        """).fetchdf()

    return clean_dataframe(df)


def clean_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert numeric, target and datetime columns.
    """

    df = df.copy()

    numeric_columns = [
        "length_of_stay_days",
        "previous_hospitalizations",
        "previous_unplanned_readmissions",
        "days_since_previous_hospitalization",
        "admissions_last_30d",
        "admissions_last_90d",
        "admissions_last_365d",
        "admission_year",
        "admission_quarter",
        "admission_weekday",
    ]

    for column in numeric_columns:
        if column not in df.columns:
            continue

        if df[column].dtype == object:
            df[column] = (
                df[column]
                .astype(str)
                .str.replace(",", ".", regex=False)
            )

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    count_columns = [
        "previous_hospitalizations",
        "previous_unplanned_readmissions",
        "admissions_last_30d",
        "admissions_last_90d",
        "admissions_last_365d",
    ]

    for column in count_columns:
        if column in df.columns:
            df[column] = (
                df[column]
                .fillna(0)
                .astype(int)
            )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column not found: {TARGET_COLUMN}"
        )

    df[TARGET_COLUMN] = (
        pd.to_numeric(
            df[TARGET_COLUMN],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    datetime_columns = [
        "admission_ts",
        "discharge_ts",
        "admission_month",
    ]

    for column in datetime_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce",
            )

    return df


def get_overall_readmission_rate(
    df: pd.DataFrame,
) -> float:
    """
    Overall unplanned readmission rate (%).
    """

    if df.empty:
        return 0.0

    return float(
        df[TARGET_COLUMN].mean() * 100
    )


def get_kpis(
    df: pd.DataFrame,
) -> dict:
    """
    Compute dashboard KPIs.
    """

    total = len(df)

    unplanned_readmissions = int(
        df[TARGET_COLUMN].sum()
    )

    rate = (
        unplanned_readmissions / total * 100
        if total > 0
        else 0.0
    )

    return {
        "total_episodes": total,
        "unplanned_readmissions":
            unplanned_readmissions,
        "readmissions":
            unplanned_readmissions,
        "rate": rate,
        "avg_los": (
            df["length_of_stay_days"].mean()
            if "length_of_stay_days" in df.columns
            else 0.0
        ),
        "avg_previous_hospitalizations": (
            df["previous_hospitalizations"].mean()
            if "previous_hospitalizations" in df.columns
            else 0.0
        ),
        "avg_admissions_last365": (
            df["admissions_last_365d"].mean()
            if "admissions_last_365d" in df.columns
            else 0.0
        ),
    }