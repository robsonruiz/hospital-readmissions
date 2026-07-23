import pandas as pd
import plotly.express as px
import streamlit as st


TARGET_COLUMN = "unplanned_readmitted_30d"


def _has_columns(
    df: pd.DataFrame,
    *columns: str,
) -> bool:
    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        st.info(
            "Missing columns: "
            + ", ".join(missing)
        )
        return False

    return True


def _rate_table(
    df: pd.DataFrame,
    group: str,
    min_episodes: int = 0,
) -> pd.DataFrame:
    result = (
        df.assign(
            **{
                group: df[group].fillna(
                    "Not informed"
                )
            }
        )
        .groupby(group, dropna=False)
        .agg(
            episodes=(TARGET_COLUMN, "size"),
            readmissions=(TARGET_COLUMN, "sum"),
        )
        .reset_index()
    )

    result = result[
        result["episodes"] >= min_episodes
    ].copy()

    result["rate"] = (
        result["readmissions"]
        / result["episodes"]
        * 100
    )

    return result


def _bar_rate(
    data: pd.DataFrame,
    category: str,
    title: str,
    category_label: str,
    top_n: int | None = None,
    overall_rate: float | None = None,
    height: int = 600,
) -> None:
    if top_n is not None:
        data = data.head(top_n)

    if data.empty:
        st.info("No data available for this chart.")
        return

    fig = px.bar(
        data,
        x="rate",
        y=category,
        orientation="h",
        text="episodes",
        title=title,
        labels={
            "rate":
                "Unplanned readmission rate (%)",
            category: category_label,
            "episodes": "Episodes",
            "readmissions":
                "Unplanned readmissions",
        },
        hover_data={
            "episodes": ":,",
            "readmissions": ":,",
            "rate": ":.2f",
        },
    )

    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
    )

    if overall_rate is not None:
        fig.add_vline(
            x=overall_rate,
            line_dash="dash",
            annotation_text=(
                f"Overall ({overall_rate:.1f}%)"
            ),
        )

    fig.update_layout(
        yaxis={
            "categoryorder": "total ascending",
            "title": None,
        },
        height=height,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


def prepare_specialty(
    df: pd.DataFrame,
    min_episodes: int,
) -> pd.DataFrame:
    return _rate_table(
        df,
        "especialidade",
        min_episodes,
    ).sort_values(
        "rate",
        ascending=False,
    )


def prepare_hospital(
    df: pd.DataFrame,
    min_episodes: int,
) -> pd.DataFrame:
    return _rate_table(
        df,
        "executante",
        min_episodes,
    ).sort_values(
        "rate",
        ascending=False,
    )


def prepare_monthly(
    df: pd.DataFrame,
) -> pd.DataFrame:
    return _rate_table(
        df,
        "admission_month",
    ).sort_values("admission_month")


def prepare_quarterly(
    df: pd.DataFrame,
) -> pd.DataFrame:
    data = (
        df.groupby(
            [
                "admission_year",
                "admission_quarter",
            ]
        )
        .agg(
            episodes=(TARGET_COLUMN, "size"),
            readmissions=(TARGET_COLUMN, "sum"),
        )
        .reset_index()
        .sort_values(
            [
                "admission_year",
                "admission_quarter",
            ]
        )
    )

    data["rate"] = (
        data["readmissions"]
        / data["episodes"]
        * 100
    )

    data["quarter"] = (
        data["admission_year"]
        .astype(int)
        .astype(str)
        + " Q"
        + data["admission_quarter"]
        .astype(int)
        .astype(str)
    )

    return data


def show_kpi_cards(
    df: pd.DataFrame,
) -> None:
    """
    Compatibility function. Prefer show_metrics()
    from metrics.py.
    """

    if not _has_columns(
        df,
        TARGET_COLUMN,
        "length_of_stay_days",
        "previous_hospitalizations",
        "previous_unplanned_readmissions",
        "admissions_last_365d",
    ):
        return

    total = len(df)
    positives = int(df[TARGET_COLUMN].sum())
    rate = positives / total * 100 if total else 0

    metrics = [
        ("Episodes", f"{total:,}"),
        (
            "Unplanned Readmissions",
            f"{positives:,}",
        ),
        (
            "Unplanned Readmission Rate",
            f"{rate:.2f}%",
        ),
        (
            "Average LOS",
            (
                f"{df['length_of_stay_days'].mean():.1f} "
                "days"
            ),
        ),
        (
            "Avg Previous Hospitalizations",
            (
                f"{df['previous_hospitalizations'].mean():.2f}"
            ),
        ),
        (
            "Avg Admissions (Last 365 Days)",
            (
                f"{df['admissions_last_365d'].mean():.2f}"
            ),
        ),
        (
            "Avg Previous Unplanned Readmissions",
            (
                f"{df['previous_unplanned_readmissions'].mean():.2f}"
            ),
        ),
    ]

    st.subheader("Overview")

    for start in range(0, len(metrics), 4):
        columns = st.columns(4)

        for column, metric in zip(
            columns,
            metrics[start:start + 4],
        ):
            column.metric(*metric)

    st.divider()


def plot_specialty_rate(
    df: pd.DataFrame,
    min_episodes: int = 100,
    top_n: int = 20,
) -> None:
    if not _has_columns(
        df,
        "especialidade",
        TARGET_COLUMN,
    ):
        return

    _bar_rate(
        prepare_specialty(
            df,
            min_episodes,
        ),
        "especialidade",
        "Unplanned Readmission Rate by Specialty",
        "Specialty",
        top_n,
        df[TARGET_COLUMN].mean() * 100,
        650,
    )


def plot_hospital_rate(
    df: pd.DataFrame,
    min_episodes: int = 100,
    top_n: int = 20,
) -> None:
    if not _has_columns(
        df,
        "executante",
        TARGET_COLUMN,
    ):
        return

    _bar_rate(
        prepare_hospital(
            df,
            min_episodes,
        ),
        "executante",
        "Unplanned Readmission Rate by Hospital",
        "Hospital",
        top_n,
        df[TARGET_COLUMN].mean() * 100,
        650,
    )


def plot_monthly_trend(
    df: pd.DataFrame,
) -> None:
    if not _has_columns(
        df,
        "admission_month",
        TARGET_COLUMN,
    ):
        return

    data = prepare_monthly(df)

    fig = px.line(
        data,
        x="admission_month",
        y="rate",
        markers=True,
        title="Monthly Unplanned Readmission Rate",
        labels={
            "admission_month": "Month",
            "rate":
                "Unplanned readmission rate (%)",
        },
        hover_data={
            "episodes": ":,",
            "readmissions": ":,",
            "rate": ":.2f",
        },
    )

    fig.update_traces(
        line_width=3,
        marker_size=7,
    )

    fig.update_layout(
        hovermode="x unified",
        height=500,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


def plot_quarterly_trend(
    df: pd.DataFrame,
) -> None:
    if not _has_columns(
        df,
        "admission_year",
        "admission_quarter",
        TARGET_COLUMN,
    ):
        return

    data = prepare_quarterly(df)

    fig = px.bar(
        data,
        x="quarter",
        y="rate",
        text="episodes",
        title="Quarterly Unplanned Readmission Rate",
        labels={
            "quarter": "Quarter",
            "rate":
                "Unplanned readmission rate (%)",
        },
        hover_data={
            "episodes": ":,",
            "readmissions": ":,",
            "rate": ":.2f",
        },
    )

    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        height=500,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


def plot_los_distribution(
    df: pd.DataFrame,
) -> None:
    if not _has_columns(
        df,
        "length_of_stay_days",
        TARGET_COLUMN,
    ):
        return

    data = df[
        df["length_of_stay_days"].notna()
        & (df["length_of_stay_days"] >= 0)
    ].copy()

    if data.empty:
        st.info("No valid length-of-stay data.")
        return

    data = data[
        data["length_of_stay_days"]
        <= data["length_of_stay_days"].quantile(
            0.99
        )
    ]

    data["status"] = (
        data[TARGET_COLUMN]
        .map({
            0: "No unplanned readmission",
            1: "Unplanned readmission",
        })
    )

    fig = px.histogram(
        data,
        x="length_of_stay_days",
        color="status",
        nbins=50,
        barmode="overlay",
        opacity=0.65,
        title="Length of Stay Distribution",
        labels={
            "length_of_stay_days":
                "Length of stay (days)",
            "count": "Episodes",
            "status": "Outcome",
        },
    )

    fig.update_layout(height=500)

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


def plot_los_vs_readmission(
    df: pd.DataFrame,
) -> None:
    if not _has_columns(
        df,
        "length_of_stay_days",
        TARGET_COLUMN,
    ):
        return

    data = (
        df.groupby(TARGET_COLUMN)
        ["length_of_stay_days"]
        .mean()
        .reset_index()
    )

    data["status"] = (
        data[TARGET_COLUMN]
        .map({
            0: "No unplanned readmission",
            1: "Unplanned readmission",
        })
    )

    fig = px.bar(
        data,
        x="status",
        y="length_of_stay_days",
        text="length_of_stay_days",
        title=(
            "Average Length of Stay by "
            "Unplanned Readmission Status"
        ),
        labels={
            "status": "Outcome",
            "length_of_stay_days":
                "Average length of stay (days)",
        },
    )

    fig.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside",
    )

    fig.update_layout(height=500)

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


def plot_icd_codes(
    df: pd.DataFrame,
    top_n: int = 20,
) -> None:
    if not _has_columns(
        df,
        "codigocid",
        TARGET_COLUMN,
    ):
        return

    data = (
        df[
            (df[TARGET_COLUMN] == 1)
            & df["codigocid"].notna()
        ]
        .groupby("codigocid")
        .size()
        .reset_index(
            name="unplanned_readmissions"
        )
        .sort_values(
            "unplanned_readmissions",
            ascending=False,
        )
        .head(top_n)
    )

    if data.empty:
        st.info("No ICD data available.")
        return

    fig = px.bar(
        data,
        x="unplanned_readmissions",
        y="codigocid",
        orientation="h",
        text="unplanned_readmissions",
        title=(
            "Top ICD Codes Associated with "
            "Unplanned Readmissions"
        ),
        labels={
            "unplanned_readmissions":
                "Unplanned readmissions",
            "codigocid": "ICD code",
        },
    )

    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
    )

    fig.update_layout(
        yaxis={
            "categoryorder": "total ascending",
            "title": None,
        },
        height=600,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


def plot_discharge_reason(
    df: pd.DataFrame,
    min_episodes: int = 100,
) -> None:
    if not _has_columns(
        df,
        "motivoalta",
        TARGET_COLUMN,
    ):
        return

    _bar_rate(
        _rate_table(
            df,
            "motivoalta",
            min_episodes,
        ).sort_values(
            "rate",
            ascending=True,
        ),
        "motivoalta",
        (
            "Unplanned Readmission Rate "
            "by Discharge Reason"
        ),
        "Discharge reason",
        height=450,
    )


def plot_discharge_reason_distribution(
    df: pd.DataFrame,
) -> None:
    if not _has_columns(
        df,
        "motivoalta",
    ):
        return

    data = (
        df.assign(
            motivoalta=df["motivoalta"]
            .fillna("Not informed")
        )
        .groupby(
            "motivoalta",
            dropna=False,
        )
        .size()
        .reset_index(name="episodes")
        .sort_values(
            "episodes",
            ascending=True,
        )
    )

    fig = px.bar(
        data,
        x="episodes",
        y="motivoalta",
        orientation="h",
        text="episodes",
        title=(
            "Distribution of Discharge Reasons "
            "among Eligible Episodes"
        ),
        labels={
            "episodes": "Episodes",
            "motivoalta": "Discharge reason",
        },
    )

    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
    )

    fig.update_layout(
        height=450,
        yaxis_title=None,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


def plot_heatmap(
    df: pd.DataFrame,
    min_episodes: int = 100,
) -> None:
    if not _has_columns(
        df,
        "executante",
        "especialidade",
        TARGET_COLUMN,
    ):
        return

    top_hospitals = (
        df["executante"]
        .value_counts()
        .head(10)
        .index
    )

    top_specialties = (
        df["especialidade"]
        .value_counts()
        .head(10)
        .index
    )

    data = (
        df[
            df["executante"].isin(
                top_hospitals
            )
            & df["especialidade"].isin(
                top_specialties
            )
        ]
        .groupby(
            [
                "executante",
                "especialidade",
            ]
        )
        .agg(
            episodes=(TARGET_COLUMN, "size"),
            readmissions=(TARGET_COLUMN, "sum"),
        )
        .reset_index()
    )

    data = data[
        data["episodes"] >= min_episodes
    ].copy()

    if data.empty:
        st.info("No data available for heatmap.")
        return

    data["rate"] = (
        data["readmissions"]
        / data["episodes"]
        * 100
    )

    pivot = data.pivot(
        index="executante",
        columns="especialidade",
        values="rate",
    )

    fig = px.imshow(
        pivot,
        aspect="auto",
        title=(
            "Unplanned Readmission Rate: "
            "Hospital × Specialty"
        ),
        labels={
            "x": "Specialty",
            "y": "Hospital",
            "color": "Rate (%)",
        },
    )

    fig.update_layout(height=700)

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


def plot_previous_hospitalizations(
    df: pd.DataFrame,
) -> None:
    if not _has_columns(
        df,
        "previous_hospitalizations",
        TARGET_COLUMN,
    ):
        return

    data = (
        _rate_table(
            df,
            "previous_hospitalizations",
        )
        .sort_values(
            "previous_hospitalizations"
        )
        .head(30)
    )

    fig = px.line(
        data,
        x="previous_hospitalizations",
        y="rate",
        markers=True,
        title=(
            "Unplanned Readmission Rate vs "
            "Previous Hospitalizations"
        ),
        labels={
            "previous_hospitalizations":
                "Previous hospitalizations",
            "rate":
                "Unplanned readmission rate (%)",
        },
        hover_data={
            "episodes": ":,",
            "readmissions": ":,",
            "rate": ":.2f",
        },
    )

    fig.update_traces(
        line_width=3,
        marker_size=7,
    )

    fig.update_layout(height=500)

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


def plot_discharge_outcomes(
    df: pd.DataFrame,
) -> None:
    plot_discharge_reason_distribution(df)


def plot_model_eligibility(
    df: pd.DataFrame,
) -> None:
    st.info(
        "ml_features already contains only "
        "eligible episodes."
    )


def plot_non_eligible_reasons(
    df: pd.DataFrame,
) -> None:
    st.info(
        "Non-eligible episodes are not included "
        "in ml_features."
    )


def plot_deaths_by_hospital(
    df: pd.DataFrame,
    min_episodes: int = 100,
    top_n: int = 20,
) -> None:
    st.info(
        "Deaths are not included in ml_features "
        "because they are not eligible to originate "
        "a future readmission."
    )