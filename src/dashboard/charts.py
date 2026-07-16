import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ==========================================================
# Helpers
# ==========================================================

def prepare_specialty(df, min_episodes):
    specialty = (
        df.groupby("especialidade")
        .agg(
            episodes=("readmitted_30d_clean", "count"),
            readmissions=("readmitted_30d_clean", "sum")
        )
        .reset_index()
    )

    specialty["rate"] = (
        specialty["readmissions"]
        / specialty["episodes"]
        * 100
    )

    specialty = specialty[
        specialty["episodes"] >= min_episodes
    ]

    specialty = specialty.sort_values(
        "rate",
        ascending=False
    )

    return specialty


def prepare_hospital(df, min_episodes):

    hospital = (
        df.groupby("executante")
        .agg(
            episodes=("readmitted_30d_clean", "count"),
            readmissions=("readmitted_30d_clean", "sum")
        )
        .reset_index()
    )

    hospital["rate"] = (
        hospital["readmissions"]
        / hospital["episodes"]
        * 100
    )

    hospital = hospital[
        hospital["episodes"] >= min_episodes
    ]

    hospital = hospital.sort_values(
        "rate",
        ascending=False
    )

    return hospital


def prepare_monthly(df):

    monthly = (
        df.groupby("admission_month")
        .agg(
            episodes=("readmitted_30d_clean", "count"),
            readmissions=("readmitted_30d_clean", "sum")
        )
        .reset_index()
    )

    monthly["rate"] = (
        monthly["readmissions"]
        / monthly["episodes"]
        * 100
    )

    return monthly.sort_values("admission_month")


def prepare_quarterly(df):

    quarterly = (
        df.groupby(
            ["admission_year", "admission_quarter"]
        )
        .agg(
            episodes=("readmitted_30d_clean", "count"),
            readmissions=("readmitted_30d_clean", "sum")
        )
        .reset_index()
    )

    quarterly["rate"] = (
        quarterly["readmissions"]
        / quarterly["episodes"]
        * 100
    )

    quarterly["quarter"] = (
        quarterly["admission_year"].astype(str)
        + " Q"
        + quarterly["admission_quarter"].astype(str)
    )

    return quarterly


# ==========================================================
# KPI Cards
# ==========================================================

def show_kpi_cards(df):

    total_episodes = len(df)

    readmissions = int(
        df["readmitted_30d_clean"].sum()
    )

    readmission_rate = (
        readmissions
        / total_episodes
        * 100
        if total_episodes > 0
        else 0
    )

    avg_los = (
        df["length_of_stay_days"]
        .dropna()
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

    avg_previous_readm = (
        df["previous_readmissions"]
        .mean()
    )

    missing_discharge = (
        df["missing_discharge"]
        .sum()
    )

    st.subheader("Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Episodes",
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

    with c4:
        st.metric(
            "Average LOS",
            f"{avg_los:.1f} days"
        )

    c5, c6, c7, c8 = st.columns(4)

    with c5:
        st.metric(
            "Previous Hospitalizations",
            f"{avg_previous:.2f}"
        )

    with c6:
        st.metric(
            "Admissions Last 365 Days",
            f"{avg_last365:.2f}"
        )

    with c7:
        st.metric(
            "Previous Readmissions",
            f"{avg_previous_readm:.2f}"
        )

    with c8:
        st.metric(
            "Missing Discharge",
            f"{missing_discharge:,}"
        )

    st.divider()

    # ==========================================================
# Readmission Rate by Specialty
# ==========================================================

def plot_specialty_rate(df, min_episodes=100, top_n=20):

    specialty = prepare_specialty(df, min_episodes)

    overall_rate = (
        df["readmitted_30d_clean"].mean()
        * 100
    )

    fig = px.bar(
        specialty.head(top_n),
        x="rate",
        y="especialidade",
        orientation="h",
        text="episodes",
        title="Readmission Rate by Specialty",
        labels={
            "rate": "Readmission Rate (%)",
            "especialidade": "Specialty"
        }
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.add_vline(
        x=overall_rate,
        line_dash="dash",
        annotation_text=f"Overall ({overall_rate:.1f}%)"
    )

    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=650
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==========================================================
# Readmission Rate by Hospital
# ==========================================================

def plot_hospital_rate(df, min_episodes=100, top_n=20):

    hospital = prepare_hospital(df, min_episodes)

    overall_rate = (
        df["readmitted_30d_clean"].mean()
        * 100
    )

    fig = px.bar(
        hospital.head(top_n),
        x="rate",
        y="executante",
        orientation="h",
        text="episodes",
        title="Readmission Rate by Hospital",
        labels={
            "rate": "Readmission Rate (%)",
            "executante": "Hospital"
        }
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.add_vline(
        x=overall_rate,
        line_dash="dash",
        annotation_text=f"Overall ({overall_rate:.1f}%)"
    )

    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=650
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==========================================================
# Monthly Trend
# ==========================================================

def plot_monthly_trend(df):

    monthly = prepare_monthly(df)

    fig = px.line(
        monthly,
        x="admission_month",
        y="rate",
        markers=True,
        title="Monthly Readmission Rate",
        labels={
            "admission_month": "Month",
            "rate": "Readmission Rate (%)"
        }
    )

    fig.update_traces(
        line_width=3,
        marker_size=8
    )

    fig.update_layout(
        hovermode="x unified",
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==========================================================
# Quarterly Trend
# ==========================================================

def plot_quarterly_trend(df):

    quarterly = prepare_quarterly(df)

    fig = px.bar(
        quarterly,
        x="quarter",
        y="rate",
        text="episodes",
        title="Quarterly Readmission Rate",
        labels={
            "quarter": "Quarter",
            "rate": "Readmission Rate (%)"
        }
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ==========================================================
# Length of Stay Distribution
# ==========================================================

def plot_los_distribution(df):

    if "length_of_stay_days" not in df.columns:
        st.info("Column length_of_stay_days not found.")
        return

    los_df = df[
        df["length_of_stay_days"].notna()
        & (df["length_of_stay_days"] >= 0)
    ].copy()

    if los_df.empty:
        st.info("No valid length of stay data available.")
        return

    upper_limit = los_df["length_of_stay_days"].quantile(0.99)

    los_df = los_df[
        los_df["length_of_stay_days"] <= upper_limit
    ]

    readmission_los = (
        los_df[los_df["readmitted_30d_clean"] == 1]
        .groupby("length_of_stay_days")
        .size()
        .reset_index(name="readmissions")
    )

    fig = px.histogram(
        los_df,
        x="length_of_stay_days",
        nbins=50,
        title="Length of Stay Distribution and Readmissions",
        labels={
            "length_of_stay_days": "Length of Stay (days)",
            "count": "Episodes"
        },
        opacity=0.75
    )

    fig_line = go.Scatter(
        x=readmission_los["length_of_stay_days"],
        y=readmission_los["readmissions"],
        name="Readmissions",
        mode="lines+markers",
        yaxis="y2"
    )

    fig.add_trace(fig_line)

    fig.update_layout(
        yaxis=dict(title="Episodes"),
        yaxis2=dict(
            title="Readmissions",
            overlaying="y",
            side="right"
        ),
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)


# ==========================================================
# LOS vs Readmission
# ==========================================================

def plot_los_vs_readmission(df):

    if "length_of_stay_days" not in df.columns:
        st.info("Column length_of_stay_days not found.")
        return

    los = (
        df.groupby("readmitted_30d_clean")
        ["length_of_stay_days"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        los,
        x="readmitted_30d_clean",
        y="length_of_stay_days",
        text="length_of_stay_days",
        title="Average Length of Stay by Readmission Status",
        labels={
            "readmitted_30d_clean": "Readmitted in 30 days",
            "length_of_stay_days": "Average LOS (days)"
        }
    )

    fig.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside"
    )

    fig.update_layout(height=500)

    st.plotly_chart(fig, use_container_width=True)


# ==========================================================
# ICD Codes
# ==========================================================

def plot_icd_codes(df, top_n=20):

    icd = (
        df[df["readmitted_30d_clean"] == 1]
        .groupby("codigocid")
        .size()
        .reset_index(name="readmissions")
        .sort_values("readmissions", ascending=False)
        .head(top_n)
    )

    if icd.empty:
        st.info("No ICD data available.")
        return

    fig = px.bar(
        icd,
        x="readmissions",
        y="codigocid",
        orientation="h",
        text="readmissions",
        title="Top ICD Codes Associated with Readmissions",
        labels={
            "readmissions": "Readmissions",
            "codigocid": "ICD Code"
        }
    )

    fig.update_traces(textposition="outside")

    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)


# ==========================================================
# Discharge Reason
# ==========================================================

def plot_discharge_reason(
    df: pd.DataFrame,
    min_episodes: int = 100
) -> None:
    """
    Taxa de reinternação por motivo de alta,
    considerando apenas episódios elegíveis.
    """

    required_columns = {
        "motivoalta",
        "readmitted_30d_clean",
        "eligible_for_readmission_model"
    }

    if not required_columns.issubset(df.columns):
        st.info(
            "Required columns for discharge reason chart "
            "were not found."
        )
        return

    eligible_df = df[
        df["eligible_for_readmission_model"] == 1
    ].copy()

    discharge = (
        eligible_df
        .assign(
            motivoalta=eligible_df["motivoalta"]
            .fillna("Não informado")
        )
        .groupby("motivoalta", dropna=False)
        .agg(
            episodes=(
                "readmitted_30d_clean",
                "size"
            ),
            readmissions=(
                "readmitted_30d_clean",
                "sum"
            )
        )
        .reset_index()
    )

    discharge = discharge[
        discharge["episodes"] >= min_episodes
    ].copy()

    if discharge.empty:
        st.info(
            "No eligible discharge reasons satisfy "
            "the minimum episode filter."
        )
        return

    discharge["rate"] = (
        discharge["readmissions"]
        / discharge["episodes"]
        * 100
    )

    discharge = discharge.sort_values(
        "rate",
        ascending=True
    )

    fig = px.bar(
        discharge,
        x="rate",
        y="motivoalta",
        orientation="h",
        text="episodes",
        title=(
            "Readmission Rate by Eligible "
            "Discharge Reason"
        ),
        labels={
            "rate": "Readmission rate (%)",
            "motivoalta": "Discharge reason",
            "episodes": "Episodes",
            "readmissions": "Readmissions"
        },
        hover_data={
            "episodes": ":,",
            "readmissions": ":,",
            "rate": ":.2f"
        }
    )

    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="outside"
    )

    fig.update_layout(
        height=450,
        yaxis_title=None
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "Episodes ending in death, transfer, or without "
        "a valid discharge date are not included because "
        "they are not eligible to originate a future "
        "readmission."
    )

def plot_discharge_reason_distribution(
    df: pd.DataFrame
) -> None:
    """
    Distribuição dos motivos de alta em todos os episódios.
    """

    if "motivoalta" not in df.columns:
        st.info("Column motivoalta not found.")
        return

    discharge_distribution = (
        df.assign(
            motivoalta=df["motivoalta"]
            .fillna("Não informado")
        )
        .groupby("motivoalta", dropna=False)
        .size()
        .reset_index(name="episodes")
        .sort_values(
            "episodes",
            ascending=True
        )
    )

    if discharge_distribution.empty:
        st.info(
            "No discharge reason data available."
        )
        return

    fig = px.bar(
        discharge_distribution,
        x="episodes",
        y="motivoalta",
        orientation="h",
        text="episodes",
        title="Distribution of Discharge Reasons",
        labels={
            "episodes": "Episodes",
            "motivoalta": "Discharge reason"
        }
    )

    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="outside"
    )

    fig.update_layout(
        height=500,
        yaxis_title=None
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

def plot_model_eligibility(
    df: pd.DataFrame
) -> None:

    if "eligible_for_readmission_model" not in df.columns:
        st.info(
            "Eligibility column not found."
        )
        return

    eligibility = (
        df["eligible_for_readmission_model"]
        .map({
            1: "Eligible",
            0: "Not eligible"
        })
        .fillna("Unknown")
        .value_counts()
        .rename_axis("status")
        .reset_index(name="episodes")
    )

    if eligibility.empty:
        st.info("No eligibility data available.")
        return

    fig = px.pie(
        eligibility,
        names="status",
        values="episodes",
        hole=0.45,
        title="Eligibility for Readmission Prediction"
    )

    fig.update_traces(
        textinfo="percent+label",
        hovertemplate=(
            "%{label}<br>"
            "Episodes: %{value:,}<br>"
            "Percentage: %{percent}"
            "<extra></extra>"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

def plot_non_eligible_reasons(
    df: pd.DataFrame
) -> None:

    required_columns = {
        "eligible_for_readmission_model",
        "motivoalta",
        "discharge_ts"
    }

    if not required_columns.issubset(df.columns):
        st.info(
            "Required columns for model eligibility "
            "chart were not found."
        )
        return

    excluded_df = df[
        df["eligible_for_readmission_model"] == 0
    ].copy()

    if excluded_df.empty:
        st.info(
            "No non-eligible episodes found."
        )
        return

    excluded_df["exclusion_reason"] = "Other"

    excluded_df.loc[
        excluded_df["motivoalta"].eq("Óbito"),
        "exclusion_reason"
    ] = "Death"

    excluded_df.loc[
        excluded_df["motivoalta"].eq("Transferência"),
        "exclusion_reason"
    ] = "Transfer"

    excluded_df.loc[
        excluded_df["discharge_ts"].isna(),
        "exclusion_reason"
    ] = "Missing discharge"

    exclusion_summary = (
        excluded_df
        .groupby("exclusion_reason")
        .size()
        .reset_index(name="episodes")
        .sort_values(
            "episodes",
            ascending=True
        )
    )

    fig = px.bar(
        exclusion_summary,
        x="episodes",
        y="exclusion_reason",
        orientation="h",
        text="episodes",
        title="Reasons for Model Ineligibility",
        labels={
            "episodes": "Episodes",
            "exclusion_reason": "Reason"
        }
    )

    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="outside"
    )

    fig.update_layout(
        yaxis_title=None
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    
# ==========================================================
# Heatmap Hospital x Specialty
# ==========================================================

def plot_heatmap(df, min_episodes=100):

    heatmap = (
        df.groupby(["executante", "especialidade"])
        .agg(
            episodes=("readmitted_30d_clean", "count"),
            readmissions=("readmitted_30d_clean", "sum")
        )
        .reset_index()
    )

    heatmap = heatmap[
        heatmap["episodes"] >= min_episodes
    ]

    if heatmap.empty:
        st.info("No data available for heatmap.")
        return

    heatmap["rate"] = (
        heatmap["readmissions"]
        / heatmap["episodes"]
        * 100
    )

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

    heatmap = heatmap[
        heatmap["executante"].isin(top_hospitals)
        & heatmap["especialidade"].isin(top_specialties)
    ]

    pivot = heatmap.pivot(
        index="executante",
        columns="especialidade",
        values="rate"
    )

    fig = px.imshow(
        pivot,
        aspect="auto",
        title="Readmission Rate Heatmap: Hospital x Specialty",
        labels=dict(
            x="Specialty",
            y="Hospital",
            color="Rate (%)"
        )
    )

    fig.update_layout(height=700)

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# Previous Hospitalizations
# ==========================================================

def plot_previous_hospitalizations(df):

    history = (
        df.groupby("previous_hospitalizations")
        .agg(
            episodes=("readmitted_30d_clean", "count"),
            readmissions=("readmitted_30d_clean", "sum")
        )
        .reset_index()
    )

    history["rate"] = (
        history["readmissions"]
        / history["episodes"]
        * 100
    )

    history = history.sort_values(
        "previous_hospitalizations"
    ).head(30)

    fig = px.line(
        history,
        x="previous_hospitalizations",
        y="rate",
        markers=True,
        title="Readmission Rate vs Previous Hospitalizations",
        labels={
            "previous_hospitalizations": "Previous Hospitalizations",
            "rate": "Readmission Rate (%)"
        }
    )

    fig.update_traces(
        line_width=3,
        marker_size=8
    )

    fig.update_layout(height=500)

    st.plotly_chart(fig, use_container_width=True)

def plot_discharge_outcomes(df):

    if "motivoalta" not in df.columns:
        st.info("Column motivoalta not found.")
        return

    outcomes = (
        df.assign(
            motivoalta=df["motivoalta"]
            .fillna("Não informado")
        )
        .groupby("motivoalta")
        .size()
        .reset_index(name="episodes")
        .sort_values(
            "episodes",
            ascending=False
        )
    )

    if outcomes.empty:
        st.info("No discharge outcome data available.")
        return

    fig = px.bar(
        outcomes,
        x="episodes",
        y="motivoalta",
        orientation="h",
        text="episodes",
        title="Hospital Episodes by Discharge Outcome",
        labels={
            "episodes": "Episodes",
            "motivoalta": "Discharge outcome"
        }
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        yaxis={
            "categoryorder": "total ascending"
        },
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

def plot_deaths_by_hospital(
    df,
    min_episodes=100,
    top_n=20
):

    hospital_summary = (
        df.groupby("executante")
        .agg(
            episodes=(
                "motivoalta",
                "count"
            ),
            deaths=(
                "motivoalta",
                lambda values: (
                    values == "Óbito"
                ).sum()
            )
        )
        .reset_index()
    )

    hospital_summary = hospital_summary[
        hospital_summary["episodes"]
        >= min_episodes
    ].copy()

    hospital_summary["death_rate"] = (
        hospital_summary["deaths"]
        / hospital_summary["episodes"]
        * 100
    )

    hospital_summary = (
        hospital_summary
        .sort_values(
            "death_rate",
            ascending=False
        )
        .head(top_n)
    )

    if hospital_summary.empty:
        st.info(
            "No hospitals satisfy the minimum episode filter."
        )
        return

    fig = px.bar(
        hospital_summary,
        x="death_rate",
        y="executante",
        orientation="h",
        text="episodes",
        title="In-Hospital Mortality by Hospital",
        labels={
            "death_rate": "Mortality (%)",
            "executante": "Hospital"
        }
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        yaxis={
            "categoryorder": "total ascending"
        },
        height=600
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )