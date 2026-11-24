import pandas as pd
import streamlit as st


TARGET_COLUMN = "unplanned_readmitted_30d"


def show_metrics(df: pd.DataFrame) -> None:
    """
    Display compact dashboard KPIs.
    """

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
        if total_episodes > 0
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

    average_los_text = (
        f"{average_los:.1f} days"
        if pd.notna(average_los)
        else "-"
    )

    # Custom KPI cards are rendered with st.html so the HTML is never
    # interpreted as Markdown/code. The title and value are intentionally
    # close together, and all cards have a solid white background.
    html = f"""
<style>
.kpi-row {{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 14px;
    margin: 0 0 14px 0;
    width: 100%;
}}
.kpi-card {{
    background: #ffffff;
    border: 1px solid #d9dee7;
    border-radius: 8px;
    min-height: 92px;
    padding: 11px 14px 10px 14px;
    box-sizing: border-box;
}}
.kpi-title {{
    color: #687386;
    font-size: 12px;
    line-height: 1;
    font-weight: 500;
    margin: 0;
    padding: 0;
}}
.kpi-value {{
    color: #111827;
    font-size: 25px;
    line-height: 1;
    font-weight: 750;
    margin: 0;
    letter-spacing: -0.4px;
}}
.kpi-combined {{
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 12px;
    margin: 0;
}}
.kpi-main {{
    color: #111827;
    font-size: 25px;
    line-height: 1;
    font-weight: 750;
    margin: 0;
    letter-spacing: -0.4px;
}}
.kpi-secondary {{
    color: #3157d5;
    font-size: 23px;
    line-height: 1;
    font-weight: 750;
    margin: 0;
    text-align: right;
    white-space: nowrap;
}}
.kpi-secondary-label {{
    color: #7b8492;
    font-size: 10px;
    line-height: 1;
    font-weight: 600;
    margin: 0 0 4px 0;
    text-align: right;
    letter-spacing: 0.15px;
}}
</style>

<div class="kpi-row">
    <div class="kpi-card">
        <div class="kpi-title">Hospitalization Episodes</div>
        <div class="kpi-value">{total_episodes:,}</div>
    </div>

    <div class="kpi-card">
        <div class="kpi-title">Unplanned Readmissions</div>
        <div class="kpi-combined">
            <div class="kpi-main">{unplanned_readmissions:,}</div>
            <div>
                <div class="kpi-secondary-label">RATE</div>
                <div class="kpi-secondary">{readmission_rate:.2f}%</div>
            </div>
        </div>
    </div>

    <div class="kpi-card">
        <div class="kpi-title">Unique Patients</div>
        <div class="kpi-combined">
            <div class="kpi-main">{unique_patients:,}</div>
            <div>
                <div class="kpi-secondary-label">WITH READMISSION</div>
                <div class="kpi-secondary">{readmitted_patients:,}</div>
            </div>
        </div>
    </div>

    <div class="kpi-card">
        <div class="kpi-title">Average Length of Stay</div>
        <div class="kpi-value">{average_los_text}</div>
    </div>
</div>
"""

    if hasattr(st, "html"):
        st.html(html)
    else:
        st.markdown(
            html.replace("<style>", "").replace("</style>", ""),
            unsafe_allow_html=True,
        )

    st.divider()

