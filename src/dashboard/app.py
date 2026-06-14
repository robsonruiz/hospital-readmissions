import duckdb
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Hospital Readmissions Dashboard",
    layout="wide"
)

@st.cache_data
def load_data():
    conn = duckdb.connect("data/hospital_readmissions.duckdb")
    df = conn.execute("SELECT * FROM ml_features").fetchdf()
    conn.close()
    return df

df = load_data()

# Sanitização de tipos
if "length_of_stay_days" in df.columns:
    if df["length_of_stay_days"].dtype == "O":
        # trocar vírgula por ponto se houver e converter
        df["length_of_stay_days"] = (
            df["length_of_stay_days"]
            .astype(str)
            .str.replace(",", ".", regex=False)
        )
    df["length_of_stay_days"] = pd.to_numeric(df["length_of_stay_days"], errors="coerce")

st.title("Hospital Readmissions Dashboard")
st.markdown("Analysis of hospital readmissions within 30 days.")

# -------------------------
# KPIs (linha única)
# -------------------------
total_episodes = len(df)
readmissions = df["readmitted_30d_clean"].sum()
readmission_rate = (readmissions / total_episodes * 100) if total_episodes else 0

k1, k2, k3 = st.columns(3)
with k1:
    st.metric("Total Episodes", f"{total_episodes:,}")
with k2:
    st.metric("Readmissions", f"{readmissions:,}")
with k3:
    st.metric("Readmission Rate", f"{readmission_rate:.2f}%")

st.divider()

# -------------------------
# Preparos de dados
# -------------------------

# Readmission by specialty
specialty = (
    df.groupby("especialidade", dropna=False)
    .agg(
        episodes=("readmitted_30d_clean", "count"),
        readmissions=("readmitted_30d_clean", "sum")
    )
    .reset_index()
)
specialty["rate"] = specialty["readmissions"] / specialty["episodes"] * 100
specialty = specialty.sort_values("rate", ascending=False)

# LOS limpo
los_clean = df["length_of_stay_days"].dropna()
# Opcional: cortar outliers extremos para melhor visual
if len(los_clean) > 0:
    upper = los_clean.quantile(0.99)
    los_clean = los_clean[(los_clean >= 0) & (los_clean <= upper)]

# Readmission vs LOS (média por status)
los = (
    df.groupby("readmitted_30d_clean")["length_of_stay_days"]
    .mean()
    .reset_index()
)

# Top ICD codes
cid = (
    df[df["readmitted_30d_clean"] == 1]
    .groupby("codigocid")
    .size()
    .reset_index(name="readmissions")
    .sort_values("readmissions", ascending=False)
    .head(20)
)

# -------------------------
# Layout em 2 colunas (2 gráficos por linha)
# Use full-width apenas para gráficos muito largos
# -------------------------

# Linha 1: Specialty (largo) em largura total
st.subheader("Readmission Rate by Specialty")
fig_specialty = px.bar(
    specialty.head(20),  # limitar a 20 ajuda a não ficar superlargo
    x="rate",
    y="especialidade",
    orientation="h",
    labels={"rate": "Readmission rate (%)", "especialidade": "Especialidade"}
)
st.plotly_chart(fig_specialty, use_container_width=True)

# Linha 2: LOS hist e Readmission vs LOS lado a lado
c1, c2 = st.columns(2)

with c1:
    st.subheader("Length of Stay Distribution & Readmissions")
    if len(los_clean) == 0:
        st.info("Sem dados válidos de length_of_stay_days para plotar.")
    else:
        import plotly.graph_objects as go

        # 1. Filtrar o DataFrame original usando as regras do seu los_clean
        # Isso garante que todos os dados tenham o mesmo tamanho e contexto
        upper_limit = df["length_of_stay_days"].quantile(0.99)
        df_filtered = df[
            df["length_of_stay_days"].notna() & 
            (df["length_of_stay_days"] >= 0) & 
            (df["length_of_stay_days"] <= upper_limit)
        ]

        # 2. Agrupar as reinternações APENAS dentro desse subset filtrado
        df_reint_los = (
            df_filtered[df_filtered["readmitted_30d_clean"] == 1]
            .groupby("length_of_stay_days")
            .size()
            .reset_index(name="qtd_reinternacoes")
        )

        # 3. Criar a base do histograma usando o DataFrame filtrado
        fig_combined = px.histogram(
            df_filtered,
            x="length_of_stay_days",
            nbins=50,
            labels={"length_of_stay_days": "Length of stay (dias)", "y": "Total de Episódios (Barras)"},
            opacity=0.75
        )

        # 4. Criar a linha de reinternações
        fig_line = go.Scatter(
            x=df_reint_los["length_of_stay_days"],
            y=df_reint_los["qtd_reinternacoes"],
            name="Reinternações",
            mode="lines+markers",
            line=dict(color="red", width=2),
            yaxis="y2"  # Vincula ao eixo secundário
        )

        # 5. Juntar a linha ao histograma e configurar os eixos
        fig_combined.add_trace(fig_line)
        fig_combined.update_layout(
            yaxis=dict(title="Total de Episódios (Barras)"),
            yaxis2=dict(
                title="Qtd de Reinternações (Linha)",
                overlaying="y",
                side="right"
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig_combined, use_container_width=True)
        
with c2:
    st.subheader("Average Length of Stay by Readmission Status")
    fig_los_bar = px.bar(
        los,
        x="readmitted_30d_clean",
        y="length_of_stay_days",
        labels={
            "readmitted_30d_clean": "Readmitido em 30d (0/1)",
            "length_of_stay_days": "Média LOS (dias)"
        }
    )
    st.plotly_chart(fig_los_bar, use_container_width=True)

# Linha 3: Top ICD codes (pode ir lado a lado com algo; se não houver, ocupar uma coluna)
c3, c4 = st.columns(2)
with c3:
    st.subheader("Top ICD Codes Associated with Readmissions")
    fig_cid = px.bar(
        cid,
        x="codigocid",
        y="readmissions",
        labels={"codigocid": "ICD", "readmissions": "Readmissions"}
    )
    st.plotly_chart(fig_cid, use_container_width=True)

with c4:
    st.subheader("Dataset Sample")
    st.dataframe(df.head(100), use_container_width=True)
