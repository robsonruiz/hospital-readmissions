from pathlib import Path
import pandas as pd
import streamlit as st

MODELS_DIR = Path("models")
METRICS_FILE = MODELS_DIR / "clustering_metrics.csv"
PROFILE_FILE = MODELS_DIR / "clustering_profiles.csv"
PCA_FILE = MODELS_DIR / "clustering_pca.csv"


def _load(path):
    return pd.read_csv(path) if path.exists() else None


def render_clustering():
    st.header("Clusterização")

    metrics = _load(METRICS_FILE)
    profile = _load(PROFILE_FILE)
    pca = _load(PCA_FILE)

    if any(x is None for x in (metrics, profile, pca)):
        st.warning(
            "Os resultados ainda não foram gerados. Execute "
            "`python -m src.analytics.run_clustering`."
        )
        return

    best = metrics.loc[metrics["silhouette_score"].idxmax()]

    c1, c2 = st.columns(2)
    c1.metric("Número de clusters", int(best["n_clusters"]))
    c2.metric("Silhouette Score", f'{best["silhouette_score"]:.3f}')

    st.subheader("Avaliação do número de clusters")
    metrics_display = metrics.copy()
    metrics_display["silhouette_score"] = metrics_display["silhouette_score"].round(3)
    st.dataframe(metrics_display, use_container_width=True, hide_index=True)

    st.subheader("Distribuição dos clusters")
    counts = profile[["cluster", "episodes"]].set_index("cluster")
    st.bar_chart(counts)

    st.subheader("Perfil dos clusters")
    display = profile.copy()
    for col in display.select_dtypes("number").columns:
        if col not in ("cluster", "episodes"):
            if "rate" in col:
                display[col] = (display[col] * 100).round(2)
            else:
                display[col] = display[col].round(2)
    st.dataframe(display, use_container_width=True, hide_index=True)

    if "unplanned_readmission_rate" in profile.columns:
        st.subheader("Taxa de reinternação não planejada por cluster")
        rates = profile[["cluster", "unplanned_readmission_rate"]].copy()
        rates["unplanned_readmission_rate"] *= 100
        rates = rates.set_index("cluster")
        st.bar_chart(rates)

    st.subheader("Visualização dos clusters")
    st.caption("Projeção bidimensional por PCA; o PCA é usado somente para visualização.")

    plot = pca[["pca_1", "pca_2", "cluster"]].copy()
    plot["cluster"] = "Cluster " + plot["cluster"].astype(str)
    st.scatter_chart(plot, x="pca_1", y="pca_2", color="cluster")
