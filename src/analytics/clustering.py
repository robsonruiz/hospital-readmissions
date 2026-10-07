from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import duckdb

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


DATABASE_PATH = Path(
    "data/hospital_readmissions.duckdb"
)

OUTPUT_DIR = Path("models")

RANDOM_STATE = 42

# Apenas uma amostra é usada para escolher o melhor K.
SELECTION_SAMPLE_SIZE = 20_000

# Amostra usada para calcular o Silhouette Score.
SILHOUETTE_SAMPLE_SIZE = 10_000

# Tamanho dos lotes do MiniBatchKMeans.
BATCH_SIZE = 4096

# Número de clusters avaliados.
K_RANGE = range(2, 7)

TARGET = "unplanned_readmitted_30d"


# ============================================================
# Variáveis utilizadas na clusterização
# ============================================================

BASE_FEATURES = [
    "previous_hospitalizations",
    "previous_unplanned_readmissions",
    "admissions_last_30d",
    "admissions_last_90d",
    "admissions_last_365d",
    "days_since_previous_hospitalization",
]


# ============================================================
# Carregamento
# ============================================================

def load_ml_features(
    database_path=DATABASE_PATH,
):

    con = duckdb.connect(
        str(database_path),
        read_only=True,
    )

    try:

        return con.execute(
            """
            SELECT *
            FROM ml_features
            """
        ).df()

    finally:

        con.close()


# ============================================================
# Seleção das variáveis
# ============================================================

def select_features(df):

    features = [
        column
        for column in BASE_FEATURES
        if column in df.columns
    ]

    if not features:

        raise ValueError(
            "Nenhuma das variáveis selecionadas "
            "para clusterização foi encontrada "
            "em ml_features."
        )

    X = df[
        features
    ].copy()

    for column in features:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X = X.fillna(
        X.median(
            numeric_only=True
        )
    )

    return X, features


# ============================================================
# Amostragem
# ============================================================

def get_sample_indices(
    n_rows,
    sample_size,
):

    sample_size = min(
        sample_size,
        n_rows,
    )

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    return rng.choice(
        n_rows,
        size=sample_size,
        replace=False,
    )


# ============================================================
# Escolha do número de clusters
# ============================================================

def choose_best_k(X):

    print(
        "\nPreparing sample for "
        "cluster selection..."
    )

    selection_indices = get_sample_indices(
        len(X),
        SELECTION_SAMPLE_SIZE,
    )

    X_selection = X.iloc[
        selection_indices
    ]

    scaler = StandardScaler()

    X_selection_scaled = scaler.fit_transform(
        X_selection
    )

    silhouette_indices = get_sample_indices(
        len(X_selection_scaled),
        SILHOUETTE_SAMPLE_SIZE,
    )

    X_silhouette = (
        X_selection_scaled[
            silhouette_indices
        ]
    )

    rows = []

    print(
        f"Selection sample: "
        f"{len(X_selection):,} episodes"
    )

    print(
        "\nEvaluating number of clusters..."
    )

    for k in K_RANGE:

        print(
            f"  K={k}"
        )

        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            batch_size=BATCH_SIZE,
            n_init=3,
            max_iter=100,
        )

        labels = model.fit_predict(
            X_selection_scaled
        )

        silhouette_labels = labels[
            silhouette_indices
        ]

        score = silhouette_score(
            X_silhouette,
            silhouette_labels,
        )

        rows.append(
            {
                "n_clusters": k,
                "silhouette_score": score,
            }
        )

        print(
            f"    Silhouette: "
            f"{score:.4f}"
        )

    metrics = pd.DataFrame(
        rows
    )

    best_k = int(
        metrics.loc[
            metrics[
                "silhouette_score"
            ].idxmax(),
            "n_clusters",
        ]
    )

    print(
        f"\nBest K: {best_k}"
    )

    return (
        best_k,
        metrics,
    )


# ============================================================
# Clusterização final
# ============================================================

def fit_final_clustering(
    X,
    n_clusters,
):

    print(
        "\nScaling full dataset..."
    )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    print(
        f"Training MiniBatchKMeans "
        f"with K={n_clusters}..."
    )

    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        batch_size=BATCH_SIZE,
        n_init=3,
        max_iter=100,
    )

    labels = model.fit_predict(
        X_scaled
    )

    return (
        model,
        scaler,
        labels,
        X_scaled,
    )


# ============================================================
# Perfil dos clusters
# ============================================================

def build_profile(
    df,
    labels,
    features,
):

    work = df.copy()

    work["cluster"] = (
        labels.astype(int)
    )

    aggregations = {
        "episodes": (
            "cluster",
            "size",
        )
    }

    for feature in features:

        aggregations[
            feature
        ] = (
            feature,
            "mean",
        )

    profile = (
        work
        .groupby(
            "cluster",
            as_index=False,
        )
        .agg(
            **aggregations
        )
    )

    # O target NÃO participa da formação
    # dos clusters. Ele é utilizado apenas
    # para caracterizar os grupos depois.

    if TARGET in work.columns:

        work["_target"] = pd.to_numeric(
            work[TARGET],
            errors="coerce",
        )

        target_profile = (
            work
            .groupby(
                "cluster",
                as_index=False,
            )["_target"]
            .mean()
            .rename(
                columns={
                    "_target":
                    "unplanned_readmission_rate"
                }
            )
        )

        profile = profile.merge(
            target_profile,
            on="cluster",
            how="left",
        )

    return profile.sort_values(
        "cluster"
    )


# ============================================================
# PCA
# ============================================================

def build_pca(
    X_scaled,
    labels,
):

    print(
        "\nCalculating PCA..."
    )

    # PCA é utilizado somente para
    # visualização.
    #
    # Para evitar custo desnecessário,
    # ajustamos o PCA em uma amostra.
    pca_sample_indices = (
        get_sample_indices(
            len(X_scaled),
            SELECTION_SAMPLE_SIZE,
        )
    )

    X_pca_sample = (
        X_scaled[
            pca_sample_indices
        ]
    )

    pca = PCA(
        n_components=2,
        random_state=RANDOM_STATE,
    )

    pca.fit(
        X_pca_sample
    )

    # Transformamos todos os registros.
    components = pca.transform(
        X_scaled
    )

    result = pd.DataFrame(
        {
            "pca_1": components[:, 0],
            "pca_2": components[:, 1],
            "cluster": labels.astype(int),
        }
    )

    explained = (
        pca.explained_variance_ratio_
    )

    return (
        result,
        explained,
    )


# ============================================================
# Execução principal
# ============================================================

def run_clustering(
    database_path=DATABASE_PATH,
    output_dir=OUTPUT_DIR,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nLoading ml_features..."
    )

    df = load_ml_features(
        database_path
    )

    print(
        f"Dataset: "
        f"{len(df):,} episodes"
    )

    X, features = select_features(
        df
    )

    print(
        "\nVariables used:"
    )

    for feature in features:

        print(
            f"  - {feature}"
        )

    # --------------------------------------------------------
    # Escolha de K
    # --------------------------------------------------------

    best_k, metrics = (
        choose_best_k(X)
    )

    # --------------------------------------------------------
    # Clusterização final
    # --------------------------------------------------------

    (
        model,
        scaler,
        labels,
        X_scaled,
    ) = fit_final_clustering(
        X,
        best_k,
    )

    # --------------------------------------------------------
    # Perfil
    # --------------------------------------------------------

    print(
        "\nBuilding cluster profiles..."
    )

    profile = build_profile(
        df,
        labels,
        features,
    )

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    pca_data, explained = (
        build_pca(
            X_scaled,
            labels,
        )
    )

    # --------------------------------------------------------
    # Resultados dos clusters
    # --------------------------------------------------------

    results = pd.DataFrame(
        {
            "row_id": np.arange(
                len(labels)
            ),
            "cluster": labels.astype(
                int
            ),
        }
    )

    pca_data.insert(
        0,
        "row_id",
        np.arange(
            len(pca_data)
        ),
    )

    # --------------------------------------------------------
    # Salvamento
    # --------------------------------------------------------

    print(
        "\nSaving results..."
    )

    metrics.to_csv(
        output_dir
        / "clustering_metrics.csv",
        index=False,
    )

    profile.to_csv(
        output_dir
        / "clustering_profiles.csv",
        index=False,
    )

    results.to_csv(
        output_dir
        / "clustering_results.csv",
        index=False,
    )

    pca_data.to_csv(
        output_dir
        / "clustering_pca.csv",
        index=False,
    )

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "features": features,
            "best_k": best_k,
            "silhouette_score": float(
                metrics.loc[
                    metrics[
                        "n_clusters"
                    ]
                    == best_k,
                    "silhouette_score",
                ].iloc[0]
            ),
        },
        output_dir
        / "clustering_model.joblib",
    )

    # --------------------------------------------------------
    # Resumo
    # --------------------------------------------------------

    print(
        "\n================================"
    )

    print(
        "CLUSTERING COMPLETED"
    )

    print(
        "================================"
    )

    print(
        f"Episodes: "
        f"{len(df):,}"
    )

    print(
        f"Best K: "
        f"{best_k}"
    )

    best_score = float(
        metrics.loc[
            metrics[
                "n_clusters"
            ]
            == best_k,
            "silhouette_score",
        ].iloc[0]
    )

    print(
        f"Silhouette Score: "
        f"{best_score:.4f}"
    )

    print(
        "PCA variance explained: "
        f"{explained[0]:.2%} + "
        f"{explained[1]:.2%}"
    )

    print(
        "\nCluster sizes:"
    )

    print(
        pd.Series(
            labels
        )
        .value_counts()
        .sort_index()
        .rename(
            "episodes"
        )
        .to_string()
    )

    print(
        "\nFiles saved:"
    )

    print(
        "  models/clustering_metrics.csv"
    )

    print(
        "  models/clustering_profiles.csv"
    )

    print(
        "  models/clustering_results.csv"
    )

    print(
        "  models/clustering_pca.csv"
    )

    print(
        "  models/clustering_model.joblib"
    )

    return {
        "best_k": best_k,
        "silhouette_score": best_score,
        "features": features,
        "metrics": metrics,
        "profile": profile,
        "pca": pca_data,
    }


if __name__ == "__main__":

    run_clustering()