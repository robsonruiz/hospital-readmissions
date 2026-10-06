from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import duckdb
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATABASE_PATH = Path("data/hospital_readmissions.duckdb")
OUTPUT_DIR = Path("models")
RANDOM_STATE = 42
K_RANGE = range(2, 7)
TARGET = "unplanned_readmitted_30d"

BASE_FEATURES = [
    "previous_hospitalizations",
    "previous_unplanned_readmissions",
    "admissions_last_30d",
    "admissions_last_90d",
    "admissions_last_365d",
    "days_since_previous_hospitalization",
]


def load_ml_features(database_path=DATABASE_PATH):
    con = duckdb.connect(str(database_path), read_only=True)
    try:
        return con.execute("SELECT * FROM ml_features").df()
    finally:
        con.close()


def select_features(df):
    features = [c for c in BASE_FEATURES if c in df.columns]
    if not features:
        raise ValueError("Nenhuma variável de clusterização foi encontrada em ml_features.")

    X = df[features].copy()
    for col in features:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    return X, features


def fit_kmeans(X, k):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=20,
    )
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    return model, scaler, labels, score, X_scaled


def choose_best_k(X):
    rows = []
    for k in K_RANGE:
        _, _, _, score, _ = fit_kmeans(X, k)
        rows.append({"n_clusters": k, "silhouette_score": score})

    metrics = pd.DataFrame(rows)
    best_k = int(metrics.loc[metrics["silhouette_score"].idxmax(), "n_clusters"])
    return best_k, metrics


def build_profile(df, labels, features):
    work = df.copy()
    work["cluster"] = labels

    aggregations = {"episodes": ("cluster", "size")}
    aggregations.update({feature: (feature, "mean") for feature in features})

    profile = work.groupby("cluster", as_index=False).agg(**aggregations)

    if TARGET in work.columns:
        work["_target"] = pd.to_numeric(work[TARGET], errors="coerce")
        target_profile = (
            work.groupby("cluster", as_index=False)["_target"]
            .mean()
            .rename(columns={"_target": "unplanned_readmission_rate"})
        )
        profile = profile.merge(target_profile, on="cluster", how="left")

    return profile.sort_values("cluster")


def build_pca(X_scaled, labels):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    components = pca.fit_transform(X_scaled)

    data = pd.DataFrame({
        "pca_1": components[:, 0],
        "pca_2": components[:, 1],
        "cluster": labels.astype(int),
    })
    return data, pca.explained_variance_ratio_


def run_clustering(database_path=DATABASE_PATH, output_dir=OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_ml_features(database_path)
    X, features = select_features(df)
    best_k, metrics = choose_best_k(X)

    model, scaler, labels, best_score, X_scaled = fit_kmeans(X, best_k)
    profile = build_profile(df, labels, features)
    pca_data, explained = build_pca(X_scaled, labels)

    results = pd.DataFrame({
        "row_id": np.arange(len(labels)),
        "cluster": labels.astype(int),
    })

    pca_data.insert(0, "row_id", np.arange(len(pca_data)))

    metrics.to_csv(output_dir / "clustering_metrics.csv", index=False)
    profile.to_csv(output_dir / "clustering_profiles.csv", index=False)
    results.to_csv(output_dir / "clustering_results.csv", index=False)
    pca_data.to_csv(output_dir / "clustering_pca.csv", index=False)

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "features": features,
            "best_k": best_k,
            "silhouette_score": best_score,
        },
        output_dir / "clustering_model.joblib",
    )

    print("\nClusterização concluída.")
    print(f"Variáveis: {', '.join(features)}")
    print(f"Melhor k: {best_k}")
    print(f"Silhouette Score: {best_score:.4f}")
    print(f"PCA: {explained[0]:.2%} + {explained[1]:.2%}")

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
