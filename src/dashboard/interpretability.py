import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.pipeline import Pipeline

from src.ml.prepare_data import load_ml_data, prepare_data


# =============================================================================
# Configuração
# =============================================================================
MODEL_DIR = PROJECT_ROOT / "models" / "trajectory_experiment"
OUTPUT_DIR = PROJECT_ROOT / "models" / "interpretability_trajectory"

SHAP_SAMPLE_SIZE = 300
SHAP_BATCH_SIZE = 50
RANDOM_STATE = 42

DATASETS = {
    "A — Baseline": {
        "key": "A_baseline",
        "path": MODEL_DIR / "a_baseline.parquet",
    },
    "B — Trajetória completa": {
        "key": "B_trajectory",
        "path": MODEL_DIR / "b_trajectory.parquet",
    },
    "C — APS": {
        "key": "C_aps",
        "path": MODEL_DIR / "c_aps.parquet",
    },
}

MODELS = {
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "CatBoost": "catboost",
}


# =============================================================================
# Dados e modelos
# =============================================================================
@st.cache_data
def load_official_cutoffs():
    df = load_ml_data()
    prepared = prepare_data(df)
    return prepared[9], prepared[10]


@st.cache_data
def load_dataset(dataset_key):
    dataset_info = next(
        item
        for item in DATASETS.values()
        if item["key"] == dataset_key
    )

    path = dataset_info["path"]

    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {path}")

    df = pd.read_parquet(path)
    df["admission_ts"] = pd.to_datetime(
        df["admission_ts"],
        errors="coerce",
    )

    if df["admission_ts"].isna().any():
        raise ValueError(
            f"{dataset_key}: admission_ts contém valores inválidos."
        )

    return df


@st.cache_resource
def load_model(dataset_key, model_key):
    model_path = (
        MODEL_DIR
        / f"{dataset_key}_tuned_{model_key}.joblib"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {model_path}"
        )

    return joblib.load(model_path)


def get_validation_data(dataset_key):
    _, validation_cutoff = load_official_cutoffs()
    df = load_dataset(dataset_key)

    dates = df["admission_ts"].dt.normalize()

    validation_mask = dates <= validation_cutoff

    # The lower bound must be the official train cutoff.
    train_cutoff, _ = load_official_cutoffs()
    validation_mask &= dates > train_cutoff

    validation_df = df.loc[validation_mask].copy()

    if validation_df.empty:
        raise ValueError(
            f"{dataset_key}: conjunto de validação vazio."
        )

    target = "unplanned_readmitted_30d"

    if target not in validation_df.columns:
        raise ValueError(
            f"{dataset_key}: coluna alvo '{target}' não encontrada."
        )

    y_valid = pd.to_numeric(
        validation_df[target],
        errors="coerce",
    )

    if y_valid.isna().any():
        raise ValueError(
            f"{dataset_key}: target contém valores ausentes."
        )

    y_valid = y_valid.astype(int)

    return validation_df, y_valid


def get_features(df):
    excluded = {
        "unplanned_readmitted_30d",
        "episode_id",
        "identificador",
        "admission_ts",
        "discharge_ts",
        "previous_discharge_ts",
        "next_admission_ts",
        "days_until_next_admission",
        "readmitted_30d_raw",
        "readmitted_30d_clean",
        "eligible_for_readmission_model",
        "admission_month",
        "admission_year",
        "admission_quarter",
        "admission_weekday",
        "municipiosolicitante",
        "municipioresidencia",
        "municipioexecutante",
    }

    return [c for c in df.columns if c not in excluded]


def prepare_validation_features(dataset_key):
    validation_df, y_valid = get_validation_data(dataset_key)
    features = get_features(validation_df)
    X_valid = validation_df[features].copy()

    return X_valid, y_valid


def sample_validation_data(X_valid, y_valid):
    if len(X_valid) <= SHAP_SAMPLE_SIZE:
        return X_valid.copy(), y_valid.copy()

    rng = np.random.RandomState(RANDOM_STATE)
    indices = rng.choice(
        len(X_valid),
        SHAP_SAMPLE_SIZE,
        replace=False,
    )

    return (
        X_valid.iloc[indices].copy(),
        y_valid.iloc[indices].copy(),
    )


def prepare_catboost_data(X, model):
    X = X.copy()

    if not hasattr(model, "get_cat_feature_indices"):
        return X

    for index in model.get_cat_feature_indices():
        column = X.columns[index]
        X[column] = X[column].fillna("missing").astype(str)

    return X


# =============================================================================
# SHAP
# =============================================================================
def normalize_shap_values(values):
    if isinstance(values, list):
        return np.asarray(
            values[1] if len(values) > 1 else values[0]
        )

    values = np.asarray(values)

    if values.ndim == 3:
        if values.shape[-1] == 2:
            return values[:, :, 1]
        if values.shape[1] == 2:
            return values[:, 1, :]

    return values


def get_pipeline_components(model):
    if not isinstance(model, Pipeline):
        return None, model

    if "preprocessor" not in model.named_steps:
        return None, model.steps[-1][1]

    return (
        model.named_steps["preprocessor"],
        model.steps[-1][1],
    )


def get_original_feature_names(preprocessor):
    feature_names = preprocessor.get_feature_names_out()
    original_columns = []

    for name in feature_names:
        clean_name = name.split("__", 1)[-1]

        matches = [
            column
            for column in preprocessor.feature_names_in_
            if clean_name == column
            or clean_name.startswith(f"{column}_")
        ]

        if matches:
            original_columns.append(
                max(matches, key=len)
            )
        else:
            original_columns.append(clean_name)

    return feature_names, original_columns


def aggregate_shap_values(
    shap_values,
    transformed_feature_names,
    original_feature_names,
    X_original,
):
    grouped = pd.DataFrame(
        shap_values,
        columns=transformed_feature_names,
        index=X_original.index,
    )

    aggregated = pd.DataFrame(index=X_original.index)

    for original_feature in dict.fromkeys(original_feature_names):
        columns = [
            transformed_feature_names[i]
            for i, name in enumerate(original_feature_names)
            if name == original_feature
        ]

        if columns:
            aggregated[original_feature] = grouped[
                columns
            ].sum(axis=1)

    return aggregated


def calculate_shap(model_name, model, X):
    import shap

    if model_name == "CatBoost":
        X_model = prepare_catboost_data(X, model)
        explainer = shap.TreeExplainer(model)

        shap_parts = []

        for start in range(0, len(X_model), SHAP_BATCH_SIZE):
            batch = X_model.iloc[
                start:start + SHAP_BATCH_SIZE
            ]

            values = normalize_shap_values(
                explainer.shap_values(batch)
            )

            shap_parts.append(
                np.asarray(values, dtype=np.float32)
            )

        shap_values = np.vstack(shap_parts)
        feature_names = list(X_model.columns)

        return (
            shap_values,
            X_model[feature_names].copy(),
        )

    preprocessor, estimator = get_pipeline_components(model)

    if preprocessor is None:
        explainer = shap.TreeExplainer(estimator)

        shap_parts = []

        for start in range(0, len(X), SHAP_BATCH_SIZE):
            batch = X.iloc[
                start:start + SHAP_BATCH_SIZE
            ]

            values = normalize_shap_values(
                explainer.shap_values(batch)
            )

            shap_parts.append(
                np.asarray(values, dtype=np.float32)
            )

        shap_values = np.vstack(shap_parts)

        return shap_values, X.copy()

    transformed_names, original_names = (
        get_original_feature_names(preprocessor)
    )

    explainer = shap.TreeExplainer(estimator)

    shap_parts = []

    for start in range(0, len(X), SHAP_BATCH_SIZE):
        batch = X.iloc[
            start:start + SHAP_BATCH_SIZE
        ]

        transformed = preprocessor.transform(batch)

        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        transformed = np.asarray(
            transformed,
            dtype=np.float32,
        )

        values = normalize_shap_values(
            explainer.shap_values(transformed)
        )

        shap_parts.append(
            np.asarray(values, dtype=np.float32)
        )

        del transformed

    shap_values_encoded = np.vstack(shap_parts)

    shap_values = aggregate_shap_values(
        shap_values_encoded,
        transformed_names,
        original_names,
        X,
    )

    return (
        shap_values.to_numpy(dtype=np.float32),
        X.copy(),
    )


def global_importance(shap_values, X_display):
    return (
        pd.DataFrame(
            {
                "feature": X_display.columns,
                "mean_abs_shap": np.abs(shap_values).mean(
                    axis=0
                ),
            }
        )
        .sort_values(
            "mean_abs_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# =============================================================================
# Predição e interações
# =============================================================================
def predict_positive_probability(model_name, model, X):
    if model_name == "CatBoost":
        X_model = prepare_catboost_data(X, model)
        return model.predict_proba(X_model)[:, 1]

    return model.predict_proba(X)[:, 1]


def feature_baselines(X):
    baselines = {}

    for column in X.columns:
        if pd.api.types.is_numeric_dtype(X[column]):
            baselines[column] = X[column].median()
        else:
            mode = X[column].mode(dropna=True)
            baselines[column] = (
                mode.iloc[0]
                if not mode.empty
                else "missing"
            )

    return baselines


def calculate_pairwise_interactions(
    model_name,
    model,
    X,
):
    if X.empty or len(X.columns) < 2:
        return pd.DataFrame(
            columns=[
                "feature_1",
                "feature_2",
                "interaction_importance",
                "mean_interaction",
            ]
        )

    baselines = feature_baselines(X)
    rows = []
    columns = list(X.columns)

    for i, feature_1 in enumerate(columns[:-1]):
        for feature_2 in columns[i + 1:]:
            original = X.copy()

            masked_1 = X.copy()
            masked_1[feature_1] = baselines[feature_1]

            masked_2 = X.copy()
            masked_2[feature_2] = baselines[feature_2]

            masked_both = X.copy()
            masked_both[feature_1] = baselines[feature_1]
            masked_both[feature_2] = baselines[feature_2]

            batch = pd.concat(
                [
                    original,
                    masked_1,
                    masked_2,
                    masked_both,
                ],
                ignore_index=True,
            )

            probabilities = predict_positive_probability(
                model_name,
                model,
                batch,
            )

            n = len(X)

            interaction = (
                probabilities[:n]
                - probabilities[n:2 * n]
                - probabilities[2 * n:3 * n]
                + probabilities[3 * n:4 * n]
            )

            rows.append(
                {
                    "feature_1": feature_1,
                    "feature_2": feature_2,
                    "interaction_importance": float(
                        np.mean(np.abs(interaction))
                    ),
                    "mean_interaction": float(
                        np.mean(interaction)
                    ),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "interaction_importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# =============================================================================
# Exportação
# =============================================================================
def make_key(dataset_key, model_key):
    return f"{dataset_key}_{model_key}"


def save_model_results(
    dataset_key,
    model_name,
    shap_values,
    X_display,
    importance,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    key = make_key(
        dataset_key,
        MODELS[model_name],
    )

    global_file = (
        OUTPUT_DIR
        / f"global_importance_{key}.csv"
    )

    shap_file = (
        OUTPUT_DIR
        / f"shap_values_{key}.csv"
    )

    result = importance.copy()
    result["dataset"] = dataset_key
    result["model"] = model_name

    result.to_csv(
        global_file,
        index=False,
    )

    rows = []

    for observation in range(len(X_display)):
        for feature_index, feature in enumerate(
            X_display.columns
        ):
            rows.append(
                {
                    "observation": observation,
                    "feature": feature,
                    "feature_value": X_display.iloc[
                        observation,
                        feature_index,
                    ],
                    "shap_value": shap_values[
                        observation,
                        feature_index,
                    ],
                }
            )

    pd.DataFrame(rows).to_csv(
        shap_file,
        index=False,
    )

    return global_file, shap_file


def save_interaction_results(
    dataset_key,
    model_name,
    interactions,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    key = make_key(
        dataset_key,
        MODELS[model_name],
    )

    interaction_file = (
        OUTPUT_DIR
        / f"interaction_importance_{key}.csv"
    )

    result = interactions.copy()
    result["dataset"] = dataset_key
    result["model"] = model_name

    result.to_csv(
        interaction_file,
        index=False,
    )

    return interaction_file


def save_comparison(results, filename):
    if not results:
        return None

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_file = OUTPUT_DIR / filename

    pd.concat(
        results,
        ignore_index=True,
    ).to_csv(
        comparison_file,
        index=False,
    )

    return comparison_file


# =============================================================================
# Visualizações
# =============================================================================
def plot_global_importance(importance, top_n):
    data = (
        importance.head(top_n)
        .sort_values("mean_abs_shap")
    )

    fig = px.bar(
        data,
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        labels={
            "mean_abs_shap": "Mean |SHAP value|",
            "feature": "Variável",
        },
        title=f"Top {top_n} variáveis por importância SHAP",
    )

    fig.update_layout(
        height=max(500, top_n * 32),
        margin=dict(
            l=10,
            r=20,
            t=60,
            b=20,
        ),
    )

    return fig


def _numeric_color_values(values):
    """Convert a feature column into numeric values for SHAP-style coloring."""
    series = pd.Series(values)

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            return numeric.fillna(numeric.median()).to_numpy(dtype=float)

    codes, _ = pd.factorize(
        series.astype("string").fillna("missing"),
        sort=True,
    )
    return codes.astype(float)


def _jitter_positions(values, seed=42):
    """Create deterministic vertical jitter so points form a beeswarm-like cloud."""
    rng = np.random.RandomState(seed)
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values

    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    if len(values) == 1:
        return rng.normal(0, 0.04, size=1)

    centered = (ranks / (len(values) - 1)) - 0.5
    return centered * 0.55 + rng.normal(0, 0.035, size=len(values))


def plot_shap_beeswarm(
    shap_values,
    X_display,
    top_features,
):
    """Create a SHAP beeswarm-style plot using the aggregated original features.

    The project aggregates encoded SHAP values back to the original feature level.
    This visualization therefore recreates the SHAP beeswarm semantics while
    preserving categorical variables that cannot be passed directly to
    shap.plots.beeswarm.
    """
    selected = top_features["feature"].tolist()
    selected = [
        feature
        for feature in selected
        if feature in X_display.columns
    ]

    if not selected:
        return go.Figure()

    # SHAP beeswarm convention: most important feature at the top.
    y_labels = list(reversed(selected))
    fig = go.Figure()

    for y_index, feature in enumerate(y_labels):
        feature_index = X_display.columns.get_loc(feature)
        shap_column = np.asarray(
            shap_values[:, feature_index],
            dtype=float,
        )
        feature_values = X_display[feature].to_numpy()
        color_values = _numeric_color_values(feature_values)

        jitter = _jitter_positions(
            shap_column,
            seed=RANDOM_STATE + y_index,
        )

        fig.add_trace(
            go.Scatter(
                x=shap_column,
                y=np.full(len(shap_column), y_index) + jitter,
                mode="markers",
                name=feature,
                showlegend=False,
                marker={
                    "size": 7,
                    "opacity": 0.72,
                    "color": color_values,
                    "colorscale": "RdBu_r",
                    "showscale": False,
                    "line": {"width": 0},
                },
                customdata=np.column_stack([
                    np.asarray(feature_values, dtype=object),
                    shap_column,
                ]),
                hovertemplate=(
                    f"<b>{feature}</b><br>"
                    "Valor: %{customdata[0]}<br>"
                    "SHAP: %{customdata[1]:.4f}<extra></extra>"
                ),
            )
        )

    fig.add_vline(
        x=0,
        line_width=1,
        line_dash="dash",
        line_color="gray",
    )

    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(y_labels))),
        ticktext=y_labels,
        title="Variável",
        showgrid=False,
    )
    fig.update_xaxes(
        title="Impacto SHAP",
        zeroline=False,
    )

    fig.update_layout(
        height=max(520, len(selected) * 42),
        margin=dict(
            l=10,
            r=20,
            t=60,
            b=20,
        ),
        title="SHAP beeswarm — impacto das variáveis na predição",
        hovermode="closest",
    )

    return fig


def plot_interaction_importance(
    interactions,
    top_n=10,
):
    data = interactions.head(top_n).copy()

    data["interaction"] = (
        data["feature_1"]
        + " × "
        + data["feature_2"]
    )

    data = data.sort_values(
        "interaction_importance"
    )

    fig = px.bar(
        data,
        x="interaction_importance",
        y="interaction",
        orientation="h",
        labels={
            "interaction_importance": "Importância da interação",
            "interaction": "Combinação de variáveis",
        },
        title="Principais interações entre variáveis",
    )

    fig.update_layout(
        height=max(450, top_n * 35),
        margin=dict(
            l=10,
            r=20,
            t=60,
            b=20,
        ),
    )

    return fig


# =============================================================================
# Execução
# =============================================================================
def run_analysis(
    dataset_key,
    model_name,
    X_sample,
):
    model_key = MODELS[model_name]
    model = load_model(
        dataset_key,
        model_key,
    )

    shap_values, X_display = calculate_shap(
        model_name,
        model,
        X_sample,
    )

    importance = global_importance(
        shap_values,
        X_display,
    )

    global_file, shap_file = save_model_results(
        dataset_key,
        model_name,
        shap_values,
        X_display,
        importance,
    )

    interactions = calculate_pairwise_interactions(
        model_name,
        model,
        X_sample,
    )

    interaction_file = save_interaction_results(
        dataset_key,
        model_name,
        interactions,
    )

    return {
        "dataset": dataset_key,
        "model": model_name,
        "importance": importance,
        "shap_values": shap_values,
        "X_display": X_display,
        "interactions": interactions,
        "global_file": global_file,
        "shap_file": shap_file,
        "interaction_file": interaction_file,
    }


def run_all_models(dataset_key, X_sample):
    results = []
    statuses = []

    for model_name in MODELS:
        try:
            result = run_analysis(
                dataset_key,
                model_name,
                X_sample,
            )

            results.append(result)

            statuses.append(
                {
                    "model": model_name,
                    "status": "Concluído",
                    "global_file": str(
                        result["global_file"]
                    ),
                    "shap_file": str(
                        result["shap_file"]
                    ),
                    "interaction_file": str(
                        result["interaction_file"]
                    ),
                }
            )

        except Exception as exc:
            statuses.append(
                {
                    "model": model_name,
                    "status": f"Erro: {exc}",
                }
            )

    importance_results = [
        result["importance"]
        for result in results
    ]

    interaction_results = [
        result["interactions"]
        for result in results
    ]

    comparison_file = save_comparison(
        importance_results,
        f"global_importance_comparison_{dataset_key}.csv",
    )

    interaction_comparison_file = save_comparison(
        interaction_results,
        f"interaction_importance_comparison_{dataset_key}.csv",
    )

    return (
        results,
        statuses,
        comparison_file,
        interaction_comparison_file,
    )


# =============================================================================
# Interface
# =============================================================================
def show_interpretability():
    st.header("Interpretabilidade")

    st.markdown(
        "Análise SHAP dos modelos tunados utilizando "
        "a validação temporal oficial."
    )

    dataset_label = st.selectbox(
        "Conjunto de dados",
        list(DATASETS.keys()),
        index=1,
    )

    dataset_key = DATASETS[dataset_label]["key"]

    train_cutoff, validation_cutoff = (
        load_official_cutoffs()
    )

    st.info(
        f"Validação temporal: "
        f"{train_cutoff.date()} até "
        f"{validation_cutoff.date()}."
    )

    X_valid, y_valid = prepare_validation_features(
        dataset_key
    )

    X_sample, y_sample = sample_validation_data(
        X_valid,
        y_valid,
    )

    st.write(
        f"Amostra SHAP: {len(X_sample):,} observações "
        f"de um total de {len(X_valid):,} na validação."
    )

    if st.button(
        "Gerar SHAP para os 3 modelos",
        type="primary",
    ):
        with st.spinner(
            "Calculando SHAP e interações..."
        ):
            (
                results,
                statuses,
                comparison_file,
                interaction_comparison_file,
            ) = run_all_models(
                dataset_key,
                X_sample,
            )

        st.subheader("Status")
        st.dataframe(
            pd.DataFrame(statuses),
            width="stretch",
            hide_index=True,
        )

        if results:
            st.success(
                "Resultados de interpretabilidade "
                "gerados com sucesso."
            )

            st.write(
                f"Comparação global: `{comparison_file}`"
            )
            st.write(
                f"Comparação de interações: "
                f"`{interaction_comparison_file}`"
            )
        else:
            st.error(
                "Nenhum modelo pôde ser processado."
            )

    st.divider()

    model_name = st.selectbox(
        "Modelo para visualização",
        list(MODELS.keys()),
        index=1,
    )

    top_n = st.slider(
        "Número de variáveis exibidas",
        5,
        20,
        15,
    )

    model_key = MODELS[model_name]
    key = make_key(
        dataset_key,
        model_key,
    )

    global_file = (
        OUTPUT_DIR
        / f"global_importance_{key}.csv"
    )

    shap_file = (
        OUTPUT_DIR
        / f"shap_values_{key}.csv"
    )

    if not global_file.exists() or not shap_file.exists():
        st.info(
            "Gere os resultados SHAP acima para "
            "visualizar este modelo e conjunto."
        )
        return

    importance = pd.read_csv(
        global_file
    )

    shap_data = pd.read_csv(
        shap_file
    )

    shap_pivot = (
        shap_data.pivot(
            index="observation",
            columns="feature",
            values="shap_value",
        )
        .sort_index()
    )

    value_pivot = (
        shap_data.pivot(
            index="observation",
            columns="feature",
            values="feature_value",
        )
        .sort_index()
    )

    available_features = [
        feature
        for feature in importance["feature"]
        if feature in shap_pivot.columns
        and feature in value_pivot.columns
    ]

    shap_values = shap_pivot[
        available_features
    ].to_numpy()

    X_display = value_pivot[
        available_features
    ]

    importance = importance[
        importance["feature"].isin(
            available_features
        )
    ].copy()

    st.subheader("Importância global")

    c1, c2 = st.columns(2)

    with c1:
        st.plotly_chart(
            plot_global_importance(
                importance,
                top_n,
            ),
            width="stretch",
        )

    with c2:
        st.dataframe(
            importance.head(top_n).assign(
                mean_abs_shap=lambda x:
                x["mean_abs_shap"].round(5)
            ),
            width="stretch",
            hide_index=True,
        )

    st.subheader("Impacto das variáveis")

    st.plotly_chart(
        plot_shap_beeswarm(
            shap_values,
            X_display,
            importance.head(top_n),
        ),
        width="stretch",
    )

    st.caption(
        "Cada ponto representa uma observação da amostra SHAP. "
        "Valores SHAP positivos aumentam a previsão de "
        "readmissão não planejada; valores negativos reduzem "
        "a previsão. A posição horizontal representa o impacto "
        "SHAP e a dispersão vertical mostra a distribuição das "
        "observações. A interpretação não representa causalidade."
    )

    interaction_file = (
        OUTPUT_DIR
        / f"interaction_importance_{key}.csv"
    )

    if interaction_file.exists():
        interactions = pd.read_csv(
            interaction_file
        )

        st.divider()
        st.subheader(
            "Principais interações entre variáveis"
        )

        interaction_top_n = st.slider(
            "Número de interações exibidas",
            5,
            15,
            10,
            key=f"interaction_top_n_{key}",
        )

        st.plotly_chart(
            plot_interaction_importance(
                interactions,
                interaction_top_n,
            ),
            width="stretch",
        )

        st.caption(
            "A importância representa o comportamento "
            "combinado das duas variáveis na previsão. "
            "Não representa causalidade."
        )

        display_interactions = interactions.head(
            interaction_top_n
        ).copy()

        display_interactions["Interação"] = (
            display_interactions["feature_1"]
            + " × "
            + display_interactions["feature_2"]
        )

        st.dataframe(
            display_interactions[
                [
                    "Interação",
                    "interaction_importance",
                    "mean_interaction",
                ]
            ].rename(
                columns={
                    "interaction_importance": "Importância",
                    "mean_interaction": "Interação média",
                }
            ).round(5),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info(
            "Gere os resultados SHAP acima para "
            "calcular as interações entre variáveis."
        )


if __name__ == "__main__":
    show_interpretability()
