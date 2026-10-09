from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.pipeline import Pipeline

from src.ml.prepare_data import load_ml_data, prepare_data


# =============================================================================
# Configuração
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = MODELS_DIR / "interpretability"

SHAP_SAMPLE_SIZE = 300
SHAP_BATCH_SIZE = 50
RANDOM_STATE = 42

MODEL_FILES = {
    "Random Forest": MODELS_DIR / "random_forest.joblib",
    "XGBoost": MODELS_DIR / "xgboost.joblib",
    "LightGBM": MODELS_DIR / "lightgbm.joblib",
    "CatBoost": MODELS_DIR / "catboost.joblib",
}


# =============================================================================
# Dados e modelos
# =============================================================================

@st.cache_data
def load_validation_data():
    df = load_ml_data()
    prepared = prepare_data(df)
    return prepared[1], prepared[4]


@st.cache_resource
def load_model(model_name):
    return joblib.load(MODEL_FILES[model_name])


def sample_validation_data(X_valid, y_valid):
    if len(X_valid) <= SHAP_SAMPLE_SIZE:
        return X_valid.copy(), y_valid.copy()

    rng = np.random.RandomState(RANDOM_STATE)
    indices = rng.choice(
        len(X_valid),
        SHAP_SAMPLE_SIZE,
        replace=False,
    )
    return X_valid.iloc[indices].copy(), y_valid.iloc[indices].copy()


def prepare_catboost_data(X, model):
    X = X.copy()

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

        aggregated[original_feature] = grouped[columns].sum(axis=1)

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

        return shap_values, X_model[feature_names].copy()

    if not isinstance(model, Pipeline):
        raise TypeError(
            f"{model_name}: o modelo carregado não é um Pipeline."
        )

    preprocessor = model.named_steps["preprocessor"]
    estimator = model.steps[-1][1]

    transformed_names, original_names = (
        get_original_feature_names(preprocessor)
    )

    explainer = shap.TreeExplainer(estimator)
    shap_parts = []

    for start in range(0, len(X), SHAP_BATCH_SIZE):
        batch = X.iloc[start:start + SHAP_BATCH_SIZE]

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

    return shap_values.to_numpy(dtype=np.float32), X.copy()


def global_importance(shap_values, X_display):
    return (
        pd.DataFrame(
            {
                "feature": X_display.columns,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            }
        )
        .sort_values(
            "mean_abs_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# =============================================================================
# Exportação
# =============================================================================

def save_model_results(
    model_name,
    shap_values,
    X_display,
    importance,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    key = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    global_file = (
        OUTPUT_DIR / f"global_importance_{key}.csv"
    )
    shap_file = (
        OUTPUT_DIR / f"shap_values_{key}.csv"
    )

    result = importance.copy()
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


def save_comparison(results):
    comparison_file = (
        OUTPUT_DIR / "global_importance_comparison.csv"
    )

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
        title=(
            f"Top {top_n} variáveis por importância SHAP"
        ),
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


def plot_shap_beeswarm(
    shap_values,
    X_display,
    top_features,
):
    selected = top_features["feature"].tolist()

    rows = []

    for feature in selected:
        feature_index = X_display.columns.get_loc(
            feature
        )

        for observation in range(len(X_display)):
            rows.append(
                {
                    "feature": feature,
                    "shap_value": shap_values[
                        observation,
                        feature_index,
                    ],
                    "feature_value": X_display.iloc[
                        observation,
                        feature_index,
                    ],
                }
            )

    data = pd.DataFrame(rows)

    order = (
        top_features.sort_values(
            "mean_abs_shap",
            ascending=True,
        )["feature"]
        .tolist()
    )

    fig = px.strip(
        data,
        x="shap_value",
        y="feature",
        orientation="h",
        category_orders={
            "feature": order,
        },
        hover_data={
            "feature_value": True,
            "shap_value": ":.4f",
        },
        labels={
            "shap_value": "SHAP value",
            "feature": "Variável",
            "feature_value": "Valor da variável",
        },
        title="Impacto das variáveis na predição",
    )

    fig.add_vline(
        x=0,
        line_width=1,
    )

    fig.update_traces(
        marker=dict(size=6),
    )

    fig.update_layout(
        height=max(500, len(selected) * 35),
        margin=dict(
            l=10,
            r=20,
            t=60,
            b=20,
        ),
    )

    return fig


# =============================================================================
# Execução dos modelos
# =============================================================================

def run_all_models(X_sample):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []
    statuses = []

    for model_name, model_path in MODEL_FILES.items():
        if not model_path.exists():
            statuses.append(
                {
                    "model": model_name,
                    "status": "Modelo não encontrado",
                }
            )
            continue

        try:
            model = load_model(model_name)

            shap_values, X_display = calculate_shap(
                model_name,
                model,
                X_sample,
            )

            importance = global_importance(
                shap_values,
                X_display,
            )

            global_file, shap_file = (
                save_model_results(
                    model_name,
                    shap_values,
                    X_display,
                    importance,
                )
            )

            result = importance.copy()
            result["model"] = model_name
            results.append(result)

            statuses.append(
                {
                    "model": model_name,
                    "status": "Concluído",
                    "global_file": str(
                        global_file
                    ),
                    "shap_file": str(
                        shap_file
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

    comparison_file = (
        save_comparison(results)
        if results
        else None
    )

    return (
        results,
        statuses,
        comparison_file,
    )


# =============================================================================
# Interface
# =============================================================================

def show_interpretability():
    st.header("Interpretabilidade")

    st.markdown(
        "Análise SHAP dos modelos utilizando a "
        "base de validação temporal."
    )

    X_valid, y_valid = load_validation_data()
    X_sample, y_sample = sample_validation_data(
        X_valid,
        y_valid,
    )

    st.info(
        f"Amostra utilizada: "
        f"{len(X_sample):,} observações da validação."
    )

    st.write(
        f"Resultados salvos em: `{OUTPUT_DIR}`"
    )

    if st.button(
        "Gerar resultados dos 4 modelos",
        type="primary",
    ):
        with st.spinner(
            "Calculando SHAP em lotes para os quatro modelos..."
        ):
            results, statuses, comparison_file = (
                run_all_models(X_sample)
            )

        st.subheader("Status")

        st.dataframe(
            pd.DataFrame(statuses),
            use_container_width=True,
            hide_index=True,
        )

        if results:
            st.success(
                "Resultados dos modelos foram "
                "gerados e salvos."
            )

            st.write(
                f"Comparação: `{comparison_file}`"
            )
        else:
            st.error(
                "Nenhum modelo pôde ser processado."
            )

    available_models = [
        name
        for name, path in MODEL_FILES.items()
        if path.exists()
    ]

    if not available_models:
        st.error(
            "Nenhum modelo treinado foi encontrado "
            "em models/."
        )
        return

    st.divider()

    model_name = st.selectbox(
        "Modelo para visualização",
        available_models,
    )

    top_n = st.slider(
        "Número de variáveis exibidas",
        5,
        20,
        15,
    )

    key = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    global_file = (
        OUTPUT_DIR / f"global_importance_{key}.csv"
    )
    shap_file = (
        OUTPUT_DIR / f"shap_values_{key}.csv"
    )

    if not global_file.exists() or not shap_file.exists():
        st.info(
            "Gere os resultados SHAP acima para "
            "visualizar este modelo."
        )
        return

    importance = pd.read_csv(global_file)
    shap_data = pd.read_csv(shap_file)

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

    shap_values = shap_pivot[
        importance["feature"].tolist()
    ].to_numpy()

    X_display = value_pivot[
        importance["feature"].tolist()
    ]

    st.subheader("Importância global")

    c1, c2 = st.columns(2)

    with c1:
        st.plotly_chart(
            plot_global_importance(
                importance,
                top_n,
            ),
            use_container_width=True,
        )

    with c2:
        st.dataframe(
            importance.head(top_n).assign(
                mean_abs_shap=lambda x:
                x["mean_abs_shap"].round(5)
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Impacto das variáveis")

    st.plotly_chart(
        plot_shap_beeswarm(
            shap_values,
            X_display,
            importance.head(top_n),
        ),
        use_container_width=True,
    )

    st.caption(
        "Cada ponto representa uma observação da "
        "amostra SHAP. O eixo horizontal indica a "
        "contribuição da variável para a predição. "
        "O valor original da variável pode ser "
        "consultado ao passar o cursor sobre o ponto."
    )


if __name__ == "__main__":
    show_interpretability()
