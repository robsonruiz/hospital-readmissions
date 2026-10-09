from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st

from sklearn.pipeline import Pipeline

from src.ml.prepare_data import load_ml_data, prepare_data


MODELS_DIR = Path("models")

MODEL_FILES = {
    "Random Forest": MODELS_DIR / "random_forest.joblib",
    "XGBoost": MODELS_DIR / "xgboost.joblib",
    "LightGBM": MODELS_DIR / "lightgbm.joblib",
    "CatBoost": MODELS_DIR / "catboost.joblib",
}

SHAP_SAMPLE_SIZE = 2000
RANDOM_STATE = 42

OUTPUT_DIR = MODELS_DIR / "interpretability"


@st.cache_data
def load_validation_data():
    """Load the same temporal validation set used by the ML pipeline."""
    df = load_ml_data()

    (
        _X_train,
        X_valid,
        _X_test,
        _y_train,
        y_valid,
        _y_test,
        _train_df,
        _validation_df,
        _test_df,
        _train_cutoff,
        _validation_cutoff,
    ) = prepare_data(df)

    return X_valid, y_valid


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

    return (
        X_valid.iloc[indices].copy(),
        y_valid.iloc[indices].copy(),
    )


def prepare_catboost_data(X, model):
    X = X.copy()

    for index in model.get_cat_feature_indices():
        column = X.columns[index]
        X[column] = X[column].fillna("missing").astype(str)

    return X


def normalize_shap_values(shap_values):
    """
    Normalize SHAP output for binary classification.

    Depending on the SHAP version/model, binary outputs may be returned
    as a list or as a 3-dimensional ndarray.
    """
    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            return np.asarray(shap_values[1])
        return np.asarray(shap_values[0])

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        if shap_values.shape[-1] == 2:
            return shap_values[:, :, 1]

        if shap_values.shape[1] == 2:
            return shap_values[:, 1, :]

    return shap_values


def calculate_shap(model_name, model, X):
    if model_name == "CatBoost":
        X_model = prepare_catboost_data(X, model)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_model)

        return (
            normalize_shap_values(shap_values),
            X_model,
            explainer,
        )

    if not isinstance(model, Pipeline):
        raise TypeError(
            f"{model_name}: o modelo carregado não é um Pipeline."
        )

    preprocessor = model.named_steps["preprocessor"]
    estimator = model.steps[-1][1]

    X_transformed = preprocessor.transform(X)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    X_display = pd.DataFrame(
        X_transformed,
        columns=feature_names,
        index=X.index,
    )

    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_transformed)

    return (
        normalize_shap_values(shap_values),
        X_display,
        explainer,
    )


def global_importance(shap_values, X_display):
    importance = pd.DataFrame(
        {
            "feature": X_display.columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    )

    return importance.sort_values(
        "mean_abs_shap",
        ascending=False,
    ).reset_index(drop=True)


def plot_global_importance(importance, top_n=15):
    data = importance.head(top_n).sort_values(
        "mean_abs_shap",
        ascending=True,
    )

    fig = px.bar(
        data,
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        labels={
            "mean_abs_shap": "Mean |SHAP value|",
            "feature": "Feature",
        },
        title=f"Top {top_n} features por importância SHAP",
    )

    fig.update_layout(
        height=max(500, top_n * 32),
        margin=dict(l=10, r=20, t=60, b=20),
    )

    return fig


def plot_shap_beeswarm(shap_values, X_display, top_features):
    selected = top_features["feature"].tolist()

    positions = [
        X_display.columns.get_loc(feature)
        for feature in selected
    ]

    shap_selected = shap_values[:, positions]
    X_selected = X_display[selected].copy()

    long_rows = []

    for column_index, feature in enumerate(selected):
        values = X_selected.iloc[:, column_index]
        shap_column = shap_selected[:, column_index]

        for value, shap_value in zip(values, shap_column):
            long_rows.append(
                {
                    "feature": feature,
                    "value": value,
                    "shap_value": shap_value,
                }
            )

    data = pd.DataFrame(long_rows)

    order = (
        top_features.sort_values(
            "mean_abs_shap",
            ascending=True,
        )["feature"]
        .tolist()
    )

    fig = px.scatter(
        data,
        x="shap_value",
        y="feature",
        color="value",
        category_orders={"feature": order},
        labels={
            "shap_value": "SHAP value",
            "feature": "Feature",
            "value": "Feature value",
        },
        title="Impacto das variáveis na predição",
        hover_data=["value", "shap_value"],
    )

    fig.add_vline(x=0, line_width=1)

    fig.update_traces(marker=dict(size=5))

    fig.update_layout(
        height=max(500, len(selected) * 35),
        margin=dict(l=10, r=20, t=60, b=20),
    )

    return fig


def plot_local_explanation(
    shap_values,
    X_display,
    row_position,
    top_n=15,
):
    row_shap = shap_values[row_position]

    local = pd.DataFrame(
        {
            "feature": X_display.columns,
            "shap_value": row_shap,
            "feature_value": X_display.iloc[row_position].values,
        }
    )

    local["abs_shap"] = local["shap_value"].abs()

    local = local.sort_values(
        "abs_shap",
        ascending=False,
    ).head(top_n)

    local = local.sort_values(
        "shap_value",
        ascending=True,
    )

    local["direction"] = np.where(
        local["shap_value"] >= 0,
        "Aumenta a predição",
        "Reduz a predição",
    )

    fig = px.bar(
        local,
        x="shap_value",
        y="feature",
        color="direction",
        orientation="h",
        hover_data=["feature_value", "shap_value"],
        labels={
            "shap_value": "SHAP value",
            "feature": "Feature",
            "feature_value": "Valor da feature",
            "direction": "Efeito",
        },
        title=f"Explicação da observação {row_position}",
    )

    fig.add_vline(x=0, line_width=1)

    fig.update_layout(
        height=max(500, top_n * 32),
        margin=dict(l=10, r=20, t=60, b=20),
    )

    return fig, local


def save_interpretability_outputs(
    model_name,
    shap_values,
    X_display,
    importance,
):
    """Save global importance and observation-level SHAP values."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_key = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    importance_to_save = importance.copy()
    importance_to_save["model"] = model_name

    importance_to_save.to_csv(
        OUTPUT_DIR / f"global_importance_{model_key}.csv",
        index=False,
    )

    shap_rows = []

    for feature_index, feature in enumerate(X_display.columns):
        shap_rows.append(
            pd.DataFrame(
                {
                    "observation": np.arange(len(X_display)),
                    "feature": feature,
                    "feature_value": X_display.iloc[:, feature_index].values,
                    "shap_value": shap_values[:, feature_index],
                }
            )
        )

    shap_long = pd.concat(
        shap_rows,
        ignore_index=True,
    )

    shap_long.to_csv(
        OUTPUT_DIR / f"shap_values_{model_key}.csv",
        index=False,
    )



def show_interpretability():
    st.header("Interpretabilidade")

    st.markdown(
        """
        A análise utiliza SHAP para explicar as predições dos modelos
        de classificação. Os resultados são calculados sobre uma amostra
        da base de validação temporal.
        """
    )

    available_models = [
        name
        for name, path in MODEL_FILES.items()
        if path.exists()
    ]

    if not available_models:
        st.error(
            "Nenhum modelo treinado foi encontrado em models/."
        )
        return

    model_name = st.selectbox(
        "Modelo",
        available_models,
    )

    top_n = st.slider(
        "Número de variáveis exibidas",
        min_value=5,
        max_value=20,
        value=15,
    )

    with st.spinner("Calculando valores SHAP..."):
        X_valid, y_valid = load_validation_data()

        X_sample, y_sample = sample_validation_data(
            X_valid,
            y_valid,
        )

        model = load_model(model_name)

        shap_values, X_display, _ = calculate_shap(
            model_name,
            model,
            X_sample,
        )

    importance = global_importance(
        shap_values,
        X_display,
    )

    save_interpretability_outputs(
        model_name,
        shap_values,
        X_display,
        importance,
    )

    st.success(
        f"Resultados salvos em: {OUTPUT_DIR}"
    )

    st.subheader("Importância global")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            plot_global_importance(
                importance,
                top_n=top_n,
            ),
            use_container_width=True,
        )

    with col2:
        st.dataframe(
            importance.head(top_n).assign(
                mean_abs_shap=lambda x: x["mean_abs_shap"].round(5)
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Impacto das variáveis")

    top_features = importance.head(top_n)

    st.plotly_chart(
        plot_shap_beeswarm(
            shap_values,
            X_display,
            top_features,
        ),
        use_container_width=True,
    )

    st.markdown(
        """
        Valores SHAP positivos indicam contribuição para aumentar a
        predição da classe positiva, enquanto valores negativos indicam
        contribuição para reduzi-la.
        """
    )

    st.subheader("Explicação individual")

    row_position = st.number_input(
        "Observação da validação",
        min_value=0,
        max_value=len(X_sample) - 1,
        value=0,
        step=1,
    )

    local_fig, local_table = plot_local_explanation(
        shap_values,
        X_display,
        int(row_position),
        top_n=top_n,
    )

    st.plotly_chart(
        local_fig,
        use_container_width=True,
    )

    st.dataframe(
        local_table[
            [
                "feature",
                "feature_value",
                "shap_value",
                "direction",
            ]
        ].round(5),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        f"A análise utiliza {len(X_sample):,} observações "
        "da base de validação."
    )
