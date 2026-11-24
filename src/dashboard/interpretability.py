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
    "B — Full trajectory": {
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


def tr(key, **kwargs):
    """Return an English UI string."""
    translations = {
        "page_title": "Interpretability analysis",
        "page_description": "SHAP analysis of tuned models using the official temporal validation set.",
        "language": "Language",
        "dataset": "Dataset",
        "dataset_a": "A — Baseline",
        "dataset_b": "B — Full trajectory",
        "dataset_c": "C — Primary care",
        "temporal_validation": "Temporal validation",
        "shap_sample": "SHAP sample: {sample:,} observations out of {total:,} in validation.",
        "generate": "Generate SHAP for all 3 models",
        "calculating": "Calculating SHAP values and interactions...",
        "status": "Status",
        "generated_success": "Interpretability results generated successfully.",
        "global_comparison": "Global comparison",
        "interaction_comparison": "Interaction comparison",
        "no_model": "No model could be processed.",
        "model": "Model for visualization",
        "features_displayed": "Number of features displayed",
        "missing_results": "Generate the SHAP results above to visualize this model and dataset.",
        "global_importance": "Global feature importance",
        "mean_shap": "Mean |SHAP| importance",
        "feature": "Feature",
        "global_caption": "Mean |SHAP| represents the average magnitude of each feature's contribution to model predictions. Higher values indicate greater influence, but do not by themselves indicate whether the contribution increases or decreases predicted risk.",
        "dependence": "Dependence of the main features",
        "dependence_caption": "Select one of the main features to explore how different values relate to its contribution to the prediction. Points above zero indicate a contribution toward higher predicted risk; points below zero indicate a contribution toward lower predicted risk. The observed relationship represents an association with the model prediction, not a causal effect.",
        "dependence_select": "Feature for dependence analysis",
        "interactions": "Main feature interactions",
        "interaction_caption": "Interaction importance represents the combined behavior of the two features in the prediction. It does not represent causality.",
        "interaction": "Interaction",
        "importance": "Importance",
        "mean_interaction": "Mean interaction",
        "no_interactions": "No interaction results available for this model and dataset.",
        "feature_value": "Feature value",
        "low": "Low",
        "high": "High",
        "outside_limit": "Outside visual limit",
        "yes": "Yes",
        "no": "No",
        "shap_value": "SHAP value",
        "contribution_risk": "Feature contribution to predicted risk",
        "each_point": "Each point represents one observation from the SHAP sample.",
        "positive_negative": "Positive SHAP values increase predicted unplanned readmission risk; negative values decrease it.",
        "color_caption": "Color represents the relative feature value (blue = low; red = high).",
        "visual_limit": "For readability, the horizontal axis is limited to the central 99% of SHAP values; extreme values are not removed and are identified in the hover information.",
        "not_causal": "The interpretation does not represent causality.",
    }
    return translations[key].format(**kwargs)

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
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_parquet(path)
    df["admission_ts"] = pd.to_datetime(
        df["admission_ts"],
        errors="coerce",
    )

    if df["admission_ts"].isna().any():
        raise ValueError(
            f"{dataset_key}: admission_ts contains invalid values."
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
            f"Model not found: {model_path}"
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
            f"{dataset_key}: validation set is empty."
        )

    target = "unplanned_readmitted_30d"

    if target not in validation_df.columns:
        raise ValueError(
            f"{dataset_key}: target column '{target}' not found."
        )

    y_valid = pd.to_numeric(
        validation_df[target],
        errors="coerce",
    )

    if y_valid.isna().any():
        raise ValueError(
            f"{dataset_key}: target contains missing values."
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
            "feature": "Feature",
        },
        title=f"Top {top_n} features by SHAP importance",
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
    """Map feature values to a 0-1 relative scale for SHAP-style coloring."""
    series = pd.Series(values)

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            numeric = numeric.fillna(numeric.median()).to_numpy(dtype=float)
        else:
            return np.full(len(series), 0.5, dtype=float)
    else:
        codes, _ = pd.factorize(
            series.astype("string").fillna("missing"),
            sort=True,
        )
        numeric = codes.astype(float)

    if len(numeric) == 0:
        return numeric

    minimum = np.nanmin(numeric)
    maximum = np.nanmax(numeric)

    if maximum == minimum:
        return np.full(len(numeric), 0.5, dtype=float)

    return (numeric - minimum) / (maximum - minimum)


def _jitter_positions(values, seed=42):
    """Create deterministic vertical jitter for a beeswarm-like cloud."""
    rng = np.random.RandomState(seed)
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return values

    # Bin SHAP values horizontally so points with similar effects
    # spread vertically instead of forming a simple diagonal rank.
    finite = np.isfinite(values)
    jitter = np.zeros(len(values), dtype=float)

    if finite.sum() == 0:
        return rng.normal(0, 0.08, size=len(values))

    finite_values = values[finite]
    if np.ptp(finite_values) == 0:
        jitter[finite] = rng.normal(0, 0.08, size=finite.sum())
        return jitter

    n_bins = max(8, min(30, int(np.sqrt(finite.sum()))))
    bins = np.linspace(
        finite_values.min(),
        finite_values.max(),
        n_bins + 1,
    )

    bin_ids = np.digitize(finite_values, bins[1:-1], right=False)

    for bin_id in np.unique(bin_ids):
        positions = np.where(finite)[0][bin_ids == bin_id]
        if len(positions) == 1:
            jitter[positions[0]] = rng.normal(0, 0.025)
        else:
            offsets = np.linspace(-0.26, 0.26, len(positions))
            offsets += rng.normal(0, 0.018, size=len(positions))
            jitter[positions] = offsets

    return jitter


def plot_shap_beeswarm(
    shap_values,
    X_display,
    top_features,
    x_quantile=0.99,
):
    """Create a clean SHAP beeswarm-style plot.

    The x-axis is visually limited to the central distribution so that
    extreme SHAP outliers do not compress the rest of the observations.
    Outliers are not removed from the underlying data; only their display
    position is clipped to the plotting range.
    """
    selected = top_features["feature"].tolist()
    selected = [
        feature
        for feature in selected
        if feature in X_display.columns
    ]

    if not selected:
        return go.Figure()

    # SHAP convention: most important feature at the top.
    y_labels = list(reversed(selected))
    fig = go.Figure()

    all_shap = []
    for feature in selected:
        feature_index = X_display.columns.get_loc(feature)
        all_shap.extend(
            np.asarray(
                shap_values[:, feature_index],
                dtype=float,
            )[np.isfinite(shap_values[:, feature_index])]
        )

    all_shap = np.asarray(all_shap, dtype=float)

    if len(all_shap) > 0:
        limit = max(
            float(np.quantile(np.abs(all_shap), x_quantile)),
            1e-6,
        )
    else:
        limit = 1.0

    # Keep zero centered and use a symmetric visual range.
    x_limit = limit * 1.05

    for y_index, feature in enumerate(y_labels):
        feature_index = X_display.columns.get_loc(feature)

        shap_column = np.asarray(
            shap_values[:, feature_index],
            dtype=float,
        )

        feature_values = X_display[feature].to_numpy()
        color_values = _numeric_color_values(feature_values)

        display_shap = np.clip(
            shap_column,
            -limit,
            limit,
        )

        jitter = _jitter_positions(
            display_shap,
            seed=RANDOM_STATE + y_index,
        )

        customdata = np.column_stack([
            np.asarray(feature_values, dtype=object),
            shap_column,
            np.where(
                np.abs(shap_column) > limit,
                tr("yes"),
                tr("no"),
            ),
        ])

        fig.add_trace(
            go.Scatter(
                x=display_shap,
                y=np.full(len(display_shap), y_index) + jitter,
                mode="markers",
                name=feature,
                showlegend=False,
                marker={
                    "size": 6,
                    "opacity": 0.72,
                    "color": color_values,
                    "colorscale": [
                        [0.0, "#2166AC"],
                        [0.5, "#F7F7F7"],
                        [1.0, "#B2182B"],
                    ],
                    "cmin": 0,
                    "cmax": 1,
                    "showscale": y_index == len(y_labels) - 1,
                    "colorbar": {
                        "title": "Feature value",
                        "tickvals": [0, 1],
                        "ticktext": ["Low", "High"],
                        "len": 0.75,
                    } if y_index == len(y_labels) - 1 else None,
                    "line": {"width": 0},
                },
                customdata=customdata,
                hovertemplate=(
                    f"<b>{feature}</b><br>"
                    "Valor: %{customdata[0]}<br>"
                    "SHAP: %{customdata[1]:.4f}<br>"
                    "Outside visual limit: %{customdata[2]}"
                    "<extra></extra>"
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
        title="Feature",
        showgrid=False,
        range=[-0.7, len(y_labels) - 0.3],
    )

    fig.update_xaxes(
        title=tr("shap_value"),
        zeroline=False,
        range=[-x_limit, x_limit],
    )

    fig.update_layout(
        height=max(500, len(selected) * 42),
        margin=dict(
            l=10,
            r=20,
            t=75,
            b=55,
        ),
        title=(
            "SHAP beeswarm — feature contributions "
            "para o risco previsto"
        ),
        hovermode="closest",
    )

    return fig


def plot_shap_dependence(
    shap_values,
    X_display,
    feature,
):
    """Plot SHAP dependence for one feature."""
    if feature not in X_display.columns:
        return go.Figure()

    feature_index = X_display.columns.get_loc(feature)
    x_values = X_display[feature]
    y_values = np.asarray(
        shap_values[:, feature_index],
        dtype=float,
    )

    if not pd.api.types.is_numeric_dtype(x_values):
        codes, labels = pd.factorize(
            x_values.astype("string").fillna("missing"),
            sort=True,
        )
        x_plot = codes.astype(float)
        x_tickvals = list(range(len(labels)))
        x_ticktext = [str(label) for label in labels]
    else:
        x_plot = pd.to_numeric(
            x_values,
            errors="coerce",
        ).to_numpy(dtype=float)
        x_tickvals = None
        x_ticktext = None

    mask = np.isfinite(x_plot) & np.isfinite(y_values)

    fig = go.Figure(
        go.Scatter(
            x=x_plot[mask],
            y=y_values[mask],
            mode="markers",
            marker={
                "size": 6,
                "opacity": 0.55,
                "color": x_plot[mask],
                "colorscale": [
                    [0.0, "#2166AC"],
                    [0.5, "#F7F7F7"],
                    [1.0, "#B2182B"],
                ],
                "showscale": False,
                "line": {"width": 0},
            },
            customdata=np.asarray(
                x_values[mask],
                dtype=object,
            ),
            hovertemplate=(
                f"<b>{feature}</b><br>"
                "Valor: %{customdata}<br>"
                "SHAP: %{y:.4f}<extra></extra>"
            ),
        )
    )

    fig.add_hline(
        y=0,
        line_width=1,
        line_dash="dash",
        line_color="gray",
    )

    fig.update_layout(
        height=350,
        margin=dict(
            l=10,
            r=20,
            t=55,
            b=45,
        ),
        title=f"SHAP dependence — {feature}",
        xaxis_title=feature,
        yaxis_title=tr("shap_value"),
    )

    if x_tickvals is not None:
        fig.update_xaxes(
            tickmode="array",
            tickvals=x_tickvals,
            ticktext=x_ticktext,
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
            "interaction_importance": "Interaction importance",
            "interaction": "Feature combination",
        },
        title="Main feature interactions",
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
                    "status": "Completed",
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
    dataset_options = {
        tr("dataset_a"): "A_baseline",
        tr("dataset_b"): "B_trajectory",
        tr("dataset_c"): "C_aps",
    }

    st.header(tr("page_title"))
    st.markdown(tr("page_description"))

    dataset_label = st.selectbox(
        tr("dataset"),
        list(dataset_options.keys()),
        index=1,
    )

    dataset_key = dataset_options[dataset_label]

    train_cutoff, validation_cutoff = (
        load_official_cutoffs()
    )

    st.info(
        f"{tr('temporal_validation')}: "
        f"{train_cutoff.date()} to "
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
        tr(
            "shap_sample",
            sample=len(X_sample),
            total=len(X_valid),
        )
    )

    if st.button(
        tr("generate"),
        type="primary",
    ):
        with st.spinner(
            tr("calculating")
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

        st.subheader(tr("status"))
        st.dataframe(
            pd.DataFrame(statuses),
            width="stretch",
            hide_index=True,
        )

        if results:
            st.success(
                tr("generated_success")
            )

            st.write(
                f"{tr('global_comparison')}: `{comparison_file}`"
            )
            st.write(
                f"Interaction comparison: "
                f"`{interaction_comparison_file}`"
            )
        else:
            st.error(
                tr("no_model")
            )

    st.divider()

    model_name = st.selectbox(
        tr("model"),
        list(MODELS.keys()),
        index=1,
    )

    top_n = st.slider(
        tr("features_displayed"),
        5,
        15,
        10,
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
            tr("missing_results")
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

    st.subheader(tr("global_importance"))

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

    st.subheader(tr("global_importance"))

    importance_plot = importance.head(top_n).copy()
    importance_plot = importance_plot.sort_values(
        "mean_abs_shap",
        ascending=True,
    )

    fig_importance = go.Figure(
        go.Bar(
            x=importance_plot["mean_abs_shap"],
            y=importance_plot["feature"],
            orientation="h",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Mean |SHAP|: %{x:.4f}<extra></extra>"
            ),
        )
    )

    fig_importance.update_layout(
        height=max(450, len(importance_plot) * 38),
        margin=dict(l=10, r=20, t=25, b=45),
        xaxis_title=tr("mean_shap"),
        yaxis_title=tr("feature"),
    )

    st.plotly_chart(
        fig_importance,
        width="stretch",
    )

    st.caption(tr("global_caption"))

    st.subheader(tr("dependence"))

    dependence_features = importance.head(
        min(5, top_n)
    )["feature"].tolist()

    st.caption(tr("dependence_caption"))

    selected_dependence_feature = st.selectbox(
        tr("dependence_select"),
        dependence_features,
        index=0,
        key=f"dependence_feature_{dataset_key}_{model_key}",
    )

    st.plotly_chart(
        plot_shap_dependence(
            shap_values,
            X_display,
            selected_dependence_feature,
        ),
        width="stretch",
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
            "Main feature interactions"
        )

        interaction_top_n = st.slider(
            "Number of interactions displayed",
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
            "Interaction importance represents the combined behavior of the two features in the prediction. "
            "It does not represent causality."
        )

        display_interactions = interactions.head(
            interaction_top_n
        ).copy()

        display_interactions[tr("interaction")] = (
            display_interactions["feature_1"]
            + " × "
            + display_interactions["feature_2"]
        )

        st.dataframe(
            display_interactions[
                [
                    tr("interaction"),
                    "interaction_importance",
                    "mean_interaction",
                ]
            ].rename(
                columns={
                    "interaction_importance": tr("importance"),
                    "mean_interaction": "Mean interaction",
                }
            ).round(5),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info(
            "Generate the SHAP results above to calculate feature interactions."
        )


if __name__ == "__main__":
    show_interpretability()
