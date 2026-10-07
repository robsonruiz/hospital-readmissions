from pathlib import Path

import pandas as pd
import streamlit as st


# ==============================================================
# CONFIGURATION
# ==============================================================

MODELS_DIR = Path("models")

VALIDATION_RESULTS = (
    MODELS_DIR / "validation_results.csv"
)

THRESHOLD_RESULTS = (
    MODELS_DIR / "threshold_analysis_validation.csv"
)

BEST_THRESHOLDS = (
    MODELS_DIR / "best_thresholds_validation.csv"
)


st.set_page_config(
    page_title="Machine Learning — Reinternações",
    page_icon="🏥",
    layout="wide",
)


# ==============================================================
# PAGE TITLE
# ==============================================================

st.title(
    "Machine Learning — Reinternações Não Planejadas"
)

st.caption(
    "Avaliação dos modelos preditivos utilizando "
    "divisão temporal e conjunto de validação."
)


# ==============================================================
# LOAD DATA
# ==============================================================

@st.cache_data
def load_validation_results():

    if not VALIDATION_RESULTS.exists():
        return None

    return pd.read_csv(
        VALIDATION_RESULTS
    )


@st.cache_data
def load_threshold_results():

    if not THRESHOLD_RESULTS.exists():
        return None

    return pd.read_csv(
        THRESHOLD_RESULTS
    )


@st.cache_data
def load_best_thresholds():

    if not BEST_THRESHOLDS.exists():
        return None

    return pd.read_csv(
        BEST_THRESHOLDS
    )


validation_results = (
    load_validation_results()
)

threshold_results = (
    load_threshold_results()
)

best_thresholds = (
    load_best_thresholds()
)


# ==============================================================
# CHECK FILES
# ==============================================================

if validation_results is None:

    st.error(
        "O arquivo "
        "models/validation_results.csv "
        "não foi encontrado."
    )

    st.info(
        "Execute primeiro o treinamento dos modelos."
    )

    st.stop()


# ==============================================================
# NORMALIZE MODEL NAMES
# ==============================================================

MODEL_NAMES = {

    "random_forest": "Random Forest",
    "random forest": "Random Forest",
    "rf": "Random Forest",

    "xgboost": "XGBoost",
    "xgb": "XGBoost",

    "lightgbm": "LightGBM",
    "lgbm": "LightGBM",

    "catboost": "CatBoost",
    "cat": "CatBoost",
}


def format_model_name(name):

    key = str(
        name
    ).strip().lower()

    return MODEL_NAMES.get(
        key,
        str(name)
        .replace("_", " ")
        .title(),
    )


if "model" in validation_results.columns:

    validation_results[
        "model_display"
    ] = (
        validation_results["model"]
        .apply(format_model_name)
    )

elif "Model" in validation_results.columns:

    validation_results[
        "model_display"
    ] = (
        validation_results["Model"]
        .apply(format_model_name)
    )

else:

    st.error(
        "A coluna 'model' não foi encontrada "
        "em validation_results.csv."
    )

    st.stop()


# ==============================================================
# METRIC CONFIGURATION
# ==============================================================

METRIC_LABELS = {

    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "precision": "Precision",
    "recall": "Recall",
    "specificity": "Specificity",
    "f1": "F1",
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced Accuracy",
    "brier_score": "Brier Score",
}


METRIC_DIRECTION = {

    "roc_auc": "max",
    "pr_auc": "max",
    "precision": "max",
    "recall": "max",
    "specificity": "max",
    "f1": "max",
    "accuracy": "max",
    "balanced_accuracy": "max",
    "brier_score": "min",
}


# ==============================================================
# FORMAT VALUES
# ==============================================================

def format_metric(value):

    if pd.isna(value):
        return "-"

    return f"{float(value):.3f}"


# ==============================================================
# HIGHLIGHT BEST RESULT
# ==============================================================

def highlight_best(
    dataframe,
    metric_columns,
):

    styles = pd.DataFrame(
        "",
        index=dataframe.index,
        columns=dataframe.columns,
    )

    for metric in metric_columns:

        if metric not in dataframe.columns:
            continue

        values = pd.to_numeric(
            dataframe[metric],
            errors="coerce",
        )

        if values.isna().all():
            continue

        if (
            METRIC_DIRECTION.get(metric)
            == "min"
        ):

            best_value = values.min()

        else:

            best_value = values.max()

        for idx in dataframe.index:

            value = values.loc[idx]

            if (
                pd.notna(value)
                and value == best_value
            ):

                styles.loc[
                    idx,
                    metric
                ] = "font-weight: bold;"

    return styles


# ==============================================================
# TABS
# ==============================================================

tab_models, tab_threshold = (
    st.tabs(
        [
            "Comparação dos modelos",
            "Threshold"
        ]
    )
)


# ==============================================================
# TAB 1 — MODEL COMPARISON
# ==============================================================

with tab_models:

    st.header(
        "Comparação dos modelos"
    )

    st.write(
        "Resultados obtidos no conjunto de "
        "validação temporal. Os valores em "
        "negrito representam o melhor resultado "
        "observado entre os modelos para cada métrica."
    )

    # ----------------------------------------------------------
    # Discrimination
    # ----------------------------------------------------------

    discrimination_metrics = [
        metric
        for metric in [
            "roc_auc",
            "pr_auc",
        ]
        if metric in validation_results.columns
    ]

    if discrimination_metrics:

        st.subheader(
            "Discriminação"
        )

        display_data = validation_results[
            [
                "model_display"
            ]
            + discrimination_metrics
        ].copy()

        display_data = display_data.rename(
            columns={
                "model_display": "Modelo",
                **{
                    metric: METRIC_LABELS[
                        metric
                    ]
                    for metric
                    in discrimination_metrics
                },
            }
        )

        for metric in discrimination_metrics:

            display_data[
                METRIC_LABELS[metric]
            ] = (
                display_data[
                    METRIC_LABELS[metric]
                ]
                .apply(format_metric)
            )

        numeric_data = validation_results[
            [
                "model_display"
            ]
            + discrimination_metrics
        ].copy()

        numeric_data = numeric_data.rename(
            columns={
                "model_display": "Modelo"
            }
        )

        styles = highlight_best(
            numeric_data,
            discrimination_metrics,
        )

        styles.columns = (
            display_data.columns
        )

        st.dataframe(
            display_data.style.apply(
                lambda _: styles,
                axis=None,
            ),
            use_container_width=True,
            hide_index=True,
        )

    # ----------------------------------------------------------
    # Classification
    # ----------------------------------------------------------

    classification_metrics = [
        metric
        for metric in [
            "precision",
            "recall",
            "specificity",
            "f1",
            "accuracy",
            "balanced_accuracy",
        ]
        if metric in validation_results.columns
    ]

    if classification_metrics:

        st.subheader(
            "Classificação"
        )

        display_data = validation_results[
            [
                "model_display"
            ]
            + classification_metrics
        ].copy()

        display_data = display_data.rename(
            columns={
                "model_display": "Modelo",
                **{
                    metric: METRIC_LABELS[
                        metric
                    ]
                    for metric
                    in classification_metrics
                },
            }
        )

        for metric in classification_metrics:

            display_data[
                METRIC_LABELS[metric]
            ] = (
                display_data[
                    METRIC_LABELS[metric]
                ]
                .apply(format_metric)
            )

        numeric_data = validation_results[
            [
                "model_display"
            ]
            + classification_metrics
        ].copy()

        numeric_data = numeric_data.rename(
            columns={
                "model_display": "Modelo"
            }
        )

        styles = highlight_best(
            numeric_data,
            classification_metrics,
        )

        styles.columns = (
            display_data.columns
        )

        st.dataframe(
            display_data.style.apply(
                lambda _: styles,
                axis=None,
            ),
            use_container_width=True,
            hide_index=True,
        )

    # ----------------------------------------------------------
    # Probability
    # ----------------------------------------------------------

    probability_metrics = [
        metric
        for metric in [
            "brier_score",
        ]
        if metric in validation_results.columns
    ]

    if probability_metrics:

        st.subheader(
            "Qualidade das probabilidades"
        )

        display_data = validation_results[
            [
                "model_display"
            ]
            + probability_metrics
        ].copy()

        display_data = display_data.rename(
            columns={
                "model_display": "Modelo",
                **{
                    metric: METRIC_LABELS[
                        metric
                    ]
                    for metric
                    in probability_metrics
                },
            }
        )

        for metric in probability_metrics:

            display_data[
                METRIC_LABELS[metric]
            ] = (
                display_data[
                    METRIC_LABELS[metric]
                ]
                .apply(format_metric)
            )

        numeric_data = validation_results[
            [
                "model_display"
            ]
            + probability_metrics
        ].copy()

        numeric_data = numeric_data.rename(
            columns={
                "model_display": "Modelo"
            }
        )

        styles = highlight_best(
            numeric_data,
            probability_metrics,
        )

        styles.columns = (
            display_data.columns
        )

        st.dataframe(
            display_data.style.apply(
                lambda _: styles,
                axis=None,
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Para ROC-AUC, PR-AUC, Precision, Recall, "
        "Specificity, F1, Accuracy e Balanced Accuracy, "
        "valores maiores são melhores. Para Brier Score, "
        "valores menores são melhores."
    )


# ==============================================================
# TAB 2 — THRESHOLD
# ==============================================================

with tab_threshold:

    st.header(
        "Threshold"
    )

    # ----------------------------------------------------------
    # Best threshold
    # ----------------------------------------------------------

    if best_thresholds is None:

        st.warning(
            "O arquivo "
            "models/best_thresholds_validation.csv "
            "não foi encontrado."
        )

    else:

        st.subheader(
            "Threshold selecionado"
        )

        best_display = (
            best_thresholds.copy()
        )

        # ------------------------------------------------------
        # Model
        # ------------------------------------------------------

        if "model" in best_display.columns:

            best_display[
                "Modelo"
            ] = (
                best_display["model"]
                .apply(format_model_name)
            )

        elif "Model" in best_display.columns:

            best_display[
                "Modelo"
            ] = (
                best_display["Model"]
                .apply(format_model_name)
            )

        # ------------------------------------------------------
        # Columns
        # ------------------------------------------------------

        columns = [
            "Modelo",
            "threshold",
            "precision",
            "recall",
            "specificity",
            "f1",
        ]

        columns = [
            column
            for column in columns
            if column in best_display.columns
        ]

        best_table = (
            best_display[
                columns
            ].copy()
        )

        best_table = (
            best_table.rename(
                columns={
                    "threshold": "Threshold",
                    "precision": "Precision",
                    "recall": "Recall",
                    "specificity": "Specificity",
                    "f1": "F1",
                }
            )
        )

        # ------------------------------------------------------
        # Format threshold
        # ------------------------------------------------------

        if "Threshold" in best_table.columns:

            best_table[
                "Threshold"
            ] = (
                pd.to_numeric(
                    best_table[
                        "Threshold"
                    ],
                    errors="coerce",
                )
                .map(
                    lambda x:
                    "-"
                    if pd.isna(x)
                    else f"{x:.2f}"
                )
            )

        # ------------------------------------------------------
        # Format metrics
        # ------------------------------------------------------

        for column in [
            "Precision",
            "Recall",
            "Specificity",
            "F1",
        ]:

            if column in best_table.columns:

                best_table[
                    column
                ] = (
                    pd.to_numeric(
                        best_table[
                            column
                        ],
                        errors="coerce",
                    )
                    .map(format_metric)
                )

        # ------------------------------------------------------
        # Display
        # ------------------------------------------------------

        st.dataframe(
            best_table,
            use_container_width=True,
            hide_index=True,
        )

    # ----------------------------------------------------------
    # Full threshold results
    # ----------------------------------------------------------

    if threshold_results is not None:

        with st.expander(
            "Ver todos os thresholds avaliados"
        ):

            threshold_display = (
                threshold_results.copy()
            )

            # --------------------------------------------------
            # Model
            # --------------------------------------------------

            if "model" in threshold_display.columns:

                threshold_display[
                    "Modelo"
                ] = (
                    threshold_display["model"]
                    .apply(format_model_name)
                )

            elif "Model" in threshold_display.columns:

                threshold_display[
                    "Modelo"
                ] = (
                    threshold_display["Model"]
                    .apply(format_model_name)
                )

            # --------------------------------------------------
            # Columns
            # --------------------------------------------------

            columns = [
                "Modelo",
                "threshold",
                "precision",
                "recall",
                "specificity",
                "f1",
                "true_positives",
                "false_positives",
                "false_negatives",
                "true_negatives",
            ]

            columns = [
                column
                for column in columns
                if column
                in threshold_display.columns
            ]

            threshold_table = (
                threshold_display[
                    columns
                ].copy()
            )

            threshold_table = (
                threshold_table.rename(
                    columns={
                        "threshold": "Threshold",
                        "precision": "Precision",
                        "recall": "Recall",
                        "specificity": "Specificity",
                        "f1": "F1",
                        "true_positives": "TP",
                        "false_positives": "FP",
                        "false_negatives": "FN",
                        "true_negatives": "TN",
                    }
                )
            )

            # --------------------------------------------------
            # Format threshold
            # --------------------------------------------------

            if "Threshold" in threshold_table.columns:

                threshold_table[
                    "Threshold"
                ] = (
                    pd.to_numeric(
                        threshold_table[
                            "Threshold"
                        ],
                        errors="coerce",
                    )
                    .map(
                        lambda x:
                        "-"
                        if pd.isna(x)
                        else f"{x:.2f}"
                    )
                )

            # --------------------------------------------------
            # Format metrics
            # --------------------------------------------------

            for column in [
                "Precision",
                "Recall",
                "Specificity",
                "F1",
            ]:

                if column in threshold_table.columns:

                    threshold_table[
                        column
                    ] = (
                        pd.to_numeric(
                            threshold_table[
                                column
                            ],
                            errors="coerce",
                        )
                        .map(format_metric)
                    )

            st.dataframe(
                threshold_table,
                use_container_width=True,
                hide_index=True,
            )