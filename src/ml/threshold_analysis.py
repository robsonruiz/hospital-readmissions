import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from catboost import CatBoostClassifier


MODELS_DIR = Path("models")


def prepare_catboost_data(X, model):

    X = X.copy()

    categorical_indices = (
        model.get_cat_feature_indices()
    )

    for idx in categorical_indices:

        column = X.columns[idx]

        X[column] = (
            X[column]
            .fillna("missing")
            .astype(str)
        )

    return X


def get_probabilities(
    model,
    X
):

    if isinstance(
        model,
        CatBoostClassifier
    ):

        X = prepare_catboost_data(
            X,
            model
        )

    return model.predict_proba(X)[:, 1]


def evaluate_thresholds(
    model,
    X_valid,
    y_valid,
    model_name,
):

    probabilities = get_probabilities(
        model,
        X_valid
    )

    thresholds = np.arange(
        0.05,
        0.96,
        0.05
    )

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_valid,
            predictions
        ).ravel()

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0
        )

        results.append({

            "model": model_name,

            "threshold": round(
                threshold,
                2
            ),

            "precision": precision_score(
                y_valid,
                predictions,
                zero_division=0
            ),

            "recall": recall_score(
                y_valid,
                predictions,
                zero_division=0
            ),

            "f1": f1_score(
                y_valid,
                predictions,
                zero_division=0
            ),

            "specificity": specificity,

            "true_negatives": tn,

            "false_positives": fp,

            "false_negatives": fn,

            "true_positives": tp,
        })

    return pd.DataFrame(results)


def run_threshold_analysis(
    trained_models,
    X_valid,
    y_valid,
):

    all_results = []

    for model_name, model in trained_models.items():

        print(
            f"\nAnalyzing thresholds: {model_name}"
        )

        results = evaluate_thresholds(
            model,
            X_valid,
            y_valid,
            model_name
        )

        all_results.append(results)

    threshold_results = pd.concat(
        all_results,
        ignore_index=True
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    threshold_results.to_csv(
        MODELS_DIR
        / "threshold_analysis_validation.csv",
        index=False
    )

    # ----------------------------------------------------------
    # Best threshold based on F1
    # ----------------------------------------------------------

    best_thresholds = (
        threshold_results
        .sort_values(
            [
                "model",
                "f1",
                "recall",
                "precision"
            ],
            ascending=[
                True,
                False,
                False,
                False
            ]
        )
        .groupby(
            "model",
            as_index=False
        )
        .first()
    )

    best_thresholds.to_csv(
        MODELS_DIR
        / "best_thresholds_validation.csv",
        index=False
    )

    print(
        "\nThreshold analysis saved:"
    )

    print(
        "  models/threshold_analysis_validation.csv"
    )

    print(
        "  models/best_thresholds_validation.csv"
    )

    return threshold_results