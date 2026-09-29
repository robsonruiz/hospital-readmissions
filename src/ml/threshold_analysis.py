import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from catboost import CatBoostClassifier


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
    X_test,
    y_test,
    model_name,
):

    probabilities = get_probabilities(
        model,
        X_test
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
            y_test,
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
                y_test,
                predictions,
                zero_division=0
            ),

            "recall": recall_score(
                y_test,
                predictions,
                zero_division=0
            ),

            "f1": f1_score(
                y_test,
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