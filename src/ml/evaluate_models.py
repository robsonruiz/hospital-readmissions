import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    brier_score_loss,
)

from catboost import CatBoostClassifier


def prepare_catboost_data(
    X,
    model
):

    X = X.copy()

    categorical = model.get_cat_feature_indices()

    # CatBoost trabalha com índices das colunas
    for idx in categorical:

        column = X.columns[idx]

        X[column] = (
            X[column]
            .fillna("missing")
            .astype(str)
        )

    return X


def evaluate_model(
    model,
    X_test,
    y_test,
    name,
):

    # -----------------------------
    # Prepare CatBoost data
    # -----------------------------

    if isinstance(
        model,
        CatBoostClassifier
    ):

        X_test = prepare_catboost_data(
            X_test,
            model
        )

    # -----------------------------
    # Predictions
    # -----------------------------

    probability = model.predict_proba(
        X_test
    )[:, 1]

    prediction = (
        probability >= 0.5
    ).astype(int)

    # -----------------------------
    # Confusion matrix
    # -----------------------------

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        prediction
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    # -----------------------------
    # Metrics
    # -----------------------------

    return {

        "model": name,

        "roc_auc":
            roc_auc_score(
                y_test,
                probability
            ),

        "pr_auc":
            average_precision_score(
                y_test,
                probability
            ),

        "precision":
            precision_score(
                y_test,
                prediction,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_test,
                prediction,
                zero_division=0
            ),

        "specificity":
            specificity,

        "f1":
            f1_score(
                y_test,
                prediction,
                zero_division=0
            ),

        "accuracy":
            accuracy_score(
                y_test,
                prediction
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_test,
                prediction
            ),

        "brier_score":
            brier_score_loss(
                y_test,
                probability
            ),

        "true_negatives": tn,

        "false_positives": fp,

        "false_negatives": fn,

        "true_positives": tp,
    }


def evaluate_models(
    trained_models,
    X_test,
    y_test,
):

    results = []

    for name, model in trained_models.items():

        print(
            f"\nEvaluating {name}..."
        )

        result = evaluate_model(
            model,
            X_test,
            y_test,
            name
        )

        results.append(result)

    return pd.DataFrame(results)