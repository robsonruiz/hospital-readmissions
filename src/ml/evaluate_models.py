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


def prepare_catboost_data(X, model):
    X = X.copy()

    categorical = model.get_cat_feature_indices()

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
    X_valid,
    y_valid,
    name,
    threshold=0.5,
):
    # -----------------------------
    # Prepare CatBoost data
    # -----------------------------

    if isinstance(model, CatBoostClassifier):
        X_valid = prepare_catboost_data(
            X_valid,
            model
        )

    # -----------------------------
    # Predictions
    # -----------------------------

    probability = model.predict_proba(
        X_valid
    )[:, 1]

    prediction = (
        probability >= threshold
    ).astype(int)

    # -----------------------------
    # Confusion matrix
    # -----------------------------

    tn, fp, fn, tp = confusion_matrix(
        y_valid,
        prediction,
        labels=[0, 1]
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
        "threshold": threshold,

        "roc_auc": roc_auc_score(
            y_valid,
            probability
        ),

        "pr_auc": average_precision_score(
            y_valid,
            probability
        ),

        "precision": precision_score(
            y_valid,
            prediction,
            zero_division=0
        ),

        "recall": recall_score(
            y_valid,
            prediction,
            zero_division=0
        ),

        "specificity": specificity,

        "f1": f1_score(
            y_valid,
            prediction,
            zero_division=0
        ),

        "accuracy": accuracy_score(
            y_valid,
            prediction
        ),

        "balanced_accuracy": balanced_accuracy_score(
            y_valid,
            prediction
        ),

        "brier_score": brier_score_loss(
            y_valid,
            probability
        ),

        "true_negatives": tn,

        "false_positives": fp,

        "false_negatives": fn,

        "true_positives": tp,
    }


def evaluate_models(
    trained_models,
    X_valid,
    y_valid,
    threshold=0.5,
):
    results = []

    for name, model in trained_models.items():
        print(
            f"\nEvaluating {name}..."
        )

        result = evaluate_model(
            model=model,
            X_valid=X_valid,
            y_valid=y_valid,
            name=name,
            threshold=threshold,
        )

        results.append(result)

    return (
        pd.DataFrame(results)
        .sort_values(
            "pr_auc",
            ascending=False
        )
        .reset_index(drop=True)
    )