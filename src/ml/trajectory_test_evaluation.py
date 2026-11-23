from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from catboost import CatBoostClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODEL_DIR = PROJECT_ROOT / "models" / "trajectory_experiment"
DATASET_DIR = MODEL_DIR

TARGET = "unplanned_readmitted_30d"

DATASET_NAME = "B_trajectory"
DATASET_FILE = "b_trajectory.parquet"

MODELS = {
    "xgboost": {
        "threshold": 0.14,
        "model_file": (
            "B_trajectory_tuned_xgboost.joblib"
        ),
    },
    "random_forest": {
        "threshold": 0.16,
        "model_file": (
            "B_trajectory_tuned_random_forest.joblib"
        ),
    },
}


def load_dataset():
    path = DATASET_DIR / DATASET_FILE

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{path}"
        )

    print(
        f"Loading dataset: {path.name}",
        flush=True,
    )

    df = pd.read_parquet(path)

    if TARGET not in df.columns:
        raise ValueError(
            f"Target '{TARGET}' not found."
        )

    if "admission_ts" not in df.columns:
        raise ValueError(
            "Column 'admission_ts' not found."
        )

    df["admission_ts"] = pd.to_datetime(
        df["admission_ts"],
        errors="coerce",
    )

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce",
    )

    df = (
        df
        .dropna(
            subset=[
                "admission_ts",
                TARGET,
            ]
        )
        .sort_values("admission_ts")
        .reset_index(drop=True)
    )

    df[TARGET] = df[TARGET].astype(int)

    return df


def load_official_temporal_cutoffs():
    from src.ml.prepare_data import (
        load_ml_data,
        prepare_data,
    )

    print()
    print(
        "Loading official temporal split...",
        flush=True,
    )

    df = load_ml_data()

    (
        _X_train,
        _X_valid,
        _X_test,
        _y_train,
        _y_valid,
        _y_test,
        _train_df,
        _validation_df,
        _test_df,
        train_cutoff,
        validation_cutoff,
    ) = prepare_data(df)

    print(
        f"TRAIN cutoff: "
        f"{train_cutoff.date()}",
        flush=True,
    )

    print(
        f"VALIDATION cutoff: "
        f"{validation_cutoff.date()}",
        flush=True,
    )

    return (
        train_cutoff,
        validation_cutoff,
    )


def split_dataset(
    df,
    train_cutoff,
    validation_cutoff,
):
    dates = df[
        "admission_ts"
    ].dt.normalize()

    train_mask = (
        dates <= train_cutoff
    )

    validation_mask = (
        (dates > train_cutoff)
        & (dates <= validation_cutoff)
    )

    test_mask = (
        dates > validation_cutoff
    )

    train = df.loc[
        train_mask
    ].copy()

    validation = df.loc[
        validation_mask
    ].copy()

    test = df.loc[
        test_mask
    ].copy()

    if train.empty:
        raise ValueError(
            "TRAIN split is empty."
        )

    if validation.empty:
        raise ValueError(
            "VALIDATION split is empty."
        )

    if test.empty:
        raise ValueError(
            "TEST split is empty."
        )

    return (
        train,
        validation,
        test,
    )


def get_features(df):
    excluded = {
        TARGET,
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

    features = [
        column
        for column in df.columns
        if column not in excluded
    ]

    return features


def prepare_xy(
    df,
    features,
):
    X = df[
        features
    ].copy()

    y = df[
        TARGET
    ].astype(int).copy()

    return X, y


def prepare_catboost_data(X):
    X = X.copy()

    categorical = (
        X.select_dtypes(
            include=[
                "object",
                "category",
            ]
        )
        .columns
        .tolist()
    )

    for column in categorical:
        X[column] = (
            X[column]
            .fillna("missing")
            .astype(str)
        )

    return X, categorical


def predict_probability(
    model,
    X,
):
    if isinstance(
        model,
        CatBoostClassifier,
    ):
        X_eval, _ = (
            prepare_catboost_data(X)
        )
    else:
        X_eval = X

    probability = model.predict_proba(
        X_eval
    )[:, 1]

    if not np.isfinite(
        probability
    ).all():
        raise ValueError(
            "Model probabilities contain "
            "NaN or infinite values."
        )

    return probability


def evaluate_model(
    model_name,
    model,
    X_test,
    y_test,
    threshold,
):
    probability = predict_probability(
        model,
        X_test,
    )

    prediction = (
        probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y_test,
            prediction,
            labels=[0, 1],
        ).ravel()
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    metrics = {
        "dataset": DATASET_NAME,
        "model": model_name,
        "threshold": threshold,
        "test_roc_auc": roc_auc_score(
            y_test,
            probability,
        ),
        "test_pr_auc": average_precision_score(
            y_test,
            probability,
        ),
        "test_brier": brier_score_loss(
            y_test,
            probability,
        ),
        "test_precision": precision_score(
            y_test,
            prediction,
            zero_division=0,
        ),
        "test_recall": recall_score(
            y_test,
            prediction,
            zero_division=0,
        ),
        "test_specificity": specificity,
        "test_f1": f1_score(
            y_test,
            prediction,
            zero_division=0,
        ),
        "test_accuracy": accuracy_score(
            y_test,
            prediction,
        ),
        "test_balanced_accuracy": (
            balanced_accuracy_score(
                y_test,
                prediction,
            )
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "test_n": len(y_test),
        "test_positives": int(y_test.sum()),
        "test_positive_rate": float(
            y_test.mean()
        ),
    }

    return (
        metrics,
        probability,
        prediction,
    )


def save_predictions(
    test,
    y_test,
    model_name,
    probability,
    prediction,
):
    output = pd.DataFrame(
        {
            "admission_ts": test[
                "admission_ts"
            ].values,
            "y_true": y_test.values,
            "predicted_probability": probability,
            "predicted_class": prediction,
        }
    )

    path = (
        MODEL_DIR
        / (
            f"B_trajectory_"
            f"{model_name}_test_predictions.csv"
        )
    )

    output.to_csv(
        path,
        index=False,
    )

    return path


def print_model_result(result):
    print()
    print("-" * 70)
    print(
        f"MODEL: "
        f"{result['model'].upper()}"
    )
    print("-" * 70)

    print(
        f"Threshold:          "
        f"{result['threshold']:.2f}"
    )

    print(
        f"ROC-AUC:            "
        f"{result['test_roc_auc']:.4f}"
    )

    print(
        f"PR-AUC:             "
        f"{result['test_pr_auc']:.4f}"
    )

    print(
        f"Brier:              "
        f"{result['test_brier']:.4f}"
    )

    print(
        f"Precision:          "
        f"{result['test_precision']:.4f}"
    )

    print(
        f"Recall:             "
        f"{result['test_recall']:.4f}"
    )

    print(
        f"Specificity:        "
        f"{result['test_specificity']:.4f}"
    )

    print(
        f"F1:                 "
        f"{result['test_f1']:.4f}"
    )

    print(
        f"Accuracy:           "
        f"{result['test_accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy:  "
        f"{result['test_balanced_accuracy']:.4f}"
    )

    print()
    print("Confusion matrix:")

    print(
        f"  TN: {result['tn']:,}"
    )

    print(
        f"  FP: {result['fp']:,}"
    )

    print(
        f"  FN: {result['fn']:,}"
    )

    print(
        f"  TP: {result['tp']:,}"
    )


def main():
    print()
    print("=" * 80)
    print(
        "FINAL TEST EVALUATION - "
        "TRAJECTORY MODEL"
    )
    print("=" * 80)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "- Models were selected using TRAIN/VALIDATION."
    )

    print(
        "- Thresholds were selected using VALIDATION."
    )

    print(
        "- TEST is being evaluated for the first time."
    )

    print(
        "- No threshold optimization will be performed on TEST."
    )

    print(
        "- No model tuning will be performed on TEST."
    )

    train_cutoff, validation_cutoff = (
        load_official_temporal_cutoffs()
    )

    df = load_dataset()

    (
        train,
        validation,
        test,
    ) = split_dataset(
        df,
        train_cutoff,
        validation_cutoff,
    )

    features = get_features(
        df
    )

    print()
    print(
        f"Dataset: {DATASET_NAME}"
    )

    print(
        f"Features: {len(features)}"
    )

    print(
        f"TRAIN: "
        f"{len(train):,} rows"
    )

    print(
        f"VALIDATION: "
        f"{len(validation):,} rows"
    )

    print(
        f"TEST: "
        f"{len(test):,} rows"
    )

    print(
        f"TEST period: "
        f"{test['admission_ts'].min().date()} "
        f"-> "
        f"{test['admission_ts'].max().date()}"
    )

    X_test, y_test = prepare_xy(
        test,
        features,
    )

    print()
    print(
        f"TEST positives: "
        f"{int(y_test.sum()):,}"
    )

    print(
        f"TEST positive rate: "
        f"{y_test.mean():.4%}"
    )

    results = []

    for model_name, config in (
        MODELS.items()
    ):
        print()
        print(
            "=" * 80
        )

        model_path = (
            MODEL_DIR
            / config["model_file"]
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found:\n"
                f"{model_path}"
            )

        print(
            f"Loading: "
            f"{model_path.name}",
            flush=True,
        )

        model = joblib.load(
            model_path
        )

        threshold = float(
            config["threshold"]
        )

        (
            result,
            probability,
            prediction,
        ) = evaluate_model(
            model_name,
            model,
            X_test,
            y_test,
            threshold,
        )

        results.append(
            result
        )

        prediction_path = (
            save_predictions(
                test,
                y_test,
                model_name,
                probability,
                prediction,
            )
        )

        print_model_result(
            result
        )

        print()
        print(
            f"Predictions saved to:"
        )

        print(
            prediction_path
        )

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            "test_pr_auc",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    results_path = (
        MODEL_DIR
        / "trajectory_test_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    print()
    print("=" * 80)
    print(
        "FINAL TEST COMPARISON"
    )
    print("=" * 80)

    display_columns = [
        "model",
        "threshold",
        "test_roc_auc",
        "test_pr_auc",
        "test_brier",
        "test_precision",
        "test_recall",
        "test_specificity",
        "test_f1",
        "test_balanced_accuracy",
    ]

    print()

    print(
        results_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    print()
    print(
        f"Results saved to:"
    )

    print(
        results_path
    )

    print()
    print(
        "TEST evaluation completed."
    )

    print(
        "No TEST-based model or threshold "
        "selection was performed."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()