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


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODEL_DIR = PROJECT_ROOT / "models" / "trajectory_experiment"

DATASET_PATH = (
    MODEL_DIR
    / "a_baseline.parquet"
)

MODEL_PATH = (
    MODEL_DIR
    / "A_baseline_tuned_xgboost.joblib"
)

OUTPUT_RESULTS_PATH = (
    MODEL_DIR
    / "A_baseline_xgboost_test_results.csv"
)

OUTPUT_PREDICTIONS_PATH = (
    MODEL_DIR
    / "A_baseline_xgboost_test_predictions.csv"
)

TARGET = "unplanned_readmitted_30d"

THRESHOLD = 0.16


def load_dataset():
    print()
    print("=" * 80)
    print("LOADING A — BASELINE DATASET")
    print("=" * 80)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    df = pd.read_parquet(DATASET_PATH)

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

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError(
            f"Target '{TARGET}' must contain only 0 and 1."
        )

    print(
        f"Dataset: {DATASET_PATH.name}"
    )
    print(
        f"Rows: {len(df):,}"
    )

    return df


def load_official_temporal_cutoffs():
    print()
    print("=" * 80)
    print("LOADING OFFICIAL TEMPORAL SPLIT")
    print("=" * 80)

    from src.ml.prepare_data import (
        load_ml_data,
        prepare_data,
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
        f"TRAIN cutoff: {train_cutoff.date()}"
    )
    print(
        f"VALIDATION cutoff: {validation_cutoff.date()}"
    )

    return train_cutoff, validation_cutoff


def split_using_official_cutoffs(
    df,
    train_cutoff,
    validation_cutoff,
):
    dates = df["admission_ts"].dt.normalize()

    train_mask = dates <= train_cutoff

    validation_mask = (
        (dates > train_cutoff)
        & (dates <= validation_cutoff)
    )

    test_mask = dates > validation_cutoff

    train = df.loc[train_mask].copy()
    validation = df.loc[validation_mask].copy()
    test = df.loc[test_mask].copy()

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

    return train, validation, test


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

    return [
        column
        for column in df.columns
        if column not in excluded
    ]


def calculate_metrics(
    y_true,
    probability,
    threshold,
):
    prediction = (
        probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        y_true,
        prediction,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        prediction,
        zero_division=0,
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    f1 = f1_score(
        y_true,
        prediction,
        zero_division=0,
    )

    accuracy = accuracy_score(
        y_true,
        prediction,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        prediction,
    )

    roc_auc = roc_auc_score(
        y_true,
        probability,
    )

    pr_auc = average_precision_score(
        y_true,
        probability,
    )

    brier = brier_score_loss(
        y_true,
        probability,
    )

    return {
        "dataset": "A_baseline",
        "model": "XGBoost",
        "threshold": threshold,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "brier_score": brier,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():
    print()
    print("=" * 80)
    print("A — BASELINE + XGBOOST — TEST EVALUATION")
    print("=" * 80)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"\nTuned model not found:\n{MODEL_PATH}"
        )

    df = load_dataset()

    train_cutoff, validation_cutoff = (
        load_official_temporal_cutoffs()
    )

    train, validation, test = (
        split_using_official_cutoffs(
            df,
            train_cutoff,
            validation_cutoff,
        )
    )

    features = get_features(df)

    print()
    print("=" * 80)
    print("DATASET STRUCTURE")
    print("=" * 80)

    print(
        f"Features: {len(features)}"
    )

    print(
        f"TRAIN: {len(train):,}"
    )

    print(
        f"VALIDATION: {len(validation):,}"
    )

    print(
        f"TEST: {len(test):,}"
    )

    X_test = test[features].copy()
    y_test = test[TARGET].astype(int).copy()

    print()
    print("=" * 80)
    print("LOADING TUNED MODEL")
    print("=" * 80)

    print(
        f"Model: {MODEL_PATH.name}"
    )

    model = joblib.load(
        MODEL_PATH
    )

    if not hasattr(
        model,
        "predict_proba",
    ):
        raise TypeError(
            "Loaded model does not have "
            "'predict_proba'."
        )

    print()
    print("=" * 80)
    print("GENERATING TEST PROBABILITIES")
    print("=" * 80)

    probability = model.predict_proba(
        X_test
    )[:, 1]

    if not np.isfinite(
        probability
    ).all():
        raise ValueError(
            "Predicted probabilities contain "
            "NaN or infinite values."
        )

    print(
        f"Generated {len(probability):,} probabilities."
    )

    print()
    print("=" * 80)
    print("TEST RESULTS")
    print("=" * 80)

    result = calculate_metrics(
        y_test,
        probability,
        THRESHOLD,
    )

    result_df = pd.DataFrame(
        [result]
    )

    print()
    print(
        f"Threshold:       {result['threshold']:.2f}"
    )
    print(
        f"ROC-AUC:         {result['roc_auc']:.4f}"
    )
    print(
        f"PR-AUC:          {result['pr_auc']:.4f}"
    )
    print(
        f"Precision:       {result['precision']:.4f}"
    )
    print(
        f"Recall:          {result['recall']:.4f}"
    )
    print(
        f"Specificity:     {result['specificity']:.4f}"
    )
    print(
        f"F1:              {result['f1']:.4f}"
    )
    print(
        f"Accuracy:        {result['accuracy']:.4f}"
    )
    print(
        f"Balanced Acc.:   {result['balanced_accuracy']:.4f}"
    )
    print(
        f"Brier:           {result['brier_score']:.4f}"
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

    result_df.to_csv(
        OUTPUT_RESULTS_PATH,
        index=False,
    )

    predictions_df = test[
        [
            "identificador",
            "admission_ts",
            TARGET,
        ]
    ].copy()

    predictions_df[
        "predicted_probability"
    ] = probability

    predictions_df[
        "predicted_class"
    ] = (
        probability >= THRESHOLD
    ).astype(int)

    predictions_df.to_csv(
        OUTPUT_PREDICTIONS_PATH,
        index=False,
    )

    print()
    print("=" * 80)
    print("FILES SAVED")
    print("=" * 80)

    print(
        OUTPUT_RESULTS_PATH
    )

    print(
        OUTPUT_PREDICTIONS_PATH
    )

    print()
    print(
        "Threshold was selected on VALIDATION "
        "and was NOT modified using TEST."
    )

    print()
    print("=" * 80)
    print("A — BASELINE TEST EVALUATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()