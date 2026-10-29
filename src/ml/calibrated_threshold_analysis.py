from pathlib import Path
import sys

import duckdb
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DB_PATH = PROJECT_ROOT / "data" / "hospital_readmissions.duckdb"
MODELS_DIR = PROJECT_ROOT / "models"

TARGET = "unplanned_readmitted_30d"

CALIBRATED_MODEL_PATH = (
    MODELS_DIR / "calibrated_catboost_sigmoid.joblib"
)

THRESHOLD_RESULTS_PATH = (
    MODELS_DIR / "calibrated_threshold_analysis_validation.csv"
)

BEST_THRESHOLD_PATH = (
    MODELS_DIR / "best_calibrated_threshold_validation.csv"
)


DROP_COLUMNS = [
    TARGET,
    "readmitted_30d_clean",
    "eligible_for_readmission_model",
    "readmitted_30d_raw",
    "next_admission_ts",
    "days_until_next_admission",
    "admission_ts",
    "discharge_ts",
    "previous_discharge_ts",
    "admission_month",
    "identificador",
    "admission_year",
    "admission_quarter",
    "admission_weekday",
    "municipiosolicitante",
    "municipioresidencia",
    "municipioexecutante",
]


def load_ml_data():
    """
    Load ml_features from DuckDB.
    """
    con = duckdb.connect(
        DB_PATH,
        read_only=True,
    )

    df = con.execute(
        """
        SELECT *
        FROM ml_features
        ORDER BY admission_ts
        """
    ).df()

    con.close()

    if df.empty:
        raise ValueError(
            "ml_features is empty."
        )

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' "
            "was not found in ml_features."
        )

    if "admission_ts" not in df.columns:
        raise ValueError(
            "Column 'admission_ts' is required "
            "for temporal splitting."
        )

    df["admission_ts"] = pd.to_datetime(
        df["admission_ts"],
        errors="coerce",
    )

    if df["admission_ts"].isna().any():
        raise ValueError(
            "admission_ts contains missing or "
            "invalid dates."
        )

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce",
    )

    if df[TARGET].isna().any():
        raise ValueError(
            f"Target '{TARGET}' contains "
            "missing values."
        )

    df[TARGET] = df[TARGET].astype(int)

    if not set(
        df[TARGET].unique()
    ).issubset({0, 1}):
        raise ValueError(
            f"Target '{TARGET}' must contain "
            "only 0 and 1."
        )

    return df


def get_temporal_cutoffs(df):
    """
    Same temporal cutoff logic used by prepare_data.py.
    """
    df = (
        df.sort_values("admission_ts")
        .reset_index(drop=True)
    )

    dates = df["admission_ts"].dt.normalize()

    n = len(df)

    train_position = (
        max(1, int(n * 0.70)) - 1
    )

    validation_position = (
        max(1, int(n * 0.85)) - 1
    )

    train_cutoff_date = (
        dates.iloc[train_position]
    )

    validation_cutoff_date = (
        dates.iloc[validation_position]
    )

    if (
        validation_cutoff_date
        <= train_cutoff_date
    ):
        unique_dates = (
            dates
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )

        train_idx = (
            unique_dates
            .searchsorted(train_cutoff_date)
        )

        valid_idx = min(
            train_idx + 1,
            len(unique_dates) - 1,
        )

        train_cutoff_date = (
            unique_dates.iloc[train_idx]
        )

        validation_cutoff_date = (
            unique_dates.iloc[valid_idx]
        )

    return (
        train_cutoff_date,
        validation_cutoff_date,
    )


def create_validation_split(df):
    """
    Reproduce the exact chronological split
    used by the project.
    """
    df = (
        df.sort_values("admission_ts")
        .reset_index(drop=True)
        .copy()
    )

    (
        train_cutoff,
        validation_cutoff,
    ) = get_temporal_cutoffs(df)

    dates = df["admission_ts"].dt.normalize()

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

    train_df = df.loc[
        train_mask
    ].copy()

    validation_df = df.loc[
        validation_mask
    ].copy()

    test_df = df.loc[
        test_mask
    ].copy()

    if (
        train_df.empty
        or validation_df.empty
        or test_df.empty
    ):
        raise ValueError(
            "Temporal split produced an "
            "empty dataset."
        )

    y_train = train_df[TARGET].copy()
    y_valid = validation_df[TARGET].copy()
    y_test = test_df[TARGET].copy()

    X_train = train_df.drop(
        columns=[
            c
            for c in DROP_COLUMNS
            if c in train_df.columns
        ]
    )

    X_valid = validation_df.drop(
        columns=[
            c
            for c in DROP_COLUMNS
            if c in validation_df.columns
        ]
    )

    X_test = test_df.drop(
        columns=[
            c
            for c in DROP_COLUMNS
            if c in test_df.columns
        ]
    )

    return (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
        train_df,
        validation_df,
        test_df,
        train_cutoff,
        validation_cutoff,
    )


def load_validation_data():
    """
    Load ml_features and reproduce the project's
    temporal split without importing prepare_data.py.
    """
    print(
        "   Loading ml_features..."
    )

    df = load_ml_data()

    print(
        f"   Loaded {len(df):,} records."
    )

    (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
        train_df,
        validation_df,
        test_df,
        train_cutoff,
        validation_cutoff,
    ) = create_validation_split(df)

    print(
        f"   Train:      {len(X_train):,}"
    )

    print(
        f"   Validation: {len(X_valid):,}"
    )

    print(
        f"   Test:       {len(X_test):,}"
    )

    print(
        f"   Validation period: "
        f"{validation_df['admission_ts'].min().date()} "
        f"-> "
        f"{validation_df['admission_ts'].max().date()}"
    )

    print(
        f"   Validation positives: "
        f"{int(y_valid.sum()):,}"
    )

    print(
        f"   Validation positive rate: "
        f"{y_valid.mean() * 100:.2f}%"
    )

    print(
        f"   Train cutoff: "
        f"{train_cutoff.date()}"
    )

    print(
        f"   Validation cutoff: "
        f"{validation_cutoff.date()}"
    )

    return X_valid, y_valid


def load_calibrated_model():
    """
    Load the CatBoost + sigmoid calibrated model.
    """
    if not CALIBRATED_MODEL_PATH.exists():
        raise FileNotFoundError(
            "\nCalibrated model not found:\n"
            f"{CALIBRATED_MODEL_PATH}\n\n"
            "Check the filename generated by "
            "calibration_analysis.py."
        )

    print(
        f"   Loading model: "
        f"{CALIBRATED_MODEL_PATH.name}"
    )

    model = joblib.load(
        CALIBRATED_MODEL_PATH
    )

    if not hasattr(
        model,
        "predict_proba",
    ):
        raise TypeError(
            "The calibrated model does not "
            "have predict_proba()."
        )

    return model


def predict_calibrated_probability(
    model,
    X_valid,
):
    """
    Generate positive-class calibrated probabilities.
    """
    probability = model.predict_proba(
        X_valid
    )

    if probability.ndim != 2:
        raise ValueError(
            "predict_proba() did not return "
            "a 2D array."
        )

    if probability.shape[1] < 2:
        raise ValueError(
            "predict_proba() does not contain "
            "two class probabilities."
        )

    probability = probability[:, 1]

    if not np.isfinite(
        probability
    ).all():
        raise ValueError(
            "Probabilities contain NaN "
            "or infinite values."
        )

    return probability


def calculate_metrics(
    y_true,
    probability,
    threshold,
):
    """
    Calculate metrics for one threshold.
    """
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

    f1 = f1_score(
        y_true,
        prediction,
        zero_division=0,
    )

    accuracy = accuracy_score(
        y_true,
        prediction,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            prediction,
        )
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_threshold_analysis(
    y_valid,
    probability,
):
    """
    Evaluate thresholds from 0.05 to 0.95
    in increments of 0.01.
    """
    thresholds = np.round(
        np.arange(
            0.05,
            0.951,
            0.01,
        ),
        2,
    )

    results = []

    for threshold in thresholds:
        results.append(
            calculate_metrics(
                y_valid,
                probability,
                threshold,
            )
        )

    return pd.DataFrame(
        results
    )


def select_best_threshold(
    results_df,
):
    """
    Select the threshold with the highest F1.

    Tie-breakers:
    1. Recall
    2. Specificity
    """
    ranked = (
        results_df
        .sort_values(
            by=[
                "f1",
                "recall",
                "specificity",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    return ranked.iloc[0]


def save_results(
    results_df,
    best_result,
):
    """
    Save threshold analysis results.
    """
    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        THRESHOLD_RESULTS_PATH,
        index=False,
    )

    pd.DataFrame(
        [best_result]
    ).to_csv(
        BEST_THRESHOLD_PATH,
        index=False,
    )


def print_results(
    results_df,
    best_result,
    y_valid,
    probability,
):
    """
    Print threshold analysis summary.
    """
    print()
    print("=" * 70)
    print(
        "CATBOOST + SIGMOID — "
        "CALIBRATED THRESHOLD ANALYSIS"
    )
    print("=" * 70)

    print()
    print(
        f"Validation records: "
        f"{len(y_valid):,}"
    )

    print(
        f"Observed positive rate: "
        f"{y_valid.mean() * 100:.2f}%"
    )

    print()
    print("Probability distribution:")

    print(
        f"  Minimum: "
        f"{probability.min():.6f}"
    )

    print(
        f"  Maximum: "
        f"{probability.max():.6f}"
    )

    print(
        f"  Mean:    "
        f"{probability.mean():.6f}"
    )

    print()
    print("-" * 70)
    print("BEST THRESHOLD")
    print("-" * 70)

    print(
        f"\nThreshold: "
        f"{best_result['threshold']:.2f}"
    )

    print(
        f"\nPrecision:          "
        f"{best_result['precision']:.4f}"
    )

    print(
        f"Recall:             "
        f"{best_result['recall']:.4f}"
    )

    print(
        f"Specificity:        "
        f"{best_result['specificity']:.4f}"
    )

    print(
        f"F1:                 "
        f"{best_result['f1']:.4f}"
    )

    print(
        f"Accuracy:           "
        f"{best_result['accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy:  "
        f"{best_result['balanced_accuracy']:.4f}"
    )

    print()
    print("Confusion matrix:")

    print(
        f"  TN: {best_result['tn']:,}"
    )

    print(
        f"  FP: {best_result['fp']:,}"
    )

    print(
        f"  FN: {best_result['fn']:,}"
    )

    print(
        f"  TP: {best_result['tp']:,}"
    )

    print()
    print("-" * 70)
    print("TOP 10 THRESHOLDS BY F1")
    print("-" * 70)

    columns = [
        "threshold",
        "precision",
        "recall",
        "specificity",
        "f1",
        "balanced_accuracy",
    ]

    top_10 = (
        results_df
        .sort_values(
            by="f1",
            ascending=False,
        )
        .head(10)
    )

    print()

    print(
        top_10[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("-" * 70)
    print("FILES SAVED")
    print("-" * 70)

    print(
        f"\n{THRESHOLD_RESULTS_PATH}"
    )

    print(
        f"{BEST_THRESHOLD_PATH}"
    )

    print()
    print(
        "TEST was NOT used to select "
        "the threshold."
    )

    print("=" * 70)
    print()


def main():
    print()
    print("=" * 70)
    print(
        "CALIBRATED THRESHOLD ANALYSIS"
    )
    print("=" * 70)

    print()
    print(
        "1. Loading validation data..."
    )

    X_valid, y_valid = (
        load_validation_data()
    )

    print()
    print(
        "2. Loading calibrated "
        "CatBoost + sigmoid..."
    )

    model = load_calibrated_model()

    print()
    print(
        "3. Generating calibrated "
        "probabilities..."
    )

    probability = (
        predict_calibrated_probability(
            model,
            X_valid,
        )
    )

    print(
        f"   Generated "
        f"{len(probability):,} probabilities."
    )

    print()
    print(
        "4. Evaluating thresholds..."
    )

    results_df = run_threshold_analysis(
        y_valid,
        probability,
    )

    print(
        f"   Evaluated "
        f"{len(results_df)} thresholds."
    )

    print()
    print(
        "5. Selecting best threshold..."
    )

    best_result = (
        select_best_threshold(
            results_df
        )
    )

    print()
    print(
        "6. Saving results..."
    )

    save_results(
        results_df,
        best_result,
    )

    print_results(
        results_df,
        best_result,
        y_valid,
        probability,
    )


if __name__ == "__main__":
    main()