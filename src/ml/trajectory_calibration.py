from pathlib import Path
import sys
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)

from catboost import CatBoostClassifier


warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODEL_DIR = PROJECT_ROOT / "models" / "trajectory_experiment"
DATASET_DIR = MODEL_DIR

RANDOM_STATE = 42
CV_SPLITS = 3
TARGET = "unplanned_readmitted_30d"

DATASETS = {
    "B_trajectory": "b_trajectory.parquet",
    "C_aps": "c_aps.parquet",
}

CANDIDATES = {
    "B_trajectory": [
        "random_forest",
        "xgboost",
        "catboost",
    ],
    "C_aps": [
        "xgboost",
    ],
}


class CalibratedModel:
    def __init__(self, base_model, calibrator, method):
        self.base_model = base_model
        self.calibrator = calibrator
        self.method = method

    def predict_proba(self, X):
        base_probability = predict_probability(
            self.base_model,
            X,
        )

        calibrated_probability = self._calibrate(
            base_probability
        )

        return np.column_stack(
            [
                1.0 - calibrated_probability,
                calibrated_probability,
            ]
        )

    def _calibrate(self, probability):
        probability = np.asarray(
            probability,
            dtype=float,
        )

        if self.method == "sigmoid":
            calibrated = self.calibrator.predict_proba(
                probability.reshape(-1, 1)
            )[:, 1]

        elif self.method == "isotonic":
            calibrated = self.calibrator.predict(
                probability
            )

        else:
            raise ValueError(
                f"Unknown calibration method: {self.method}"
            )

        return np.clip(
            calibrated,
            0.0,
            1.0,
        )


def load_dataset(path: Path) -> pd.DataFrame:
    print(
        f"  Loading: {path.name}",
        flush=True,
    )

    df = pd.read_parquet(path)

    if TARGET not in df.columns:
        raise ValueError(
            f"Target '{TARGET}' not found in {path.name}."
        )

    if "admission_ts" not in df.columns:
        raise ValueError(
            f"Column 'admission_ts' not found in {path.name}."
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

    return df


def load_official_temporal_cutoffs():
    print()
    print(
        "Loading official temporal split...",
        flush=True,
    )

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
        f"Official TRAIN cutoff: "
        f"{train_cutoff.date()}",
        flush=True,
    )

    print(
        f"Official VALIDATION cutoff: "
        f"{validation_cutoff.date()}",
        flush=True,
    )

    return (
        train_cutoff,
        validation_cutoff,
    )


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

    return [
        column
        for column in df.columns
        if column not in excluded
    ]


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


def load_tuned_model(
    dataset_name,
    model_name,
):
    model_path = (
        MODEL_DIR
        / f"{dataset_name}_tuned_{model_name}.joblib"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Tuned model not found:\n"
            f"{model_path}"
        )

    print(
        f"  Loading model: "
        f"{model_path.name}",
        flush=True,
    )

    return joblib.load(
        model_path
    )


def generate_temporal_oof_predictions(
    model,
    X_train,
    y_train,
):
    """
    Generate out-of-fold probabilities using
    TimeSeriesSplit.

    Each validation fold is predicted only by
    a model trained on observations occurring
    earlier in time.
    """

    print(
        "  Generating temporal OOF "
        "predictions...",
        flush=True,
    )

    tscv = TimeSeriesSplit(
        n_splits=CV_SPLITS
    )

    oof_probability = np.full(
        len(X_train),
        np.nan,
        dtype=float,
    )

    for fold, (
        train_idx,
        validation_idx,
    ) in enumerate(
        tscv.split(X_train),
        start=1,
    ):
        print(
            f"    Fold {fold}: "
            f"train={len(train_idx):,}, "
            f"validation={len(validation_idx):,}",
            flush=True,
        )

        fold_model = clone_model(
            model
        )

        X_fold_train = (
            X_train.iloc[train_idx]
        )

        y_fold_train = (
            y_train.iloc[train_idx]
        )

        X_fold_valid = (
            X_train.iloc[validation_idx]
        )

        fold_model.fit(
            X_fold_train,
            y_fold_train,
        )

        probability = predict_probability(
            fold_model,
            X_fold_valid,
        )

        oof_probability[
            validation_idx
        ] = probability

    valid_mask = np.isfinite(
        oof_probability
    )

    if valid_mask.sum() == 0:
        raise ValueError(
            "No valid temporal OOF predictions "
            "were generated."
        )

    print(
        f"  Valid OOF predictions: "
        f"{valid_mask.sum():,}",
        flush=True,
    )

    return (
        oof_probability,
        valid_mask,
    )


def clone_model(model):
    """
    Create a fresh model with the same
    parameters as the tuned model.

    This avoids fitting the already-fitted
    object used for validation.
    """

    from sklearn.base import clone

    return clone(model)


def fit_sigmoid_calibrator(
    probability,
    y,
):
    print(
        "  Fitting sigmoid calibrator...",
        flush=True,
    )

    calibrator = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
    )

    calibrator.fit(
        probability.reshape(-1, 1),
        y,
    )

    return calibrator


def fit_isotonic_calibrator(
    probability,
    y,
):
    print(
        "  Fitting isotonic calibrator...",
        flush=True,
    )

    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )

    calibrator.fit(
        probability,
        y,
    )

    return calibrator


def fit_final_base_model(
    model,
    X_train,
    y_train,
):
    print(
        "  Fitting final base model "
        "on complete TRAIN...",
        flush=True,
    )

    final_model = clone_model(
        model
    )

    final_model.fit(
        X_train,
        y_train,
    )

    return final_model


def calculate_probability_metrics(
    y_true,
    probability,
):
    return {
        "roc_auc": roc_auc_score(
            y_true,
            probability,
        ),
        "pr_auc": average_precision_score(
            y_true,
            probability,
        ),
        "brier_score": brier_score_loss(
            y_true,
            probability,
        ),
    }


def calculate_threshold_metrics(
    y_true,
    probability,
    threshold,
):
    prediction = (
        probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            prediction,
            labels=[0, 1],
        ).ravel()
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "threshold": threshold,
        "precision": precision_score(
            y_true,
            prediction,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            prediction,
            zero_division=0,
        ),
        "specificity": specificity,
        "f1": f1_score(
            y_true,
            prediction,
            zero_division=0,
        ),
        "accuracy": accuracy_score(
            y_true,
            prediction,
        ),
        "balanced_accuracy": (
            balanced_accuracy_score(
                y_true,
                prediction,
            )
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def analyze_thresholds(
    y_true,
    probability,
):
    thresholds = np.round(
        np.arange(
            0.05,
            0.951,
            0.01,
        ),
        2,
    )

    rows = []

    for threshold in thresholds:
        metrics = calculate_threshold_metrics(
            y_true,
            probability,
            float(threshold),
        )

        rows.append(
            metrics
        )

    result = pd.DataFrame(
        rows
    )

    return (
        result
        .sort_values(
            [
                "f1",
                "recall",
                "precision",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )


def select_best_threshold(
    threshold_results,
):
    return (
        threshold_results
        .sort_values(
            [
                "f1",
                "recall",
                "precision",
            ],
            ascending=False,
        )
        .iloc[0]
        .copy()
    )


def calibrate_candidate(
    dataset_name,
    model_name,
    train,
    validation,
    features,
):
    print()
    print("=" * 80)
    print(
        f"CALIBRATION: "
        f"{dataset_name} + {model_name}"
    )
    print("=" * 80)

    X_train, y_train = prepare_xy(
        train,
        features,
    )

    X_valid, y_valid = prepare_xy(
        validation,
        features,
    )

    tuned_model = load_tuned_model(
        dataset_name,
        model_name,
    )

    print()
    print(
        "Step 1 - Temporal OOF "
        "predictions..."
    )

    (
        oof_probability,
        valid_mask,
    ) = generate_temporal_oof_predictions(
        tuned_model,
        X_train,
        y_train,
    )

    oof_probability_valid = (
        oof_probability[
            valid_mask
        ]
    )

    y_oof = (
        y_train
        .iloc[
            np.where(valid_mask)[0]
        ]
        .reset_index(drop=True)
    )

    print()
    print(
        "Step 2 - OOF calibration "
        "quality..."
    )

    oof_metrics = (
        calculate_probability_metrics(
            y_oof,
            oof_probability_valid,
        )
    )

    print(
        f"  OOF ROC-AUC: "
        f"{oof_metrics['roc_auc']:.4f}"
    )

    print(
        f"  OOF PR-AUC:  "
        f"{oof_metrics['pr_auc']:.4f}"
    )

    print(
        f"  OOF Brier:   "
        f"{oof_metrics['brier_score']:.4f}"
    )

    print()
    print(
        "Step 3 - Fitting calibrators..."
    )

    sigmoid_calibrator = (
        fit_sigmoid_calibrator(
            oof_probability_valid,
            y_oof,
        )
    )

    isotonic_calibrator = (
        fit_isotonic_calibrator(
            oof_probability_valid,
            y_oof,
        )
    )

    print()
    print(
        "Step 4 - Fitting final base "
        "model on complete TRAIN..."
    )

    final_base_model = (
        fit_final_base_model(
            tuned_model,
            X_train,
            y_train,
        )
    )

    print()
    print(
        "Step 5 - Generating VALIDATION "
        "probabilities..."
    )

    base_probability = predict_probability(
        final_base_model,
        X_valid,
    )

    calibrated_models = {
        "sigmoid": CalibratedModel(
            final_base_model,
            sigmoid_calibrator,
            "sigmoid",
        ),
        "isotonic": CalibratedModel(
            final_base_model,
            isotonic_calibrator,
            "isotonic",
        ),
    }

    all_results = []
    threshold_results_all = []

    for method, calibrated_model in (
        calibrated_models.items()
    ):
        print()
        print(
            f"Step 6 - Evaluating "
            f"{method} calibration..."
        )

        probability = (
            calibrated_model
            .predict_proba(X_valid)[:, 1]
        )

        probability_metrics = (
            calculate_probability_metrics(
                y_valid,
                probability,
            )
        )

        threshold_results = (
            analyze_thresholds(
                y_valid,
                probability,
            )
        )

        best_threshold = (
            select_best_threshold(
                threshold_results
            )
        )

        all_results.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "calibration": method,
                "validation_roc_auc": (
                    probability_metrics[
                        "roc_auc"
                    ]
                ),
                "validation_pr_auc": (
                    probability_metrics[
                        "pr_auc"
                    ]
                ),
                "validation_brier": (
                    probability_metrics[
                        "brier_score"
                    ]
                ),
                "best_threshold": float(
                    best_threshold[
                        "threshold"
                    ]
                ),
                "precision": float(
                    best_threshold[
                        "precision"
                    ]
                ),
                "recall": float(
                    best_threshold[
                        "recall"
                    ]
                ),
                "specificity": float(
                    best_threshold[
                        "specificity"
                    ]
                ),
                "f1": float(
                    best_threshold[
                        "f1"
                    ]
                ),
                "accuracy": float(
                    best_threshold[
                        "accuracy"
                    ]
                ),
                "balanced_accuracy": float(
                    best_threshold[
                        "balanced_accuracy"
                    ]
                ),
                "tn": int(
                    best_threshold[
                        "tn"
                    ]
                ),
                "fp": int(
                    best_threshold[
                        "fp"
                    ]
                ),
                "fn": int(
                    best_threshold[
                        "fn"
                    ]
                ),
                "tp": int(
                    best_threshold[
                        "tp"
                    ]
                ),
            }
        )

        threshold_results = (
            threshold_results
            .copy()
        )

        threshold_results.insert(
            0,
            "dataset",
            dataset_name,
        )

        threshold_results.insert(
            1,
            "model",
            model_name,
        )

        threshold_results.insert(
            2,
            "calibration",
            method,
        )

        threshold_results_all.append(
            threshold_results
        )

        print(
            f"  {method.upper()}"
        )

        print(
            f"    ROC-AUC: "
            f"{probability_metrics['roc_auc']:.4f}"
        )

        print(
            f"    PR-AUC:  "
            f"{probability_metrics['pr_auc']:.4f}"
        )

        print(
            f"    Brier:   "
            f"{probability_metrics['brier_score']:.4f}"
        )

        print(
            f"    Best threshold: "
            f"{best_threshold['threshold']:.2f}"
        )

        print(
            f"    F1: "
            f"{best_threshold['f1']:.4f}"
        )

        print(
            f"    Recall: "
            f"{best_threshold['recall']:.4f}"
        )

        print(
            f"    Precision: "
            f"{best_threshold['precision']:.4f}"
        )

        model_path = (
            MODEL_DIR
            / (
                f"{dataset_name}_"
                f"{model_name}_"
                f"calibrated_"
                f"{method}.joblib"
            )
        )

        joblib.dump(
            calibrated_model,
            model_path,
        )

        print(
            f"    Saved: "
            f"{model_path.name}"
        )

    base_metrics = (
        calculate_probability_metrics(
            y_valid,
            base_probability,
        )
    )

    base_threshold_results = (
        analyze_thresholds(
            y_valid,
            base_probability,
        )
    )

    base_best_threshold = (
        select_best_threshold(
            base_threshold_results
        )
    )

    all_results.append(
        {
            "dataset": dataset_name,
            "model": model_name,
            "calibration": "none",
            "validation_roc_auc": (
                base_metrics["roc_auc"]
            ),
            "validation_pr_auc": (
                base_metrics["pr_auc"]
            ),
            "validation_brier": (
                base_metrics["brier_score"]
            ),
            "best_threshold": float(
                base_best_threshold[
                    "threshold"
                ]
            ),
            "precision": float(
                base_best_threshold[
                    "precision"
                ]
            ),
            "recall": float(
                base_best_threshold[
                    "recall"
                ]
            ),
            "specificity": float(
                base_best_threshold[
                    "specificity"
                ]
            ),
            "f1": float(
                base_best_threshold[
                    "f1"
                ]
            ),
            "accuracy": float(
                base_best_threshold[
                    "accuracy"
                ]
            ),
            "balanced_accuracy": float(
                base_best_threshold[
                    "balanced_accuracy"
                ]
            ),
            "tn": int(
                base_best_threshold["tn"]
            ),
            "fp": int(
                base_best_threshold["fp"]
            ),
            "fn": int(
                base_best_threshold["fn"]
            ),
            "tp": int(
                base_best_threshold["tp"]
            ),
        }
    )

    return (
        all_results,
        threshold_results_all,
    )


def main():
    print()
    print("=" * 80)
    print(
        "TEMPORAL CALIBRATION - "
        "TRAJECTORY EXPERIMENT"
    )
    print("=" * 80)

    print()
    print(
        "Candidates:"
    )

    for dataset_name, models in (
        CANDIDATES.items()
    ):
        print(
            f"  {dataset_name}: "
            f"{', '.join(models)}"
        )

    print()
    print(
        "TEST will NOT be evaluated."
    )

    train_cutoff, validation_cutoff = (
        load_official_temporal_cutoffs()
    )

    all_results = []
    all_threshold_results = []

    for dataset_name, filename in (
        DATASETS.items()
    ):
        dataset_path = (
            DATASET_DIR / filename
        )

        df = load_dataset(
            dataset_path
        )

        train, validation, test = (
            split_using_official_cutoffs(
                df,
                train_cutoff,
                validation_cutoff,
            )
        )

        features = get_features(
            df
        )

        print()
        print(
            f"{dataset_name}:"
        )

        print(
            f"  Features: "
            f"{len(features)}"
        )

        print(
            f"  TRAIN: "
            f"{len(train):,}"
        )

        print(
            f"  VALIDATION: "
            f"{len(validation):,}"
        )

        print(
            f"  TEST: "
            f"{len(test):,} "
            f"(not evaluated)"
        )

        for model_name in CANDIDATES[
            dataset_name
        ]:
            (
                results,
                threshold_results,
            ) = calibrate_candidate(
                dataset_name,
                model_name,
                train,
                validation,
                features,
            )

            all_results.extend(
                results
            )

            all_threshold_results.extend(
                threshold_results
            )

    results_df = pd.DataFrame(
        all_results
    )

    threshold_df = pd.concat(
        all_threshold_results,
        ignore_index=True,
    )

    results_df = (
        results_df
        .sort_values(
            [
                "dataset",
                "f1",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    threshold_path = (
        MODEL_DIR
        / "trajectory_calibration_thresholds_validation.csv"
    )

    results_path = (
        MODEL_DIR
        / "trajectory_calibration_results_validation.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    threshold_df.to_csv(
        threshold_path,
        index=False,
    )

    print()
    print("=" * 80)
    print(
        "CALIBRATION RESULTS"
    )
    print("=" * 80)

    display_columns = [
        "dataset",
        "model",
        "calibration",
        "validation_roc_auc",
        "validation_pr_auc",
        "validation_brier",
        "best_threshold",
        "precision",
        "recall",
        "specificity",
        "f1",
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
        f"Results saved to:\n"
        f"{results_path}"
    )

    print()
    print(
        f"Threshold results saved to:\n"
        f"{threshold_path}"
    )

    print()
    print(
        "TEST was NOT evaluated."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()