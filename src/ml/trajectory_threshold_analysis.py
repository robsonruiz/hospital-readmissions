from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "trajectory_experiment"
)

DATASET_DIR = MODEL_DIR

TARGET = "unplanned_readmitted_30d"

DATASETS = {
    "A_baseline": "a_baseline.parquet",
    "B_trajectory": "b_trajectory.parquet",
    "C_aps": "c_aps.parquet",
}

MODEL_NAMES = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
]

THRESHOLDS = np.arange(
    0.05,
    0.96,
    0.01,
)


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
        f"VALIDATION cutoff: "
        f"{validation_cutoff.date()}"
    )

    return train_cutoff, validation_cutoff


def load_dataset(
    dataset_path,
    train_cutoff,
    validation_cutoff,
):
    df = pd.read_parquet(
        dataset_path
    )

    if TARGET not in df.columns:
        raise ValueError(
            f"Target '{TARGET}' not found in "
            f"{dataset_path.name}."
        )

    if "admission_ts" not in df.columns:
        raise ValueError(
            f"'admission_ts' not found in "
            f"{dataset_path.name}."
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

    dates = df["admission_ts"].dt.normalize()

    validation_mask = (
        (dates > train_cutoff)
        & (dates <= validation_cutoff)
    )

    validation = df.loc[
        validation_mask
    ].copy()

    if validation.empty:
        raise ValueError(
            f"Validation split is empty for "
            f"{dataset_path.name}."
        )

    return validation


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

    return X


def predict_probability(
    model,
    X,
):
    model_class_name = (
        model.__class__.__name__
    )

    if model_class_name == "CatBoostClassifier":
        X_eval = prepare_catboost_data(X)
    else:
        X_eval = X

    return model.predict_proba(
        X_eval
    )[:, 1]


def calculate_metrics(
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


def analyze_model(
    dataset_name,
    model_name,
    model,
    X_valid,
    y_valid,
):
    probability = predict_probability(
        model,
        X_valid,
    )

    results = []

    for threshold in THRESHOLDS:
        metrics = calculate_metrics(
            y_true=y_valid,
            probability=probability,
            threshold=float(threshold),
        )

        metrics.update(
            {
                "dataset": dataset_name,
                "model": model_name,
            }
        )

        results.append(metrics)

    return pd.DataFrame(results)


def main():
    print()
    print("=" * 80)
    print("TRAJECTORY THRESHOLD ANALYSIS")
    print("=" * 80)

    print()
    print("Methodology:")
    print("- Same official temporal split")
    print("- Validation only")
    print("- Thresholds: 0.05 to 0.95")
    print("- Step: 0.01")
    print("- Selection criterion: maximum F1")
    print("- TEST is NOT evaluated")

    train_cutoff, validation_cutoff = (
        load_official_temporal_cutoffs()
    )

    all_results = []
    best_results = []

    for dataset_name, filename in DATASETS.items():

        print()
        print("=" * 80)
        print(
            f"DATASET: {dataset_name}"
        )
        print("=" * 80)

        dataset_path = (
            DATASET_DIR / filename
        )

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found:\n"
                f"{dataset_path}"
            )

        validation = load_dataset(
            dataset_path,
            train_cutoff,
            validation_cutoff,
        )

        features = get_features(
            validation
        )

        X_valid = validation[
            features
        ].copy()

        y_valid = validation[
            TARGET
        ].astype(int)

        print(
            f"Validation rows: "
            f"{len(validation):,}"
        )

        print(
            f"Features: "
            f"{len(features)}"
        )

        for model_name in MODEL_NAMES:

            model_path = (
                MODEL_DIR
                / f"{dataset_name}_tuned_"
                f"{model_name}.joblib"
            )

            if not model_path.exists():
                print()
                print(
                    f"WARNING: Model not found: "
                    f"{model_path.name}"
                )
                continue

            print()
            print(
                f"Analyzing: "
                f"{model_name}"
            )

            model = joblib.load(
                model_path
            )

            model_results = analyze_model(
                dataset_name=dataset_name,
                model_name=model_name,
                model=model,
                X_valid=X_valid,
                y_valid=y_valid,
            )

            all_results.append(
                model_results
            )

            best = (
                model_results
                .sort_values(
                    [
                        "f1",
                        "recall",
                        "precision",
                    ],
                    ascending=False,
                )
                .iloc[0]
                .to_dict()
            )

            best_results.append(
                best
            )

            print(
                f"  Best threshold: "
                f"{best['threshold']:.2f}"
            )

            print(
                f"  Precision: "
                f"{best['precision']:.4f}"
            )

            print(
                f"  Recall: "
                f"{best['recall']:.4f}"
            )

            print(
                f"  Specificity: "
                f"{best['specificity']:.4f}"
            )

            print(
                f"  F1: "
                f"{best['f1']:.4f}"
            )

    if not all_results:
        raise ValueError(
            "No threshold results were generated."
        )

    all_results_df = pd.concat(
        all_results,
        ignore_index=True,
    )

    best_results_df = pd.DataFrame(
        best_results
    )

    all_results_df = (
        all_results_df[
            [
                "dataset",
                "model",
                "threshold",
                "precision",
                "recall",
                "specificity",
                "f1",
                "accuracy",
                "balanced_accuracy",
                "tn",
                "fp",
                "fn",
                "tp",
            ]
        ]
        .sort_values(
            [
                "dataset",
                "model",
                "threshold",
            ]
        )
        .reset_index(drop=True)
    )

    best_results_df = (
        best_results_df[
            [
                "dataset",
                "model",
                "threshold",
                "precision",
                "recall",
                "specificity",
                "f1",
                "accuracy",
                "balanced_accuracy",
                "tn",
                "fp",
                "fn",
                "tp",
            ]
        ]
        .sort_values(
            "f1",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    all_results_path = (
        MODEL_DIR
        / "trajectory_threshold_analysis_validation.csv"
    )

    best_results_path = (
        MODEL_DIR
        / "trajectory_best_thresholds_validation.csv"
    )

    all_results_df.to_csv(
        all_results_path,
        index=False,
    )

    best_results_df.to_csv(
        best_results_path,
        index=False,
    )

    print()
    print("=" * 80)
    print("BEST THRESHOLDS")
    print("=" * 80)

    print(
        best_results_df[
            [
                "dataset",
                "model",
                "threshold",
                "precision",
                "recall",
                "specificity",
                "f1",
            ]
        ].to_string(
            index=False,
        )
    )

    print()
    print("=" * 80)
    print("BEST MODEL PER DATASET")
    print("=" * 80)

    best_per_dataset = (
        best_results_df
        .sort_values(
            "f1",
            ascending=False,
        )
        .groupby(
            "dataset",
            as_index=False,
        )
        .first()
    )

    print(
        best_per_dataset[
            [
                "dataset",
                "model",
                "threshold",
                "precision",
                "recall",
                "specificity",
                "f1",
            ]
        ].to_string(
            index=False,
        )
    )

    print()
    print("=" * 80)
    print("FILES SAVED")
    print("=" * 80)

    print(
        all_results_path
    )

    print(
        best_results_path
    )

    print()
    print(
        "THRESHOLD ANALYSIS COMPLETED."
    )
    print(
        "TEST was not evaluated."
    )


if __name__ == "__main__":
    main()