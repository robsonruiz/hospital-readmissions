import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
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

MODEL_DIR = PROJECT_ROOT / "models" / "trajectory_experiment"

DATASETS = {
    "A_baseline": MODEL_DIR / "a_baseline.parquet",
    "B_trajectory": MODEL_DIR / "b_trajectory.parquet",
    "C_aps": MODEL_DIR / "c_aps.parquet",
}

MODELS = ["xgboost", "random_forest", "catboost"]

# Thresholds selected exclusively on the validation set.
THRESHOLDS = {
    ("A_baseline", "xgboost"): 0.16,
    ("A_baseline", "random_forest"): 0.16,
    ("A_baseline", "catboost"): 0.65,
    ("B_trajectory", "xgboost"): 0.14,
    ("B_trajectory", "random_forest"): 0.16,
    ("B_trajectory", "catboost"): 0.61,
    # C thresholds selected in the validation threshold analysis.
    ("C_aps", "xgboost"): 0.18,
    ("C_aps", "random_forest"): 0.16,
    ("C_aps", "catboost"): 0.61,
}


def get_temporal_split():
    from src.ml.prepare_data import load_ml_data, prepare_data

    df = load_ml_data()

    (
        _X_train,
        _X_valid,
        _X_test,
        _y_train,
        _y_valid,
        y_test,
        _train_df,
        _validation_df,
        test_df,
        train_cutoff,
        validation_cutoff,
    ) = prepare_data(df)

    return (
        y_test.reset_index(drop=True),
        train_cutoff,
        validation_cutoff,
    )


def get_features(df):
    excluded = {
        "unplanned_readmitted_30d",
        "episode_id",
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

    return [c for c in df.columns if c not in excluded]


def load_test_data(dataset_name, validation_cutoff):
    path = DATASETS[dataset_name]

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_parquet(path)
    df["admission_ts"] = pd.to_datetime(
        df["admission_ts"],
        errors="coerce",
    )

    if df["admission_ts"].isna().any():
        raise ValueError(
            f"{dataset_name} contains invalid or missing admission_ts."
        )

    test_mask = df["admission_ts"].dt.normalize() > validation_cutoff
    test_df = df.loc[test_mask].copy().reset_index(drop=True)

    if test_df.empty:
        raise ValueError(f"TEST split is empty for {dataset_name}.")

    return test_df


def evaluate_model(dataset_name, model_name, validation_cutoff):
    test_df = load_test_data(dataset_name, validation_cutoff)

    target = "unplanned_readmitted_30d"

    if target not in test_df.columns:
        raise ValueError(
            f"Target '{target}' not found in {dataset_name}."
        )

    y_test = pd.to_numeric(
        test_df[target],
        errors="coerce",
    )

    if y_test.isna().any():
        raise ValueError(
            f"Target contains missing values in {dataset_name}."
        )

    y_test = y_test.astype(int)

    features = get_features(test_df)
    X_test = test_df[features].copy()

    model_path = (
        MODEL_DIR
        / f"{dataset_name}_tuned_{model_name}.joblib"
    )

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)

    probabilities = model.predict_proba(X_test)[:, 1]

    threshold = THRESHOLDS[(dataset_name, model_name)]
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    result = {
        "dataset": dataset_name,
        "model": model_name,
        "threshold": threshold,
        "test_rows": len(y_test),
        "test_positives": int(y_test.sum()),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "specificity": (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0.0
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_test,
            predictions,
        ),
        "brier_score": brier_score_loss(
            y_test,
            probabilities,
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    prediction_path = (
        MODEL_DIR
        / f"{dataset_name}_{model_name}_test_predictions.csv"
    )

    prediction_columns = [
        c
        for c in ["identificador", "admission_ts"]
        if c in test_df.columns
    ]

    predictions_df = test_df[prediction_columns].copy()
    predictions_df["y_true"] = y_test.to_numpy()
    predictions_df["probability"] = probabilities
    predictions_df["prediction"] = predictions

    predictions_df.to_csv(
        prediction_path,
        index=False,
    )

    return result


def main():
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        _y_test,
        train_cutoff,
        validation_cutoff,
    ) = get_temporal_split()

    print("\n=== FINAL TEST EVALUATION: A vs B vs C ===")
    print(f"Train cutoff:      {train_cutoff.date()}")
    print(f"Validation cutoff: {validation_cutoff.date()}")
    print("\nDatasets:")
    print("A = baseline hospital history")
    print("B = baseline + complete trajectory")
    print("C = baseline + APS trajectory")
    print("\nModels:")
    print("XGBoost + Random Forest + CatBoost")
    print("\nTEST is evaluated only after model/threshold selection.")
    print("No retuning or threshold adjustment is performed on TEST.")

    results = []

    for dataset_name in DATASETS:
        for model_name in MODELS:
            print(
                f"\nEvaluating "
                f"{dataset_name} + {model_name}..."
            )

            result = evaluate_model(
                dataset_name,
                model_name,
                validation_cutoff,
            )

            results.append(result)

            print(
                f"ROC-AUC={result['roc_auc']:.4f} | "
                f"PR-AUC={result['pr_auc']:.4f} | "
                f"F1={result['f1']:.4f} | "
                f"Recall={result['recall']:.4f} | "
                f"Precision={result['precision']:.4f}"
            )

    results_df = pd.DataFrame(results)

    dataset_order = [
        "A_baseline",
        "B_trajectory",
        "C_aps",
    ]

    model_order = [
        "xgboost",
        "random_forest",
        "catboost",
    ]

    results_df["dataset"] = pd.Categorical(
        results_df["dataset"],
        categories=dataset_order,
        ordered=True,
    )

    results_df["model"] = pd.Categorical(
        results_df["model"],
        categories=model_order,
        ordered=True,
    )

    results_df = results_df.sort_values(
        ["dataset", "model"]
    ).reset_index(drop=True)

    output_path = (
        MODEL_DIR
        / "trajectory_test_results_all_models.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    metrics = [
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "specificity",
        "f1",
        "brier_score",
    ]

    comparison = results_df.pivot(
        index="model",
        columns="dataset",
        values=metrics,
    )

    comparison_path = (
        MODEL_DIR
        / "trajectory_test_comparison_all_models.csv"
    )

    comparison.to_csv(
        comparison_path
    )

    print("\n=== FINAL RESULTS ===")
    print(
        results_df[
            [
                "dataset",
                "model",
                "threshold",
                "roc_auc",
                "pr_auc",
                "precision",
                "recall",
                "specificity",
                "f1",
                "brier_score",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved: {output_path}")
    print(f"Saved: {comparison_path}")


if __name__ == "__main__":
    main()
