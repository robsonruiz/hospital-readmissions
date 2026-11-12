from pathlib import Path
import sys
import time
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
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

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODEL_DIR = PROJECT_ROOT / "models" / "trajectory_experiment"
DATASET_DIR = MODEL_DIR

RANDOM_STATE = 42
CV_SPLITS = 3
N_ITER = 10

TARGET = "unplanned_readmitted_30d"

DATASETS = {
    "A_baseline": "dataset_A_baseline.parquet",
    "B_trajectory": "dataset_B_trajectory.parquet",
    "C_aps": "dataset_C_aps.parquet",
}


def load_dataset(path: Path) -> pd.DataFrame:
    print(f"  Loading: {path.name}", flush=True)

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
    print("Loading official temporal split...", flush=True)

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
        f"Official TRAIN cutoff: {train_cutoff.date()}",
        flush=True,
    )
    print(
        f"Official VALIDATION cutoff: {validation_cutoff.date()}",
        flush=True,
    )

    return train_cutoff, validation_cutoff


def split_using_official_cutoffs(
    df: pd.DataFrame,
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
        raise ValueError("TRAIN split is empty.")

    if validation.empty:
        raise ValueError("VALIDATION split is empty.")

    if test.empty:
        raise ValueError("TEST split is empty.")

    return train, validation, test


def get_features(df: pd.DataFrame):
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
    df: pd.DataFrame,
    features,
):
    X = df[features].copy()

    y = df[TARGET].astype(int).copy()

    return X, y


def build_preprocessor(X: pd.DataFrame):
    numeric_features = (
        X.select_dtypes(
            include=[
                "number",
                "bool",
            ]
        )
        .columns
        .tolist()
    )

    categorical_features = (
        X.select_dtypes(
            include=[
                "object",
                "category",
            ]
        )
        .columns
        .tolist()
    )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    transformers = []

    if numeric_features:
        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    if not transformers:
        raise ValueError(
            "No numeric or categorical features were found."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def build_sklearn_models():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "decision_tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        ),
        "lightgbm": LGBMClassifier(
            objective="binary",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        ),
    }


def build_search_spaces():
    return {
        "logistic_regression": {
            "model__C": np.logspace(
                -3,
                1,
                8,
            ),
            "model__solver": [
                "liblinear",
                "lbfgs",
            ],
        },
        "decision_tree": {
            "model__max_depth": [
                3,
                5,
                8,
                12,
                16,
                None,
            ],
            "model__min_samples_leaf": [
                2,
                5,
                10,
                20,
                30,
            ],
            "model__min_samples_split": [
                2,
                5,
                10,
                20,
            ],
            "model__criterion": [
                "gini",
                "entropy",
            ],
        },
        "random_forest": {
            "model__n_estimators": [
                100,
                200,
                300,
            ],
            "model__max_depth": [
                5,
                8,
                12,
                16,
                None,
            ],
            "model__min_samples_leaf": [
                2,
                5,
                10,
                20,
            ],
            "model__max_features": [
                "sqrt",
                "log2",
                0.5,
            ],
        },
        "xgboost": {
            "model__n_estimators": [
                100,
                200,
                300,
            ],
            "model__max_depth": [
                3,
                4,
                5,
                6,
            ],
            "model__learning_rate": [
                0.01,
                0.03,
                0.05,
                0.1,
            ],
            "model__subsample": [
                0.7,
                0.8,
                0.9,
                1.0,
            ],
            "model__colsample_bytree": [
                0.7,
                0.8,
                0.9,
                1.0,
            ],
            "model__min_child_weight": [
                1,
                3,
                5,
                10,
            ],
        },
        "lightgbm": {
            "model__n_estimators": [
                100,
                200,
                300,
            ],
            "model__learning_rate": [
                0.01,
                0.03,
                0.05,
                0.1,
            ],
            "model__num_leaves": [
                7,
                15,
                31,
                63,
            ],
            "model__max_depth": [
                -1,
                5,
                8,
                12,
            ],
            "model__min_child_samples": [
                10,
                20,
                30,
                50,
            ],
            "model__subsample": [
                0.7,
                0.8,
                0.9,
                1.0,
            ],
        },
    }


def tune_sklearn_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    models = build_sklearn_models()
    search_spaces = build_search_spaces()

    if model_name not in models:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    print(
        f"    Building {model_name} pipeline...",
        flush=True,
    )

    preprocessor = build_preprocessor(
        X_train
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                models[model_name],
            ),
        ]
    )

    cv = TimeSeriesSplit(
        n_splits=CV_SPLITS
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=search_spaces[
            model_name
        ],
        n_iter=N_ITER,
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    search.fit(
        X_train,
        y_train,
    )

    return (
        search.best_estimator_,
        float(search.best_score_),
        search.best_params_,
    )


def build_catboost():
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=RANDOM_STATE,
        verbose=False,
        thread_count=-1,
    )


def prepare_catboost_data(
    X: pd.DataFrame,
):
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


def tune_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    X_train_cb, categorical = (
        prepare_catboost_data(
            X_train
        )
    )

    model = build_catboost()

    params = {
        "iterations": [
            100,
            200,
            300,
        ],
        "depth": [
            4,
            5,
            6,
            7,
            8,
        ],
        "learning_rate": [
            0.01,
            0.03,
            0.05,
            0.1,
        ],
        "l2_leaf_reg": [
            1,
            3,
            5,
            10,
        ],
        "random_strength": [
            0,
            0.5,
            1,
            2,
        ],
    }

    cv = TimeSeriesSplit(
        n_splits=CV_SPLITS
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=N_ITER,
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
        verbose=1,
    )

    search.fit(
        X_train_cb,
        y_train,
        cat_features=categorical,
    )

    return (
        search.best_estimator_,
        float(search.best_score_),
        search.best_params_,
    )


def predict_probability(
    model,
    X: pd.DataFrame,
):
    if isinstance(
        model,
        CatBoostClassifier,
    ):
        X_eval, _ = (
            prepare_catboost_data(
                X
            )
        )
    else:
        X_eval = X

    return model.predict_proba(
        X_eval
    )[:, 1]


def evaluate_validation(
    model,
    X_valid,
    y_valid,
):
    probability = predict_probability(
        model,
        X_valid,
    )

    threshold = 0.5

    prediction = (
        probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y_valid,
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
        "roc_auc": roc_auc_score(
            y_valid,
            probability,
        ),
        "pr_auc": average_precision_score(
            y_valid,
            probability,
        ),
        "precision": precision_score(
            y_valid,
            prediction,
            zero_division=0,
        ),
        "recall": recall_score(
            y_valid,
            prediction,
            zero_division=0,
        ),
        "specificity": specificity,
        "f1": f1_score(
            y_valid,
            prediction,
            zero_division=0,
        ),
        "accuracy": accuracy_score(
            y_valid,
            prediction,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_valid,
            prediction,
        ),
        "brier_score": brier_score_loss(
            y_valid,
            probability,
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": threshold,
    }


def run_model(
    dataset_name,
    model_name,
    X_train,
    y_train,
    X_valid,
    y_valid,
    features,
):
    print()
    print(
        "-" * 70,
        flush=True,
    )
    print(
        f"MODEL: {model_name}",
        flush=True,
    )
    print(
        "-" * 70,
        flush=True,
    )

    start = time.time()

    if model_name == "catboost":
        model, cv_pr_auc, best_params = (
            tune_catboost(
                X_train,
                y_train,
            )
        )
    else:
        model, cv_pr_auc, best_params = (
            tune_sklearn_model(
                model_name,
                X_train,
                y_train,
            )
        )

    elapsed = (
        time.time() - start
    )

    print(
        f"    Best CV PR-AUC: {cv_pr_auc:.4f}",
        flush=True,
    )

    print(
        f"    Time: {elapsed / 60:.2f} min",
        flush=True,
    )

    print(
        "    Evaluating validation...",
        flush=True,
    )

    validation = evaluate_validation(
        model,
        X_valid,
        y_valid,
    )

    print(
        f"    Validation ROC-AUC: "
        f"{validation['roc_auc']:.4f}",
        flush=True,
    )

    print(
        f"    Validation PR-AUC: "
        f"{validation['pr_auc']:.4f}",
        flush=True,
    )

    print(
        f"    Validation F1 @ 0.5: "
        f"{validation['f1']:.4f}",
        flush=True,
    )

    print(
        f"    Validation Recall @ 0.5: "
        f"{validation['recall']:.4f}",
        flush=True,
    )

    model_path = (
        MODEL_DIR
        / f"{dataset_name}_tuned_{model_name}.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    params_path = (
        MODEL_DIR
        / f"{dataset_name}_tuned_{model_name}_params.json"
    )

    with open(
        params_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            best_params,
            file,
            indent=2,
            default=str,
        )

    return {
        "dataset": dataset_name,
        "model": model_name,
        "n_features": len(features),
        "cv_pr_auc": cv_pr_auc,
        "validation_roc_auc": validation[
            "roc_auc"
        ],
        "validation_pr_auc": validation[
            "pr_auc"
        ],
        "validation_precision": validation[
            "precision"
        ],
        "validation_recall": validation[
            "recall"
        ],
        "validation_specificity": validation[
            "specificity"
        ],
        "validation_f1": validation[
            "f1"
        ],
        "validation_accuracy": validation[
            "accuracy"
        ],
        "validation_balanced_accuracy": validation[
            "balanced_accuracy"
        ],
        "validation_brier": validation[
            "brier_score"
        ],
        "validation_tn": validation[
            "tn"
        ],
        "validation_fp": validation[
            "fp"
        ],
        "validation_fn": validation[
            "fn"
        ],
        "validation_tp": validation[
            "tp"
        ],
        "validation_threshold": validation[
            "threshold"
        ],
        "elapsed_minutes": elapsed / 60,
        "best_params": json.dumps(
            best_params,
            default=str,
        ),
    }


def run_dataset_experiment(
    dataset_name,
    dataset_path,
    train_cutoff,
    validation_cutoff,
):
    print()
    print("=" * 80)
    print(
        f"DATASET: {dataset_name}",
        flush=True,
    )
    print("=" * 80)

    df = load_dataset(
        dataset_path
    )

    features = get_features(
        df
    )

    print(
        f"Rows: {len(df):,}",
        flush=True,
    )

    print(
        f"Features ({len(features)}):",
        flush=True,
    )

    for feature in features:
        print(
            f"  - {feature}",
            flush=True,
        )

    train, validation, test = (
        split_using_official_cutoffs(
            df,
            train_cutoff,
            validation_cutoff,
        )
    )

    print()
    print(
        f"TRAIN:      {len(train):,}",
        flush=True,
    )

    print(
        f"VALIDATION: {len(validation):,}",
        flush=True,
    )

    print(
        f"TEST:       {len(test):,}",
        flush=True,
    )

    print(
        f"Train dates: "
        f"{train['admission_ts'].min().date()} "
        f"-> "
        f"{train['admission_ts'].max().date()}",
        flush=True,
    )

    print(
        f"Validation dates: "
        f"{validation['admission_ts'].min().date()} "
        f"-> "
        f"{validation['admission_ts'].max().date()}",
        flush=True,
    )

    print(
        f"Test dates: "
        f"{test['admission_ts'].min().date()} "
        f"-> "
        f"{test['admission_ts'].max().date()}",
        flush=True,
    )

    X_train, y_train = prepare_xy(
        train,
        features,
    )

    X_valid, y_valid = prepare_xy(
        validation,
        features,
    )

    print()
    print(
        f"Train positive rate: "
        f"{y_train.mean():.4%}",
        flush=True,
    )

    print(
        f"Validation positive rate: "
        f"{y_valid.mean():.4%}",
        flush=True,
    )

    model_names = [
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "xgboost",
        "lightgbm",
        "catboost",
    ]

    dataset_results = []

    for model_name in model_names:
        result = run_model(
            dataset_name=dataset_name,
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            features=features,
        )

        dataset_results.append(
            result
        )

    validation_results = pd.DataFrame(
        dataset_results
    )

    validation_results = (
        validation_results
        .sort_values(
            "validation_pr_auc",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    tuning_results = validation_results[
        [
            "dataset",
            "model",
            "n_features",
            "cv_pr_auc",
            "elapsed_minutes",
            "best_params",
        ]
    ].copy()

    validation_results.to_csv(
        MODEL_DIR
        / f"{dataset_name}_validation_results.csv",
        index=False,
    )

    tuning_results.to_csv(
        MODEL_DIR
        / f"{dataset_name}_tuning_results.csv",
        index=False,
    )

    print()
    print(
        f"VALIDATION RESULTS - {dataset_name}",
        flush=True,
    )

    print(
        validation_results[
            [
                "model",
                "validation_roc_auc",
                "validation_pr_auc",
                "validation_f1",
                "validation_recall",
                "validation_brier",
            ]
        ].to_string(
            index=False
        ),
        flush=True,
    )

    return (
        tuning_results,
        validation_results,
    )


def main():
    print()
    print("=" * 80)
    print("TRAJECTORY EXPERIMENT")
    print("=" * 80)
    print(
        "A = Baseline",
        flush=True,
    )
    print(
        "B = Baseline + Trajectory",
        flush=True,
    )
    print(
        "C = Baseline + APS",
        flush=True,
    )
    print()
    print(
        "Methodology:",
        flush=True,
    )
    print(
        "- Same official temporal split",
        flush=True,
    )
    print(
        "- TimeSeriesSplit(n_splits=3)",
        flush=True,
    )
    print(
        "- RandomizedSearchCV",
        flush=True,
    )
    print(
        "- 10 iterations per model",
        flush=True,
    )
    print(
        "- Scoring: PR-AUC",
        flush=True,
    )
    print(
        "- Validation used for comparison",
        flush=True,
    )
    print(
        "- TEST is NOT evaluated in this stage",
        flush=True,
    )
    print("=" * 80)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_cutoff, validation_cutoff = (
        load_official_temporal_cutoffs()
    )

    all_tuning_results = []
    all_validation_results = []

    for dataset_name, filename in DATASETS.items():
        dataset_path = (
            DATASET_DIR / filename
        )

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found:\n"
                f"{dataset_path}"
            )

        (
            tuning_results,
            validation_results,
        ) = run_dataset_experiment(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            train_cutoff=train_cutoff,
            validation_cutoff=validation_cutoff,
        )

        all_tuning_results.append(
            tuning_results
        )

        all_validation_results.append(
            validation_results
        )

    tuning_comparison = pd.concat(
        all_tuning_results,
        ignore_index=True,
    )

    validation_comparison = pd.concat(
        all_validation_results,
        ignore_index=True,
    )

    tuning_comparison = (
        tuning_comparison
        .sort_values(
            "cv_pr_auc",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    validation_comparison = (
        validation_comparison
        .sort_values(
            "validation_pr_auc",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    tuning_comparison.to_csv(
        MODEL_DIR
        / "trajectory_tuning_comparison.csv",
        index=False,
    )

    validation_comparison.to_csv(
        MODEL_DIR
        / "trajectory_validation_comparison.csv",
        index=False,
    )

    print()
    print("=" * 80)
    print("FINAL VALIDATION COMPARISON")
    print("=" * 80)

    print(
        validation_comparison[
            [
                "dataset",
                "model",
                "n_features",
                "validation_roc_auc",
                "validation_pr_auc",
                "validation_f1",
                "validation_recall",
                "validation_brier",
            ]
        ].to_string(
            index=False
        ),
        flush=True,
    )

    print()
    print("=" * 80)
    print("BEST MODEL PER DATASET")
    print("=" * 80)

    best_per_dataset = (
        validation_comparison
        .sort_values(
            "validation_pr_auc",
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
                "validation_roc_auc",
                "validation_pr_auc",
                "validation_f1",
                "validation_brier",
            ]
        ].to_string(
            index=False
        ),
        flush=True,
    )

    print()
    print("=" * 80)
    print("FILES SAVED")
    print("=" * 80)

    print(
        MODEL_DIR
        / "trajectory_tuning_comparison.csv",
        flush=True,
    )

    print(
        MODEL_DIR
        / "trajectory_validation_comparison.csv",
        flush=True,
    )

    print()
    print("TRAJECTORY EXPERIMENT COMPLETED.", flush=True)


if __name__ == "__main__":
    main()