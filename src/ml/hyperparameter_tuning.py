from pathlib import Path
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
CV_SPLITS = 3
N_ITER = 10


def get_columns(X):
    categorical = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical = X.select_dtypes(
        exclude=["object", "category"]
    ).columns.tolist()

    return numerical, categorical


def build_preprocessor(X):
    numerical, categorical = get_columns(X)

    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ])

    return ColumnTransformer([
        (
            "num",
            numerical_pipeline,
            numerical
        ),
        (
            "cat",
            categorical_pipeline,
            categorical
        )
    ])


def build_search_spaces():
    return {
        "logistic_regression": {
            "model": LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ),
            "params": {
                "model__C": np.logspace(-3, 1, 8),
                "model__solver": [
                    "liblinear",
                    "lbfgs"
                ]
            }
        },
        "decision_tree": {
            "model": DecisionTreeClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE
            ),
            "params": {
                "model__max_depth": [
                    3,
                    5,
                    8,
                    12,
                    16,
                    None
                ],
                "model__min_samples_leaf": [
                    2,
                    5,
                    10,
                    20,
                    30
                ],
                "model__min_samples_split": [
                    2,
                    5,
                    10,
                    20
                ],
                "model__criterion": [
                    "gini",
                    "entropy"
                ]
            }
        },
        "random_forest": {
            "model": RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            "params": {
                "model__n_estimators": [
                    100,
                    200,
                    300
                ],
                "model__max_depth": [
                    5,
                    8,
                    12,
                    16,
                    None
                ],
                "model__min_samples_leaf": [
                    2,
                    5,
                    10,
                    20
                ],
                "model__max_features": [
                    "sqrt",
                    "log2",
                    0.5
                ]
            }
        },
        "xgboost": {
            "model": XGBClassifier(
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            "params": {
                "model__n_estimators": [
                    100,
                    200,
                    300
                ],
                "model__max_depth": [
                    3,
                    4,
                    5,
                    6
                ],
                "model__learning_rate": [
                    0.01,
                    0.03,
                    0.05,
                    0.1
                ],
                "model__subsample": [
                    0.7,
                    0.8,
                    0.9,
                    1.0
                ],
                "model__colsample_bytree": [
                    0.7,
                    0.8,
                    0.9,
                    1.0
                ],
                "model__min_child_weight": [
                    1,
                    3,
                    5,
                    10
                ]
            }
        },
        "lightgbm": {
            "model": LGBMClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=-1
            ),
            "params": {
                "model__n_estimators": [
                    100,
                    200,
                    300
                ],
                "model__learning_rate": [
                    0.01,
                    0.03,
                    0.05,
                    0.1
                ],
                "model__num_leaves": [
                    7,
                    15,
                    31,
                    63
                ],
                "model__max_depth": [
                    -1,
                    5,
                    8,
                    12
                ],
                "model__min_child_samples": [
                    10,
                    20,
                    30,
                    50
                ],
                "model__subsample": [
                    0.7,
                    0.8,
                    0.9,
                    1.0
                ]
            }
        }
    }


def prepare_catboost_data(X):
    X = X.copy()

    categorical = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    for column in categorical:
        X[column] = (
            X[column]
            .fillna("missing")
            .astype(str)
        )

    return X, categorical


def build_catboost_search():
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=RANDOM_STATE,
        verbose=False,
        thread_count=-1
    )

    params = {
        "iterations": [
            100,
            200,
            300
        ],
        "depth": [
            4,
            5,
            6,
            7,
            8
        ],
        "learning_rate": [
            0.01,
            0.03,
            0.05,
            0.1
        ],
        "l2_leaf_reg": [
            1,
            3,
            5,
            10
        ],
        "random_strength": [
            0,
            0.5,
            1,
            2
        ]
    }

    return model, params


def tune_sklearn_models(X_train, y_train):
    search_spaces = build_search_spaces()

    cv = TimeSeriesSplit(
        n_splits=CV_SPLITS
    )

    results = {}
    best_models = {}

    for name, config in search_spaces.items():
        print(f"\n{'=' * 70}")
        print(f"Optimizing {name}...")

        start = time.time()

        pipeline = Pipeline([
            (
                "preprocessor",
                build_preprocessor(X_train)
            ),
            (
                "model",
                config["model"]
            )
        ])

        search = RandomizedSearchCV(
            pipeline,
            config["params"],
            n_iter=N_ITER,
            scoring="average_precision",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=1,
            refit=True,
            verbose=1
        )

        search.fit(
            X_train,
            y_train
        )

        elapsed = time.time() - start

        best_models[name] = (
            search.best_estimator_
        )

        joblib.dump(
            search.best_estimator_,
            MODEL_DIR / f"tuned_{name}.joblib"
        )

        results[name] = {
            "model": name,
            "best_cv_pr_auc": search.best_score_,
            "best_params": json.dumps(
                search.best_params_,
                default=str
            ),
            "elapsed_minutes": elapsed / 60
        }

        print(
            f"Best CV PR-AUC: "
            f"{search.best_score_:.4f}"
        )

        print(
            f"Time: "
            f"{elapsed / 60:.2f} minutes"
        )

        print(
            f"Best parameters: "
            f"{search.best_params_}"
        )

    return best_models, results


def tune_catboost(X_train, y_train):
    print(f"\n{'=' * 70}")
    print("Optimizing catboost...")

    start = time.time()

    X_train_cb, categorical = (
        prepare_catboost_data(X_train)
    )

    model, params = (
        build_catboost_search()
    )

    cv = TimeSeriesSplit(
        n_splits=CV_SPLITS
    )

    search = RandomizedSearchCV(
        model,
        params,
        n_iter=N_ITER,
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
        verbose=1
    )

    search.fit(
        X_train_cb,
        y_train,
        cat_features=categorical
    )

    elapsed = time.time() - start

    joblib.dump(
        search.best_estimator_,
        MODEL_DIR / "tuned_catboost.joblib"
    )

    result = {
        "model": "catboost",
        "best_cv_pr_auc": search.best_score_,
        "best_params": json.dumps(
            search.best_params_,
            default=str
        ),
        "elapsed_minutes": elapsed / 60
    }

    print(
        f"Best CV PR-AUC: "
        f"{search.best_score_:.4f}"
    )

    print(
        f"Time: "
        f"{elapsed / 60:.2f} minutes"
    )

    print(
        f"Best parameters: "
        f"{search.best_params_}"
    )

    return (
        search.best_estimator_,
        result
    )


def evaluate_tuned_models(
    models,
    X_valid,
    y_valid
):
    rows = []

    for name, model in models.items():
        X_eval = X_valid

        if name == "catboost":
            X_eval, _ = (
                prepare_catboost_data(
                    X_valid
                )
            )

        probabilities = (
            model
            .predict_proba(X_eval)[:, 1]
        )

        rows.append({
            "model": name,
            "roc_auc": roc_auc_score(
                y_valid,
                probabilities
            ),
            "pr_auc": average_precision_score(
                y_valid,
                probabilities
            )
        })

    return pd.DataFrame(
        rows
    ).sort_values(
        "pr_auc",
        ascending=False
    )


def run_hyperparameter_tuning(
    X_train,
    y_train,
    X_valid=None,
    y_valid=None
):
    print(
        "\n" + "=" * 70
    )

    print(
        "HYPERPARAMETER TUNING"
    )

    print(
        "=" * 70
    )

    print(
        f"\nCV strategy: "
        f"TimeSeriesSplit ({CV_SPLITS} splits)"
    )

    print(
        f"RandomizedSearch iterations "
        f"per model: {N_ITER}"
    )

    print(
        "Optimization metric: PR-AUC"
    )

    print(
        "Validation and test sets "
        "remain untouched during tuning."
    )

    tuned_models, results = (
        tune_sklearn_models(
            X_train,
            y_train
        )
    )

    catboost_model, catboost_result = (
        tune_catboost(
            X_train,
            y_train
        )
    )

    tuned_models["catboost"] = (
        catboost_model
    )

    results["catboost"] = (
        catboost_result
    )

    results_df = pd.DataFrame(
        list(results.values())
    ).sort_values(
        "best_cv_pr_auc",
        ascending=False
    )

    results_df.to_csv(
        MODEL_DIR /
        "hyperparameter_tuning_results.csv",
        index=False
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "BEST CV RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        results_df[
            [
                "model",
                "best_cv_pr_auc",
                "elapsed_minutes"
            ]
        ].to_string(
            index=False
        )
    )

    if (
        X_valid is not None
        and y_valid is not None
    ):
        validation_results = (
            evaluate_tuned_models(
                tuned_models,
                X_valid,
                y_valid
            )
        )

        validation_results.to_csv(
            MODEL_DIR /
            "validation_results_tuned.csv",
            index=False
        )

        print(
            "\n" + "=" * 70
        )

        print(
            "VALIDATION RESULTS - "
            "TUNED MODELS"
        )

        print(
            "=" * 70
        )

        print(
            validation_results.to_string(
                index=False
            )
        )

    return {
        "models": tuned_models,
        "tuning_results": results_df
    }


if __name__ == "__main__":
    from src.ml.prepare_data import (
        load_ml_data,
        prepare_data
    )

    df = load_ml_data()

    prepared = prepare_data(df)

    X_train = prepared[0]
    X_valid = prepared[1]
    X_test = prepared[2]
    y_train = prepared[3]
    y_valid = prepared[4]
    y_test = prepared[5]

    run_hyperparameter_tuning(
        X_train,
        y_train,
        X_valid,
        y_valid
    )