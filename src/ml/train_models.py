from pathlib import Path
import time

import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


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

    return ColumnTransformer(
        [
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
        ]
    )


def train_models(X_train, y_train):

    print("\nPreparing preprocessing...")

    preprocessor = build_preprocessor(X_train)

    print(
        f"Numerical variables: "
        f"{len(get_columns(X_train)[0])}"
    )

    print(
        f"Categorical variables: "
        f"{len(get_columns(X_train)[1])}"
    )

    models = {

        "random_forest": RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "xgboost": XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ),

        "lightgbm": LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=15,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
    }

    trained = {}

    for name, model in models.items():

        print(
            f"\n{'=' * 50}"
        )

        print(
            f"Training {name}..."
        )

        start = time.time()

        pipeline = Pipeline([
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ])

        pipeline.fit(
            X_train,
            y_train
        )

        elapsed = time.time() - start

        trained[name] = pipeline

        joblib.dump(
            pipeline,
            MODEL_DIR / f"{name}.joblib"
        )

        print(
            f"{name} completed "
            f"in {elapsed / 60:.2f} minutes."
        )

    return trained


def train_catboost(X_train, y_train):

    print(
        f"\n{'=' * 50}"
    )

    print(
        "Training catboost..."
    )

    start = time.time()

    X_train = X_train.copy()

    categorical = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    X_train[categorical] = (
        X_train[categorical]
        .fillna("missing")
        .astype(str)
    )

    model = CatBoostClassifier(
        iterations=100,
        depth=5,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=False,
        thread_count=-1
    )

    model.fit(
        X_train,
        y_train,
        cat_features=categorical
    )

    elapsed = time.time() - start

    joblib.dump(
        model,
        MODEL_DIR / "catboost.joblib"
    )

    print(
        f"catboost completed "
        f"in {elapsed / 60:.2f} minutes."
    )

    return model