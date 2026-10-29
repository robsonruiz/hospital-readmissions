from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.model_selection import TimeSeriesSplit

from catboost import CatBoostClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODELS_DIR = PROJECT_ROOT / "models"


MODEL_FILES = {
    "catboost": "tuned_catboost.joblib",
    "xgboost": "tuned_xgboost.joblib",
}


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


def prepare_model_data(X, model):
    if isinstance(model, CatBoostClassifier):
        return prepare_catboost_data(
            X,
            model
        )

    return X.copy()


def load_tuned_models():
    models = {}

    for name, filename in MODEL_FILES.items():

        path = MODELS_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Model not found:\n{path}"
            )

        print(
            f"Loading {name}..."
        )

        models[name] = joblib.load(
            path
        )

    return models


def fit_base_model(
    model,
    X,
    y,
):
    X = prepare_model_data(
        X,
        model
    )

    fitted_model = clone(model)

    if isinstance(
        model,
        CatBoostClassifier
    ):
        categorical_features = (
            model.get_cat_feature_indices()
        )

        fitted_model.fit(
            X,
            y,
            cat_features=categorical_features,
        )

    else:
        fitted_model.fit(
            X,
            y,
        )

    return fitted_model


def get_probability(
    model,
    X,
):
    X = prepare_model_data(
        X,
        model
    )

    return model.predict_proba(
        X
    )[:, 1]


def generate_oof_probabilities(
    model,
    X_train,
    y_train,
    n_splits=3,
):
    print(
        "\nGenerating temporal OOF probabilities..."
    )

    tscv = TimeSeriesSplit(
        n_splits=n_splits
    )

    oof_probability = np.full(
        len(y_train),
        np.nan,
        dtype=float,
    )

    for fold, (
        train_idx,
        calibration_idx,
    ) in enumerate(
        tscv.split(X_train),
        start=1,
    ):

        print(
            f"  Fold {fold}/{n_splits}"
        )

        X_fold_train = X_train.iloc[
            train_idx
        ]

        y_fold_train = y_train.iloc[
            train_idx
        ]

        X_fold_calibration = X_train.iloc[
            calibration_idx
        ]

        fitted_model = fit_base_model(
            model,
            X_fold_train,
            y_fold_train,
        )

        probability = get_probability(
            fitted_model,
            X_fold_calibration,
        )

        oof_probability[
            calibration_idx
        ] = probability

    valid_mask = ~np.isnan(
        oof_probability
    )

    return (
        oof_probability,
        valid_mask,
    )


def fit_calibrator(
    probability,
    y,
    method,
):
    probability = np.asarray(
        probability,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=int,
    )

    if method == "sigmoid":

        calibrator = LogisticRegression(
            max_iter=1000,
            random_state=42,
        )

        calibrator.fit(
            probability.reshape(-1, 1),
            y,
        )

        return calibrator

    if method == "isotonic":

        calibrator = IsotonicRegression(
            out_of_bounds="clip",
        )

        calibrator.fit(
            probability,
            y,
        )

        return calibrator

    raise ValueError(
        f"Unknown calibration method: {method}"
    )


def apply_calibrator(
    calibrator,
    probability,
    method,
):
    probability = np.asarray(
        probability,
        dtype=float,
    )

    if method == "sigmoid":

        return calibrator.predict_proba(
            probability.reshape(-1, 1)
        )[:, 1]

    if method == "isotonic":

        return calibrator.predict(
            probability
        )

    raise ValueError(
        f"Unknown calibration method: {method}"
    )


class CalibratedModel:
    def __init__(
        self,
        base_model,
        calibrator,
        method,
    ):
        self.base_model = base_model
        self.calibrator = calibrator
        self.method = method

    def predict_proba(
        self,
        X,
    ):
        probability = get_probability(
            self.base_model,
            X,
        )

        calibrated_probability = (
            apply_calibrator(
                self.calibrator,
                probability,
                self.method,
            )
        )

        return np.column_stack(
            [
                1 - calibrated_probability,
                calibrated_probability,
            ]
        )

    def predict(
        self,
        X,
    ):
        probability = self.predict_proba(
            X
        )[:, 1]

        return (
            probability >= 0.5
        ).astype(int)


def calibrate_model(
    model,
    X_train,
    y_train,
    method,
):
    print(
        f"\nCalibration method: {method}"
    )

    oof_probability, valid_mask = (
        generate_oof_probabilities(
            model=model,
            X_train=X_train,
            y_train=y_train,
            n_splits=3,
        )
    )

    calibration_probability = (
        oof_probability[valid_mask]
    )

    calibration_y = (
        y_train.iloc[valid_mask]
    )

    print(
        f"Calibration samples: "
        f"{len(calibration_y):,}"
    )

    calibrator = fit_calibrator(
        probability=calibration_probability,
        y=calibration_y,
        method=method,
    )

    print(
        "\nFitting final base model "
        "on complete training set..."
    )

    final_base_model = fit_base_model(
        model,
        X_train,
        y_train,
    )

    calibrated_model = CalibratedModel(
        base_model=final_base_model,
        calibrator=calibrator,
        method=method,
    )

    return calibrated_model


def evaluate_model(
    model,
    X_valid,
    y_valid,
    name,
):
    probability = model.predict_proba(
        X_valid
    )[:, 1]

    result = {
        "model": name,
        "roc_auc": roc_auc_score(
            y_valid,
            probability,
        ),
        "pr_auc": average_precision_score(
            y_valid,
            probability,
        ),
        "brier_score": brier_score_loss(
            y_valid,
            probability,
        ),
    }

    return (
        result,
        probability,
    )


def save_calibration_curve(
    y_valid,
    probability,
    model_name,
    method,
):
    fraction_positive, mean_predicted = (
        calibration_curve(
            y_valid,
            probability,
            n_bins=10,
            strategy="quantile",
        )
    )

    curve = pd.DataFrame(
        {
            "model": model_name,
            "method": method,
            "mean_predicted_probability": (
                mean_predicted
            ),
            "fraction_positive": (
                fraction_positive
            ),
        }
    )

    output_path = (
        MODELS_DIR
        / f"calibration_curve_{model_name}_{method}.csv"
    )

    curve.to_csv(
        output_path,
        index=False,
    )

    return curve


def run_calibration_analysis(
    X_train,
    y_train,
    X_valid,
    y_valid,
):
    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    models = load_tuned_models()

    results = []
    curves = []

    for model_name, model in models.items():

        print(
            "\n" + "=" * 70
        )

        print(
            f"CALIBRATION - {model_name.upper()}"
        )

        print(
            "=" * 70
        )

        for method in [
            "sigmoid",
            "isotonic",
        ]:

            print(
                f"\n--- {method.upper()} ---"
            )

            calibrated_model = calibrate_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                method=method,
            )

            result, probability = (
                evaluate_model(
                    calibrated_model,
                    X_valid,
                    y_valid,
                    f"{model_name}_{method}",
                )
            )

            results.append(
                result
            )

            curve = save_calibration_curve(
                y_valid=y_valid,
                probability=probability,
                model_name=model_name,
                method=method,
            )

            curves.append(
                curve
            )

            model_path = (
                MODELS_DIR
                / f"calibrated_{model_name}_{method}.joblib"
            )

            joblib.dump(
                calibrated_model,
                model_path,
            )

            print(
                f"ROC-AUC: "
                f"{result['roc_auc']:.3f}"
            )

            print(
                f"PR-AUC: "
                f"{result['pr_auc']:.3f}"
            )

            print(
                f"Brier Score: "
                f"{result['brier_score']:.3f}"
            )

            print(
                f"Saved: {model_path}"
            )

    results_df = (
        pd.DataFrame(results)
        .sort_values(
            "brier_score",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    results_path = (
        MODELS_DIR
        / "calibration_results_validation.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    curves_df = pd.concat(
        curves,
        ignore_index=True,
    )

    curves_path = (
        MODELS_DIR
        / "calibration_curves_validation.csv"
    )

    curves_df.to_csv(
        curves_path,
        index=False,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "CALIBRATION RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        f"\nResults saved to:"
        f"\n  {results_path}"
    )

    print(
        f"\nCurves saved to:"
        f"\n  {curves_path}"
    )

    return (
        results_df,
        curves_df,
    )