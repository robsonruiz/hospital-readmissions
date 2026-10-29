from pathlib import Path
import sys
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.prepare_data import (
    load_ml_data,
    prepare_data,
    print_split_summary,
)
from src.ml.evaluate_models import evaluate_models
from src.ml.threshold_analysis import run_threshold_analysis


MODELS_DIR = PROJECT_ROOT / "models"


TUNED_MODEL_FILES = {
    "logistic_regression": "tuned_logistic_regression.joblib",
    "decision_tree": "tuned_decision_tree.joblib",
    "random_forest": "tuned_random_forest.joblib",
    "xgboost": "tuned_xgboost.joblib",
    "lightgbm": "tuned_lightgbm.joblib",
    "catboost": "tuned_catboost.joblib",
}


def load_tuned_models():
    trained_models = {}

    print("\nLoading tuned models...")

    for name, filename in TUNED_MODEL_FILES.items():
        model_path = MODELS_DIR / filename

        if not model_path.exists():
            raise FileNotFoundError(
                f"Tuned model not found:\n  {model_path}"
            )

        print(f"  Loading {name}...")

        trained_models[name] = joblib.load(
            model_path
        )

    print("\nAll tuned models loaded successfully.")

    return trained_models


def run_ml():
    print("\n" + "=" * 70)
    print("MACHINE LEARNING PIPELINE")
    print("=" * 70)

    print("\n=== STEP 1 - Loading ML data ===")

    df = load_ml_data()

    print(f"Loaded: {len(df):,} records")

    print("\n=== STEP 2 - Creating temporal split ===")

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
    ) = prepare_data(df)

    print_split_summary(
        train_df,
        validation_df,
        test_df,
        train_cutoff,
        validation_cutoff,
    )

    print("\nFeature shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_valid: {X_valid.shape}")
    print(f"  X_test:  {X_test.shape}")

    print("\n=== STEP 3 - Loading tuned models ===")

    trained_models = load_tuned_models()

    print("\n=== STEP 4 - Evaluating tuned models on validation set ===")

    validation_results = evaluate_models(
        trained_models,
        X_valid,
        y_valid,
        threshold=0.5,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_results_path = (
        MODELS_DIR / "validation_results_tuned_full.csv"
    )

    validation_results.to_csv(
        validation_results_path,
        index=False,
    )

    print(
        f"\nFull validation results saved to:"
        f"\n  {validation_results_path}"
    )

    print("\nValidation results - tuned models:")

    print(
        validation_results
        .sort_values(
            "pr_auc",
            ascending=False,
        )
        .to_string(index=False)
    )

    print("\n=== STEP 5 - Threshold analysis ===")

    print(
        "\nRunning threshold analysis using "
        "the validation set only..."
    )

    threshold_results = run_threshold_analysis(
        trained_models,
        X_valid,
        y_valid,
    )

    print("\nThreshold analysis completed.")

    print("\nGenerated threshold files:")
    print(
        "  models/threshold_analysis_validation.csv"
    )
    print(
        "  models/best_thresholds_validation.csv"
    )

    print("\n=== STEP 6 - Test set ===")

    print(
        "The test set remains completely untouched."
    )

    print(
        "It was NOT used for:"
        "\n  - hyperparameter tuning"
        "\n  - model selection"
        "\n  - threshold selection"
        "\n  - model comparison"
    )

    print(f"\nTest records: {len(X_test):,}")

    print(
        f"Test positive rate: "
        f"{y_test.mean() * 100:.2f}%"
    )

    print("\n" + "=" * 70)
    print("MACHINE LEARNING PIPELINE COMPLETED")
    print("=" * 70)

    return {
        "trained_models": trained_models,
        "validation_results": validation_results,
        "threshold_results": threshold_results,
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test,
        "train_df": train_df,
        "validation_df": validation_df,
        "test_df": test_df,
        "train_cutoff": train_cutoff,
        "validation_cutoff": validation_cutoff,
    }


if __name__ == "__main__":
    run_ml()