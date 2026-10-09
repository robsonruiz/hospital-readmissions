from pathlib import Path

from src.ml.prepare_data import (
    load_ml_data,
    prepare_data,
    print_split_summary,
)
from src.ml.train_models import (
    train_models,
    train_catboost,
)
from src.ml.evaluate_models import evaluate_models
from src.ml.threshold_analysis import run_threshold_analysis


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"


def run_ml():
    print("\n" + "=" * 70)
    print("MACHINE LEARNING PIPELINE")
    print("=" * 70)

    # 1. Load data
    print("\n=== STEP 1 - Loading ML data ===")

    df = load_ml_data()

    print(f"Loaded: {len(df):,} records")

    # 2. Temporal split
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

    # 3. Train models
    print("\n=== STEP 3 - Training models ===")

    print("\nTraining Random Forest, XGBoost and LightGBM...")

    trained_models = train_models(
        X_train,
        y_train,
    )

    print("\nTraining CatBoost...")

    trained_models["catboost"] = train_catboost(
        X_train,
        y_train,
    )

    print("\nAll models trained successfully.")

    # 4. Validation
    print("\n=== STEP 4 - Evaluating models on validation set ===")

    validation_results = evaluate_models(
        trained_models,
        X_valid,
        y_valid,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_results_path = (
        MODELS_DIR / "validation_results.csv"
    )

    validation_results.to_csv(
        validation_results_path,
        index=False,
    )

    print(
        f"\nValidation results saved to:"
        f"\n  {validation_results_path}"
    )

    print("\nValidation results:")

    print(
        validation_results
        .sort_values(
            "pr_auc",
            ascending=False,
        )
        .to_string(index=False)
    )

    # 5. Threshold analysis
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

    # 6. Test set
    print("\n=== STEP 6 - Test set ===")

    print(
        "The test set was NOT used for model selection, "
        "threshold selection or hyperparameter decisions."
    )

    print(f"Test records: {len(X_test):,}")

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