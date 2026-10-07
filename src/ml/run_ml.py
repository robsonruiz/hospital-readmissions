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


MODELS_DIR = Path("models")


def run_ml():

    print("=" * 70)
    print("MACHINE LEARNING PIPELINE")
    print("=" * 70)

    # ==============================================================
    # 1. LOAD DATA
    # ==============================================================

    print("\n1 - Loading ml_features...")

    df = load_ml_data()

    print(f"Loaded: {len(df):,} records")

    # ==============================================================
    # 2. TEMPORAL SPLIT
    # ==============================================================

    print("\n2 - Creating temporal train/validation/test split...")

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

    # ==============================================================
    # 3. TRAIN MODELS
    # ==============================================================

    print(
        "\n3 - Training Random Forest, XGBoost and LightGBM..."
    )

    trained_models = train_models(
        X_train,
        y_train,
    )

    print("\n4 - Training CatBoost...")

    trained_models["catboost"] = train_catboost(
        X_train,
        y_train,
    )

    # ==============================================================
    # 4. VALIDATION
    # ==============================================================

    print(
        "\n5 - Evaluating models on VALIDATION set..."
    )

    validation_results = evaluate_models(
        trained_models,
        X_valid,
        y_valid,
    )

    # ==============================================================
    # 5. SAVE VALIDATION RESULTS
    # ==============================================================

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

    # ==============================================================
    # 6. DISPLAY VALIDATION RESULTS
    # ==============================================================

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    print(
        validation_results
        .sort_values(
            "pr_auc",
            ascending=False,
        )
        .to_string(index=False)
    )

    # ==============================================================
    # 7. THRESHOLD ANALYSIS
    # ==============================================================

    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS")
    print("=" * 70)

    print(
        "\nRunning threshold analysis using "
        "the VALIDATION set only..."
    )

    threshold_results = run_threshold_analysis(
        trained_models,
        X_valid,
        y_valid,
    )

    print(
        "\nThreshold analysis completed."
    )

    # ==============================================================
    # 8. TEST SET REMAINS UNTOUCHED
    # ==============================================================

    print("\n" + "=" * 70)
    print("TEST SET")
    print("=" * 70)

    print(
        "The test set was NOT used for model selection, "
        "threshold selection or hyperparameter decisions."
    )

    print(
        f"Test records: {len(X_test):,}"
    )

    print(
        f"Test positive rate: "
        f"{y_test.mean() * 100:.2f}%"
    )

    # ==============================================================
    # 9. FINISHED
    # ==============================================================

    print("\n" + "=" * 70)
    print("MACHINE LEARNING PIPELINE COMPLETED")
    print("=" * 70)

    print("\nGenerated files:")

    print(
        "  models/validation_results.csv"
    )

    print(
        "  models/threshold_analysis_validation.csv"
    )

    print(
        "  models/best_thresholds_validation.csv"
    )

    print(
        "\nThe test set remains untouched "
        "for the final evaluation."
    )

    # ==============================================================
    # RETURN RESULTS
    # ==============================================================

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


def main():

    run_ml()


if __name__ == "__main__":

    main()