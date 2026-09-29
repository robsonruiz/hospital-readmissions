from src.ml.prepare_data import (
    load_ml_data,
    prepare_data,
    print_split_summary,
)
from src.ml.train_models import train_models, train_catboost
from src.ml.evaluate_models import evaluate_models


def main():
    print("=" * 70)
    print("MACHINE LEARNING PIPELINE")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n1 - Loading ml_features...")
    df = load_ml_data()

    print(f"Loaded: {len(df):,} records")

    # ------------------------------------------------------------------
    # 2. Temporal split
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 3. Train models
    # ------------------------------------------------------------------
    print("\n3 - Training Random Forest, XGBoost and LightGBM...")

    trained_models = train_models(
        X_train,
        y_train,
    )

    print("\n4 - Training CatBoost...")

    trained_models["catboost"] = train_catboost(
        X_train,
        y_train,
    )

    # ------------------------------------------------------------------
    # 4. Evaluate on validation set
    # ------------------------------------------------------------------
    # IMPORTANT:
    # The validation set is used for model/threshold decisions.
    # The test set remains untouched until the final evaluation.
    print("\n5 - Evaluating models on VALIDATION set...")

    validation_results = evaluate_models(
        trained_models,
        X_valid,
        y_valid,
    )

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    print(
        validation_results
        .sort_values("pr_auc", ascending=False)
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    # 5. Test set remains untouched
    # ------------------------------------------------------------------
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
        f"Test positive rate: {y_test.mean() * 100:.2f}%"
    )

    print("\nPipeline completed.")

if __name__ == "__main__":
    main()
