from pathlib import Path
from src.ml.prepare_data import load_ml_data, prepare_data, print_split_summary
from src.ml.train_models import train_models, train_catboost
from src.ml.evaluate_models import evaluate_models

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

def main():
    df = load_ml_data()
    (X_train, X_valid, X_test, y_train, y_valid, y_test,
     train_df, validation_df, test_df, train_cutoff, validation_cutoff) = prepare_data(df)

    print_split_summary(train_df, validation_df, test_df, train_cutoff, validation_cutoff)
    print(f"\nFeature shapes: train={X_train.shape}, valid={X_valid.shape}, test={X_test.shape}")

    models = train_models(X_train, y_train)
    models["catboost"] = train_catboost(X_train, y_train)

    results = evaluate_models(models, X_valid, y_valid)
    results.to_csv(MODEL_DIR / "validation_results.csv", index=False)

    print("\nVALIDATION RESULTS")
    print(results.sort_values("pr_auc", ascending=False).to_string(index=False))
    print(f"\nSaved: {MODEL_DIR / 'validation_results.csv'}")
    print("\nTest set was NOT used for model/threshold decisions.")

if __name__ == "__main__":
    main()
