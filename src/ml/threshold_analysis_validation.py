from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from catboost import CatBoostClassifier

OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)
THRESHOLDS = np.arange(0.05, 0.96, 0.05)

def prepare_catboost_data(X, model):
    X = X.copy()
    for idx in model.get_cat_feature_indices():
        col = X.columns[idx]
        X[col] = X[col].fillna("missing").astype(str)
    return X

def get_probabilities(model, X):
    if isinstance(model, CatBoostClassifier):
        X = prepare_catboost_data(X, model)
    return model.predict_proba(X)[:, 1]

def evaluate_thresholds(model, X_valid, y_valid, model_name):
    probabilities = get_probabilities(model, X_valid)
    rows = []
    for threshold in THRESHOLDS:
        pred = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_valid, pred, labels=[0, 1]).ravel()
        rows.append({
            "model": model_name,
            "threshold": round(float(threshold), 2),
            "precision": precision_score(y_valid, pred, zero_division=0),
            "recall": recall_score(y_valid, pred, zero_division=0),
            "f1": f1_score(y_valid, pred, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
        })
    return pd.DataFrame(rows)

def run_threshold_analysis(trained_models, X_valid, y_valid):
    results = pd.concat(
        [evaluate_thresholds(model, X_valid, y_valid, name)
         for name, model in trained_models.items()],
        ignore_index=True,
    )
    results.to_csv(OUTPUT_DIR / "threshold_analysis_validation.csv", index=False)

    best = (
        results.sort_values(["model", "f1"], ascending=[True, False])
        .groupby("model", as_index=False).first()
    )
    best.to_csv(OUTPUT_DIR / "best_thresholds_validation.csv", index=False)
    print(best[["model","threshold","precision","recall","f1","specificity"]].to_string(index=False))
    return results, best

def main():
    from src.ml.prepare_data import load_ml_data, prepare_data
    from src.ml.train_models import train_models, train_catboost

    df = load_ml_data()
    X_train, X_valid, X_test, y_train, y_valid, y_test, *_ = prepare_data(df)

    models = train_models(X_train, y_train)
    models["catboost"] = train_catboost(X_train, y_train)

    run_threshold_analysis(models, X_valid, y_valid)
    print("\nTest set was not used.")

if __name__ == "__main__":
    main()
