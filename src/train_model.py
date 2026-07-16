import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "src" / "data" / "port_data.csv"
MODEL_DIR = PROJECT_ROOT / "models"

RANDOM_SEED = 42
N_SPLITS = 5

LEAKAGE_COLUMNS = ["delay_hours", "is_delayed"]


# load .csv file
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


# load features X and labels y
def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    encoded = pd.get_dummies(df, columns=["vessel_type", "weather"])
    y = encoded["is_delayed"]
    X = encoded.drop(columns=LEAKAGE_COLUMNS)
    return X, y


# 5-fold cross validation
def cross_validate(X: pd.DataFrame, y: pd.Series) -> dict:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    oof_proba = np.zeros(len(X))
    fold_metrics = []

    for fold_index, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestClassifier(random_state=RANDOM_SEED)
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        oof_proba[test_idx] = y_proba

        fold_metrics.append(
            {
                "fold": fold_index,
                "roc_auc": float(roc_auc_score(y_test, y_proba)),
                "f1": float(f1_score(y_test, y_pred)),
            }
        )

    oof_pred = (oof_proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, oof_pred).ravel()

    return {
        "folds": fold_metrics,
        "mean_roc_auc": float(np.mean([m["roc_auc"] for m in fold_metrics])),
        "mean_f1": float(np.mean([m["f1"] for m in fold_metrics])),
        "out_of_fold_roc_auc": float(roc_auc_score(y, oof_proba)),
        "out_of_fold_f1": float(f1_score(y, oof_pred)),
        "out_of_fold_confusion_matrix": {
            "threshold": 0.5,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


# train final model on all data
def fit_final_model(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    model = RandomForestClassifier(random_state=RANDOM_SEED)
    model.fit(X, y)
    return model


# save model + metric + schema
def save_artifacts(model: RandomForestClassifier, X: pd.DataFrame, cv_results: dict) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_DIR / "model.joblib")

    schema = {"feature_columns": list(X.columns)}
    (MODEL_DIR / "feature_schema.json").write_text(json.dumps(schema, indent=2))

    (MODEL_DIR / "metrics.json").write_text(json.dumps(cv_results, indent=2))


def print_summary(cv_results: dict) -> None:
    print("=" * 60)
    print("Model Training Summary (5-Fold CV)")
    print("=" * 60)

    for fold in cv_results["folds"]:
        print(f"Fold {fold['fold']}: ROC-AUC={fold['roc_auc']:.3f}  F1={fold['f1']:.3f}")

    print(f"\nMean ROC-AUC: {cv_results['mean_roc_auc']:.3f}")
    print(f"Mean F1:      {cv_results['mean_f1']:.3f}")

    print("\nOut-of-fold metrics:")
    print(f"ROC-AUC: {cv_results['out_of_fold_roc_auc']:.3f}")
    print(f"F1:      {cv_results['out_of_fold_f1']:.3f}")

    cm = cv_results["out_of_fold_confusion_matrix"]
    print(f"\nConfusion matrix @ threshold={cm['threshold']}:")
    print(f"  TN={cm['tn']}  FP={cm['fp']}")
    print(f"  FN={cm['fn']}  TP={cm['tp']}")


def main() -> None:
    df = load_data()
    X, y = build_features(df)

    cv_results = cross_validate(X, y)
    final_model = fit_final_model(X, y)

    save_artifacts(final_model, X, cv_results)
    print_summary(cv_results)


if __name__ == "__main__":
    main()
