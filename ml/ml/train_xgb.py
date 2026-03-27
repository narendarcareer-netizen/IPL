import os

import mlflow
import mlflow.xgboost
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from xgboost import XGBClassifier

from ml.feature_config import FEATURE_COLUMNS, HISTORICAL_OUTPUT_PATH


def main():
    if not os.path.exists(HISTORICAL_OUTPUT_PATH):
        raise FileNotFoundError(f"Features file not found: {HISTORICAL_OUTPUT_PATH}")

    df = pd.read_csv(HISTORICAL_OUTPUT_PATH)

    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[FEATURE_COLUMNS].fillna(0.0)
    y = df["team1_won"].astype(int)

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.025,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_weight=4,
        reg_lambda=1.5,
        gamma=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    ll = log_loss(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    acc = accuracy_score(y_test, preds)

    print(f"Log loss: {ll:.4f}")
    print(f"Brier score: {brier:.4f}")
    print(f"Accuracy: {acc:.4f}")

    mlflow.set_experiment("ipl_prediction")
    with mlflow.start_run(run_name="xgb_prematch_richer_features_v1"):
        mlflow.log_params({
            "model_type": "xgboost",
            "feature_count": len(FEATURE_COLUMNS),
            "n_estimators": 500,
            "max_depth": 3,
            "learning_rate": 0.025,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "min_child_weight": 4,
            "reg_lambda": 1.5,
            "gamma": 0.1,
        })
        mlflow.log_metrics({
            "log_loss": ll,
            "brier_score": brier,
            "accuracy": acc,
        })
        mlflow.xgboost.log_model(model, artifact_path="model")

    print("XGBoost model logged to MLflow")


if __name__ == "__main__":
    main()
