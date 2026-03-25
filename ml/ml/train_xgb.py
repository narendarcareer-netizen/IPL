import os
import pandas as pd
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score

FEATURES_PATH = "/app/ml/artifacts/pre_match_features.csv"


def main():
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Features file not found: {FEATURES_PATH}")

    df = pd.read_csv(FEATURES_PATH)

    feature_cols = [
    "elo_diff",
    "batting_strength_diff",
    "bowling_strength_diff",
    "all_rounder_balance_diff",
    "spin_strength_diff",
    "pace_strength_diff",
    "death_overs_strength_diff",
    "venue_win_bias_diff",
    "probable_xi_count_diff",
    "probable_xi_batting_form_diff",
    "probable_xi_bowling_form_diff",
    "top_order_strength_diff",
    "middle_order_strength_diff",
    "death_bowling_strength_diff",
]

    X = df[feature_cols].fillna(0.0)
    y = df["team1_won"].astype(int)

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
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

    with mlflow.start_run(run_name="xgb_prematch_baseline"):
        mlflow.log_params({
            "model_type": "xgboost",
            "feature_count": len(feature_cols),
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
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