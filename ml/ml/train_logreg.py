import os
from pathlib import Path
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from sklearn.model_selection import train_test_split

FEATURES_PATH = "/app/ml/artifacts/pre_match_features.csv"
MODEL_PATH = "/app/ml/artifacts/logreg_prematch.joblib"


def main():
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Features file not found: {FEATURES_PATH}")

    df = pd.read_csv(FEATURES_PATH)

    print("CSV columns:")
    print(df.columns.tolist())

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

    for bad_col in [
        "winner_team_id",
        "team1_won",
        "match_id",
        "start_time_utc",
        "toss_winner_team_id",
        "toss_decision",
    ]:
        if bad_col in feature_cols:
            raise ValueError(f"Leakage column found in feature_cols: {bad_col}")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    print("\nTraining sample:")
    print(df[feature_cols + ["team1_won"]].head(10).to_string(index=False))

    X = df[feature_cols].fillna(0.0)
    y = df["team1_won"].astype(int)

    suspicious = []
    for col in feature_cols:
        try:
            corr = abs(pd.Series(X[col]).corr(y))
            if pd.notna(corr) and corr > 0.98:
                suspicious.append((col, corr))
        except Exception:
            pass

    if suspicious:
        raise ValueError(f"Potential leakage: suspiciously high target correlation: {suspicious}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    ll = log_loss(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    acc = accuracy_score(y_test, preds)

    print(f"\nLog loss: {ll:.4f}")
    print(f"Brier score: {brier:.4f}")
    print(f"Accuracy: {acc:.4f}")

    Path("/app/ml/artifacts").mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "metrics": {
                "log_loss": ll,
                "brier_score": brier,
                "accuracy": acc,
            },
        },
        MODEL_PATH,
    )
    print(f"Model saved to {MODEL_PATH}")

    mlflow.set_experiment("ipl_prediction")
    with mlflow.start_run(run_name="logreg_prematch_leakcheck_v2"):
        mlflow.log_params({
            "model_type": "logistic_regression",
            "feature_count": len(feature_cols),
        })
        mlflow.log_metrics({
            "log_loss": ll,
            "brier_score": brier,
            "accuracy": acc,
        })
        mlflow.sklearn.log_model(model, artifact_path="model")

    print("Model logged to MLflow")


if __name__ == "__main__":
    main()