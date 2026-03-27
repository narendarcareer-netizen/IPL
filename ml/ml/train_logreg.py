import os
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.feature_config import (
    STAGES,
    feature_columns_for_stage,
    historical_output_for_stage,
    model_artifact_for_stage,
    model_name_for_stage,
    normalize_stage,
)

DATABASE_URL = os.environ["DATABASE_URL"]


def register_model_version(
    engine,
    model_name: str,
    stage: str,
    run_id: str | None,
    artifact_uri: str,
    metrics: dict,
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO model_versions (
                    model_name,
                    stage,
                    created_at_utc,
                    mlflow_run_id,
                    artifact_uri,
                    metrics_json
                )
                VALUES (
                    :model_name,
                    :stage,
                    NOW(),
                    :mlflow_run_id,
                    :artifact_uri,
                    CAST(:metrics_json AS jsonb)
                )
            """),
            {
                "model_name": model_name,
                "stage": stage,
                "mlflow_run_id": run_id,
                "artifact_uri": artifact_uri,
                "metrics_json": json.dumps(metrics),
            },
        )


def train_stage(engine, stage: str) -> None:
    stage = normalize_stage(stage)
    features_path = historical_output_for_stage(stage)
    model_path = model_artifact_for_stage(stage)
    model_name = model_name_for_stage(stage)
    feature_cols = feature_columns_for_stage(stage)

    if not os.path.exists(features_path):
        print(f"Skipping {stage}: features file not found at {features_path}")
        return

    df = pd.read_csv(features_path)
    if df.empty:
        print(f"Skipping {stage}: no historical rows available")
        return

    missing = [column for column in feature_cols if column not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns for {stage}: {missing}")
    if "team1_won" not in df.columns:
        raise ValueError(f"Historical features for {stage} must include team1_won")

    X = df[feature_cols].fillna(0.0)
    y = df["team1_won"].astype(int)

    suspicious = []
    for column in feature_cols:
        corr = abs(pd.Series(X[column]).corr(y))
        if pd.notna(corr) and corr > 0.98:
            suspicious.append((column, corr))
    if suspicious:
        raise ValueError(f"Potential leakage for {stage}: suspiciously high target correlation: {suspicious}")

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=2000, C=0.7)),
    ])
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    ll = log_loss(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    acc = accuracy_score(y_test, preds)

    metrics = {
        "log_loss": float(ll),
        "brier_score": float(brier),
        "accuracy": float(acc),
    }

    print(f"[{stage}] Log loss: {ll:.4f}")
    print(f"[{stage}] Brier score: {brier:.4f}")
    print(f"[{stage}] Accuracy: {acc:.4f}")

    Path("/app/ml/artifacts").mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "stage": stage,
        "model_name": model_name,
    }
    joblib.dump(payload, model_path)
    print(f"[{stage}] Model saved to {model_path}")

    mlflow.set_experiment("ipl_prediction")
    with mlflow.start_run(run_name=f"{model_name}_richer_features_v2") as run:
        mlflow.log_params({
            "model_type": "logistic_regression",
            "stage": stage,
            "feature_count": len(feature_cols),
            "scaled": True,
        })
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")

        artifact_uri = f"{run.info.artifact_uri}/model"
        register_model_version(
            engine=engine,
            model_name=model_name,
            stage=stage,
            run_id=run.info.run_id,
            artifact_uri=artifact_uri,
            metrics=metrics,
        )

    print(f"[{stage}] Model logged to MLflow and registered in model_versions")


def main():
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    for stage in STAGES:
        train_stage(engine, stage)


if __name__ == "__main__":
    main()
