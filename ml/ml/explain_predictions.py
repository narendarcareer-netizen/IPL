import json
import os

import joblib
import pandas as pd
import shap
from sqlalchemy import create_engine, text

from ml.feature_config import STAGES, model_artifact_for_stage, upcoming_output_for_stage

DATABASE_URL = os.environ["DATABASE_URL"]


def explain_stage(engine, stage: str) -> int:
    model_path = model_artifact_for_stage(stage)
    features_path = upcoming_output_for_stage(stage)

    if not os.path.exists(model_path):
        print(f"Skipping explanations for {stage}: model file not found at {model_path}")
        return 0
    if not os.path.exists(features_path):
        print(f"Skipping explanations for {stage}: features file not found at {features_path}")
        return 0

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    df = pd.read_csv(features_path)

    if df.empty:
        print(f"Skipping explanations for {stage}: no features found.")
        return 0

    X = df[feature_cols].fillna(0.0)
    scaler = getattr(model, "named_steps", {}).get("scaler")
    logreg = getattr(model, "named_steps", {}).get("logreg")
    if scaler is not None and logreg is not None:
        X_model = scaler.transform(X)
        explainer = shap.LinearExplainer(logreg, X_model)
        shap_result = explainer(X_model)
    else:
        explainer = shap.Explainer(model, X)
        shap_result = explainer(X)

    rows = []
    for i in range(len(df)):
        match_id = int(df.iloc[i]["match_id"])

        with engine.begin() as conn:
            pred_row = conn.execute(
                text("""
                    SELECT prediction_id
                    FROM predictions
                    WHERE match_id = :match_id
                      AND stage = :stage
                    ORDER BY created_at_utc DESC, prediction_id DESC
                    LIMIT 1
                """),
                {"match_id": match_id, "stage": stage},
            ).first()

        if not pred_row:
            print(f"Skipping match_id={match_id}, stage={stage}: no prediction row found")
            continue

        prediction_id = int(pred_row[0])
        shap_values = shap_result.values[i]
        base_value = shap_result.base_values[i]

        if hasattr(base_value, "item"):
            base_value = base_value.item()

        pairs = []
        for fname, fval in zip(feature_cols, shap_values):
            if hasattr(fval, "item"):
                fval = fval.item()
            pairs.append({
                "feature_name": fname,
                "impact_value": float(fval),
            })

        top_features = sorted(pairs, key=lambda x: abs(x["impact_value"]), reverse=True)[:5]
        rows.append({
            "prediction_id": prediction_id,
            "method": "shap",
            "base_value": float(base_value),
            "shap_values_json": json.dumps(pairs),
            "top_features_json": json.dumps(top_features),
        })

    if not rows:
        print(f"No explanations generated for stage={stage}.")
        return 0

    with engine.begin() as conn:
        for row in rows:
            conn.execute(
                text("""
                    DELETE FROM explanations
                    WHERE prediction_id = :prediction_id
                """),
                {"prediction_id": row["prediction_id"]},
            )
            conn.execute(
                text("""
                    INSERT INTO explanations (
                        prediction_id,
                        method,
                        base_value,
                        shap_values_json,
                        top_features_json
                    )
                    VALUES (
                        :prediction_id,
                        :method,
                        :base_value,
                        CAST(:shap_values_json AS jsonb),
                        CAST(:top_features_json AS jsonb)
                    )
                """),
                row,
            )

    print(f"SHAP explanations stored successfully for {len(rows)} {stage} predictions.")
    return int(len(rows))


def main():
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    total = 0
    for stage in STAGES:
        total += explain_stage(engine, stage)
    print(f"Stored explanations for {total} predictions across stages.")


if __name__ == "__main__":
    main()
