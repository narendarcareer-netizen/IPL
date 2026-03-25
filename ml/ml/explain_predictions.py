import os
import json
import joblib
import pandas as pd
import shap
from sqlalchemy import create_engine, text

DATABASE_URL = os.environ["DATABASE_URL"]
MODEL_PATH = "/app/ml/artifacts/logreg_prematch.joblib"
FEATURES_PATH = "/app/ml/artifacts/upcoming_match_features.csv"


def main():
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Upcoming features file not found: {FEATURES_PATH}")

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    df = pd.read_csv(FEATURES_PATH)

    if df.empty:
        print("No features found.")
        return

    X = df[feature_cols].fillna(0.0)

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
                    ORDER BY created_at_utc DESC, prediction_id DESC
                    LIMIT 1
                """),
                {"match_id": match_id},
            ).first()

        if not pred_row:
            print(f"Skipping match_id={match_id}: no prediction row found")
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

        top_features = sorted(
            pairs,
            key=lambda x: abs(x["impact_value"]),
            reverse=True
        )[:5]

        rows.append({
            "prediction_id": prediction_id,
            "method": "shap",
            "base_value": float(base_value),
            "shap_values_json": json.dumps(pairs),
            "top_features_json": json.dumps(top_features),
        })

    if not rows:
        print("No explanations generated.")
        return

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM explanations"))

        for row in rows:
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

    print(f"SHAP explanations stored successfully for {len(rows)} predictions.")


if __name__ == "__main__":
    main()