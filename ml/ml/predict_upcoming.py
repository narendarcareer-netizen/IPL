import json
import os

import joblib
import pandas as pd
from sqlalchemy import create_engine, text

from ml.feature_config import (
    STAGES,
    feature_columns_for_stage,
    model_artifact_for_stage,
    model_name_for_stage,
    normalize_stage,
    upcoming_output_for_stage,
)

DATABASE_URL = os.environ["DATABASE_URL"]


def get_latest_model_version_id(engine, model_name: str) -> int:
    with engine.begin() as conn:
        row = conn.execute(
            text("""
                SELECT model_version_id
                FROM model_versions
                WHERE model_name = :model_name
                ORDER BY created_at_utc DESC, model_version_id DESC
                LIMIT 1
            """),
            {"model_name": model_name},
        ).first()

    if not row:
        raise ValueError(f"No model_versions row found for model_name={model_name}")

    return int(row[0])


def predict_stage(engine, stage: str) -> int:
    stage = normalize_stage(stage)
    model_path = model_artifact_for_stage(stage)
    features_path = upcoming_output_for_stage(stage)
    model_name = model_name_for_stage(stage)

    if not os.path.exists(model_path):
        print(f"Skipping {stage}: model file not found at {model_path}")
        return 0
    if not os.path.exists(features_path):
        print(f"Skipping {stage}: upcoming features file not found at {features_path}")
        return 0

    df = pd.read_csv(features_path)
    if df.empty:
        print(f"Skipping {stage}: no upcoming rows available")
        return 0

    payload = joblib.load(model_path)
    model = payload["model"]
    trained_feature_cols = payload.get("feature_cols", feature_columns_for_stage(stage))
    model_version_id = get_latest_model_version_id(engine, model_name)

    X = df[trained_feature_cols].fillna(0.0)
    probs = model.predict_proba(X)[:, 1]
    confidence = abs(probs - 0.5) * 2.0

    out = df.copy()
    out["team1_win_prob"] = probs
    out["team2_win_prob"] = 1.0 - probs
    out["confidence_score"] = confidence
    out["model_name"] = model_name

    with engine.begin() as conn:
        for _, row in out.iterrows():
            cutoff_time_utc = pd.to_datetime(row["start_time_utc"], utc=True).to_pydatetime()
            bookmaker_prob_team1 = row.get("bookmaker_prob_team1")
            bookmaker_prob_team1 = None if pd.isna(bookmaker_prob_team1) else float(bookmaker_prob_team1)
            bookmaker_prob_team2 = None
            model_edge_team1 = None
            model_edge_team2 = None

            if bookmaker_prob_team1 is not None:
                bookmaker_prob_team2 = float(1.0 - bookmaker_prob_team1)
                model_edge_team1 = float(row["team1_win_prob"] - bookmaker_prob_team1)
                model_edge_team2 = float(row["team2_win_prob"] - bookmaker_prob_team2)

            feature_payload = {
                column: float(row[column])
                for column in trained_feature_cols
                if column in row and pd.notna(row[column])
            }

            conn.execute(
                text("""
                    INSERT INTO predictions (
                        match_id,
                        model_version_id,
                        stage,
                        cutoff_time_utc,
                        created_at_utc,
                        team1_win_prob,
                        team2_win_prob,
                        bookmaker_prob_team1,
                        bookmaker_prob_team2,
                        model_edge_team1,
                        model_edge_team2,
                        confidence_score,
                        features_json,
                        model_name
                    )
                    VALUES (
                        :match_id,
                        :model_version_id,
                        :stage,
                        :cutoff_time_utc,
                        NOW(),
                        :team1_win_prob,
                        :team2_win_prob,
                        :bookmaker_prob_team1,
                        :bookmaker_prob_team2,
                        :model_edge_team1,
                        :model_edge_team2,
                        :confidence_score,
                        CAST(:features_json AS jsonb),
                        :model_name
                    )
                """),
                {
                    "match_id": int(row["match_id"]),
                    "model_version_id": model_version_id,
                    "stage": stage,
                    "cutoff_time_utc": cutoff_time_utc,
                    "team1_win_prob": float(row["team1_win_prob"]),
                    "team2_win_prob": float(row["team2_win_prob"]),
                    "bookmaker_prob_team1": bookmaker_prob_team1,
                    "bookmaker_prob_team2": bookmaker_prob_team2,
                    "model_edge_team1": model_edge_team1,
                    "model_edge_team2": model_edge_team2,
                    "confidence_score": float(row["confidence_score"]),
                    "features_json": json.dumps(feature_payload),
                    "model_name": row["model_name"],
                },
            )

    preview_cols = [
        "match_id",
        "stage",
        "team1_name",
        "team2_name",
        "team1_win_prob",
        "team2_win_prob",
        "confidence_score",
        "bookmaker_prob_team1",
        "model_name",
    ]
    preview_cols = [column for column in preview_cols if column in out.columns]
    print(out[preview_cols].head(20).to_string(index=False))
    print(f"\nInserted {stage} predictions for {len(out)} matches")
    return int(len(out))


def main():
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    total = 0
    for stage in STAGES:
        total += predict_stage(engine, stage)
    print(f"\nInserted predictions across stages for {total} rows")


if __name__ == "__main__":
    main()
