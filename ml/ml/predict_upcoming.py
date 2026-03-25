import os
import joblib
import pandas as pd
from sqlalchemy import create_engine, text

DATABASE_URL = os.environ["DATABASE_URL"]

MODEL_PATH = "/app/ml/artifacts/logreg_prematch.joblib"
FEATURES_PATH = "/app/ml/artifacts/upcoming_match_features.csv"


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


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Upcoming features file not found: {FEATURES_PATH}")

    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    payload = joblib.load(MODEL_PATH)
    model = payload["model"]
    trained_feature_cols = payload["feature_cols"]

    model_name = "logreg_prematch"
    model_version_id = get_latest_model_version_id(engine, model_name)

    df = pd.read_csv(FEATURES_PATH)

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
                    "stage": "pre_toss",
                    "cutoff_time_utc": cutoff_time_utc,
                    "team1_win_prob": float(row["team1_win_prob"]),
                    "team2_win_prob": float(row["team2_win_prob"]),
                    "bookmaker_prob_team1": None,
                    "bookmaker_prob_team2": None,
                    "model_edge_team1": None,
                    "model_edge_team2": None,
                    "confidence_score": float(row["confidence_score"]),
                    "features_json": pd.Series({
                        "elo_diff": row.get("elo_diff"),
                        "batting_strength_diff": row.get("batting_strength_diff"),
                        "bowling_strength_diff": row.get("bowling_strength_diff"),
                        "all_rounder_balance_diff": row.get("all_rounder_balance_diff"),
                        "spin_strength_diff": row.get("spin_strength_diff"),
                        "pace_strength_diff": row.get("pace_strength_diff"),
                        "death_overs_strength_diff": row.get("death_overs_strength_diff"),
                        "probable_xi_batting_form_diff": row.get("probable_xi_batting_form_diff"),
                        "probable_xi_bowling_form_diff": row.get("probable_xi_bowling_form_diff"),
                        "probable_xi_runs_avg_diff": row.get("probable_xi_runs_avg_diff"),
                        "probable_xi_sr_diff": row.get("probable_xi_sr_diff"),
                        "probable_xi_wkts_diff": row.get("probable_xi_wkts_diff"),
                        "probable_xi_econ_diff": row.get("probable_xi_econ_diff"),
                    }).dropna().to_json(),
                    "model_name": row["model_name"],
                },
            )

    print(out[[
        "match_id",
        "team1_name",
        "team2_name",
        "team1_win_prob",
        "team2_win_prob",
        "confidence_score",
        "model_name",
    ]].head(20).to_string(index=False))

    print(f"\nInserted predictions for {len(out)} matches")


if __name__ == "__main__":
    main()