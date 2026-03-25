from pathlib import Path
import joblib
import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

MODEL_PATH = Path("/app/ml/artifacts/logreg_prematch.joblib")


def elo_prob(r1: float, r2: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r2 - r1) / 400.0))


def load_local_logreg():
    if not MODEL_PATH.exists():
        return None
    payload = joblib.load(MODEL_PATH)
    return payload


def get_team_snapshot(db: Session, team_id: int) -> dict:
    row = db.execute(
        text("""
            SELECT
                COALESCE(elo_rating, 1500) AS elo_rating,
                COALESCE(batting_strength, 0) AS batting_strength,
                COALESCE(bowling_strength, 0) AS bowling_strength,
                COALESCE(all_rounder_balance, 0) AS all_rounder_balance,
                COALESCE(spin_strength, 0) AS spin_strength,
                COALESCE(pace_strength, 0) AS pace_strength,
                COALESCE(death_overs_strength, 0) AS death_overs_strength
            FROM team_form_snapshots
            WHERE team_id = :team_id
            ORDER BY as_of_time_utc DESC
            LIMIT 1
        """),
        {"team_id": team_id},
    ).mappings().first()

    if not row:
        return {
            "elo_rating": 1500.0,
            "batting_strength": 0.0,
            "bowling_strength": 0.0,
            "all_rounder_balance": 0.0,
            "spin_strength": 0.0,
            "pace_strength": 0.0,
            "death_overs_strength": 0.0,
        }

    return dict(row)


def predict_match(db: Session, match_id: int, stage: str, cutoff_time_utc, model_uri: str | None = None) -> dict:
    match = db.execute(
        text("""
            SELECT
                m.match_id,
                m.team1_id,
                m.team2_id,
                t1.name AS team1_name,
                t2.name AS team2_name
            FROM matches m
            JOIN teams t1 ON m.team1_id = t1.team_id
            JOIN teams t2 ON m.team2_id = t2.team_id
            WHERE m.match_id = :match_id
        """),
        {"match_id": match_id},
    ).mappings().first()

    if not match:
        raise ValueError(f"Match {match_id} not found")

    team1 = get_team_snapshot(db, match["team1_id"])
    team2 = get_team_snapshot(db, match["team2_id"])

    features = {
        "elo_diff": float(team1["elo_rating"] - team2["elo_rating"]),
        "batting_strength_diff": float(team1["batting_strength"] - team2["batting_strength"]),
        "bowling_strength_diff": float(team1["bowling_strength"] - team2["bowling_strength"]),
        "all_rounder_balance_diff": float(team1["all_rounder_balance"] - team2["all_rounder_balance"]),
        "spin_strength_diff": float(team1["spin_strength"] - team2["spin_strength"]),
        "pace_strength_diff": float(team1["pace_strength"] - team2["pace_strength"]),
        "death_overs_strength_diff": float(team1["death_overs_strength"] - team2["death_overs_strength"]),
    }

    payload = load_local_logreg()

    if payload is not None:
        model = payload["model"]
        feature_cols = payload["feature_cols"]
        X = np.array([[features[col] for col in feature_cols]], dtype=float)
        p1 = float(model.predict_proba(X)[0, 1])
        model_name = "logreg_prematch"
    else:
        p1 = float(elo_prob(team1["elo_rating"], team2["elo_rating"]))
        model_name = "elo_baseline"

    p2 = 1.0 - p1
    confidence = abs(p1 - 0.5) * 2.0

    return {
        "match_id": match["match_id"],
        "stage": stage,
        "model_name": model_name,
        "team1_name": match["team1_name"],
        "team2_name": match["team2_name"],
        "team1_rating": float(team1["elo_rating"]),
        "team2_rating": float(team2["elo_rating"]),
        "team1_win_prob": p1,
        "team2_win_prob": p2,
        "confidence_score": confidence,
        "features": features,
        "cutoff_time_utc": cutoff_time_utc.isoformat(),
    }