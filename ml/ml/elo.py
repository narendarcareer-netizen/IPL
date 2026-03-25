import os
from math import log
import pandas as pd
from sqlalchemy import create_engine, text

DATABASE_URL = os.environ["DATABASE_URL"]

BASE_ELO = 1500.0
K_FACTOR = 20.0
HOME_ADVANTAGE = 0.0


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def fetch_ipl_results(engine) -> pd.DataFrame:
    q = text("""
        SELECT
            match_id,
            start_time_utc,
            team1_id,
            team2_id,
            winner_team_id,
            completed,
            competition
        FROM matches
        WHERE competition = 'Indian Premier League'
          AND completed = true
          AND winner_team_id IS NOT NULL
        ORDER BY start_time_utc ASC, match_id ASC
    """)
    return pd.read_sql(q, engine)


def backtest_elo(df: pd.DataFrame):
    ratings: dict[int, float] = {}
    rows: list[dict] = []

    for _, row in df.iterrows():
        team1_id = int(row["team1_id"])
        team2_id = int(row["team2_id"])
        winner_team_id = int(row["winner_team_id"])

        r1 = ratings.get(team1_id, BASE_ELO)
        r2 = ratings.get(team2_id, BASE_ELO)

        p1 = expected_score(r1 + HOME_ADVANTAGE, r2)
        team1_won = 1 if winner_team_id == team1_id else 0

        rows.append({
            "match_id": int(row["match_id"]),
            "start_time_utc": row["start_time_utc"],
            "team1_id": team1_id,
            "team2_id": team2_id,
            "team1_elo_pre": r1,
            "team2_elo_pre": r2,
            "team1_win_prob": p1,
            "team1_won": team1_won,
        })

        s1 = float(team1_won)
        s2 = 1.0 - s1

        new_r1 = r1 + K_FACTOR * (s1 - p1)
        new_r2 = r2 + K_FACTOR * (s2 - (1.0 - p1))

        ratings[team1_id] = new_r1
        ratings[team2_id] = new_r2

    return pd.DataFrame(rows), ratings


def save_team_elos(engine, ratings: dict[int, float]):
    with engine.begin() as conn:
        for team_id, rating in ratings.items():
            conn.execute(
                text("""
                    INSERT INTO team_elo_ratings (team_id, rating)
                    VALUES (:team_id, :rating)
                    ON CONFLICT (team_id)
                    DO UPDATE SET
                        rating = EXCLUDED.rating,
                        updated_at_utc = NOW()
                """),
                {"team_id": team_id, "rating": rating},
            )


def main():
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    matches = fetch_ipl_results(engine)

    if matches.empty:
        print("No completed IPL matches with winner_team_id found.")
        return

    preds, ratings = backtest_elo(matches)

    print(preds.head(10).to_string(index=False))

    eps = 1e-9
    log_loss = -(
        preds["team1_won"] * preds["team1_win_prob"].apply(lambda x: log(x + eps))
        + (1 - preds["team1_won"]) * preds["team1_win_prob"].apply(lambda x: log(1 - x + eps))
    ).mean()

    brier = ((preds["team1_win_prob"] - preds["team1_won"]) ** 2).mean()

    save_team_elos(engine, ratings)

    print(f"\nMatches evaluated: {len(preds)}")
    print(f"Log loss: {log_loss:.4f}")
    print(f"Brier score: {brier:.4f}")
    print(f"Saved Elo ratings for {len(ratings)} teams")


if __name__ == "__main__":
    main()