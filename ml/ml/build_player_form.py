import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

DATABASE_URL = os.environ["DATABASE_URL"]


def safe_div(a, b):
    return a / b if b else 0.0


def competition_weight(competition: str | None) -> float:
    name = str(competition or "").strip().lower()
    if not name:
        return 0.9
    if "world cup" in name and "20" in name:
        return 1.15
    if "ipl" in name or "indian premier league" in name:
        return 1.10
    if "t20i" in name or "international" in name:
        return 1.05
    if any(token in name for token in ("big bash", "psl", "bpl", "sa20", "ilt20", "blast", "super league")):
        return 0.98
    return 0.92


def weighted_mean(series: pd.Series, weights: np.ndarray) -> float:
    values = series.fillna(0).astype(float).to_numpy()
    if len(values) == 0:
        return 0.0
    return float(np.average(values, weights=weights))


def main():
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    q = text("""
        SELECT
            pms.match_id,
            pms.player_id,
            m.start_time_utc,
            m.competition,
            m.match_type,
            pms.runs,
            pms.balls_faced,
            pms.fours,
            pms.sixes,
            pms.overs_bowled,
            pms.runs_conceded,
            pms.wickets
        FROM player_match_stats pms
        JOIN matches m
          ON pms.match_id = m.match_id
        WHERE COALESCE(m.completed, false) = true
          AND LOWER(COALESCE(m.match_type, '')) = 't20'
        ORDER BY pms.player_id, m.start_time_utc
    """)

    df = pd.read_sql(q, engine)

    if df.empty:
        print("No player_match_stats found.")
        return

    df["start_time_utc"] = pd.to_datetime(df["start_time_utc"], utc=True)

    rows = []

    for player_id, g in df.groupby("player_id", sort=True):
        g = g.sort_values("start_time_utc").reset_index(drop=True)

        for i in range(len(g)):
            hist = g.iloc[max(0, i - 10):i].copy()

            if hist.empty:
                rows.append({
                    "player_id": int(player_id),
                    "as_of_time_utc": g.iloc[i]["start_time_utc"].to_pydatetime(),
                    "horizon_matches": 5,
                    "availability_flag": True,
                    "snapshot_time_utc": g.iloc[i]["start_time_utc"].to_pydatetime(),
                    "matches_used": 0,
                    "batting_form_score": 0.0,
                    "bowling_form_score": 0.0,
                    "batting_runs_avg": 0.0,
                    "batting_strike_rate": 0.0,
                    "batting_boundary_pct": 0.0,
                    "dismissal_rate": 0.0,
                    "bowling_wkts_avg": 0.0,
                    "bowling_economy": 0.0,
                    "bowling_strike_rate": 0.0,
                    "fielding_score": 0.0,
                })
                continue

            recency_weights = np.power(0.86, np.arange(len(hist))[::-1])
            competition_weights = hist["competition"].apply(competition_weight).to_numpy(dtype=float)
            weights = recency_weights * competition_weights
            weight_sum = float(weights.sum()) if len(weights) else 0.0

            weighted_runs = float((hist["runs"].fillna(0).astype(float).to_numpy() * weights).sum())
            weighted_balls_faced = float((hist["balls_faced"].fillna(0).astype(float).to_numpy() * weights).sum())
            weighted_boundaries = float(
                ((hist["fours"].fillna(0).astype(float) + hist["sixes"].fillna(0).astype(float)).to_numpy() * weights).sum()
            )
            weighted_overs = float((hist["overs_bowled"].fillna(0).astype(float).to_numpy() * weights).sum())
            weighted_runs_conceded = float((hist["runs_conceded"].fillna(0).astype(float).to_numpy() * weights).sum())
            weighted_wickets = float((hist["wickets"].fillna(0).astype(float).to_numpy() * weights).sum())

            runs_avg = weighted_mean(hist["runs"], weights) if weighted_balls_faced > 0 else 0.0
            sr = safe_div(weighted_runs * 100.0, weighted_balls_faced) if weighted_balls_faced > 0 else 0.0
            boundary_pct = safe_div(weighted_boundaries, weighted_balls_faced) if weighted_balls_faced > 0 else 0.0
            dismissal_rate = weighted_mean((hist["balls_faced"].fillna(0) > 0).astype(float), weights)

            wkts_avg = weighted_mean(hist["wickets"], weights) if weighted_overs > 0 else 0.0
            econ = safe_div(weighted_runs_conceded, weighted_overs) if weighted_overs > 0 else 0.0
            balls_bowled = weighted_overs * 6.0
            bowling_sr = safe_div(balls_bowled, weighted_wickets) if weighted_wickets > 0 else 0.0

            forty_plus_rate = weighted_mean((hist["runs"].fillna(0) >= 40).astype(float), weights)
            three_wkt_rate = weighted_mean((hist["wickets"].fillna(0) >= 3).astype(float), weights)

            batting_form_score = 0.0
            if weighted_balls_faced > 0:
                batting_form_score = (
                    0.44 * runs_avg
                    + 0.20 * (sr / 100.0)
                    + 0.12 * (boundary_pct * 100.0)
                    + 6.0 * forty_plus_rate
                    + 0.08 * min(weight_sum, 6.0)
                )

            bowling_form_score = 0.0
            if weighted_overs > 0:
                bowling_form_score = (
                    0.50 * wkts_avg
                    + 0.24 * max(0.0, 8.5 - econ)
                    + 0.16 * max(0.0, 24.0 - bowling_sr) / 6.0
                    + 4.0 * three_wkt_rate
                )

            rows.append({
                "player_id": int(player_id),
                "as_of_time_utc": g.iloc[i]["start_time_utc"].to_pydatetime(),
                "horizon_matches": 5,
                "availability_flag": True,
                "snapshot_time_utc": g.iloc[i]["start_time_utc"].to_pydatetime(),
                "matches_used": int(len(hist)),
                "batting_form_score": float(batting_form_score),
                "bowling_form_score": float(bowling_form_score),
                "batting_runs_avg": float(runs_avg),
                "batting_strike_rate": float(sr),
                "batting_boundary_pct": float(boundary_pct),
                "dismissal_rate": float(dismissal_rate),
                "bowling_wkts_avg": float(wkts_avg),
                "bowling_economy": float(econ),
                "bowling_strike_rate": float(bowling_sr),
                "fielding_score": 0.0,
            })

    out = pd.DataFrame(rows)

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM player_form_snapshots"))
        for _, row in out.iterrows():
            conn.execute(
                text("""
                    INSERT INTO player_form_snapshots (
                        player_id,
                        as_of_time_utc,
                        horizon_matches,
                        batting_form_score,
                        bowling_form_score,
                        availability_flag,
                        snapshot_time_utc,
                        matches_used,
                        batting_runs_avg,
                        batting_strike_rate,
                        batting_boundary_pct,
                        dismissal_rate,
                        bowling_wkts_avg,
                        bowling_economy,
                        bowling_strike_rate,
                        fielding_score
                    )
                    VALUES (
                        :player_id,
                        :as_of_time_utc,
                        :horizon_matches,
                        :batting_form_score,
                        :bowling_form_score,
                        :availability_flag,
                        :snapshot_time_utc,
                        :matches_used,
                        :batting_runs_avg,
                        :batting_strike_rate,
                        :batting_boundary_pct,
                        :dismissal_rate,
                        :bowling_wkts_avg,
                        :bowling_economy,
                        :bowling_strike_rate,
                        :fielding_score
                    )
                """),
                row.to_dict(),
            )

    print(f"Inserted player form snapshots: {len(out)}")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
