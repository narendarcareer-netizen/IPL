import os
import pandas as pd
from sqlalchemy import create_engine, text

DATABASE_URL = os.environ["DATABASE_URL"]


def safe_div(a, b):
    return a / b if b else 0.0


def main():
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    q = text("""
        SELECT
            pms.match_id,
            pms.player_id,
            m.start_time_utc,
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
        WHERE m.competition = 'Indian Premier League'
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
            hist = g.iloc[max(0, i - 5):i]

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

            runs_avg = hist["runs"].mean()
            sr = safe_div(hist["runs"].sum() * 100.0, hist["balls_faced"].sum())
            boundaries = hist["fours"].sum() + hist["sixes"].sum()
            boundary_pct = safe_div(boundaries, hist["balls_faced"].sum())
            dismissal_rate = safe_div((hist["balls_faced"] > 0).sum(), len(hist))

            wkts_avg = hist["wickets"].mean()
            overs = hist["overs_bowled"].sum()
            econ = safe_div(hist["runs_conceded"].sum(), overs)
            balls_bowled = overs * 6.0
            bowling_sr = safe_div(balls_bowled, hist["wickets"].sum()) if hist["wickets"].sum() > 0 else 0.0

            batting_form_score = 0.6 * runs_avg + 0.4 * (sr / 100.0)
            bowling_form_score = 0.7 * wkts_avg + 0.3 * max(0.0, 10.0 - econ)

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