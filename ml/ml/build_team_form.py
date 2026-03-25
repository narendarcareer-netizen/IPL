import os
import pandas as pd
from sqlalchemy import create_engine, text

DATABASE_URL = os.environ["DATABASE_URL"]


def main():
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    q = text("""
        WITH latest_player_form AS (
            SELECT DISTINCT ON (pfs.player_id)
                pfs.player_id,
                pfs.as_of_time_utc,
                pfs.matches_used,
                pfs.batting_form_score,
                pfs.bowling_form_score,
                pfs.batting_runs_avg,
                pfs.batting_strike_rate,
                pfs.bowling_wkts_avg,
                pfs.bowling_economy
            FROM player_form_snapshots pfs
            ORDER BY pfs.player_id, pfs.as_of_time_utc DESC
        ),
        latest_team_players AS (
            SELECT
                pms.team_id,
                pms.player_id,
                MAX(m.start_time_utc) AS last_match_time
            FROM player_match_stats pms
            JOIN matches m
              ON pms.match_id = m.match_id
            WHERE m.competition = 'Indian Premier League'
            GROUP BY pms.team_id, pms.player_id
        )
        SELECT
            ltp.team_id,
            ltp.player_id,
            lpf.as_of_time_utc,
            lpf.matches_used,
            COALESCE(lpf.batting_form_score, 0) AS batting_form_score,
            COALESCE(lpf.bowling_form_score, 0) AS bowling_form_score,
            COALESCE(lpf.batting_runs_avg, 0) AS batting_runs_avg,
            COALESCE(lpf.batting_strike_rate, 0) AS batting_strike_rate,
            COALESCE(lpf.bowling_wkts_avg, 0) AS bowling_wkts_avg,
            COALESCE(lpf.bowling_economy, 0) AS bowling_economy,
            COALESCE(ter.rating, 1500) AS elo_rating
        FROM latest_team_players ltp
        LEFT JOIN latest_player_form lpf
          ON ltp.player_id = lpf.player_id
        LEFT JOIN team_elo_ratings ter
          ON ltp.team_id = ter.team_id
        ORDER BY ltp.team_id, ltp.player_id
    """)

    df = pd.read_sql(q, engine)

    if df.empty:
        print("No player/team form data found.")
        return

    rows = []

    for team_id, g in df.groupby("team_id", sort=True):
        g = g.copy()
        g["combined_form"] = g["batting_form_score"] + g["bowling_form_score"]
        xi = g.sort_values("combined_form", ascending=False).head(11)

        batting_strength = xi["batting_form_score"].sum()
        bowling_strength = xi["bowling_form_score"].sum()

        all_rounder_balance = float(
            ((xi["batting_form_score"] > 0) & (xi["bowling_form_score"] > 0)).sum()
        )

        spin_strength = bowling_strength * 0.5
        pace_strength = bowling_strength * 0.5

        death_overs_strength = float(
            xi["bowling_wkts_avg"].sum()
            + (10.0 - xi["bowling_economy"].clip(upper=10)).sum()
        )

        powerplay_strength = float(bowling_strength * 0.30)
        middle_strength = float(bowling_strength * 0.40)
        death_strength = float(bowling_strength * 0.30)

        snapshot_time = pd.Timestamp.utcnow().to_pydatetime()
        elo_rating = float(xi["elo_rating"].iloc[0]) if not xi.empty else 1500.0

        rows.append({
            "team_id": int(team_id),
            "as_of_time_utc": snapshot_time,
            "horizon_matches": 5,
            "elo_rating": elo_rating,
            "powerplay_strength": powerplay_strength,
            "middle_strength": middle_strength,
            "death_strength": death_strength,
            "snapshot_time_utc": snapshot_time,
            "matches_used": int(xi["matches_used"].fillna(0).max() if not xi.empty else 0),
            "batting_strength": float(batting_strength),
            "bowling_strength": float(bowling_strength),
            "all_rounder_balance": all_rounder_balance,
            "spin_strength": float(spin_strength),
            "pace_strength": float(pace_strength),
            "death_overs_strength": float(death_overs_strength),
        })

    out = pd.DataFrame(rows)

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM team_form_snapshots"))
        for _, row in out.iterrows():
            conn.execute(
                text("""
                    INSERT INTO team_form_snapshots (
                        team_id,
                        as_of_time_utc,
                        horizon_matches,
                        elo_rating,
                        powerplay_strength,
                        middle_strength,
                        death_strength,
                        snapshot_time_utc,
                        matches_used,
                        batting_strength,
                        bowling_strength,
                        all_rounder_balance,
                        spin_strength,
                        pace_strength,
                        death_overs_strength
                    )
                    VALUES (
                        :team_id,
                        :as_of_time_utc,
                        :horizon_matches,
                        :elo_rating,
                        :powerplay_strength,
                        :middle_strength,
                        :death_strength,
                        :snapshot_time_utc,
                        :matches_used,
                        :batting_strength,
                        :bowling_strength,
                        :all_rounder_balance,
                        :spin_strength,
                        :pace_strength,
                        :death_overs_strength
                    )
                """),
                row.to_dict(),
            )

    print(f"Inserted team form snapshots: {len(out)}")
    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()