import os
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text

DATABASE_URL = os.environ["DATABASE_URL"]
OUT_PATH = "/app/ml/artifacts/upcoming_match_features.csv"


def main():
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    q = text("""
        WITH latest_team_form AS (
            SELECT DISTINCT ON (team_id)
                team_id,
                elo_rating,
                batting_strength,
                bowling_strength,
                all_rounder_balance,
                spin_strength,
                pace_strength,
                death_overs_strength
            FROM team_form_snapshots
            ORDER BY team_id, as_of_time_utc DESC
        ),
        latest_player_form AS (
            SELECT DISTINCT ON (player_id)
                player_id,
                batting_form_score,
                bowling_form_score,
                batting_runs_avg,
                batting_strike_rate,
                bowling_wkts_avg,
                bowling_economy
            FROM player_form_snapshots
            ORDER BY player_id, as_of_time_utc DESC
        ),
        xi_agg AS (
            SELECT
                px.match_id,
                px.team_id,
                COUNT(*) AS xi_count,
                SUM(COALESCE(lpf.batting_form_score, 0)) AS xi_batting_form,
                SUM(COALESCE(lpf.bowling_form_score, 0)) AS xi_bowling_form,
                SUM(COALESCE(lpf.batting_runs_avg, 0)) AS xi_batting_runs_avg_sum,
                AVG(COALESCE(lpf.batting_strike_rate, 0)) AS xi_batting_strike_rate_avg,
                SUM(COALESCE(lpf.bowling_wkts_avg, 0)) AS xi_bowling_wkts_avg_sum,
                AVG(COALESCE(lpf.bowling_economy, 0)) AS xi_bowling_economy_avg,
                SUM(CASE WHEN px.batting_order_hint <= 3 THEN COALESCE(lpf.batting_form_score, 0) ELSE 0 END) AS top_order_strength,
                SUM(CASE WHEN px.batting_order_hint BETWEEN 4 AND 7 THEN COALESCE(lpf.batting_form_score, 0) ELSE 0 END) AS middle_order_strength,
                SUM(CASE WHEN COALESCE(lpf.bowling_economy, 0) > 0 THEN (10 - LEAST(COALESCE(lpf.bowling_economy, 0), 10)) ELSE 0 END) AS death_bowling_strength
            FROM probable_xi px
            LEFT JOIN latest_player_form lpf
              ON px.player_id = lpf.player_id
            GROUP BY px.match_id, px.team_id
        )
        SELECT
            m.match_id,
            m.start_time_utc,
            m.team1_id,
            m.team2_id,
            t1.name AS team1_name,
            t2.name AS team2_name,

            COALESCE(tf1.elo_rating, 1500) AS team1_elo,
            COALESCE(tf2.elo_rating, 1500) AS team2_elo,

            COALESCE(tf1.batting_strength, 0) AS team1_batting_strength,
            COALESCE(tf2.batting_strength, 0) AS team2_batting_strength,
            COALESCE(tf1.bowling_strength, 0) AS team1_bowling_strength,
            COALESCE(tf2.bowling_strength, 0) AS team2_bowling_strength,
            COALESCE(tf1.all_rounder_balance, 0) AS team1_all_rounder_balance,
            COALESCE(tf2.all_rounder_balance, 0) AS team2_all_rounder_balance,
            COALESCE(tf1.spin_strength, 0) AS team1_spin_strength,
            COALESCE(tf2.spin_strength, 0) AS team2_spin_strength,
            COALESCE(tf1.pace_strength, 0) AS team1_pace_strength,
            COALESCE(tf2.pace_strength, 0) AS team2_pace_strength,
            COALESCE(tf1.death_overs_strength, 0) AS team1_death_overs_strength,
            COALESCE(tf2.death_overs_strength, 0) AS team2_death_overs_strength,

            COALESCE(x1.xi_count, 0) AS team1_xi_count,
            COALESCE(x2.xi_count, 0) AS team2_xi_count,
            COALESCE(x1.xi_batting_form, 0) AS team1_xi_batting_form,
            COALESCE(x2.xi_batting_form, 0) AS team2_xi_batting_form,
            COALESCE(x1.xi_bowling_form, 0) AS team1_xi_bowling_form,
            COALESCE(x2.xi_bowling_form, 0) AS team2_xi_bowling_form,
            COALESCE(x1.xi_batting_runs_avg_sum, 0) AS team1_xi_batting_runs_avg_sum,
            COALESCE(x2.xi_batting_runs_avg_sum, 0) AS team2_xi_batting_runs_avg_sum,
            COALESCE(x1.xi_batting_strike_rate_avg, 0) AS team1_xi_batting_strike_rate_avg,
            COALESCE(x2.xi_batting_strike_rate_avg, 0) AS team2_xi_batting_strike_rate_avg,
            COALESCE(x1.xi_bowling_wkts_avg_sum, 0) AS team1_xi_bowling_wkts_avg_sum,
            COALESCE(x2.xi_bowling_wkts_avg_sum, 0) AS team2_xi_bowling_wkts_avg_sum,
            COALESCE(x1.xi_bowling_economy_avg, 0) AS team1_xi_bowling_economy_avg,
            COALESCE(x2.xi_bowling_economy_avg, 0) AS team2_xi_bowling_economy_avg,
            COALESCE(x1.top_order_strength, 0) AS team1_top_order_strength,
            COALESCE(x2.top_order_strength, 0) AS team2_top_order_strength,
            COALESCE(x1.middle_order_strength, 0) AS team1_middle_order_strength,
            COALESCE(x2.middle_order_strength, 0) AS team2_middle_order_strength,
            COALESCE(x1.death_bowling_strength, 0) AS team1_death_bowling_strength,
            COALESCE(x2.death_bowling_strength, 0) AS team2_death_bowling_strength
        FROM matches m
        JOIN teams t1 ON m.team1_id = t1.team_id
        JOIN teams t2 ON m.team2_id = t2.team_id
        LEFT JOIN latest_team_form tf1 ON m.team1_id = tf1.team_id
        LEFT JOIN latest_team_form tf2 ON m.team2_id = tf2.team_id
        LEFT JOIN xi_agg x1 ON m.match_id = x1.match_id AND m.team1_id = x1.team_id
        LEFT JOIN xi_agg x2 ON m.match_id = x2.match_id AND m.team2_id = x2.team_id
        WHERE m.competition = 'Indian Premier League'
          AND m.completed = false
        ORDER BY m.start_time_utc ASC, m.match_id ASC
    """)

    df = pd.read_sql(q, engine)

    if df.empty:
        print("No future IPL matches found.")
        return

    df["elo_diff"] = df["team1_elo"] - df["team2_elo"]
    df["batting_strength_diff"] = df["team1_batting_strength"] - df["team2_batting_strength"]
    df["bowling_strength_diff"] = df["team1_bowling_strength"] - df["team2_bowling_strength"]
    df["all_rounder_balance_diff"] = df["team1_all_rounder_balance"] - df["team2_all_rounder_balance"]
    df["spin_strength_diff"] = df["team1_spin_strength"] - df["team2_spin_strength"]
    df["pace_strength_diff"] = df["team1_pace_strength"] - df["team2_pace_strength"]
    df["death_overs_strength_diff"] = df["team1_death_overs_strength"] - df["team2_death_overs_strength"]

    df["probable_xi_count_diff"] = df["team1_xi_count"] - df["team2_xi_count"]
    df["probable_xi_batting_form_diff"] = df["team1_xi_batting_form"] - df["team2_xi_batting_form"]
    df["probable_xi_bowling_form_diff"] = df["team1_xi_bowling_form"] - df["team2_xi_bowling_form"]
    df["probable_xi_runs_avg_diff"] = df["team1_xi_batting_runs_avg_sum"] - df["team2_xi_batting_runs_avg_sum"]
    df["probable_xi_sr_diff"] = df["team1_xi_batting_strike_rate_avg"] - df["team2_xi_batting_strike_rate_avg"]
    df["probable_xi_wkts_diff"] = df["team1_xi_bowling_wkts_avg_sum"] - df["team2_xi_bowling_wkts_avg_sum"]
    df["probable_xi_econ_diff"] = df["team2_xi_bowling_economy_avg"] - df["team1_xi_bowling_economy_avg"]
    df["top_order_strength_diff"] = df["team1_top_order_strength"] - df["team2_top_order_strength"]
    df["middle_order_strength_diff"] = df["team1_middle_order_strength"] - df["team2_middle_order_strength"]
    df["death_bowling_strength_diff"] = df["team1_death_bowling_strength"] - df["team2_death_bowling_strength"]
    df["venue_win_bias_diff"] = 0.0

    Path("/app/ml/artifacts").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Saved upcoming features to {OUT_PATH}")
    print(df[[
        "match_id",
        "team1_name",
        "team2_name",
        "elo_diff",
        "batting_strength_diff",
        "bowling_strength_diff",
        "probable_xi_count_diff",
        "probable_xi_batting_form_diff",
        "probable_xi_bowling_form_diff",
        "top_order_strength_diff",
        "middle_order_strength_diff",
        "death_bowling_strength_diff",
    ]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()