import os
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text

DATABASE_URL = os.environ["DATABASE_URL"]
OUT_PATH = "/app/ml/artifacts/pre_match_features.csv"


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
        match_base AS (
            SELECT
                m.match_id,
                m.start_time_utc,
                m.venue_id,
                m.team1_id,
                m.team2_id,
                m.winner_team_id,
                m.toss_winner_team_id,
                m.toss_decision
            FROM matches m
            WHERE m.competition = 'Indian Premier League'
              AND m.completed = true
              AND m.winner_team_id IS NOT NULL
        ),
        team_recent_form AS (
            SELECT
                x.match_id,
                x.team_id,
                AVG(x.won) AS recent_win_pct_5
            FROM (
                SELECT
                    mb.match_id,
                    t.team_id,
                    CASE WHEN mb.winner_team_id = t.team_id THEN 1.0 ELSE 0.0 END AS won,
                    ROW_NUMBER() OVER (
                        PARTITION BY mb.match_id, t.team_id
                        ORDER BY prev.start_time_utc DESC, prev.match_id DESC
                    ) AS rn
                FROM match_base mb
                JOIN LATERAL (
                    SELECT team_id
                    FROM (VALUES (mb.team1_id), (mb.team2_id)) AS tt(team_id)
                ) t ON TRUE
                JOIN match_base prev
                  ON prev.start_time_utc < mb.start_time_utc
                 AND (prev.team1_id = t.team_id OR prev.team2_id = t.team_id)
            ) x
            WHERE x.rn <= 5
            GROUP BY x.match_id, x.team_id
        ),
        venue_team_hist AS (
            SELECT
                mb.match_id,
                t.team_id,
                AVG(CASE WHEN prev.winner_team_id = t.team_id THEN 1.0 ELSE 0.0 END) AS venue_win_bias
            FROM match_base mb
            JOIN LATERAL (
                SELECT team_id
                FROM (VALUES (mb.team1_id), (mb.team2_id)) AS tt(team_id)
            ) t ON TRUE
            JOIN match_base prev
              ON prev.venue_id = mb.venue_id
             AND prev.start_time_utc < mb.start_time_utc
             AND (prev.team1_id = t.team_id OR prev.team2_id = t.team_id)
            GROUP BY mb.match_id, t.team_id
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
        xi_enriched AS (
            SELECT
                px.match_id,
                px.team_id,
                px.batting_order_hint,
                COALESCE(lpf.batting_form_score, 0) AS batting_form_score,
                COALESCE(lpf.bowling_form_score, 0) AS bowling_form_score,
                COALESCE(lpf.batting_runs_avg, 0) AS batting_runs_avg,
                COALESCE(lpf.batting_strike_rate, 0) AS batting_strike_rate,
                COALESCE(lpf.bowling_wkts_avg, 0) AS bowling_wkts_avg,
                COALESCE(lpf.bowling_economy, 0) AS bowling_economy
            FROM probable_xi px
            LEFT JOIN latest_player_form lpf
              ON px.player_id = lpf.player_id
        ),
        xi_agg AS (
            SELECT
                match_id,
                team_id,
                COUNT(*) AS xi_count,
                SUM(batting_form_score) AS xi_batting_form,
                SUM(bowling_form_score) AS xi_bowling_form,
                SUM(batting_runs_avg) AS xi_batting_runs_avg_sum,
                AVG(batting_strike_rate) AS xi_batting_strike_rate_avg,
                SUM(bowling_wkts_avg) AS xi_bowling_wkts_avg_sum,
                AVG(bowling_economy) AS xi_bowling_economy_avg,
                SUM(CASE WHEN batting_order_hint <= 3 THEN batting_form_score ELSE 0 END) AS top_order_strength,
                SUM(CASE WHEN batting_order_hint BETWEEN 4 AND 7 THEN batting_form_score ELSE 0 END) AS middle_order_strength,
                SUM(CASE WHEN bowling_form_score > 0 THEN bowling_form_score ELSE 0 END) AS bowling_options_strength,
                SUM(CASE WHEN bowling_economy > 0 THEN (10 - LEAST(bowling_economy, 10)) ELSE 0 END) AS death_bowling_strength
            FROM xi_enriched
            GROUP BY match_id, team_id
        )
        SELECT
            mb.match_id,
            mb.start_time_utc,
            mb.team1_id,
            mb.team2_id,
            mb.winner_team_id,
            mb.venue_id,
            mb.toss_winner_team_id,
            mb.toss_decision,

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

            COALESCE(rf1.recent_win_pct_5, 0.5) AS team1_recent_win_pct_5,
            COALESCE(rf2.recent_win_pct_5, 0.5) AS team2_recent_win_pct_5,
            COALESCE(vh1.venue_win_bias, 0.5) AS team1_venue_win_bias,
            COALESCE(vh2.venue_win_bias, 0.5) AS team2_venue_win_bias,

            COALESCE(x1.xi_count, 0) AS team1_xi_count,
            COALESCE(x2.xi_count, 0) AS team2_xi_count,
            COALESCE(x1.xi_batting_form, 0) AS team1_xi_batting_form,
            COALESCE(x2.xi_batting_form, 0) AS team2_xi_batting_form,
            COALESCE(x1.xi_bowling_form, 0) AS team1_xi_bowling_form,
            COALESCE(x2.xi_bowling_form, 0) AS team2_xi_bowling_form,
            COALESCE(x1.top_order_strength, 0) AS team1_top_order_strength,
            COALESCE(x2.top_order_strength, 0) AS team2_top_order_strength,
            COALESCE(x1.middle_order_strength, 0) AS team1_middle_order_strength,
            COALESCE(x2.middle_order_strength, 0) AS team2_middle_order_strength,
            COALESCE(x1.death_bowling_strength, 0) AS team1_death_bowling_strength,
            COALESCE(x2.death_bowling_strength, 0) AS team2_death_bowling_strength
        FROM match_base mb
        LEFT JOIN latest_team_form tf1 ON mb.team1_id = tf1.team_id
        LEFT JOIN latest_team_form tf2 ON mb.team2_id = tf2.team_id
        LEFT JOIN team_recent_form rf1 ON mb.match_id = rf1.match_id AND mb.team1_id = rf1.team_id
        LEFT JOIN team_recent_form rf2 ON mb.match_id = rf2.match_id AND mb.team2_id = rf2.team_id
        LEFT JOIN venue_team_hist vh1 ON mb.match_id = vh1.match_id AND mb.team1_id = vh1.team_id
        LEFT JOIN venue_team_hist vh2 ON mb.match_id = vh2.match_id AND mb.team2_id = vh2.team_id
        LEFT JOIN xi_agg x1 ON mb.match_id = x1.match_id AND mb.team1_id = x1.team_id
        LEFT JOIN xi_agg x2 ON mb.match_id = x2.match_id AND mb.team2_id = x2.team_id
        ORDER BY mb.start_time_utc ASC, mb.match_id ASC
    """)

    df = pd.read_sql(q, engine)

    if df.empty:
        print("No completed IPL matches found.")
        return

    df["elo_diff"] = df["team1_elo"] - df["team2_elo"]
    df["batting_strength_diff"] = df["team1_batting_strength"] - df["team2_batting_strength"]
    df["bowling_strength_diff"] = df["team1_bowling_strength"] - df["team2_bowling_strength"]
    df["all_rounder_balance_diff"] = df["team1_all_rounder_balance"] - df["team2_all_rounder_balance"]
    df["spin_strength_diff"] = df["team1_spin_strength"] - df["team2_spin_strength"]
    df["pace_strength_diff"] = df["team1_pace_strength"] - df["team2_pace_strength"]
    df["death_overs_strength_diff"] = df["team1_death_overs_strength"] - df["team2_death_overs_strength"]

    df["recent_win_pct_diff"] = df["team1_recent_win_pct_5"] - df["team2_recent_win_pct_5"]
    df["venue_win_bias_diff"] = df["team1_venue_win_bias"] - df["team2_venue_win_bias"]
    df["probable_xi_count_diff"] = df["team1_xi_count"] - df["team2_xi_count"]
    df["probable_xi_batting_form_diff"] = df["team1_xi_batting_form"] - df["team2_xi_batting_form"]
    df["probable_xi_bowling_form_diff"] = df["team1_xi_bowling_form"] - df["team2_xi_bowling_form"]
    df["top_order_strength_diff"] = df["team1_top_order_strength"] - df["team2_top_order_strength"]
    df["middle_order_strength_diff"] = df["team1_middle_order_strength"] - df["team2_middle_order_strength"]
    df["death_bowling_strength_diff"] = df["team1_death_bowling_strength"] - df["team2_death_bowling_strength"]

    df["team1_won"] = (df["winner_team_id"] == df["team1_id"]).astype(int)

    Path("/app/ml/artifacts").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Saved features to {OUT_PATH}")
    print(df[[
        "match_id",
        "elo_diff",
        "recent_win_pct_diff",
        "venue_win_bias_diff",
        "top_order_strength_diff",
        "middle_order_strength_diff",
        "death_bowling_strength_diff",
        "team1_won",
    ]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()