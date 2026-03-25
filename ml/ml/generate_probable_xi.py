import os
import pandas as pd
from sqlalchemy import create_engine, text

DATABASE_URL = os.environ["DATABASE_URL"]


def main():
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    matches_q = text("""
        SELECT
            match_id,
            team1_id,
            team2_id,
            start_time_utc
        FROM matches
        WHERE competition = 'Indian Premier League'
          AND completed = false
        ORDER BY start_time_utc ASC, match_id ASC
        LIMIT 50
    """)

    matches_df = pd.read_sql(matches_q, engine)

    if matches_df.empty:
        print("No future IPL matches found to generate probable XI for.")
        return

    players_q = text("""
        WITH squad_pool AS (
            SELECT
                s.season,
                s.team_id,
                s.player_id,
                COALESCE(s.role, 'unknown') AS squad_role,
                COALESCE(s.is_overseas, false) AS is_overseas
            FROM squads s
            WHERE s.season = 2026
        ),
        recent_usage AS (
            SELECT
                pms.team_id,
                pms.player_id,
                COUNT(*) AS recent_matches_used,
                MAX(m.start_time_utc) AS last_match_time
            FROM player_match_stats pms
            JOIN matches m
              ON pms.match_id = m.match_id
            WHERE m.competition = 'Indian Premier League'
              AND m.completed = true
            GROUP BY pms.team_id, pms.player_id
        ),
        latest_player_form AS (
            SELECT DISTINCT ON (player_id)
                player_id,
                COALESCE(batting_form_score, 0) AS batting_form_score,
                COALESCE(bowling_form_score, 0) AS bowling_form_score,
                COALESCE(batting_runs_avg, 0) AS batting_runs_avg,
                COALESCE(bowling_wkts_avg, 0) AS bowling_wkts_avg
            FROM player_form_snapshots
            ORDER BY player_id, as_of_time_utc DESC
        )
        SELECT
            sp.team_id,
            sp.player_id,
            sp.squad_role,
            sp.is_overseas,
            COALESCE(ru.recent_matches_used, 0) AS recent_matches_used,
            ru.last_match_time,
            COALESCE(lpf.batting_form_score, 0) AS batting_form_score,
            COALESCE(lpf.bowling_form_score, 0) AS bowling_form_score,
            COALESCE(lpf.batting_runs_avg, 0) AS batting_runs_avg,
            COALESCE(lpf.bowling_wkts_avg, 0) AS bowling_wkts_avg
        FROM squad_pool sp
        LEFT JOIN recent_usage ru
          ON sp.team_id = ru.team_id
         AND sp.player_id = ru.player_id
        LEFT JOIN latest_player_form lpf
          ON sp.player_id = lpf.player_id
    """)

    players_df = pd.read_sql(players_q, engine)

    if players_df.empty:
        print("No 2026 squad/player data found.")
        return

    players_df["combined_score"] = (
        0.35 * players_df["batting_form_score"].fillna(0)
        + 0.25 * players_df["bowling_form_score"].fillna(0)
        + 0.15 * players_df["batting_runs_avg"].fillna(0)
        + 0.15 * players_df["bowling_wkts_avg"].fillna(0)
        + 2.00 * players_df["recent_matches_used"].fillna(0)
    )

    rows = []

    for _, m in matches_df.iterrows():
        match_id = int(m["match_id"])
        as_of_time_utc = pd.to_datetime(m["start_time_utc"], utc=True).to_pydatetime()

        for team_id in [int(m["team1_id"]), int(m["team2_id"])]:
            team_players = players_df[players_df["team_id"] == team_id].copy()

            if team_players.empty:
                continue

            selected = []

            # 1 wicketkeeper if possible
            wk = team_players[team_players["squad_role"].isin(["wk_batter"])] \
                .sort_values("combined_score", ascending=False).head(1)
            if not wk.empty:
                selected.append(wk.iloc[0]["player_id"])

            # 3 more batters / wk-batters
            batters = team_players[
                team_players["squad_role"].isin(["batter", "wk_batter"])
                & ~team_players["player_id"].isin(selected)
            ].sort_values("combined_score", ascending=False).head(3)
            selected.extend(batters["player_id"].tolist())

            # 2 all-rounders
            all_rounders = team_players[
                (team_players["squad_role"] == "all_rounder")
                & ~team_players["player_id"].isin(selected)
            ].sort_values("combined_score", ascending=False).head(2)
            selected.extend(all_rounders["player_id"].tolist())

            # 4 bowlers
            bowlers = team_players[
                (team_players["squad_role"] == "bowler")
                & ~team_players["player_id"].isin(selected)
            ].sort_values("combined_score", ascending=False).head(4)
            selected.extend(bowlers["player_id"].tolist())

            # fill remaining spots by best score
            remaining_needed = 11 - len(selected)
            if remaining_needed > 0:
                fillers = team_players[
                    ~team_players["player_id"].isin(selected)
                ].sort_values("combined_score", ascending=False).head(remaining_needed)
                selected.extend(fillers["player_id"].tolist())

            # cap to 4 overseas by swapping out lowest-scoring overseas extras
            chosen = team_players[team_players["player_id"].isin(selected)].copy()
            overseas_count = int(chosen["is_overseas"].sum())

            if overseas_count > 4:
                chosen = chosen.sort_values(["is_overseas", "combined_score"], ascending=[False, True])
                overseas_to_remove = overseas_count - 4
                remove_ids = chosen[chosen["is_overseas"]].head(overseas_to_remove)["player_id"].tolist()

                selected = [pid for pid in selected if pid not in remove_ids]

                local_fillers = team_players[
                    (~team_players["is_overseas"])
                    & (~team_players["player_id"].isin(selected))
                ].sort_values("combined_score", ascending=False).head(overseas_to_remove)

                selected.extend(local_fillers["player_id"].tolist())

            final_df = team_players[team_players["player_id"].isin(selected)].copy()
            final_df = final_df.sort_values("combined_score", ascending=False).head(11)

            for rank, (_, r) in enumerate(final_df.iterrows(), start=1):
                rows.append({
                    "match_id": match_id,
                    "team_id": team_id,
                    "player_id": int(r["player_id"]),
                    "source": "squad_role_balanced_form",
                    "as_of_time_utc": as_of_time_utc,
                    "batting_order_hint": rank,
                    "confidence": 0.78,
                })

    xi_df = pd.DataFrame(rows)

    if xi_df.empty:
        print("No probable XI rows generated.")
        return

    with engine.begin() as conn:
        conn.execute(
            text("""
                DELETE FROM probable_xi
                WHERE match_id IN (
                    SELECT match_id
                    FROM matches
                    WHERE competition = 'Indian Premier League'
                      AND completed = false
                )
            """)
        )

        for _, row in xi_df.iterrows():
            conn.execute(
                text("""
                    INSERT INTO probable_xi (
                        match_id,
                        team_id,
                        player_id,
                        source,
                        as_of_time_utc,
                        batting_order_hint,
                        confidence
                    )
                    VALUES (
                        :match_id,
                        :team_id,
                        :player_id,
                        :source,
                        :as_of_time_utc,
                        :batting_order_hint,
                        :confidence
                    )
                """),
                row.to_dict(),
            )

    print(f"Inserted probable XI rows: {len(xi_df)}")
    print(xi_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()