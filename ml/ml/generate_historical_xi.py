import os
from sqlalchemy import create_engine, text

DATABASE_URL = os.environ["DATABASE_URL"]

def main():
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    with engine.begin() as conn:

        # Optional: clear old data (safe if rebuilding)
        conn.execute(text("DELETE FROM probable_xi"))

        print("Generating historical probable XI...")

        conn.execute(text("""
            INSERT INTO probable_xi (
                match_id,
                team_id,
                player_id,
                source,
                as_of_time_utc,
                batting_order_hint,
                confidence,
                created_at_utc
            )
            WITH ranked_players AS (
                SELECT
                    pms.match_id,
                    pms.team_id,
                    pms.player_id,
                    (COALESCE(pms.runs,0) + COALESCE(pms.wickets,0)*20) AS impact_score,
                    ROW_NUMBER() OVER (
                        PARTITION BY pms.match_id, pms.team_id
                        ORDER BY (COALESCE(pms.runs,0) + COALESCE(pms.wickets,0)*20) DESC
                    ) AS rnk,
                    m.start_time_utc
                FROM player_match_stats pms
                JOIN matches m ON pms.match_id = m.match_id
            )
            SELECT
                match_id,
                team_id,
                player_id,
                'historical_actual_performance' AS source,
                start_time_utc AS as_of_time_utc,
                rnk AS batting_order_hint,
                0.9 AS confidence,
                NOW()
            FROM ranked_players
            WHERE rnk <= 11
        """))

    print("Historical XI generated successfully!")

if __name__ == "__main__":
    main()