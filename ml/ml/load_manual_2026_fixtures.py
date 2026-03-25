import os
from sqlalchemy import create_engine, text

DATABASE_URL = os.environ["DATABASE_URL"]


FIXTURES_2026 = [
    {
        "team1": "Royal Challengers Bengaluru",
        "team2": "Kolkata Knight Riders",
        "start_time_utc": "2026-03-22T14:00:00+00:00",
        "venue": "M. Chinnaswamy Stadium",
        "city": "Bengaluru",
    },
    {
        "team1": "Sunrisers Hyderabad",
        "team2": "Rajasthan Royals",
        "start_time_utc": "2026-03-23T10:00:00+00:00",
        "venue": "Rajiv Gandhi International Stadium",
        "city": "Hyderabad",
    },
    {
        "team1": "Chennai Super Kings",
        "team2": "Mumbai Indians",
        "start_time_utc": "2026-03-23T14:00:00+00:00",
        "venue": "M. A. Chidambaram Stadium",
        "city": "Chennai",
    },
    {
        "team1": "Delhi Capitals",
        "team2": "Lucknow Super Giants",
        "start_time_utc": "2026-03-24T14:00:00+00:00",
        "venue": "Arun Jaitley Stadium",
        "city": "Delhi",
    },
    {
        "team1": "Punjab Kings",
        "team2": "Gujarat Titans",
        "start_time_utc": "2026-03-25T14:00:00+00:00",
        "venue": "PCA New Stadium",
        "city": "Mullanpur",
    },
]


def get_engine():
    return create_engine(DATABASE_URL, pool_pre_ping=True)


def upsert_team(conn, name: str):
    row = conn.execute(
        text("""
            INSERT INTO teams (name)
            VALUES (:name)
            ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
            RETURNING team_id
        """),
        {"name": name},
    )
    return row.scalar()


def upsert_venue(conn, name: str, city: str | None = None, country: str | None = "India"):
    row = conn.execute(
        text("""
            INSERT INTO venues (name, city, country)
            VALUES (:name, :city, :country)
            ON CONFLICT (name) DO UPDATE
            SET city = COALESCE(EXCLUDED.city, venues.city),
                country = COALESCE(EXCLUDED.country, venues.country)
            RETURNING venue_id
        """),
        {"name": name, "city": city, "country": country},
    )
    return row.scalar()


def insert_future_match(conn, fixture: dict):
    team1_id = upsert_team(conn, fixture["team1"])
    team2_id = upsert_team(conn, fixture["team2"])
    venue_id = upsert_venue(conn, fixture["venue"], fixture.get("city"), "India")

    conn.execute(
        text("""
            INSERT INTO matches (
                season,
                match_type,
                start_time_utc,
                venue_id,
                team1_id,
                team2_id,
                completed,
                competition
            )
            VALUES (
                2026,
                't20',
                :start_time_utc,
                :venue_id,
                :team1_id,
                :team2_id,
                false,
                'Indian Premier League'
            )
            ON CONFLICT DO NOTHING
        """),
        {
            "start_time_utc": fixture["start_time_utc"],
            "venue_id": venue_id,
            "team1_id": team1_id,
            "team2_id": team2_id,
        },
    )


def main():
    engine = get_engine()

    with engine.begin() as conn:
        for fixture in FIXTURES_2026:
            insert_future_match(conn, fixture)

    print(f"Inserted/checked {len(FIXTURES_2026)} manual 2026 fixtures.")


if __name__ == "__main__":
    main()