import os
from datetime import datetime, timedelta, timezone

import httpx
from sqlalchemy import create_engine, text

from etl.odds import (
    ODDS_API_BASE_URL,
    ODDS_API_KEY,
    ODDS_MARKETS,
    ODDS_REGIONS,
    ODDS_SPORT_KEY,
    has_real_api_key,
    normalize_team_name,
    parse_api_time,
)

DATABASE_URL = os.environ["DATABASE_URL"]
COMPETITION = "Indian Premier League"

CANONICAL_TEAM_NAMES = {
    "royalchallengersbengaluru": "Royal Challengers Bengaluru",
    "sunrisershyderabad": "Sunrisers Hyderabad",
    "chennaisuperkings": "Chennai Super Kings",
    "mumbaiindians": "Mumbai Indians",
    "kolkataknightriders": "Kolkata Knight Riders",
    "rajasthanroyals": "Rajasthan Royals",
    "delhicapitals": "Delhi Capitals",
    "punjabkings": "Punjab Kings",
    "gujarattitans": "Gujarat Titans",
    "lucknowsupergiants": "Lucknow Super Giants",
    "deccanchargers": "Deccan Chargers",
    "risingpunesupergiant": "Rising Pune Supergiant",
}

OFFICIAL_PHASE_1_FIXTURES = [
    ("2026-03-28T14:00:00+00:00", "Royal Challengers Bengaluru", "Sunrisers Hyderabad", "Bengaluru"),
    ("2026-03-29T14:00:00+00:00", "Mumbai Indians", "Kolkata Knight Riders", "Mumbai"),
    ("2026-03-30T14:00:00+00:00", "Rajasthan Royals", "Chennai Super Kings", "Guwahati"),
    ("2026-03-31T14:00:00+00:00", "Punjab Kings", "Gujarat Titans", "New Chandigarh"),
    ("2026-04-01T14:00:00+00:00", "Lucknow Super Giants", "Delhi Capitals", "Lucknow"),
    ("2026-04-02T14:00:00+00:00", "Kolkata Knight Riders", "Sunrisers Hyderabad", "Kolkata"),
    ("2026-04-03T14:00:00+00:00", "Chennai Super Kings", "Punjab Kings", "Chennai"),
    ("2026-04-04T10:00:00+00:00", "Delhi Capitals", "Mumbai Indians", "Delhi"),
    ("2026-04-04T14:00:00+00:00", "Gujarat Titans", "Rajasthan Royals", "Ahmedabad"),
    ("2026-04-05T10:00:00+00:00", "Sunrisers Hyderabad", "Lucknow Super Giants", "Hyderabad"),
    ("2026-04-05T14:00:00+00:00", "Royal Challengers Bengaluru", "Chennai Super Kings", "Bengaluru"),
    ("2026-04-06T14:00:00+00:00", "Kolkata Knight Riders", "Punjab Kings", "Kolkata"),
    ("2026-04-07T14:00:00+00:00", "Rajasthan Royals", "Mumbai Indians", "Guwahati"),
    ("2026-04-08T14:00:00+00:00", "Delhi Capitals", "Gujarat Titans", "Delhi"),
    ("2026-04-09T14:00:00+00:00", "Kolkata Knight Riders", "Lucknow Super Giants", "Kolkata"),
    ("2026-04-10T14:00:00+00:00", "Rajasthan Royals", "Royal Challengers Bengaluru", "Guwahati"),
    ("2026-04-11T10:00:00+00:00", "Punjab Kings", "Sunrisers Hyderabad", "New Chandigarh"),
    ("2026-04-11T14:00:00+00:00", "Chennai Super Kings", "Delhi Capitals", "Chennai"),
    ("2026-04-12T10:00:00+00:00", "Lucknow Super Giants", "Gujarat Titans", "Lucknow"),
    ("2026-04-12T14:00:00+00:00", "Mumbai Indians", "Royal Challengers Bengaluru", "Mumbai"),
]

OFFICIAL_FIXTURE_LOOKUP = {
    (
        datetime.fromisoformat(start_time_utc).astimezone(timezone.utc).isoformat(),
        normalize_team_name(team1_name),
        normalize_team_name(team2_name),
    ): {
        "venue_city": venue_city,
    }
    for start_time_utc, team1_name, team2_name, venue_city in OFFICIAL_PHASE_1_FIXTURES
}


def canonical_team_name(name: str) -> str:
    normalized = normalize_team_name(name)
    return CANONICAL_TEAM_NAMES.get(normalized, (name or "").strip())


def get_engine():
    return create_engine(DATABASE_URL, pool_pre_ping=True)


def attach_official_metadata(fixture: dict) -> dict:
    key = (
        fixture["start_time_utc"].astimezone(timezone.utc).isoformat(),
        normalize_team_name(fixture["team1_name"]),
        normalize_team_name(fixture["team2_name"]),
    )
    official = OFFICIAL_FIXTURE_LOOKUP.get(key)
    if not official:
        return fixture

    enriched = fixture.copy()
    enriched.update(official)
    return enriched


def fetch_live_fixture_events() -> list[dict]:
    if not has_real_api_key(ODDS_API_KEY):
        raise ValueError("ODDS_API_KEY is not configured")

    url = f"{ODDS_API_BASE_URL}/sports/{ODDS_SPORT_KEY}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": ODDS_REGIONS,
        "markets": ODDS_MARKETS,
        "oddsFormat": "decimal",
    }

    with httpx.Client(timeout=30.0) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        events = response.json()

    fixtures = []
    for event in events:
        home_team = canonical_team_name(event.get("home_team", ""))
        away_team = canonical_team_name(event.get("away_team", ""))
        start_time_utc = parse_api_time(event["commence_time"])
        fixture = (
            {
                "source_event_id": event.get("id"),
                "season": int(start_time_utc.year),
                "start_time_utc": start_time_utc,
                "team1_name": home_team,
                "team2_name": away_team,
            }
        )
        fixtures.append(attach_official_metadata(fixture))

    fixtures.sort(key=lambda row: (row["start_time_utc"], row["team1_name"], row["team2_name"]))
    return fixtures


def upsert_team(conn, name: str) -> int:
    result = conn.execute(
        text(
            """
            INSERT INTO teams (name)
            VALUES (:name)
            ON CONFLICT (name) DO UPDATE
            SET name = EXCLUDED.name
            RETURNING team_id
            """
        ),
        {"name": name},
    )
    return int(result.scalar())


def resolve_venue_id(conn, venue_city: str | None) -> int | None:
    if not venue_city:
        return None

    existing = conn.execute(
        text(
            """
            SELECT
                v.venue_id
            FROM venues v
            LEFT JOIN matches m
              ON m.venue_id = v.venue_id
            WHERE LOWER(COALESCE(v.city, '')) = LOWER(:venue_city)
               OR LOWER(v.name) = LOWER(:venue_city)
            GROUP BY v.venue_id
            ORDER BY COUNT(m.match_id) DESC, v.venue_id ASC
            LIMIT 1
            """
        ),
        {"venue_city": venue_city},
    ).scalar()
    if existing is not None:
        return int(existing)

    inserted = conn.execute(
        text(
            """
            INSERT INTO venues (name, city, country)
            VALUES (:name, :city, 'India')
            ON CONFLICT (name) DO UPDATE
            SET city = COALESCE(EXCLUDED.city, venues.city),
                country = COALESCE(EXCLUDED.country, venues.country)
            RETURNING venue_id
            """
        ),
        {"name": venue_city, "city": venue_city},
    ).scalar()
    return int(inserted)


def fetch_existing_scope_matches(conn, season: int, window_start_utc, window_end_utc) -> list[dict]:
    rows = conn.execute(
        text(
            """
            SELECT
                m.match_id,
                m.season,
                m.start_time_utc,
                m.venue_id,
                m.team1_id,
                m.team2_id,
                t1.name AS team1_name,
                t2.name AS team2_name
            FROM matches m
            JOIN teams t1 ON m.team1_id = t1.team_id
            JOIN teams t2 ON m.team2_id = t2.team_id
            WHERE m.competition = :competition
              AND m.season = :season
              AND COALESCE(m.completed, false) = false
              AND m.start_time_utc >= :window_start_utc
              AND m.start_time_utc <= :window_end_utc
            ORDER BY m.start_time_utc ASC, m.match_id ASC
            """
        ),
        {
            "competition": COMPETITION,
            "season": season,
            "window_start_utc": window_start_utc,
            "window_end_utc": window_end_utc,
        },
    ).mappings().all()
    return [dict(row) for row in rows]


def find_existing_fixture(source_fixture: dict, existing_matches: list[dict]) -> dict | None:
    source_keys = {
        normalize_team_name(source_fixture["team1_name"]),
        normalize_team_name(source_fixture["team2_name"]),
    }
    source_time = source_fixture["start_time_utc"]

    best_match = None
    best_seconds = None
    for candidate in existing_matches:
        candidate_keys = {
            normalize_team_name(candidate["team1_name"]),
            normalize_team_name(candidate["team2_name"]),
        }
        if candidate_keys != source_keys:
            continue
        delta_seconds = abs((candidate["start_time_utc"] - source_time).total_seconds())
        if delta_seconds > 18 * 3600:
            continue
        if best_seconds is None or delta_seconds < best_seconds:
            best_match = candidate
            best_seconds = delta_seconds
    return best_match


def update_match(conn, match_id: int, fixture: dict, team1_id: int, team2_id: int, venue_id: int | None):
    conn.execute(
        text(
            """
            UPDATE matches
            SET season = :season,
                match_type = 't20',
                start_time_utc = :start_time_utc,
                venue_id = :venue_id,
                team1_id = :team1_id,
                team2_id = :team2_id,
                completed = false,
                competition = :competition,
                toss_time_utc = NULL,
                toss_winner_team_id = NULL,
                toss_decision = NULL,
                winner_team_id = NULL,
                result_type = NULL,
                win_margin = NULL
            WHERE match_id = :match_id
            """
        ),
        {
            "match_id": match_id,
            "season": fixture["season"],
            "start_time_utc": fixture["start_time_utc"],
            "venue_id": venue_id,
            "team1_id": team1_id,
            "team2_id": team2_id,
            "competition": COMPETITION,
        },
    )


def insert_match(conn, fixture: dict, team1_id: int, team2_id: int, venue_id: int | None) -> int:
    result = conn.execute(
        text(
            """
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
                :season,
                't20',
                :start_time_utc,
                :venue_id,
                :team1_id,
                :team2_id,
                false,
                :competition
            )
            RETURNING match_id
            """
        ),
        {
            "season": fixture["season"],
            "start_time_utc": fixture["start_time_utc"],
            "venue_id": venue_id,
            "team1_id": team1_id,
            "team2_id": team2_id,
            "competition": COMPETITION,
        },
    )
    return int(result.scalar())


def delete_match_dependencies(conn, match_id: int):
    conn.execute(
        text(
            """
            DELETE FROM explanations
            WHERE prediction_id IN (
                SELECT prediction_id
                FROM predictions
                WHERE match_id = :match_id
            )
            """
        ),
        {"match_id": match_id},
    )
    for table_name in (
        "predictions",
        "odds_snapshots",
        "probable_xi",
        "confirmed_xi",
        "ball_by_ball",
        "player_match_stats",
        "team_match_stats",
    ):
        conn.execute(
            text(f"DELETE FROM {table_name} WHERE match_id = :match_id"),
            {"match_id": match_id},
        )
    conn.execute(text("DELETE FROM matches WHERE match_id = :match_id"), {"match_id": match_id})


def sync_upcoming_fixtures_from_odds(delete_stale: bool = True):
    fixtures = fetch_live_fixture_events()
    if not fixtures:
        print("sync_upcoming_fixtures_from_odds skipped: no live IPL events returned from Odds API")
        return

    season = int(fixtures[0]["season"])
    window_start_utc = fixtures[0]["start_time_utc"] - timedelta(days=14)
    window_end_utc = fixtures[-1]["start_time_utc"] + timedelta(days=14)
    engine = get_engine()

    inserted = 0
    updated = 0
    deleted = 0
    kept_match_ids: set[int] = set()

    with engine.begin() as conn:
        existing_matches = fetch_existing_scope_matches(conn, season, window_start_utc, window_end_utc)

        for fixture in fixtures:
            team1_id = upsert_team(conn, fixture["team1_name"])
            team2_id = upsert_team(conn, fixture["team2_name"])
            venue_id = resolve_venue_id(conn, fixture.get("venue_city"))

            existing = find_existing_fixture(fixture, existing_matches)
            if existing:
                update_match(conn, int(existing["match_id"]), fixture, team1_id, team2_id, venue_id)
                kept_match_ids.add(int(existing["match_id"]))
                updated += 1
                continue

            match_id = insert_match(conn, fixture, team1_id, team2_id, venue_id)
            kept_match_ids.add(match_id)
            inserted += 1

        if delete_stale:
            for existing in existing_matches:
                match_id = int(existing["match_id"])
                if match_id in kept_match_ids:
                    continue
                delete_match_dependencies(conn, match_id)
                deleted += 1

    print(
        "sync_upcoming_fixtures_from_odds complete: "
        f"fetched {len(fixtures)} live fixtures, inserted {inserted}, updated {updated}, deleted {deleted} stale matches"
    )
    for fixture in fixtures:
        print(
            f"{fixture['start_time_utc'].astimezone(timezone.utc).isoformat()} | "
            f"{fixture['team1_name']} vs {fixture['team2_name']} | "
            f"venue={fixture.get('venue_city') or 'unknown'}"
        )
