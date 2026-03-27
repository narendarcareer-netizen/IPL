import os
import re
from datetime import datetime, timedelta, timezone

import httpx
from sqlalchemy import create_engine, text

DATABASE_URL = os.environ["DATABASE_URL"]
ODDS_API_KEY = (os.environ.get("ODDS_API_KEY") or "").strip()
ODDS_API_BASE_URL = (os.environ.get("ODDS_API_BASE_URL") or "https://api.the-odds-api.com/v4").strip().rstrip("/")
ODDS_SPORT_KEY = (os.environ.get("ODDS_SPORT_KEY") or "cricket_ipl").strip()
ODDS_REGIONS = os.environ.get("ODDS_REGIONS", "uk,eu,au")
ODDS_MARKETS = os.environ.get("ODDS_MARKETS", "h2h")

TEAM_ALIASES = {
    "royalchallengersbengaluru": "royalchallengersbengaluru",
    "royalchallengersbangalore": "royalchallengersbengaluru",
    "rcb": "royalchallengersbengaluru",
    "sunrisershyderabad": "sunrisershyderabad",
    "srh": "sunrisershyderabad",
    "chennaisuperkings": "chennaisuperkings",
    "csk": "chennaisuperkings",
    "mumbaiindians": "mumbaiindians",
    "mi": "mumbaiindians",
    "kolkataknightriders": "kolkataknightriders",
    "kkr": "kolkataknightriders",
    "rajasthanroyals": "rajasthanroyals",
    "rr": "rajasthanroyals",
    "delhicapitals": "delhicapitals",
    "delhidaredevils": "delhicapitals",
    "dc": "delhicapitals",
    "punjabkings": "punjabkings",
    "kingsxipunjab": "punjabkings",
    "pbks": "punjabkings",
    "kxip": "punjabkings",
    "gujarattitans": "gujarattitans",
    "gt": "gujarattitans",
    "lucknowsupergiants": "lucknowsupergiants",
    "lucknowsupergiants": "lucknowsupergiants",
    "lsg": "lucknowsupergiants",
    "deccanchargers": "deccanchargers",
    "risingpunesupergiant": "risingpunesupergiant",
    "rps": "risingpunesupergiant",
}


def normalize_team_name(name: str) -> str:
    compact = re.sub(r"[^a-z0-9]+", "", (name or "").lower())
    return TEAM_ALIASES.get(compact, compact)


def has_real_api_key(value: str) -> bool:
    if not value:
        return False
    upper = value.upper()
    return not (upper.startswith("YOUR_") or "PLACEHOLDER" in upper or value == "changeme")


def parse_api_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def fetch_upcoming_db_matches(engine):
    window_end = datetime.now(timezone.utc) + timedelta(days=14)
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT
                    m.match_id,
                    m.start_time_utc,
                    m.team1_id,
                    m.team2_id,
                    t1.name AS team1_name,
                    t2.name AS team2_name
                FROM matches m
                JOIN teams t1 ON m.team1_id = t1.team_id
                JOIN teams t2 ON m.team2_id = t2.team_id
                WHERE m.competition = 'Indian Premier League'
                  AND m.completed = false
                  AND m.start_time_utc <= :window_end
                ORDER BY m.start_time_utc ASC, m.match_id ASC
            """),
            {"window_end": window_end},
        ).mappings().all()
    return list(rows)


def fetch_historical_db_matches(engine, limit: int, season_from: int | None = None):
    params = {"limit": limit}
    season_filter = ""
    if season_from is not None:
        season_filter = "AND m.season >= :season_from"
        params["season_from"] = season_from

    with engine.begin() as conn:
        rows = conn.execute(
            text(f"""
                SELECT
                    m.match_id,
                    m.season,
                    m.start_time_utc,
                    m.team1_id,
                    m.team2_id,
                    t1.name AS team1_name,
                    t2.name AS team2_name
                FROM matches m
                JOIN teams t1 ON m.team1_id = t1.team_id
                JOIN teams t2 ON m.team2_id = t2.team_id
                WHERE m.competition = 'Indian Premier League'
                  AND m.completed = true
                  {season_filter}
                  AND NOT EXISTS (
                    SELECT 1
                    FROM odds_snapshots os
                    WHERE os.match_id = m.match_id
                  )
                ORDER BY m.start_time_utc DESC, m.match_id DESC
                LIMIT :limit
            """),
            params,
        ).mappings().all()
    return list(rows)


def find_matching_db_match(event: dict, db_matches: list[dict]):
    event_time = parse_api_time(event["commence_time"])
    event_keys = {normalize_team_name(event.get("home_team", "")), normalize_team_name(event.get("away_team", ""))}

    best_match = None
    best_seconds = None
    for candidate in db_matches:
        candidate_keys = {
            normalize_team_name(candidate["team1_name"]),
            normalize_team_name(candidate["team2_name"]),
        }
        if event_keys != candidate_keys:
            continue
        time_delta = abs((candidate["start_time_utc"] - event_time).total_seconds())
        if time_delta > 18 * 3600:
            continue
        if best_seconds is None or time_delta < best_seconds:
            best_match = candidate
            best_seconds = time_delta
    return best_match


def upsert_odds_snapshot(conn, match_row: dict, bookmaker: dict, outcome: dict, implied_prob_raw: float, implied_prob_norm: float, overround: float, as_of_time_utc: datetime):
    selection_key = normalize_team_name(outcome.get("name", ""))
    team1_key = normalize_team_name(match_row["team1_name"])
    team2_key = normalize_team_name(match_row["team2_name"])

    selection_team_id = None
    if selection_key == team1_key:
        selection_team_id = int(match_row["team1_id"])
    elif selection_key == team2_key:
        selection_team_id = int(match_row["team2_id"])
    else:
        return False

    conn.execute(
        text("""
            INSERT INTO odds_snapshots (
                match_id,
                provider,
                bookmaker,
                market_key,
                selection_team_id,
                odds_decimal,
                implied_prob_raw,
                implied_prob_norm,
                overround,
                captured_at_utc,
                as_of_time_utc
            )
            VALUES (
                :match_id,
                :provider,
                :bookmaker,
                :market_key,
                :selection_team_id,
                :odds_decimal,
                :implied_prob_raw,
                :implied_prob_norm,
                :overround,
                :captured_at_utc,
                :as_of_time_utc
            )
        """),
        {
            "match_id": int(match_row["match_id"]),
            "provider": "the_odds_api",
            "bookmaker": bookmaker.get("key") or bookmaker.get("title") or "unknown",
            "market_key": "h2h",
            "selection_team_id": selection_team_id,
            "odds_decimal": float(outcome["price"]),
            "implied_prob_raw": float(implied_prob_raw),
            "implied_prob_norm": float(implied_prob_norm),
            "overround": float(overround),
            "captured_at_utc": datetime.now(timezone.utc),
            "as_of_time_utc": as_of_time_utc,
        },
    )
    return True


def refresh_odds():
    if not has_real_api_key(ODDS_API_KEY):
        print("refresh_odds skipped: ODDS_API_KEY not configured")
        return

    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    db_matches = fetch_upcoming_db_matches(engine)
    if not db_matches:
        print("refresh_odds skipped: no upcoming IPL matches in DB")
        return

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

    inserted = 0
    matched = 0
    with engine.begin() as conn:
        for event in events:
            match_row = find_matching_db_match(event, db_matches)
            if not match_row:
                continue
            matched += 1
            as_of_time_utc = datetime.now(timezone.utc)
            for bookmaker in event.get("bookmakers", []):
                market = next((item for item in bookmaker.get("markets", []) if item.get("key") == "h2h"), None)
                if not market:
                    continue
                outcomes = [item for item in market.get("outcomes", []) if item.get("price")]
                if len(outcomes) < 2:
                    continue
                raw_probs = [1.0 / float(outcome["price"]) for outcome in outcomes]
                overround = sum(raw_probs)
                if overround <= 0:
                    continue
                for outcome, implied_prob_raw in zip(outcomes, raw_probs):
                    implied_prob_norm = implied_prob_raw / overround
                    if upsert_odds_snapshot(
                        conn=conn,
                        match_row=match_row,
                        bookmaker=bookmaker,
                        outcome=outcome,
                        implied_prob_raw=implied_prob_raw,
                        implied_prob_norm=implied_prob_norm,
                        overround=overround,
                        as_of_time_utc=as_of_time_utc,
                    ):
                        inserted += 1

    print(
        f"refresh_odds complete: matched {matched} events, inserted {inserted} odds rows using sport key {ODDS_SPORT_KEY}"
    )


def backfill_historical_odds(limit: int = 25, hours_before_start: int = 6, season_from: int | None = 2020):
    if not has_real_api_key(ODDS_API_KEY):
        print("backfill_historical_odds skipped: ODDS_API_KEY not configured")
        return

    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    db_matches = fetch_historical_db_matches(engine, limit=limit, season_from=season_from)
    if not db_matches:
        print("backfill_historical_odds skipped: no eligible completed matches without odds snapshots")
        return

    inserted = 0
    matched = 0
    failed = 0

    with httpx.Client(timeout=30.0) as client, engine.begin() as conn:
        for match_row in db_matches:
            snapshot_time = match_row["start_time_utc"] - timedelta(hours=hours_before_start)
            params = {
                "apiKey": ODDS_API_KEY,
                "regions": ODDS_REGIONS,
                "markets": ODDS_MARKETS,
                "oddsFormat": "decimal",
                "date": snapshot_time.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
            url = f"{ODDS_API_BASE_URL}/historical/sports/{ODDS_SPORT_KEY}/odds"
            try:
                response = client.get(url, params=params)
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                failed += 1
                body = exc.response.text[:300] if exc.response is not None else ""
                print(
                    f"backfill_historical_odds failed for match {match_row['match_id']} with status "
                    f"{exc.response.status_code if exc.response is not None else 'unknown'}: {body}"
                )
                if exc.response is not None and exc.response.status_code in (401, 402, 403):
                    print("Historical odds likely require a paid plan or different quota. Stopping backfill.")
                    break
                continue
            except Exception as exc:
                failed += 1
                print(f"backfill_historical_odds request error for match {match_row['match_id']}: {exc}")
                continue

            payload = response.json()
            events = payload.get("data") if isinstance(payload, dict) else payload
            if not isinstance(events, list):
                continue
            event = None
            if events:
                event = next(
                    (item for item in events if find_matching_db_match(item, [match_row])),
                    None,
                )
            if not event:
                continue
            matched += 1

            as_of_time_utc = snapshot_time.astimezone(timezone.utc)
            for bookmaker in event.get("bookmakers", []):
                market = next((item for item in bookmaker.get("markets", []) if item.get("key") == "h2h"), None)
                if not market:
                    continue
                outcomes = [item for item in market.get("outcomes", []) if item.get("price")]
                if len(outcomes) < 2:
                    continue
                raw_probs = [1.0 / float(outcome["price"]) for outcome in outcomes]
                overround = sum(raw_probs)
                if overround <= 0:
                    continue
                for outcome, implied_prob_raw in zip(outcomes, raw_probs):
                    implied_prob_norm = implied_prob_raw / overround
                    if upsert_odds_snapshot(
                        conn=conn,
                        match_row=match_row,
                        bookmaker=bookmaker,
                        outcome=outcome,
                        implied_prob_raw=implied_prob_raw,
                        implied_prob_norm=implied_prob_norm,
                        overround=overround,
                        as_of_time_utc=as_of_time_utc,
                    ):
                        inserted += 1

    print(
        "backfill_historical_odds complete: "
        f"requested {len(db_matches)} matches, matched {matched}, inserted {inserted}, failed {failed}"
    )
