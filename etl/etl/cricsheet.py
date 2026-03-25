from pathlib import Path
import json
import os
from sqlalchemy import create_engine, text

DATABASE_URL = os.environ["DATABASE_URL"]


def get_engine():
    return create_engine(DATABASE_URL, pool_pre_ping=True)


def upsert_team(conn, name: str):
    if not name:
        return None
    result = conn.execute(
        text("""
            INSERT INTO teams (name)
            VALUES (:name)
            ON CONFLICT (name) DO UPDATE
            SET name = EXCLUDED.name
            RETURNING team_id
        """),
        {"name": name},
    )
    return result.scalar()


def upsert_player(conn, name: str):
    if not name:
        return None
    result = conn.execute(
        text("""
            INSERT INTO players (full_name)
            VALUES (:name)
            ON CONFLICT (full_name) DO UPDATE
            SET full_name = EXCLUDED.full_name
            RETURNING player_id
        """),
        {"name": name},
    )
    return result.scalar()


def upsert_venue(conn, name: str, city: str | None = None, country: str | None = None):
    if not name:
        return None
    result = conn.execute(
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
    return result.scalar()


def insert_match(
    conn,
    season: int,
    start_time: str,
    venue_id: int,
    team1_id: int,
    team2_id: int,
    competition: str,
    winner_team_id: int | None,
    toss_winner_team_id: int | None,
    toss_decision: str | None,
    result_type: str | None,
    win_margin: int | None,
    completed: bool,
):
    result = conn.execute(
        text("""
            INSERT INTO matches (
                season,
                match_type,
                start_time_utc,
                venue_id,
                team1_id,
                team2_id,
                toss_winner_team_id,
                toss_decision,
                winner_team_id,
                result_type,
                win_margin,
                completed,
                competition
            )
            VALUES (
                :season,
                't20',
                :start_time,
                :venue_id,
                :team1_id,
                :team2_id,
                :toss_winner_team_id,
                :toss_decision,
                :winner_team_id,
                :result_type,
                :win_margin,
                :completed,
                :competition
            )
            ON CONFLICT DO NOTHING
            RETURNING match_id
        """),
        {
            "season": season,
            "start_time": start_time,
            "venue_id": venue_id,
            "team1_id": team1_id,
            "team2_id": team2_id,
            "toss_winner_team_id": toss_winner_team_id,
            "toss_decision": toss_decision,
            "winner_team_id": winner_team_id,
            "result_type": result_type,
            "win_margin": win_margin,
            "completed": completed,
            "competition": competition,
        },
    )

    match_id = result.scalar()
    if match_id:
        return match_id

    existing = conn.execute(
        text("""
            SELECT match_id
            FROM matches
            WHERE season = :season
              AND start_time_utc = :start_time
              AND team1_id = :team1_id
              AND team2_id = :team2_id
              AND competition = :competition
            LIMIT 1
        """),
        {
            "season": season,
            "start_time": start_time,
            "team1_id": team1_id,
            "team2_id": team2_id,
            "competition": competition,
        },
    ).scalar()

    return existing


def parse_match_file(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    info = data.get("info", {})
    teams = info.get("teams", [])
    if len(teams) != 2:
        return None

    venue_name = info.get("venue")
    city = info.get("city")
    dates = info.get("dates", [])
    start_time = f"{dates[0]}T00:00:00+00:00" if dates else None

    season_raw = info.get("season")
    try:
        season = int(str(season_raw)[:4]) if season_raw else None
    except Exception:
        season = None

    competition = (info.get("event") or {}).get("name") or info.get("competition") or ""

    toss = info.get("toss", {}) or {}
    toss_winner_name = toss.get("winner")
    toss_decision = toss.get("decision")

    outcome = info.get("outcome", {}) or {}
    winner_name = outcome.get("winner")
    result = outcome.get("result")
    by = outcome.get("by", {}) or {}

    result_type = None
    win_margin = None

    if "runs" in by:
        result_type = "runs"
        win_margin = by.get("runs")
    elif "wickets" in by:
        result_type = "wickets"
        win_margin = by.get("wickets")
    elif result:
        result_type = result

    completed = bool(winner_name or result)

    team_players = info.get("players", {}) or {}

    return {
        "team1_name": teams[0],
        "team2_name": teams[1],
        "venue_name": venue_name,
        "city": city,
        "country": "India",
        "start_time": start_time,
        "season": season,
        "competition": competition,
        "winner_name": winner_name,
        "toss_winner_name": toss_winner_name,
        "toss_decision": toss_decision,
        "result_type": result_type,
        "win_margin": win_margin,
        "completed": completed,
        "team1_players": team_players.get(teams[0], []),
        "team2_players": team_players.get(teams[1], []),
        "raw_data": data,
    }


def extract_player_stats(data, team1_name, team2_name, team1_id, team2_id, conn):
    innings = data.get("innings", [])
    stats = {}

    def ensure_stat(key):
        if key not in stats:
            stats[key] = {
                "runs": 0,
                "balls": 0,
                "fours": 0,
                "sixes": 0,
                "dismissed": False,
                "wickets": 0,
                "balls_bowled": 0,
                "runs_conceded": 0,
            }

    for inn in innings:
        innings_team_name = inn.get("team")

        if innings_team_name == team1_name:
            batting_team_id = team1_id
            bowling_team_id = team2_id
        elif innings_team_name == team2_name:
            batting_team_id = team2_id
            bowling_team_id = team1_id
        else:
            continue

        for over in inn.get("overs", []):
            for d in over.get("deliveries", []):
                batter = d.get("batter")
                bowler = d.get("bowler")

                runs = d.get("runs", {}).get("batter", 0)
                total_runs = d.get("runs", {}).get("total", 0)

                if batter:
                    batter_id = upsert_player(conn, batter)
                    batter_key = (batter_id, batting_team_id)
                    ensure_stat(batter_key)

                    stats[batter_key]["runs"] += runs
                    stats[batter_key]["balls"] += 1

                    if runs == 4:
                        stats[batter_key]["fours"] += 1
                    elif runs == 6:
                        stats[batter_key]["sixes"] += 1

                if bowler:
                    bowler_id = upsert_player(conn, bowler)
                    bowler_key = (bowler_id, bowling_team_id)
                    ensure_stat(bowler_key)

                    stats[bowler_key]["balls_bowled"] += 1
                    stats[bowler_key]["runs_conceded"] += total_runs

                for w in d.get("wickets", []):
                    player_out = w.get("player_out")
                    if player_out:
                        out_id = upsert_player(conn, player_out)
                        out_key = (out_id, batting_team_id)
                        ensure_stat(out_key)
                        stats[out_key]["dismissed"] = True

                    if bowler:
                        bowler_id = upsert_player(conn, bowler)
                        bowler_key = (bowler_id, bowling_team_id)
                        ensure_stat(bowler_key)
                        stats[bowler_key]["wickets"] += 1

    return stats


def load_cricsheet(path: str):
    base = Path(path)
    if not base.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    files = list(base.rglob("*.json"))
    if not files:
        raise ValueError(f"No JSON files found under: {path}")

    engine = get_engine()
    processed_matches = 0
    skipped_files = 0

    with engine.begin() as conn:
        for file_path in files:
            try:
                parsed = parse_match_file(file_path)
                if not parsed:
                    skipped_files += 1
                    continue

                if not parsed["season"] or not parsed["start_time"]:
                    skipped_files += 1
                    continue

                competition = (parsed.get("competition") or "").lower()
                if "indian premier league" not in competition and "ipl" not in competition:
                    skipped_files += 1
                    continue

                team1_id = upsert_team(conn, parsed["team1_name"])
                team2_id = upsert_team(conn, parsed["team2_name"])

                for player_name in parsed.get("team1_players", []):
                    upsert_player(conn, player_name)

                for player_name in parsed.get("team2_players", []):
                    upsert_player(conn, player_name)

                winner_team_id = None
                if parsed.get("winner_name") == parsed["team1_name"]:
                    winner_team_id = team1_id
                elif parsed.get("winner_name") == parsed["team2_name"]:
                    winner_team_id = team2_id

                toss_winner_team_id = None
                if parsed.get("toss_winner_name") == parsed["team1_name"]:
                    toss_winner_team_id = team1_id
                elif parsed.get("toss_winner_name") == parsed["team2_name"]:
                    toss_winner_team_id = team2_id

                venue_id = upsert_venue(
                    conn,
                    parsed["venue_name"],
                    parsed["city"],
                    parsed["country"],
                )

                match_id = insert_match(
                    conn,
                    season=parsed["season"],
                    start_time=parsed["start_time"],
                    venue_id=venue_id,
                    team1_id=team1_id,
                    team2_id=team2_id,
                    competition=parsed["competition"],
                    winner_team_id=winner_team_id,
                    toss_winner_team_id=toss_winner_team_id,
                    toss_decision=parsed["toss_decision"],
                    result_type=parsed["result_type"],
                    win_margin=parsed["win_margin"],
                    completed=parsed["completed"],
                )

                if not match_id:
                    skipped_files += 1
                    continue

                stats = extract_player_stats(
                    parsed["raw_data"],
                    parsed["team1_name"],
                    parsed["team2_name"],
                    team1_id,
                    team2_id,
                    conn,
                )

                for (player_id, team_id), s in stats.items():
                    conn.execute(
                        text("""
                            INSERT INTO player_match_stats (
                                match_id,
                                player_id,
                                team_id,
                                runs,
                                balls_faced,
                                fours,
                                sixes,
                                wickets,
                                overs_bowled,
                                runs_conceded
                            )
                            VALUES (
                                :match_id,
                                :player_id,
                                :team_id,
                                :runs,
                                :balls_faced,
                                :fours,
                                :sixes,
                                :wickets,
                                :overs_bowled,
                                :runs_conceded
                            )
                            ON CONFLICT (match_id, player_id) DO NOTHING
                        """),
                        {
                            "match_id": match_id,
                            "player_id": player_id,
                            "team_id": team_id,
                            "runs": s["runs"],
                            "balls_faced": s["balls"],
                            "fours": s["fours"],
                            "sixes": s["sixes"],
                            "wickets": s["wickets"],
                            "overs_bowled": round(s["balls_bowled"] / 6.0,1),
                            "runs_conceded": s["runs_conceded"],
                        },
                    )

                processed_matches += 1

            except Exception as e:
                skipped_files += 1
                print(f"Failed for {file_path.name}: {e}")

    print(f"Processed IPL matches: {processed_matches}")
    print(f"Skipped files: {skipped_files}")