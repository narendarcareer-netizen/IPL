import os
import re
import httpx
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, text

DATABASE_URL = os.environ["DATABASE_URL"]

TEAM_PAGES = {
    "Royal Challengers Bengaluru": "https://www.iplt20.com/teams/royal-challengers-bengaluru/squad/2026",
    "Chennai Super Kings": "https://www.iplt20.com/teams/chennai-super-kings/squad/2026",
    "Mumbai Indians": "https://www.iplt20.com/teams/mumbai-indians/squad/2026",
    "Kolkata Knight Riders": "https://www.iplt20.com/teams/kolkata-knight-riders/squad/2026",
    "Delhi Capitals": "https://www.iplt20.com/teams/delhi-capitals/squad/2026",
    "Punjab Kings": "https://www.iplt20.com/teams/punjab-kings/squad/2026",
    "Rajasthan Royals": "https://www.iplt20.com/teams/rajasthan-royals/squad/2026",
    "Lucknow Super Giants": "https://www.iplt20.com/teams/lucknow-super-giants/squad/2026",
    "Gujarat Titans": "https://www.iplt20.com/teams/gujarat-titans/squad/2026",
    "Sunrisers Hyderabad": "https://www.iplt20.com/teams/sunrisers-hyderabad/squad/2026",
}


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


def upsert_player(conn, full_name: str):
    row = conn.execute(
        text("""
            INSERT INTO players (full_name)
            VALUES (:full_name)
            ON CONFLICT (full_name) DO UPDATE SET full_name = EXCLUDED.full_name
            RETURNING player_id
        """),
        {"full_name": full_name},
    )
    return row.scalar()


def insert_squad_row(conn, season: int, team_id: int, player_id: int, role: str, is_overseas: bool):
    conn.execute(
        text("""
            INSERT INTO squads (season, team_id, player_id, role, is_overseas)
            VALUES (:season, :team_id, :player_id, :role, :is_overseas)
            ON CONFLICT DO NOTHING
        """),
        {
            "season": season,
            "team_id": team_id,
            "player_id": player_id,
            "role": role,
            "is_overseas": is_overseas,
        },
    )


def is_likely_overseas(name: str) -> bool:
    indian_surnames = {
        "Kohli", "Patidar", "Padikkal", "Sharma", "Kumar", "Pandya", "Singh",
        "Yadav", "Dar", "Deswal", "Ostwal", "Malhotra", "Chouhan", "Iyer",
        "Jitesh", "Swapnil", "Bhuvneshwar", "Suyash", "Abhinandan", "Venkatesh",
        "Rajat", "Virat", "Devdutt", "Krunal", "Vicky", "Mangesh", "Satvik",
        "Rasikh", "Kanishk", "Vihaan"
    }
    parts = name.split()
    return not any(p in indian_surnames for p in parts)


def normalize_role(section: str, player_role: str) -> str:
    s = (section or "").strip().lower()
    p = (player_role or "").strip().lower()

    if "wk-batter" in p:
        return "wk_batter"
    if "batter" in s:
        return "batter"
    if "all rounder" in s or "all-rounder" in p:
        return "all_rounder"
    if "bowler" in s:
        return "bowler"
    return "unknown"


def main():
    engine = get_engine()
    season = 2026
    inserted = 0

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM squads WHERE season = :season"), {"season": season})

        for team_name, url in TEAM_PAGES.items():
            html = httpx.get(url, timeout=30).text
            soup = BeautifulSoup(html, "lxml")

            team_id = upsert_team(conn, team_name)

            texts = [t.strip() for t in soup.get_text("\n", strip=True).split("\n") if t.strip()]

            current_section = None
            i = 0
            seen_players = set()

            while i < len(texts):
                token = texts[i]

                if token in {"Batters", "All Rounders", "Bowlers"}:
                    current_section = token
                    i += 1
                    continue

                if current_section and i + 1 < len(texts):
                    name = texts[i]
                    role_line = texts[i + 1]

                    valid_role_lines = {"Batter", "WK-Batter", "All-Rounder", "Bowler"}

                    if role_line in valid_role_lines and name not in {
                        "Captain", "Coach", "Owner", "Venue", "Squad", "Fixtures", "Results",
                        "Videos", "News", "Archive", "Season 2026"
                    }:
                        if name not in seen_players:
                            player_id = upsert_player(conn, name)
                            role = normalize_role(current_section, role_line)
                            overseas = is_likely_overseas(name)
                            insert_squad_row(conn, season, team_id, player_id, role, overseas)
                            seen_players.add(name)
                            inserted += 1
                        i += 2
                        continue

                i += 1

    print(f"Inserted squad rows for season {season}: {inserted}")


if __name__ == "__main__":
    main()