import os
from datetime import timezone

import pandas as pd
from sqlalchemy import create_engine, text

from ml.feature_builder import (
    build_candidate_rows,
    classify_pitch,
    prepare_context,
    select_lineup,
)

DATABASE_URL = os.environ["DATABASE_URL"]


def build_candidate_pool(context, team_id: int, season: int, cutoff_time_utc: pd.Timestamp) -> list[int]:
    squad_frame = context.squads_by_team_season.get((int(season), int(team_id)), pd.DataFrame())
    squad_ids = squad_frame["player_id"].dropna().astype(int).tolist() if not squad_frame.empty else []

    history_frame = context.player_history_by_team.get(int(team_id), pd.DataFrame())
    if history_frame.empty or "start_time_utc" not in history_frame.columns:
        return sorted(set(squad_ids))

    recent_team_history = history_frame[history_frame["start_time_utc"] < cutoff_time_utc].sort_values(
        "start_time_utc",
        ascending=False,
    )
    recent_player_ids = recent_team_history["player_id"].dropna().astype(int).drop_duplicates().head(18).tolist()
    return sorted(set(squad_ids).union(recent_player_ids))


def add_continuity_boosts(context, team_id: int, cutoff_time_utc: pd.Timestamp, candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    history_frame = context.player_history_by_team.get(int(team_id), pd.DataFrame())
    if history_frame.empty or "start_time_utc" not in history_frame.columns:
        candidates["continuity_boost"] = 0.0
        candidates["played_last_match"] = False
        return candidates

    recent = history_frame[history_frame["start_time_utc"] < cutoff_time_utc].sort_values(
        "start_time_utc",
        ascending=False,
    )
    if recent.empty:
        candidates["continuity_boost"] = 0.0
        candidates["played_last_match"] = False
        return candidates

    last_match_id = recent.iloc[0]["match_id"]
    appearance_last_8 = recent.head(120).groupby("player_id").size().to_dict()
    appearance_last_3_matches = (
        recent.drop_duplicates(subset=["match_id"]).head(3)["match_id"].tolist()
    )
    recent_last_3 = recent[recent["match_id"].isin(appearance_last_3_matches)]
    appearance_last_3 = recent_last_3.groupby("player_id").size().to_dict()
    last_seen = recent.groupby("player_id")["start_time_utc"].max().to_dict()

    boosts = []
    played_last_flags = []
    for _, row in candidates.iterrows():
        player_id = int(row["player_id"])
        boost = 0.0
        boost += 0.80 * min(appearance_last_8.get(player_id, 0), 8)
        boost += 1.40 * min(appearance_last_3.get(player_id, 0), 3)
        played_last = bool(
            not recent[(recent["player_id"] == player_id) & (recent["match_id"] == last_match_id)].empty
        )
        if played_last:
            boost += 2.0
        last_seen_time = last_seen.get(player_id)
        if last_seen_time is not None:
            gap_days = max(0.0, (cutoff_time_utc - last_seen_time).total_seconds() / 86400.0)
            boost += max(0.0, 10.0 - min(gap_days, 10.0)) * 0.18
        role = row.get("role")
        if role == "wk_batter":
            boost += 0.5
        if role == "all_rounder":
            boost += 0.35
        boosts.append(boost)
        played_last_flags.append(played_last)

    candidates = candidates.copy()
    candidates["continuity_boost"] = boosts
    candidates["played_last_match"] = played_last_flags
    candidates["combined_score"] = candidates["combined_score"] + candidates["continuity_boost"]
    return candidates


def assign_batting_order(lineup: pd.DataFrame) -> pd.DataFrame:
    if lineup.empty:
        return lineup

    lineup = lineup.copy()
    batting_role_priority = {
        "wk_batter": 1,
        "batter": 1,
        "all_rounder": 2,
        "bowler": 3,
    }
    lineup["batting_sort"] = lineup["role"].map(batting_role_priority).fillna(4)
    lineup["batting_score"] = (
        0.65 * lineup["batting_form_score"].fillna(0)
        + 0.20 * lineup["batting_runs_avg"].fillna(0)
        + 0.10 * (lineup["batting_strike_rate"].fillna(0) / 100.0)
        + 0.05 * (lineup["batting_boundary_pct"].fillna(0) * 100.0)
    )
    lineup = lineup.sort_values(
        ["batting_sort", "batting_score", "combined_score"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    lineup["batting_order_hint"] = range(1, len(lineup) + 1)
    return lineup


def choose_captain(lineup: pd.DataFrame) -> int | None:
    if lineup.empty:
        return None
    candidates = lineup.copy()
    candidates["captain_score"] = (
        0.55 * candidates["recent_matches_used"].fillna(0)
        + 0.25 * candidates["combined_score"].fillna(0)
        + 0.20 * candidates["continuity_boost"].fillna(0)
    )
    preferred = candidates[candidates["role"].isin(["all_rounder", "batter", "wk_batter"])]
    if not preferred.empty:
        candidates = preferred
    return int(candidates.sort_values("captain_score", ascending=False).iloc[0]["player_id"])


def main():
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    context = prepare_context(engine)
    matches_df = context.upcoming_matches.sort_values(["start_time_utc", "match_id"]).head(50).copy()

    if matches_df.empty:
        print("No future IPL matches found.")
        return

    rows = []
    for _, match_row in matches_df.iterrows():
        match_id = int(match_row["match_id"])
        pitch_type = classify_pitch(context.venue_names.get(match_row.get("venue_id"), ""))
        as_of_time_utc = pd.to_datetime(match_row["start_time_utc"], utc=True).to_pydatetime()
        season = int(match_row["season"])

        for team_id in [int(match_row["team1_id"]), int(match_row["team2_id"])]:
            player_ids = build_candidate_pool(
                context=context,
                team_id=team_id,
                season=season,
                cutoff_time_utc=match_row["start_time_utc"],
            )
            if not player_ids:
                continue

            candidates = build_candidate_rows(
                context=context,
                team_id=team_id,
                season=season,
                cutoff_time_utc=match_row["start_time_utc"],
                pitch_type=pitch_type,
                player_ids=player_ids,
            )
            candidates = add_continuity_boosts(
                context=context,
                team_id=team_id,
                cutoff_time_utc=match_row["start_time_utc"],
                candidates=candidates,
            )

            lineup = select_lineup(candidates, pitch_type)
            lineup = assign_batting_order(lineup)
            captain_id = choose_captain(lineup)

            wk_candidates = lineup[lineup["role"] == "wk_batter"].copy()
            wicketkeeper_id = None
            if not wk_candidates.empty:
                wicketkeeper_id = int(wk_candidates.sort_values("combined_score", ascending=False).iloc[0]["player_id"])
            elif not lineup.empty:
                wicketkeeper_id = int(lineup.sort_values("batting_score", ascending=False).iloc[0]["player_id"])

            confidence = min(
                0.95,
                0.58
                + 0.02 * lineup["recent_matches_used"].fillna(0).mean()
                + 0.03 * lineup["played_last_match"].fillna(False).mean(),
            )

            for _, player_row in lineup.iterrows():
                rows.append({
                    "match_id": match_id,
                    "team_id": team_id,
                    "player_id": int(player_row["player_id"]),
                    "is_captain": int(player_row["player_id"]) == captain_id,
                    "is_wicketkeeper": int(player_row["player_id"]) == wicketkeeper_id,
                    "source": "xi_v4_recent_continuity",
                    "as_of_time_utc": as_of_time_utc,
                    "batting_order_hint": int(player_row["batting_order_hint"]),
                    "confidence": float(confidence),
                })

    xi_df = pd.DataFrame(rows)
    if xi_df.empty:
        print("No XI generated.")
        return

    with engine.begin() as conn:
        conn.execute(text("""
            DELETE FROM probable_xi
            WHERE match_id IN (
                SELECT match_id
                FROM matches
                WHERE competition = 'Indian Premier League'
                  AND completed = false
            )
        """))

        for _, row in xi_df.iterrows():
            conn.execute(
                text("""
                    INSERT INTO probable_xi (
                        match_id,
                        team_id,
                        player_id,
                        is_captain,
                        is_wicketkeeper,
                        source,
                        as_of_time_utc,
                        batting_order_hint,
                        confidence
                    )
                    VALUES (
                        :match_id,
                        :team_id,
                        :player_id,
                        :is_captain,
                        :is_wicketkeeper,
                        :source,
                        :as_of_time_utc,
                        :batting_order_hint,
                        :confidence
                    )
                """),
                row.to_dict(),
            )

    print(f"Inserted XI rows: {len(xi_df)}")
    print(xi_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
