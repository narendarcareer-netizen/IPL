from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from ml.feature_config import ALL_FEATURE_COLUMNS, OPTIONAL_CONTEXT_COLUMNS, normalize_stage

BASE_ELO = 1500.0
K_FACTOR = 20.0
DEFAULT_BOOK_PROB = 0.5
PITCH_SPIN_KEYWORDS = ("chennai", "delhi", "lucknow", "chepauk")
PITCH_PACE_KEYWORDS = ("mumbai", "bengaluru", "bangalore", "hyderabad", "mohali")
EXTERNAL_CONTEXT_PATH = Path("/data/external_match_context.csv")


@dataclass
class BuildContext:
    matches: pd.DataFrame
    completed_matches: pd.DataFrame
    upcoming_matches: pd.DataFrame
    team_history_by_team: dict[int, pd.DataFrame]
    player_history_by_team: dict[int, pd.DataFrame]
    player_form_by_player: dict[int, pd.DataFrame]
    probable_xi: pd.DataFrame
    confirmed_xi: pd.DataFrame
    squads_by_team_season: dict[tuple[int, int], pd.DataFrame]
    players_lookup: dict[int, dict]
    team_names: dict[int, str]
    venue_names: dict[int, str]
    odds: pd.DataFrame
    external_context: pd.DataFrame
    elo_pre_by_match: dict[int, dict]
    latest_completed_elo: dict[int, float]


def safe_div(numerator, denominator, default=0.0):
    if denominator is None or pd.isna(denominator):
        return default
    if float(denominator) == 0.0:
        return default
    return float(numerator) / float(denominator)


def overs_to_balls(overs_value) -> int:
    if overs_value is None or pd.isna(overs_value):
        return 0
    overs_float = float(overs_value)
    whole_overs = int(overs_float)
    partial_balls = int(round((overs_float - whole_overs) * 10))
    return whole_overs * 6 + partial_balls


def normalize_role(role_value) -> str:
    if role_value is None or pd.isna(role_value):
        return "unknown"
    role = str(role_value).strip().lower()
    mapping = {
        "wk": "wk_batter",
        "keeper": "wk_batter",
        "wicketkeeper": "wk_batter",
        "wk_batter": "wk_batter",
        "bat": "batter",
        "batsman": "batter",
        "batting": "batter",
        "batter": "batter",
        "bowl": "bowler",
        "bowler": "bowler",
        "ar": "all_rounder",
        "all_rounder": "all_rounder",
        "all-rounder": "all_rounder",
    }
    return mapping.get(role, role)


def classify_bowling_style(style_value) -> str:
    if style_value is None or pd.isna(style_value):
        return "unknown"
    style = str(style_value).strip().lower()
    if any(token in style for token in ("leg", "off", "orthodox", "spin", "slow")):
        return "spin"
    if style in {"os", "ls", "lbg", "ob"}:
        return "spin"
    if any(token in style for token in ("fast", "medium", "rf", "lf", "rm", "lm", "pace", "seam")):
        return "pace"
    return "unknown"


def classify_pitch(venue_name: str) -> str:
    name = (venue_name or "").lower()
    if any(token in name for token in PITCH_SPIN_KEYWORDS):
        return "spin"
    if any(token in name for token in PITCH_PACE_KEYWORDS):
        return "pace"
    return "neutral"


def compute_expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def load_table(engine: Engine, table_name: str) -> pd.DataFrame:
    return pd.read_sql(text(f"SELECT * FROM {table_name}"), engine)


def prepare_frame_dates(frame: pd.DataFrame, date_columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in date_columns:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], utc=True, errors="coerce")
    return out


def load_external_context() -> pd.DataFrame:
    if not EXTERNAL_CONTEXT_PATH.exists():
        return pd.DataFrame(columns=["match_id", *OPTIONAL_CONTEXT_COLUMNS])
    df = pd.read_csv(EXTERNAL_CONTEXT_PATH)
    if "match_id" not in df.columns:
        return pd.DataFrame(columns=["match_id", *OPTIONAL_CONTEXT_COLUMNS])
    for column in OPTIONAL_CONTEXT_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    return df[["match_id", *OPTIONAL_CONTEXT_COLUMNS]].drop_duplicates(subset=["match_id"])


def build_elo_history(completed_matches: pd.DataFrame) -> tuple[dict[int, dict], dict[int, float]]:
    ratings: dict[int, float] = {}
    pre_by_match: dict[int, dict] = {}

    ordered = completed_matches.sort_values(["start_time_utc", "match_id"]).reset_index(drop=True)
    for _, row in ordered.iterrows():
        team1_id = int(row["team1_id"])
        team2_id = int(row["team2_id"])
        team1_pre = ratings.get(team1_id, BASE_ELO)
        team2_pre = ratings.get(team2_id, BASE_ELO)
        pre_by_match[int(row["match_id"])] = {
            "team1_elo": team1_pre,
            "team2_elo": team2_pre,
        }

        winner_team_id = row.get("winner_team_id")
        if pd.isna(winner_team_id):
            continue

        team1_won = 1.0 if int(winner_team_id) == team1_id else 0.0
        expected_team1 = compute_expected_score(team1_pre, team2_pre)
        expected_team2 = 1.0 - expected_team1

        ratings[team1_id] = team1_pre + K_FACTOR * (team1_won - expected_team1)
        ratings[team2_id] = team2_pre + K_FACTOR * ((1.0 - team1_won) - expected_team2)

    return pre_by_match, ratings


def prepare_context(engine: Engine) -> BuildContext:
    matches = prepare_frame_dates(load_table(engine, "matches"), ["start_time_utc", "toss_time_utc"])
    teams = load_table(engine, "teams")
    venues = load_table(engine, "venues")
    players = load_table(engine, "players")
    squads = load_table(engine, "squads")
    player_match_stats = load_table(engine, "player_match_stats")
    team_match_stats = load_table(engine, "team_match_stats")
    player_form = prepare_frame_dates(load_table(engine, "player_form_snapshots"), ["as_of_time_utc", "snapshot_time_utc"])
    probable_xi = prepare_frame_dates(load_table(engine, "probable_xi"), ["as_of_time_utc", "captured_at_utc", "created_at_utc"])
    confirmed_xi = prepare_frame_dates(load_table(engine, "confirmed_xi"), ["as_of_time_utc", "captured_at_utc"])
    odds = prepare_frame_dates(load_table(engine, "odds_snapshots"), ["captured_at_utc", "as_of_time_utc"])
    external_context = load_external_context()

    matches = matches[matches.get("competition", "") == "Indian Premier League"].copy()
    matches = matches.sort_values(["start_time_utc", "match_id"]).reset_index(drop=True)
    completed_matches = matches[matches["completed"].fillna(False) & matches["winner_team_id"].notna()].copy()
    upcoming_matches = matches[~matches["completed"].fillna(False)].copy()

    player_history = player_match_stats.merge(
        matches[["match_id", "season", "start_time_utc"]],
        on="match_id",
        how="left",
    )
    player_history["balls_faced"] = player_history.get("balls_faced", 0).fillna(0)
    player_history["batting_strike_rate"] = player_history.apply(
        lambda row: safe_div(row.get("runs", 0) * 100.0, row.get("balls_faced", 0), 0.0),
        axis=1,
    )
    player_history["batting_boundary_pct"] = player_history.apply(
        lambda row: safe_div(row.get("fours", 0) + row.get("sixes", 0), row.get("balls_faced", 0), 0.0),
        axis=1,
    )
    player_history["balls_bowled"] = player_history.get("overs_bowled", 0).fillna(0).apply(overs_to_balls)
    player_history["bowling_economy"] = player_history.apply(
        lambda row: safe_div(row.get("runs_conceded", 0), row["balls_bowled"] / 6.0, 0.0),
        axis=1,
    )
    player_history = player_history.sort_values(["team_id", "start_time_utc", "match_id"])

    team_history = team_match_stats.merge(
        completed_matches[
            ["match_id", "season", "start_time_utc", "venue_id", "team1_id", "team2_id", "winner_team_id"]
        ],
        on="match_id",
        how="inner",
    )
    team_history["opponent_team_id"] = np.where(
        team_history["team_id"] == team_history["team1_id"],
        team_history["team2_id"],
        team_history["team1_id"],
    )
    team_history["won"] = (team_history["winner_team_id"] == team_history["team_id"]).astype(float)
    team_history["overs_faced_balls"] = team_history.get("overs_faced", 0).fillna(0).apply(overs_to_balls)
    team_history["run_rate"] = team_history.apply(
        lambda row: safe_div(row.get("runs_scored", 0), row["overs_faced_balls"] / 6.0, 0.0),
        axis=1,
    )
    team_history["wicket_margin"] = (
        team_history.get("wickets_taken", 0).fillna(0) - team_history.get("wickets_lost", 0).fillna(0)
    )
    team_history = team_history.sort_values(["team_id", "start_time_utc", "match_id"])

    player_form_defaults = {
        "matches_used": 0,
        "batting_form_score": 0.0,
        "bowling_form_score": 0.0,
        "batting_runs_avg": 0.0,
        "batting_strike_rate": 0.0,
        "batting_boundary_pct": 0.0,
        "bowling_wkts_avg": 0.0,
        "bowling_economy": 0.0,
    }
    for column, default in player_form_defaults.items():
        if column not in player_form.columns:
            player_form[column] = default
    player_form = player_form.sort_values(["player_id", "as_of_time_utc"])

    team_names = teams.set_index("team_id")["name"].to_dict() if not teams.empty else {}
    venue_names = venues.set_index("venue_id")["name"].to_dict() if not venues.empty else {}

    players_lookup = {}
    if not players.empty:
        players_frame = players.copy()
        if "bowling_style" not in players_frame.columns:
            players_frame["bowling_style"] = None
        if "country" not in players_frame.columns:
            players_frame["country"] = None
        players_frame["bowling_type"] = players_frame["bowling_style"].apply(classify_bowling_style)
        players_lookup = players_frame.set_index("player_id").to_dict(orient="index")

    if "role" not in squads.columns:
        squads["role"] = None
    if "is_overseas" not in squads.columns:
        squads["is_overseas"] = False
    squads_by_team_season = {
        (int(season), int(team_id)): g.copy()
        for (season, team_id), g in squads.groupby(["season", "team_id"], dropna=False)
    }

    player_form_by_player = {
        int(player_id): g.sort_values("as_of_time_utc").reset_index(drop=True)
        for player_id, g in player_form.groupby("player_id", dropna=False)
    }
    player_history_by_team = {
        int(team_id): g.sort_values("start_time_utc").reset_index(drop=True)
        for team_id, g in player_history.groupby("team_id", dropna=False)
    }
    team_history_by_team = {
        int(team_id): g.sort_values("start_time_utc").reset_index(drop=True)
        for team_id, g in team_history.groupby("team_id", dropna=False)
    }

    elo_pre_by_match, latest_completed_elo = build_elo_history(completed_matches)

    return BuildContext(
        matches=matches,
        completed_matches=completed_matches,
        upcoming_matches=upcoming_matches,
        team_history_by_team=team_history_by_team,
        player_history_by_team=player_history_by_team,
        player_form_by_player=player_form_by_player,
        probable_xi=probable_xi,
        confirmed_xi=confirmed_xi,
        squads_by_team_season=squads_by_team_season,
        players_lookup=players_lookup,
        team_names=team_names,
        venue_names=venue_names,
        odds=odds,
        external_context=external_context,
        elo_pre_by_match=elo_pre_by_match,
        latest_completed_elo=latest_completed_elo,
    )


def latest_player_snapshot(context: BuildContext, player_id: int, team_id: int, cutoff_time_utc: pd.Timestamp) -> dict:
    snapshot_frame = context.player_form_by_player.get(int(player_id))
    if snapshot_frame is not None and not snapshot_frame.empty:
        hist = snapshot_frame[snapshot_frame["as_of_time_utc"] <= cutoff_time_utc]
        if not hist.empty:
            row = hist.iloc[-1]
            return {
                "matches_used": int(row.get("matches_used", 0) or 0),
                "batting_form_score": float(row.get("batting_form_score", 0.0) or 0.0),
                "bowling_form_score": float(row.get("bowling_form_score", 0.0) or 0.0),
                "batting_runs_avg": float(row.get("batting_runs_avg", 0.0) or 0.0),
                "batting_strike_rate": float(row.get("batting_strike_rate", 0.0) or 0.0),
                "batting_boundary_pct": float(row.get("batting_boundary_pct", 0.0) or 0.0),
                "bowling_wkts_avg": float(row.get("bowling_wkts_avg", 0.0) or 0.0),
                "bowling_economy": float(row.get("bowling_economy", 0.0) or 0.0),
            }

    history_frame = context.player_history_by_team.get(int(team_id), pd.DataFrame())
    if history_frame.empty or "start_time_utc" not in history_frame.columns or "player_id" not in history_frame.columns:
        history_frame = pd.DataFrame(columns=["player_id", "start_time_utc"])
    hist = history_frame[
        (history_frame["player_id"] == int(player_id)) & (history_frame["start_time_utc"] < cutoff_time_utc)
    ].sort_values("start_time_utc", ascending=False).head(5)
    if hist.empty:
        return {
            "matches_used": 0,
            "batting_form_score": 0.0,
            "bowling_form_score": 0.0,
            "batting_runs_avg": 0.0,
            "batting_strike_rate": 0.0,
            "batting_boundary_pct": 0.0,
            "bowling_wkts_avg": 0.0,
            "bowling_economy": 0.0,
        }

    runs_avg = float(hist.get("runs", 0).fillna(0).mean())
    balls_faced = float(hist.get("balls_faced", 0).fillna(0).sum())
    strike_rate = safe_div(hist.get("runs", 0).fillna(0).sum() * 100.0, balls_faced, 0.0)
    boundaries = hist.get("fours", 0).fillna(0).sum() + hist.get("sixes", 0).fillna(0).sum()
    boundary_pct = safe_div(boundaries, balls_faced, 0.0)
    wickets_avg = float(hist.get("wickets", 0).fillna(0).mean())
    bowling_balls = hist.get("balls_bowled", 0).fillna(0).sum()
    economy = safe_div(hist.get("runs_conceded", 0).fillna(0).sum(), bowling_balls / 6.0, 0.0)
    batting_form = 0.60 * runs_avg + 0.25 * (strike_rate / 100.0) + 0.15 * (boundary_pct * 100.0)
    bowling_form = 0.65 * wickets_avg + 0.35 * max(0.0, 10.0 - economy)

    return {
        "matches_used": int(len(hist)),
        "batting_form_score": float(batting_form),
        "bowling_form_score": float(bowling_form),
        "batting_runs_avg": runs_avg,
        "batting_strike_rate": float(strike_rate),
        "batting_boundary_pct": float(boundary_pct),
        "bowling_wkts_avg": wickets_avg,
        "bowling_economy": float(economy),
    }


def infer_role(player_id: int, squad_row: pd.Series | None, player_history: pd.DataFrame) -> str:
    if squad_row is not None and "role" in squad_row.index:
        role = normalize_role(squad_row.get("role"))
        if role != "unknown":
            return role

    balls_faced = 0.0
    balls_bowled = 0.0
    if not player_history.empty and "balls_faced" in player_history.columns:
        balls_faced = float(player_history["balls_faced"].fillna(0).sum())
    if not player_history.empty and "balls_bowled" in player_history.columns:
        balls_bowled = float(player_history["balls_bowled"].fillna(0).sum())

    if balls_bowled >= 18 and balls_faced >= 18:
        return "all_rounder"
    if balls_bowled >= 18:
        return "bowler"
    if balls_faced > 0:
        return "batter"

    player_meta = squad_row.to_dict() if squad_row is not None else {}
    if player_meta.get("is_wicketkeeper"):
        return "wk_batter"
    return "batter"


def is_overseas_player(player_meta: dict, squad_row: pd.Series | None) -> bool:
    if squad_row is not None and "is_overseas" in squad_row.index and not pd.isna(squad_row["is_overseas"]):
        return bool(squad_row["is_overseas"])
    country = str(player_meta.get("country") or "").strip().lower()
    return bool(country) and country != "india"


def compute_candidate_score(snapshot: dict, role: str, bowling_type: str, pitch_type: str, last_played_days: float) -> float:
    score = (
        0.32 * snapshot["batting_form_score"]
        + 0.28 * snapshot["bowling_form_score"]
        + 0.10 * snapshot["batting_runs_avg"]
        + 0.08 * (snapshot["batting_strike_rate"] / 100.0)
        + 0.07 * (snapshot["batting_boundary_pct"] * 100.0)
        + 0.08 * snapshot["bowling_wkts_avg"]
        + 0.07 * max(0.0, 8.5 - snapshot["bowling_economy"])
        + 0.10 * min(snapshot["matches_used"], 5)
    )
    if role == "wk_batter":
        score *= 1.02
    if role == "all_rounder":
        score *= 1.08
    if pitch_type == "spin" and bowling_type == "spin":
        score *= 1.06
    if pitch_type == "pace" and bowling_type == "pace":
        score *= 1.06
    if last_played_days >= 0:
        score += max(0.0, 14.0 - min(last_played_days, 14.0)) * 0.15
    return float(score)


def select_lineup(candidates: pd.DataFrame, pitch_type: str) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()

    if pitch_type == "spin":
        target = {"wk_batter": 1, "batter": 4, "all_rounder": 3, "bowler": 3}
    elif pitch_type == "pace":
        target = {"wk_batter": 1, "batter": 4, "all_rounder": 2, "bowler": 4}
    else:
        target = {"wk_batter": 1, "batter": 4, "all_rounder": 2, "bowler": 3}

    ranked = candidates.sort_values(
        ["combined_score", "recent_matches_used", "batting_form_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    selected_ids: list[int] = []
    for role_name, count in target.items():
        role_pool = ranked[(ranked["role"] == role_name) & (~ranked["player_id"].isin(selected_ids))]
        if role_name == "wk_batter" and role_pool.empty:
            role_pool = ranked[~ranked["player_id"].isin(selected_ids)]
        selected_ids.extend(role_pool.head(count)["player_id"].astype(int).tolist())

    remaining_needed = max(0, 11 - len(selected_ids))
    if remaining_needed:
        filler_pool = ranked[~ranked["player_id"].isin(selected_ids)]
        selected_ids.extend(filler_pool.head(remaining_needed)["player_id"].astype(int).tolist())

    lineup = ranked[ranked["player_id"].isin(selected_ids)].copy()

    bowling_roles = {"bowler", "all_rounder"}
    if (lineup["role"].isin(bowling_roles)).sum() < 5:
        needed = 5 - int((lineup["role"].isin(bowling_roles)).sum())
        extra_pool = ranked[
            (ranked["role"].isin(bowling_roles)) & (~ranked["player_id"].isin(selected_ids))
        ].head(needed)
        selected_ids.extend(extra_pool["player_id"].astype(int).tolist())
        lineup = ranked[ranked["player_id"].isin(selected_ids)].copy()

    while lineup["is_overseas"].sum() > 4:
        removable = lineup[lineup["is_overseas"]].sort_values("combined_score", ascending=True)
        if removable.empty:
            break
        remove_id = int(removable.iloc[0]["player_id"])
        replacement = ranked[
            (~ranked["is_overseas"])
            & (~ranked["player_id"].isin(lineup["player_id"].astype(int).tolist()))
        ].head(1)
        lineup = lineup[lineup["player_id"] != remove_id]
        if not replacement.empty:
            lineup = pd.concat([lineup, replacement], ignore_index=True)

    lineup = lineup.drop_duplicates(subset=["player_id"]).copy()
    role_priority = {"wk_batter": 1, "batter": 2, "all_rounder": 3, "bowler": 4}
    if "batting_order_hint" in lineup.columns:
        lineup["sort_order"] = pd.to_numeric(lineup["batting_order_hint"], errors="coerce").fillna(99).astype(int)
        lineup = lineup.sort_values(["sort_order", "combined_score"], ascending=[True, False])
    else:
        lineup["sort_order"] = lineup["role"].map(role_priority).fillna(99)
        lineup = lineup.sort_values(["sort_order", "batting_form_score", "combined_score"], ascending=[True, False, False])
    lineup = lineup.head(11).reset_index(drop=True)
    lineup["batting_order_hint"] = range(1, len(lineup) + 1)
    return lineup


def summarize_lineup(lineup: pd.DataFrame) -> dict:
    if lineup.empty:
        return {
            "xi_count": 0.0,
            "batting_strength": 0.0,
            "bowling_strength": 0.0,
            "all_rounder_balance": 0.0,
            "spin_strength": 0.0,
            "pace_strength": 0.0,
            "death_overs_strength": 0.0,
            "powerplay_strength": 0.0,
            "xi_batting_form": 0.0,
            "xi_bowling_form": 0.0,
            "xi_runs_avg_sum": 0.0,
            "xi_strike_rate_avg": 0.0,
            "xi_boundary_pct_avg": 0.0,
            "xi_experience": 0.0,
            "top_order_strength": 0.0,
            "middle_order_strength": 0.0,
            "death_bowling_strength": 0.0,
        }

    bowling_econ_component = lineup["bowling_economy"].fillna(0).apply(lambda value: max(0.0, 10.0 - value)).sum()
    death_bowling_strength = float(
        bowling_econ_component + 0.75 * lineup["bowling_wkts_avg"].fillna(0).sum()
    )
    batting_strength = float(
        lineup["batting_form_score"].fillna(0).sum()
        + 0.35 * lineup["batting_runs_avg"].fillna(0).sum()
        + 0.15 * lineup["batting_boundary_pct"].fillna(0).mean() * 100.0
    )
    bowling_strength = float(
        lineup["bowling_form_score"].fillna(0).sum()
        + 0.50 * lineup["bowling_wkts_avg"].fillna(0).sum()
        + 0.20 * bowling_econ_component
    )
    spin_strength = float(
        lineup.loc[lineup["bowling_type"] == "spin", "bowling_form_score"].fillna(0).sum()
    )
    pace_strength = float(
        lineup.loc[lineup["bowling_type"] == "pace", "bowling_form_score"].fillna(0).sum()
    )
    return {
        "xi_count": float(len(lineup)),
        "batting_strength": batting_strength,
        "bowling_strength": bowling_strength,
        "all_rounder_balance": float(
            ((lineup["role"] == "all_rounder") | (
                (lineup["batting_form_score"].fillna(0) > 0.0)
                & (lineup["bowling_form_score"].fillna(0) > 0.0)
            )).sum()
        ),
        "spin_strength": spin_strength,
        "pace_strength": pace_strength,
        "death_overs_strength": death_bowling_strength,
        "powerplay_strength": float(0.60 * pace_strength + 0.40 * bowling_strength),
        "xi_batting_form": float(lineup["batting_form_score"].fillna(0).sum()),
        "xi_bowling_form": float(lineup["bowling_form_score"].fillna(0).sum()),
        "xi_runs_avg_sum": float(lineup["batting_runs_avg"].fillna(0).sum()),
        "xi_strike_rate_avg": float(lineup["batting_strike_rate"].fillna(0).mean()),
        "xi_boundary_pct_avg": float(lineup["batting_boundary_pct"].fillna(0).mean()),
        "xi_experience": float(lineup["recent_matches_used"].fillna(0).sum()),
        "top_order_strength": float(
            lineup.loc[lineup["batting_order_hint"] <= 3, "batting_form_score"].fillna(0).sum()
        ),
        "middle_order_strength": float(
            lineup.loc[
                lineup["batting_order_hint"].between(4, 7),
                "batting_form_score",
            ].fillna(0).sum()
        ),
        "death_bowling_strength": death_bowling_strength,
    }


def build_candidate_rows(
    context: BuildContext,
    team_id: int,
    season: int,
    cutoff_time_utc: pd.Timestamp,
    pitch_type: str,
    player_ids: list[int],
    preferred_order: list[int] | None = None,
) -> pd.DataFrame:
    history_frame = context.player_history_by_team.get(int(team_id), pd.DataFrame())
    if history_frame.empty or "start_time_utc" not in history_frame.columns or "player_id" not in history_frame.columns:
        history_frame = pd.DataFrame(columns=["player_id", "start_time_utc"])
    squad_frame = context.squads_by_team_season.get((int(season), int(team_id)), pd.DataFrame())
    squad_by_player = {}
    if not squad_frame.empty:
        squad_by_player = {int(row["player_id"]): row for _, row in squad_frame.iterrows()}

    rows = []
    order_lookup = {}
    if preferred_order:
        order_lookup = {int(player_id): idx + 1 for idx, player_id in enumerate(preferred_order)}

    for player_id in player_ids:
        recent_player_history = history_frame[
            (history_frame["player_id"] == int(player_id)) & (history_frame["start_time_utc"] < cutoff_time_utc)
        ].sort_values("start_time_utc", ascending=False).head(5)
        snapshot = latest_player_snapshot(context, int(player_id), int(team_id), cutoff_time_utc)
        squad_row = squad_by_player.get(int(player_id))
        player_meta = context.players_lookup.get(int(player_id), {})
        role = infer_role(int(player_id), squad_row, recent_player_history)
        bowling_type = str(player_meta.get("bowling_type") or "unknown")
        last_played_days = -1.0
        if not recent_player_history.empty:
            last_played_days = (
                (cutoff_time_utc - recent_player_history.iloc[0]["start_time_utc"]).total_seconds() / 86400.0
            )

        rows.append({
            "player_id": int(player_id),
            "role": role,
            "bowling_type": bowling_type,
            "is_overseas": is_overseas_player(player_meta, squad_row),
            "recent_matches_used": float(snapshot["matches_used"]),
            "batting_form_score": float(snapshot["batting_form_score"]),
            "bowling_form_score": float(snapshot["bowling_form_score"]),
            "batting_runs_avg": float(snapshot["batting_runs_avg"]),
            "batting_strike_rate": float(snapshot["batting_strike_rate"]),
            "batting_boundary_pct": float(snapshot["batting_boundary_pct"]),
            "bowling_wkts_avg": float(snapshot["bowling_wkts_avg"]),
            "bowling_economy": float(snapshot["bowling_economy"]),
            "batting_order_hint": order_lookup.get(int(player_id)),
            "combined_score": compute_candidate_score(
                snapshot=snapshot,
                role=role,
                bowling_type=bowling_type,
                pitch_type=pitch_type,
                last_played_days=last_played_days,
            ),
        })

    return pd.DataFrame(rows)


def add_continuity_boosts(
    context: BuildContext,
    team_id: int,
    cutoff_time_utc: pd.Timestamp,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    history_frame = context.player_history_by_team.get(int(team_id), pd.DataFrame())
    if history_frame.empty or "start_time_utc" not in history_frame.columns:
        out = candidates.copy()
        out["continuity_boost"] = 0.0
        out["played_last_match"] = False
        return out

    recent = history_frame[history_frame["start_time_utc"] < cutoff_time_utc].sort_values(
        "start_time_utc",
        ascending=False,
    )
    if recent.empty:
        out = candidates.copy()
        out["continuity_boost"] = 0.0
        out["played_last_match"] = False
        return out

    last_match_id = recent.iloc[0]["match_id"]
    appearance_last_8 = recent.head(120).groupby("player_id").size().to_dict()
    recent_match_ids = recent.drop_duplicates(subset=["match_id"]).head(3)["match_id"].tolist()
    recent_last_3 = recent[recent["match_id"].isin(recent_match_ids)]
    appearance_last_3 = recent_last_3.groupby("player_id").size().to_dict()
    last_seen = recent.groupby("player_id")["start_time_utc"].max().to_dict()

    out = candidates.copy()
    boosts = []
    last_flags = []
    for _, row in out.iterrows():
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
        last_flags.append(played_last)

    out["continuity_boost"] = boosts
    out["played_last_match"] = last_flags
    out["combined_score"] = out["combined_score"] + out["continuity_boost"]
    return out


def build_historical_lineup_summary(
    context: BuildContext,
    team_id: int,
    season: int,
    cutoff_time_utc: pd.Timestamp,
    pitch_type: str,
) -> dict:
    team_history = context.team_history_by_team.get(int(team_id), pd.DataFrame())
    if team_history.empty or "start_time_utc" not in team_history.columns:
        team_history = pd.DataFrame(columns=["match_id", "start_time_utc"])
    previous_matches = team_history[team_history["start_time_utc"] < cutoff_time_utc].sort_values(
        "start_time_utc",
        ascending=False,
    )
    recent_match_ids = previous_matches["match_id"].drop_duplicates().head(8).astype(int).tolist()

    history_frame = context.player_history_by_team.get(int(team_id), pd.DataFrame())
    candidate_pool = history_frame[history_frame["match_id"].isin(recent_match_ids)].copy()
    player_ids = candidate_pool["player_id"].dropna().astype(int).drop_duplicates().tolist()

    squad_frame = context.squads_by_team_season.get((int(season), int(team_id)), pd.DataFrame())
    if not squad_frame.empty:
        player_ids = sorted(set(player_ids).union(squad_frame["player_id"].dropna().astype(int).tolist()))

    if not player_ids:
        historical_players = history_frame[history_frame["start_time_utc"] < cutoff_time_utc]
        player_ids = historical_players["player_id"].dropna().astype(int).drop_duplicates().tolist()

    candidates = build_candidate_rows(
        context=context,
        team_id=int(team_id),
        season=int(season),
        cutoff_time_utc=cutoff_time_utc,
        pitch_type=pitch_type,
        player_ids=player_ids,
    )
    candidates = add_continuity_boosts(
        context=context,
        team_id=int(team_id),
        cutoff_time_utc=cutoff_time_utc,
        candidates=candidates,
    )
    lineup = select_lineup(candidates, pitch_type)
    return summarize_lineup(lineup)


def build_upcoming_lineup_summary(
    context: BuildContext,
    match_id: int,
    team_id: int,
    season: int,
    cutoff_time_utc: pd.Timestamp,
    pitch_type: str,
) -> dict:
    probable_xi = context.probable_xi.copy()
    if probable_xi.empty or "as_of_time_utc" not in probable_xi.columns:
        return build_historical_lineup_summary(context, team_id, season, cutoff_time_utc, pitch_type)

    xi_rows = probable_xi[
        (probable_xi["match_id"] == int(match_id))
        & (probable_xi["team_id"] == int(team_id))
        & (probable_xi["as_of_time_utc"] <= cutoff_time_utc)
    ].copy()

    if xi_rows.empty:
        return build_historical_lineup_summary(context, team_id, season, cutoff_time_utc, pitch_type)

    latest_snapshot_time = xi_rows["as_of_time_utc"].max()
    xi_rows = xi_rows[xi_rows["as_of_time_utc"] == latest_snapshot_time].copy()
    if "batting_order_hint" in xi_rows.columns:
        xi_rows = xi_rows.sort_values(["batting_order_hint", "probable_xi_id"])
    player_ids = xi_rows["player_id"].dropna().astype(int).tolist()
    preferred_order = player_ids

    candidates = build_candidate_rows(
        context=context,
        team_id=int(team_id),
        season=int(season),
        cutoff_time_utc=cutoff_time_utc,
        pitch_type=pitch_type,
        player_ids=player_ids,
        preferred_order=preferred_order,
    )
    lineup = select_lineup(candidates, pitch_type)
    return summarize_lineup(lineup)


def latest_lineup_rows(
    lineup_frame: pd.DataFrame,
    match_id: int,
    team_id: int,
    cutoff_time_utc: pd.Timestamp,
) -> pd.DataFrame:
    if lineup_frame.empty or "as_of_time_utc" not in lineup_frame.columns:
        return pd.DataFrame()

    rows = lineup_frame[
        (lineup_frame["match_id"] == int(match_id))
        & (lineup_frame["team_id"] == int(team_id))
        & (lineup_frame["as_of_time_utc"] <= cutoff_time_utc)
    ].copy()
    if rows.empty:
        return rows

    if "captured_at_utc" in rows.columns and rows["captured_at_utc"].notna().any():
        latest_captured = rows["captured_at_utc"].dropna().max()
        rows = rows[rows["captured_at_utc"] == latest_captured].copy()

    latest_asof = rows["as_of_time_utc"].max()
    rows = rows[rows["as_of_time_utc"] == latest_asof].copy()
    return rows


def probable_order_for_match(
    context: BuildContext,
    match_id: int,
    team_id: int,
    cutoff_time_utc: pd.Timestamp,
) -> list[int]:
    rows = latest_lineup_rows(context.probable_xi, match_id, team_id, cutoff_time_utc)
    if rows.empty:
        return []

    sort_columns = []
    if "batting_order_hint" in rows.columns:
        sort_columns.append("batting_order_hint")
    if "probable_xi_id" in rows.columns:
        sort_columns.append("probable_xi_id")
    if sort_columns:
        rows = rows.sort_values(sort_columns)

    return rows["player_id"].dropna().astype(int).tolist()


def confirmed_xi_available(
    context: BuildContext,
    match_id: int,
    team_id: int,
    cutoff_time_utc: pd.Timestamp,
) -> bool:
    return not latest_lineup_rows(context.confirmed_xi, match_id, team_id, cutoff_time_utc).empty


def build_historical_confirmed_lineup_summary(
    context: BuildContext,
    match_id: int,
    team_id: int,
    season: int,
    cutoff_time_utc: pd.Timestamp,
    pitch_type: str,
) -> dict:
    history_frame = context.player_history_by_team.get(int(team_id), pd.DataFrame())
    actual_rows = history_frame[history_frame["match_id"] == int(match_id)].copy()
    player_ids = actual_rows["player_id"].dropna().astype(int).drop_duplicates().tolist()
    if not player_ids:
        return build_historical_lineup_summary(context, team_id, season, cutoff_time_utc, pitch_type)

    preferred_order = probable_order_for_match(context, match_id, team_id, cutoff_time_utc)
    ordered_player_ids = [player_id for player_id in preferred_order if player_id in set(player_ids)]
    ordered_player_ids.extend(player_id for player_id in player_ids if player_id not in ordered_player_ids)

    candidates = build_candidate_rows(
        context=context,
        team_id=int(team_id),
        season=int(season),
        cutoff_time_utc=cutoff_time_utc,
        pitch_type=pitch_type,
        player_ids=ordered_player_ids,
        preferred_order=ordered_player_ids,
    )
    candidates = add_continuity_boosts(
        context=context,
        team_id=int(team_id),
        cutoff_time_utc=cutoff_time_utc,
        candidates=candidates,
    )
    lineup = select_lineup(candidates, pitch_type)
    return summarize_lineup(lineup)


def build_upcoming_confirmed_lineup_summary(
    context: BuildContext,
    match_id: int,
    team_id: int,
    season: int,
    cutoff_time_utc: pd.Timestamp,
    pitch_type: str,
) -> dict:
    rows = latest_lineup_rows(context.confirmed_xi, match_id, team_id, cutoff_time_utc)
    if rows.empty:
        return build_upcoming_lineup_summary(context, match_id, team_id, season, cutoff_time_utc, pitch_type)

    player_ids = rows["player_id"].dropna().astype(int).tolist()
    preferred_order = probable_order_for_match(context, match_id, team_id, cutoff_time_utc)
    ordered_player_ids = [player_id for player_id in preferred_order if player_id in set(player_ids)]
    ordered_player_ids.extend(player_id for player_id in player_ids if player_id not in ordered_player_ids)

    candidates = build_candidate_rows(
        context=context,
        team_id=int(team_id),
        season=int(season),
        cutoff_time_utc=cutoff_time_utc,
        pitch_type=pitch_type,
        player_ids=ordered_player_ids,
        preferred_order=ordered_player_ids,
    )
    lineup = select_lineup(candidates, pitch_type)
    return summarize_lineup(lineup)


def toss_available(match_row: pd.Series) -> bool:
    return pd.notna(match_row.get("toss_winner_team_id")) and bool(str(match_row.get("toss_decision") or "").strip())


def batting_first_team_id(match_row: pd.Series) -> int | None:
    toss_winner = match_row.get("toss_winner_team_id")
    decision = str(match_row.get("toss_decision") or "").strip().lower()
    if pd.isna(toss_winner) or not decision:
        return None

    toss_winner = int(toss_winner)
    team1_id = int(match_row["team1_id"])
    team2_id = int(match_row["team2_id"])
    if decision == "bat":
        return toss_winner
    if decision in {"field", "bowl"}:
        return team2_id if toss_winner == team1_id else team1_id
    return None


def venue_chasing_bias(context: BuildContext, venue_id, cutoff_time_utc: pd.Timestamp) -> float:
    if venue_id is None or pd.isna(venue_id):
        return 0.0

    venue_matches = context.completed_matches[
        (context.completed_matches["venue_id"] == venue_id)
        & (context.completed_matches["start_time_utc"] < cutoff_time_utc)
        & context.completed_matches["toss_winner_team_id"].notna()
        & context.completed_matches["toss_decision"].notna()
        & context.completed_matches["winner_team_id"].notna()
    ].sort_values("start_time_utc", ascending=False).head(25)

    if venue_matches.empty:
        return 0.0

    chasing_wins = []
    for _, venue_match in venue_matches.iterrows():
        batting_first_team = batting_first_team_id(venue_match)
        if batting_first_team is None:
            continue
        chasing_wins.append(float(int(venue_match["winner_team_id"]) != int(batting_first_team)))

    if not chasing_wins:
        return 0.0

    return float(np.mean(chasing_wins) - 0.5)


def compute_toss_features(
    context: BuildContext,
    match_row: pd.Series,
    stage: str,
    cutoff_time_utc: pd.Timestamp,
) -> dict:
    if stage == "pre_toss" or not toss_available(match_row):
        return {
            "team1_won_toss": 0.0,
            "team1_bats_first": 0.0,
            "team1_chasing_bias_edge": 0.0,
            "venue_chasing_bias": 0.0,
        }

    team1_id = int(match_row["team1_id"])
    toss_winner = int(match_row["toss_winner_team_id"])
    batting_first_team = batting_first_team_id(match_row)
    team1_bats_first = 0.0
    if batting_first_team is not None:
        team1_bats_first = 1.0 if int(batting_first_team) == team1_id else -1.0

    chasing_bias = venue_chasing_bias(context, match_row.get("venue_id"), cutoff_time_utc)
    team1_chasing_bias_edge = float(chasing_bias if team1_bats_first < 0 else -chasing_bias if team1_bats_first > 0 else 0.0)

    return {
        "team1_won_toss": 1.0 if toss_winner == team1_id else -1.0,
        "team1_bats_first": float(team1_bats_first),
        "team1_chasing_bias_edge": float(team1_chasing_bias_edge),
        "venue_chasing_bias": float(chasing_bias),
    }


def summarize_team_history(
    context: BuildContext,
    team_id: int,
    cutoff_time_utc: pd.Timestamp,
    venue_id,
    elo_rating: float,
) -> dict:
    team_history = context.team_history_by_team.get(int(team_id), pd.DataFrame())
    if team_history.empty or "start_time_utc" not in team_history.columns:
        return {
            "elo_rating": float(elo_rating),
            "recent_win_pct": 0.5,
            "venue_win_bias": 0.5,
            "rest_days": 7.0,
            "recent_run_rate": 0.0,
            "recent_wicket_margin": 0.0,
        }
    recent = team_history[team_history["start_time_utc"] < cutoff_time_utc].sort_values(
        "start_time_utc",
        ascending=False,
    )
    last_five = recent.head(5)
    venue_recent = recent[recent["venue_id"] == venue_id].head(8)

    rest_days = 7.0
    if not recent.empty:
        rest_days = max(
            0.0,
            (cutoff_time_utc - recent.iloc[0]["start_time_utc"]).total_seconds() / 86400.0,
        )

    return {
        "elo_rating": float(elo_rating),
        "recent_win_pct": float(last_five["won"].mean()) if not last_five.empty else 0.5,
        "venue_win_bias": float(venue_recent["won"].mean()) if not venue_recent.empty else 0.5,
        "rest_days": float(rest_days),
        "recent_run_rate": float(last_five["run_rate"].mean()) if not last_five.empty else 0.0,
        "recent_wicket_margin": float(last_five["wicket_margin"].mean()) if not last_five.empty else 0.0,
    }


def head_to_head_win_pct(context: BuildContext, team1_id: int, team2_id: int, cutoff_time_utc: pd.Timestamp) -> float:
    matches = context.completed_matches
    pair_matches = matches[
        (matches["start_time_utc"] < cutoff_time_utc)
        & (
            ((matches["team1_id"] == int(team1_id)) & (matches["team2_id"] == int(team2_id)))
            | ((matches["team1_id"] == int(team2_id)) & (matches["team2_id"] == int(team1_id)))
        )
    ].sort_values("start_time_utc", ascending=False).head(5)
    if pair_matches.empty:
        return 0.5
    return float((pair_matches["winner_team_id"] == int(team1_id)).mean())


def compute_bookmaker_features(
    context: BuildContext,
    match_id: int,
    team1_id: int,
    team2_id: int,
    cutoff_time_utc: pd.Timestamp,
) -> dict:
    odds = context.odds
    if odds.empty:
        return {
            "bookmaker_prob_team1": DEFAULT_BOOK_PROB,
            "bookmaker_prob_team2": DEFAULT_BOOK_PROB,
            "bookmaker_prob_diff": 0.0,
            "bookmaker_market_confidence": 0.0,
        }

    odds = odds[
        (odds["match_id"] == int(match_id))
        & (odds["as_of_time_utc"] <= cutoff_time_utc)
        & (odds.get("market_key", "h2h") == "h2h")
    ].copy()
    if odds.empty:
        return {
            "bookmaker_prob_team1": DEFAULT_BOOK_PROB,
            "bookmaker_prob_team2": DEFAULT_BOOK_PROB,
            "bookmaker_prob_diff": 0.0,
            "bookmaker_market_confidence": 0.0,
        }

    latest_asof = odds["as_of_time_utc"].max()
    odds = odds[odds["as_of_time_utc"] == latest_asof].copy()

    if "implied_prob_norm" not in odds.columns:
        odds["implied_prob_norm"] = np.nan
    if "bookmaker" not in odds.columns:
        odds["bookmaker"] = "unknown"
    if "implied_prob_raw" not in odds.columns:
        odds["implied_prob_raw"] = odds["odds_decimal"].apply(lambda value: safe_div(1.0, value, np.nan))

    if odds["implied_prob_norm"].isna().all():
        raw_totals = odds.groupby("bookmaker")["implied_prob_raw"].transform("sum")
        odds["implied_prob_norm"] = odds.apply(
            lambda row: safe_div(row["implied_prob_raw"], raw_totals.loc[row.name], np.nan),
            axis=1,
        )

    by_team = odds.groupby("selection_team_id")["implied_prob_norm"].median()
    team1_prob = by_team.get(int(team1_id), np.nan)
    team2_prob = by_team.get(int(team2_id), np.nan)

    if pd.isna(team1_prob) and pd.isna(team2_prob):
        team1_prob = DEFAULT_BOOK_PROB
        team2_prob = DEFAULT_BOOK_PROB
    elif pd.isna(team1_prob):
        team2_prob = float(team2_prob)
        team1_prob = 1.0 - team2_prob
    elif pd.isna(team2_prob):
        team1_prob = float(team1_prob)
        team2_prob = 1.0 - team1_prob
    else:
        total = float(team1_prob) + float(team2_prob)
        if total <= 0:
            team1_prob = DEFAULT_BOOK_PROB
            team2_prob = DEFAULT_BOOK_PROB
        else:
            team1_prob = float(team1_prob) / total
            team2_prob = float(team2_prob) / total

    return {
        "bookmaker_prob_team1": float(team1_prob),
        "bookmaker_prob_team2": float(team2_prob),
        "bookmaker_prob_diff": float(team1_prob - team2_prob),
        "bookmaker_market_confidence": float(abs(team1_prob - 0.5) * 2.0),
    }


def lookup_external_context(context: BuildContext, match_id: int) -> dict:
    if context.external_context.empty:
        return {column: np.nan for column in OPTIONAL_CONTEXT_COLUMNS}
    rows = context.external_context[context.external_context["match_id"] == int(match_id)]
    if rows.empty:
        return {column: np.nan for column in OPTIONAL_CONTEXT_COLUMNS}
    row = rows.iloc[-1]
    return {column: row.get(column, np.nan) for column in OPTIONAL_CONTEXT_COLUMNS}


def build_feature_row(
    match_row: pd.Series,
    stage: str,
    team1_context: dict,
    team2_context: dict,
    team1_lineup: dict,
    team2_lineup: dict,
    head_to_head_team1: float,
    bookmaker: dict,
    toss: dict,
    extra_context: dict,
) -> dict:
    team1_batting_vs_bowling = team1_lineup["batting_strength"] - team2_lineup["bowling_strength"]
    team2_batting_vs_bowling = team2_lineup["batting_strength"] - team1_lineup["bowling_strength"]
    team1_top_vs_powerplay = team1_lineup["top_order_strength"] - team2_lineup["powerplay_strength"]
    team2_top_vs_powerplay = team2_lineup["top_order_strength"] - team1_lineup["powerplay_strength"]
    team1_death_matchup = (team1_lineup["middle_order_strength"] + team1_lineup["xi_strike_rate_avg"] / 100.0) - team2_lineup["death_overs_strength"]
    team2_death_matchup = (team2_lineup["middle_order_strength"] + team2_lineup["xi_strike_rate_avg"] / 100.0) - team1_lineup["death_overs_strength"]

    row = {
        "match_id": int(match_row["match_id"]),
        "season": int(match_row["season"]),
        "start_time_utc": match_row["start_time_utc"],
        "team1_id": int(match_row["team1_id"]),
        "team2_id": int(match_row["team2_id"]),
        "team1_name": match_row["team1_name"],
        "team2_name": match_row["team2_name"],
        "venue_id": match_row.get("venue_id"),
        "venue_name": match_row.get("venue_name"),
        "stage": stage,
        "team1_elo": float(team1_context["elo_rating"]),
        "team2_elo": float(team2_context["elo_rating"]),
        "bookmaker_prob_team2": float(bookmaker["bookmaker_prob_team2"]),
        "elo_diff": float(team1_context["elo_rating"] - team2_context["elo_rating"]),
        "batting_strength_diff": float(team1_lineup["batting_strength"] - team2_lineup["batting_strength"]),
        "bowling_strength_diff": float(team1_lineup["bowling_strength"] - team2_lineup["bowling_strength"]),
        "all_rounder_balance_diff": float(team1_lineup["all_rounder_balance"] - team2_lineup["all_rounder_balance"]),
        "spin_strength_diff": float(team1_lineup["spin_strength"] - team2_lineup["spin_strength"]),
        "pace_strength_diff": float(team1_lineup["pace_strength"] - team2_lineup["pace_strength"]),
        "death_overs_strength_diff": float(team1_lineup["death_overs_strength"] - team2_lineup["death_overs_strength"]),
        "team_recent_win_pct_diff": float(team1_context["recent_win_pct"] - team2_context["recent_win_pct"]),
        "head_to_head_win_pct_diff": float(head_to_head_team1 - (1.0 - head_to_head_team1)),
        "venue_win_bias_diff": float(team1_context["venue_win_bias"] - team2_context["venue_win_bias"]),
        "rest_days_diff": float(team1_context["rest_days"] - team2_context["rest_days"]),
        "recent_run_rate_diff": float(team1_context["recent_run_rate"] - team2_context["recent_run_rate"]),
        "recent_wicket_margin_diff": float(team1_context["recent_wicket_margin"] - team2_context["recent_wicket_margin"]),
        "probable_xi_count_diff": float(team1_lineup["xi_count"] - team2_lineup["xi_count"]),
        "probable_xi_batting_form_diff": float(team1_lineup["xi_batting_form"] - team2_lineup["xi_batting_form"]),
        "probable_xi_bowling_form_diff": float(team1_lineup["xi_bowling_form"] - team2_lineup["xi_bowling_form"]),
        "probable_xi_runs_avg_diff": float(team1_lineup["xi_runs_avg_sum"] - team2_lineup["xi_runs_avg_sum"]),
        "probable_xi_strike_rate_diff": float(team1_lineup["xi_strike_rate_avg"] - team2_lineup["xi_strike_rate_avg"]),
        "probable_xi_boundary_pct_diff": float(team1_lineup["xi_boundary_pct_avg"] - team2_lineup["xi_boundary_pct_avg"]),
        "probable_xi_experience_diff": float(team1_lineup["xi_experience"] - team2_lineup["xi_experience"]),
        "top_order_strength_diff": float(team1_lineup["top_order_strength"] - team2_lineup["top_order_strength"]),
        "middle_order_strength_diff": float(team1_lineup["middle_order_strength"] - team2_lineup["middle_order_strength"]),
        "death_bowling_strength_diff": float(team1_lineup["death_bowling_strength"] - team2_lineup["death_bowling_strength"]),
        "batting_vs_bowling_form_diff": float(team1_batting_vs_bowling - team2_batting_vs_bowling),
        "top_order_vs_powerplay_diff": float(team1_top_vs_powerplay - team2_top_vs_powerplay),
        "death_matchup_diff": float(team1_death_matchup - team2_death_matchup),
        "bookmaker_prob_team1": float(bookmaker["bookmaker_prob_team1"]),
        "bookmaker_prob_diff": float(bookmaker["bookmaker_prob_diff"]),
        "bookmaker_market_confidence": float(bookmaker["bookmaker_market_confidence"]),
        "team1_won_toss": float(toss["team1_won_toss"]),
        "team1_bats_first": float(toss["team1_bats_first"]),
        "team1_chasing_bias_edge": float(toss["team1_chasing_bias_edge"]),
        "venue_chasing_bias": float(toss["venue_chasing_bias"]),
    }
    row.update(extra_context)
    return row


def finalize_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in ALL_FEATURE_COLUMNS:
        if column not in out.columns:
            default_value = DEFAULT_BOOK_PROB if column == "bookmaker_prob_team1" else 0.0
            out[column] = default_value
    for column in OPTIONAL_CONTEXT_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
    return out


def build_historical_feature_frame(engine: Engine, stage: str = "pre_toss") -> pd.DataFrame:
    stage = normalize_stage(stage)
    context = prepare_context(engine)
    rows = []

    for _, match_row in context.completed_matches.iterrows():
        cutoff_time_utc = match_row["start_time_utc"]
        team1_id = int(match_row["team1_id"])
        team2_id = int(match_row["team2_id"])
        venue_name = context.venue_names.get(match_row.get("venue_id"), "")
        pitch_type = classify_pitch(venue_name)
        pre_match_elo = context.elo_pre_by_match.get(int(match_row["match_id"]), {})

        team1_context = summarize_team_history(
            context=context,
            team_id=team1_id,
            cutoff_time_utc=cutoff_time_utc,
            venue_id=match_row.get("venue_id"),
            elo_rating=pre_match_elo.get("team1_elo", BASE_ELO),
        )
        team2_context = summarize_team_history(
            context=context,
            team_id=team2_id,
            cutoff_time_utc=cutoff_time_utc,
            venue_id=match_row.get("venue_id"),
            elo_rating=pre_match_elo.get("team2_elo", BASE_ELO),
        )
        if stage == "confirmed_xi":
            team1_lineup = build_historical_confirmed_lineup_summary(
                context=context,
                match_id=int(match_row["match_id"]),
                team_id=team1_id,
                season=int(match_row["season"]),
                cutoff_time_utc=cutoff_time_utc,
                pitch_type=pitch_type,
            )
            team2_lineup = build_historical_confirmed_lineup_summary(
                context=context,
                match_id=int(match_row["match_id"]),
                team_id=team2_id,
                season=int(match_row["season"]),
                cutoff_time_utc=cutoff_time_utc,
                pitch_type=pitch_type,
            )
        else:
            team1_lineup = build_historical_lineup_summary(
                context=context,
                team_id=team1_id,
                season=int(match_row["season"]),
                cutoff_time_utc=cutoff_time_utc,
                pitch_type=pitch_type,
            )
            team2_lineup = build_historical_lineup_summary(
                context=context,
                team_id=team2_id,
                season=int(match_row["season"]),
                cutoff_time_utc=cutoff_time_utc,
                pitch_type=pitch_type,
            )
        bookmaker = compute_bookmaker_features(
            context=context,
            match_id=int(match_row["match_id"]),
            team1_id=team1_id,
            team2_id=team2_id,
            cutoff_time_utc=cutoff_time_utc,
        )
        toss = compute_toss_features(context, match_row, stage, cutoff_time_utc)
        enriched_match = match_row.copy()
        enriched_match["team1_name"] = context.team_names.get(team1_id, str(team1_id))
        enriched_match["team2_name"] = context.team_names.get(team2_id, str(team2_id))
        enriched_match["venue_name"] = venue_name
        row = build_feature_row(
            match_row=enriched_match,
            stage=stage,
            team1_context=team1_context,
            team2_context=team2_context,
            team1_lineup=team1_lineup,
            team2_lineup=team2_lineup,
            head_to_head_team1=head_to_head_win_pct(context, team1_id, team2_id, cutoff_time_utc),
            bookmaker=bookmaker,
            toss=toss,
            extra_context=lookup_external_context(context, int(match_row["match_id"])),
        )
        row["winner_team_id"] = int(match_row["winner_team_id"])
        row["team1_won"] = int(int(match_row["winner_team_id"]) == team1_id)
        rows.append(row)

    frame = pd.DataFrame(rows)
    return finalize_feature_frame(frame)


def build_upcoming_feature_frame(
    engine: Engine,
    stage: str = "pre_toss",
    strict_stage_inputs: bool = False,
) -> pd.DataFrame:
    stage = normalize_stage(stage)
    context = prepare_context(engine)
    rows = []

    for _, match_row in context.upcoming_matches.sort_values(["start_time_utc", "match_id"]).iterrows():
        cutoff_time_utc = match_row["start_time_utc"]
        team1_id = int(match_row["team1_id"])
        team2_id = int(match_row["team2_id"])
        venue_name = context.venue_names.get(match_row.get("venue_id"), "")
        pitch_type = classify_pitch(venue_name)

        team1_context = summarize_team_history(
            context=context,
            team_id=team1_id,
            cutoff_time_utc=cutoff_time_utc,
            venue_id=match_row.get("venue_id"),
            elo_rating=context.latest_completed_elo.get(team1_id, BASE_ELO),
        )
        team2_context = summarize_team_history(
            context=context,
            team_id=team2_id,
            cutoff_time_utc=cutoff_time_utc,
            venue_id=match_row.get("venue_id"),
            elo_rating=context.latest_completed_elo.get(team2_id, BASE_ELO),
        )
        if stage == "post_toss" and strict_stage_inputs and not toss_available(match_row):
            continue
        if stage == "confirmed_xi" and strict_stage_inputs:
            if not toss_available(match_row):
                continue
            if not confirmed_xi_available(context, int(match_row["match_id"]), team1_id, cutoff_time_utc):
                continue
            if not confirmed_xi_available(context, int(match_row["match_id"]), team2_id, cutoff_time_utc):
                continue

        if stage == "confirmed_xi":
            team1_lineup = build_upcoming_confirmed_lineup_summary(
                context=context,
                match_id=int(match_row["match_id"]),
                team_id=team1_id,
                season=int(match_row["season"]),
                cutoff_time_utc=cutoff_time_utc,
                pitch_type=pitch_type,
            )
            team2_lineup = build_upcoming_confirmed_lineup_summary(
                context=context,
                match_id=int(match_row["match_id"]),
                team_id=team2_id,
                season=int(match_row["season"]),
                cutoff_time_utc=cutoff_time_utc,
                pitch_type=pitch_type,
            )
        else:
            team1_lineup = build_upcoming_lineup_summary(
                context=context,
                match_id=int(match_row["match_id"]),
                team_id=team1_id,
                season=int(match_row["season"]),
                cutoff_time_utc=cutoff_time_utc,
                pitch_type=pitch_type,
            )
            team2_lineup = build_upcoming_lineup_summary(
                context=context,
                match_id=int(match_row["match_id"]),
                team_id=team2_id,
                season=int(match_row["season"]),
                cutoff_time_utc=cutoff_time_utc,
                pitch_type=pitch_type,
            )
        bookmaker = compute_bookmaker_features(
            context=context,
            match_id=int(match_row["match_id"]),
            team1_id=team1_id,
            team2_id=team2_id,
            cutoff_time_utc=cutoff_time_utc,
        )
        toss = compute_toss_features(context, match_row, stage, cutoff_time_utc)
        enriched_match = match_row.copy()
        enriched_match["team1_name"] = context.team_names.get(team1_id, str(team1_id))
        enriched_match["team2_name"] = context.team_names.get(team2_id, str(team2_id))
        enriched_match["venue_name"] = venue_name
        rows.append(
            build_feature_row(
                match_row=enriched_match,
                stage=stage,
                team1_context=team1_context,
                team2_context=team2_context,
                team1_lineup=team1_lineup,
                team2_lineup=team2_lineup,
                head_to_head_team1=head_to_head_win_pct(context, team1_id, team2_id, cutoff_time_utc),
                bookmaker=bookmaker,
                toss=toss,
                extra_context=lookup_external_context(context, int(match_row["match_id"])),
            )
        )

    frame = pd.DataFrame(rows)
    return finalize_feature_frame(frame)
