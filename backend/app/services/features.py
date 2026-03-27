from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from sqlalchemy import text
from sqlalchemy.orm import Session

PITCH_SPIN_KEYWORDS = ("chennai", "delhi", "lucknow", "chepauk")
PITCH_PACE_KEYWORDS = ("mumbai", "bengaluru", "bangalore", "hyderabad", "mohali", "chandigarh")

FEATURE_COLUMNS = [
    "elo_diff",
    "batting_strength_diff",
    "bowling_strength_diff",
    "all_rounder_balance_diff",
    "spin_strength_diff",
    "pace_strength_diff",
    "death_overs_strength_diff",
    "team_recent_win_pct_diff",
    "head_to_head_win_pct_diff",
    "venue_win_bias_diff",
    "rest_days_diff",
    "recent_run_rate_diff",
    "recent_wicket_margin_diff",
    "probable_xi_count_diff",
    "probable_xi_batting_form_diff",
    "probable_xi_bowling_form_diff",
    "probable_xi_runs_avg_diff",
    "probable_xi_strike_rate_diff",
    "probable_xi_boundary_pct_diff",
    "probable_xi_experience_diff",
    "top_order_strength_diff",
    "middle_order_strength_diff",
    "death_bowling_strength_diff",
    "batting_vs_bowling_form_diff",
    "top_order_vs_powerplay_diff",
    "death_matchup_diff",
    "bookmaker_prob_team1",
    "bookmaker_prob_diff",
    "bookmaker_market_confidence",
    "team1_won_toss",
    "team1_bats_first",
    "team1_chasing_bias_edge",
    "venue_chasing_bias",
]


@dataclass
class FeatureBundle:
    features: dict
    requested_stage: str
    effective_stage: str
    cutoff_time_utc: datetime
    team1_name: str
    team2_name: str
    team1_rating: float
    team2_rating: float
    venue_name: str
    venue_city: str
    pitch_type: str
    team1_players: list[dict]
    team2_players: list[dict]
    toss_available: bool
    team1_lineup_source: str
    team2_lineup_source: str


def _normalize_stage(stage: str | None) -> str:
    normalized = str(stage or "pre_toss").strip().lower()
    if normalized == "post_lineup":
        return "confirmed_xi"
    if normalized not in {"pre_toss", "post_toss", "confirmed_xi"}:
        raise ValueError(f"Unsupported stage: {stage}")
    return normalized


def _to_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def _normalize_numeric_mapping(values: dict | None) -> dict:
    if not values:
        return {}
    normalized: dict = {}
    for key, value in values.items():
        if isinstance(value, (int, float, Decimal)) or value is None:
            normalized[key] = _to_float(value)
        else:
            normalized[key] = value
    return normalized


def safe_div(numerator, denominator, default=0.0):
    if denominator is None:
        return default
    denominator = _to_float(denominator, 0.0)
    if denominator == 0.0:
        return default
    return _to_float(numerator, 0.0) / denominator


def _normalize_role(
    role_value,
    is_wicketkeeper: bool = False,
    batting_form_score: float = 0.0,
    bowling_form_score: float = 0.0,
    bowling_wkts_avg: float = 0.0,
    bowling_style: str = "",
) -> str:
    if role_value is not None:
        role = str(role_value).strip().lower()
        mapping = {
            "wk": "wk_batter",
            "keeper": "wk_batter",
            "wicketkeeper": "wk_batter",
            "wk_batter": "wk_batter",
            "bat": "batter",
            "batter": "batter",
            "batsman": "batter",
            "bowl": "bowler",
            "bowler": "bowler",
            "ar": "all_rounder",
            "all_rounder": "all_rounder",
            "all-rounder": "all_rounder",
        }
        normalized = mapping.get(role, role)
        if normalized != "unknown":
            return normalized
    if is_wicketkeeper:
        return "wk_batter"
    if bowling_wkts_avg >= 0.9 or (bowling_style and bowling_style != "unknown" and bowling_form_score >= 1.25):
        if batting_form_score >= 12.0:
            return "all_rounder"
        return "bowler"
    if bowling_form_score > 1.5 and batting_form_score > 12.0:
        return "all_rounder"
    if bowling_form_score > batting_form_score:
        return "bowler"
    return "batter"


def _classify_pitch(venue_name: str, venue_city: str = "") -> str:
    haystack = f"{venue_name or ''} {venue_city or ''}".lower()
    if any(token in haystack for token in PITCH_SPIN_KEYWORDS):
        return "spin"
    if any(token in haystack for token in PITCH_PACE_KEYWORDS):
        return "pace"
    return "neutral"


def _player_batting_power(player: dict) -> float:
    return float(
        0.55 * _to_float(player.get("batting_form_score"))
        + 0.25 * _to_float(player.get("batting_runs_avg"))
        + 0.12 * (_to_float(player.get("batting_strike_rate")) / 100.0)
        + 0.08 * (_to_float(player.get("batting_boundary_pct")) * 100.0)
    )


def _player_bowling_power(player: dict) -> float:
    bowling_form_score = _to_float(player.get("bowling_form_score"))
    bowling_wkts_avg = _to_float(player.get("bowling_wkts_avg"))
    economy = _to_float(player.get("bowling_economy"))
    if bowling_form_score <= 0.0 and bowling_wkts_avg <= 0.0 and economy <= 0.0:
        return 0.0
    return float(
        0.55 * bowling_form_score
        + 0.25 * bowling_wkts_avg
        + 0.20 * max(0.0, 8.5 - economy)
    )


def _default_team_snapshot():
    return {
        "elo_rating": 1500.0,
        "batting_strength": 0.0,
        "bowling_strength": 0.0,
        "all_rounder_balance": 0.0,
        "spin_strength": 0.0,
        "pace_strength": 0.0,
        "death_overs_strength": 0.0,
        "powerplay_strength": 0.0,
    }


def _default_xi_summary():
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


def _latest_lineup_rows(
    db: Session,
    table_name: str,
    match_id: int,
    team_id: int,
    cutoff_time_utc: datetime,
) -> list[dict]:
    rows = db.execute(
        text(f"""
            WITH latest_snapshot AS (
                SELECT
                    as_of_time_utc,
                    captured_at_utc
                FROM {table_name}
                WHERE match_id = :match_id
                  AND team_id = :team_id
                  AND captured_at_utc <= :cutoff_time_utc
                ORDER BY captured_at_utc DESC, as_of_time_utc DESC
                LIMIT 1
            )
            SELECT *
            FROM {table_name}
            WHERE match_id = :match_id
              AND team_id = :team_id
              AND (as_of_time_utc, captured_at_utc) IN (
                SELECT as_of_time_utc, captured_at_utc
                FROM latest_snapshot
              )
        """),
        {
            "match_id": match_id,
            "team_id": team_id,
            "cutoff_time_utc": cutoff_time_utc,
        },
    ).mappings().all()
    return [dict(row) for row in rows]


def _probable_order_lookup(
    db: Session,
    match_id: int,
    team_id: int,
    cutoff_time_utc: datetime,
) -> dict[int, int]:
    rows = _latest_lineup_rows(db, "probable_xi", match_id, team_id, cutoff_time_utc)
    if not rows:
        return {}

    ordered = sorted(
        rows,
        key=lambda row: (
            int(row.get("batting_order_hint") or 99),
            int(row.get("probable_xi_id") or 0),
        ),
    )
    return {
        int(row["player_id"]): index + 1
        for index, row in enumerate(ordered)
        if row.get("player_id") is not None
    }


def _lineup_players_to_summary(players: list[dict]) -> dict:
    if not players:
        return _default_xi_summary()

    sorted_players = sorted(players, key=lambda player: float(player.get("batting_order_hint") or 99))
    bowling_econ_component = sum(max(0.0, 10.0 - _to_float(player.get("bowling_economy"))) for player in players)
    death_bowling_strength = float(
        bowling_econ_component + 0.75 * sum(_to_float(player.get("bowling_wkts_avg")) for player in players)
    )
    batting_strength = float(
        sum(_to_float(player.get("batting_form_score")) for player in players)
        + 0.35 * sum(_to_float(player.get("batting_runs_avg")) for player in players)
        + 0.15
        * safe_div(
            sum(_to_float(player.get("batting_boundary_pct")) for player in players),
            len(players),
            0.0,
        )
        * 100.0
    )
    bowling_strength = float(
        sum(_to_float(player.get("bowling_form_score")) for player in players)
        + 0.50 * sum(_to_float(player.get("bowling_wkts_avg")) for player in players)
        + 0.20 * bowling_econ_component
    )
    spin_strength = float(
        sum(
            _to_float(player.get("bowling_form_score"))
            for player in players
            if "spin" in str(player.get("bowling_style") or "").lower()
        )
    )
    pace_strength = float(
        sum(
            _to_float(player.get("bowling_form_score"))
            for player in players
            if any(token in str(player.get("bowling_style") or "").lower() for token in ("fast", "medium", "pace", "seam"))
        )
    )

    top_order = [player for player in sorted_players if int(player.get("batting_order_hint") or 99) <= 3]
    middle_order = [
        player
        for player in sorted_players
        if 4 <= int(player.get("batting_order_hint") or 99) <= 7
    ]

    return {
        "xi_count": float(len(players)),
        "xi_batting_form": float(sum(_to_float(player.get("batting_form_score")) for player in players)),
        "xi_bowling_form": float(sum(_to_float(player.get("bowling_form_score")) for player in players)),
        "xi_runs_avg_sum": float(sum(_to_float(player.get("batting_runs_avg")) for player in players)),
        "xi_strike_rate_avg": float(
            safe_div(sum(_to_float(player.get("batting_strike_rate")) for player in players), len(players), 0.0)
        ),
        "xi_boundary_pct_avg": float(
            safe_div(sum(_to_float(player.get("batting_boundary_pct")) for player in players), len(players), 0.0)
        ),
        "xi_experience": float(sum(_to_float(player.get("matches_used")) for player in players)),
        "top_order_strength": float(sum(_to_float(player.get("batting_form_score")) for player in top_order)),
        "middle_order_strength": float(sum(_to_float(player.get("batting_form_score")) for player in middle_order)),
        "death_bowling_strength": death_bowling_strength,
        "batting_strength": batting_strength,
        "bowling_strength": bowling_strength,
        "all_rounder_balance": float(
            sum(
                1
                for player in players
                if player.get("role") == "all_rounder"
                or (
                    _to_float(player.get("batting_form_score")) > 0.0
                    and _to_float(player.get("bowling_form_score")) > 0.0
                )
            )
        ),
        "spin_strength": spin_strength,
        "pace_strength": pace_strength,
        "death_overs_strength": death_bowling_strength,
        "powerplay_strength": float(0.60 * pace_strength + 0.40 * bowling_strength),
    }


def _toss_available(match: dict) -> bool:
    return match.get("toss_winner_team_id") is not None and bool(str(match.get("toss_decision") or "").strip())


def _batting_first_team_id(match: dict) -> int | None:
    toss_winner = match.get("toss_winner_team_id")
    decision = str(match.get("toss_decision") or "").strip().lower()
    if toss_winner is None or not decision:
        return None

    toss_winner = int(toss_winner)
    team1_id = int(match["team1_id"])
    team2_id = int(match["team2_id"])
    if decision == "bat":
        return toss_winner
    if decision in {"field", "bowl"}:
        return team2_id if toss_winner == team1_id else team1_id
    return None


def _venue_chasing_bias(db: Session, venue_id: int | None, cutoff_time_utc: datetime) -> float:
    if venue_id is None:
        return 0.0

    rows = db.execute(
        text("""
            SELECT
                match_id,
                team1_id,
                team2_id,
                toss_winner_team_id,
                toss_decision,
                winner_team_id
            FROM matches
            WHERE competition = 'Indian Premier League'
              AND completed = true
              AND start_time_utc < :cutoff_time_utc
              AND venue_id = :venue_id
              AND toss_winner_team_id IS NOT NULL
              AND toss_decision IS NOT NULL
              AND winner_team_id IS NOT NULL
            ORDER BY start_time_utc DESC, match_id DESC
            LIMIT 25
        """),
        {"cutoff_time_utc": cutoff_time_utc, "venue_id": venue_id},
    ).mappings().all()

    if not rows:
        return 0.0

    chasing_wins = []
    for row in rows:
        batting_first_team = _batting_first_team_id(dict(row))
        if batting_first_team is None:
            continue
        chasing_wins.append(float(int(row["winner_team_id"]) != int(batting_first_team)))

    if not chasing_wins:
        return 0.0

    return float(sum(chasing_wins) / len(chasing_wins) - 0.5)


def _build_toss_features(match: dict, stage: str, db: Session, cutoff_time_utc: datetime) -> dict:
    if stage == "pre_toss" or not _toss_available(match):
        return {
            "team1_won_toss": 0.0,
            "team1_bats_first": 0.0,
            "team1_chasing_bias_edge": 0.0,
            "venue_chasing_bias": 0.0,
        }

    team1_id = int(match["team1_id"])
    toss_winner = int(match["toss_winner_team_id"])
    batting_first_team = _batting_first_team_id(match)
    team1_bats_first = 0.0
    if batting_first_team is not None:
        team1_bats_first = 1.0 if int(batting_first_team) == team1_id else -1.0

    venue_chasing_bias = _venue_chasing_bias(db, match.get("venue_id"), cutoff_time_utc)
    chasing_edge = venue_chasing_bias if team1_bats_first < 0 else -venue_chasing_bias if team1_bats_first > 0 else 0.0
    return {
        "team1_won_toss": 1.0 if toss_winner == team1_id else -1.0,
        "team1_bats_first": float(team1_bats_first),
        "team1_chasing_bias_edge": float(chasing_edge),
        "venue_chasing_bias": float(venue_chasing_bias),
    }


def get_team_snapshot(db: Session, team_id: int, cutoff_time_utc: datetime) -> dict:
    row = db.execute(
        text("""
            SELECT
                COALESCE(elo_rating, 1500) AS elo_rating,
                COALESCE(batting_strength, 0) AS batting_strength,
                COALESCE(bowling_strength, 0) AS bowling_strength,
                COALESCE(all_rounder_balance, 0) AS all_rounder_balance,
                COALESCE(spin_strength, 0) AS spin_strength,
                COALESCE(pace_strength, 0) AS pace_strength,
                COALESCE(death_overs_strength, 0) AS death_overs_strength,
                COALESCE(powerplay_strength, 0) AS powerplay_strength
            FROM team_form_snapshots
            WHERE team_id = :team_id
              AND as_of_time_utc <= :cutoff_time_utc
            ORDER BY as_of_time_utc DESC
            LIMIT 1
        """),
        {"team_id": team_id, "cutoff_time_utc": cutoff_time_utc},
    ).mappings().first()
    return _normalize_numeric_mapping(dict(row)) if row else _default_team_snapshot()


def get_team_recent_context(db: Session, team_id: int, venue_id: int | None, cutoff_time_utc: datetime) -> dict:
    recent = db.execute(
        text("""
            SELECT
                AVG(CASE WHEN recent_matches.winner_team_id = :team_id THEN 1.0 ELSE 0.0 END) AS recent_win_pct,
                AVG(COALESCE(recent_matches.runs_scored, 0) / NULLIF(COALESCE(recent_matches.overs_faced, 0), 0)) AS recent_run_rate,
                AVG(COALESCE(recent_matches.wickets_taken, 0) - COALESCE(recent_matches.wickets_lost, 0)) AS recent_wicket_margin,
                EXTRACT(EPOCH FROM (:cutoff_time_utc - MAX(recent_matches.start_time_utc))) / 86400.0 AS rest_days
            FROM (
                SELECT
                    tms.runs_scored,
                    tms.overs_faced,
                    tms.wickets_taken,
                    tms.wickets_lost,
                    m.winner_team_id,
                    m.start_time_utc
                FROM team_match_stats tms
                JOIN matches m ON tms.match_id = m.match_id
                WHERE tms.team_id = :team_id
                  AND m.competition = 'Indian Premier League'
                  AND m.completed = true
                  AND m.start_time_utc < :cutoff_time_utc
                ORDER BY m.start_time_utc DESC, m.match_id DESC
                LIMIT 5
            ) recent_matches
        """),
        {"team_id": team_id, "cutoff_time_utc": cutoff_time_utc},
    ).mappings().first()

    venue_row = db.execute(
        text("""
            SELECT
                AVG(CASE WHEN winner_team_id = :team_id THEN 1.0 ELSE 0.0 END) AS venue_win_bias
            FROM matches
            WHERE competition = 'Indian Premier League'
              AND completed = true
              AND start_time_utc < :cutoff_time_utc
              AND venue_id = :venue_id
              AND (team1_id = :team_id OR team2_id = :team_id)
        """),
        {"team_id": team_id, "cutoff_time_utc": cutoff_time_utc, "venue_id": venue_id},
    ).mappings().first()

    return {
        "recent_win_pct": float((recent or {}).get("recent_win_pct") or 0.5),
        "recent_run_rate": float((recent or {}).get("recent_run_rate") or 0.0),
        "recent_wicket_margin": float((recent or {}).get("recent_wicket_margin") or 0.0),
        "rest_days": float((recent or {}).get("rest_days") or 7.0),
        "venue_win_bias": float((venue_row or {}).get("venue_win_bias") or 0.5),
    }


def get_head_to_head_win_pct(db: Session, team1_id: int, team2_id: int, cutoff_time_utc: datetime) -> float:
    row = db.execute(
        text("""
            SELECT
                AVG(CASE WHEN winner_team_id = :team1_id THEN 1.0 ELSE 0.0 END) AS team1_h2h_win_pct
            FROM (
                SELECT winner_team_id
                FROM matches
                WHERE competition = 'Indian Premier League'
                  AND completed = true
                  AND start_time_utc < :cutoff_time_utc
                  AND (
                    (team1_id = :team1_id AND team2_id = :team2_id)
                    OR
                    (team1_id = :team2_id AND team2_id = :team1_id)
                  )
                ORDER BY start_time_utc DESC, match_id DESC
                LIMIT 5
            ) recent_h2h
        """),
        {"team1_id": team1_id, "team2_id": team2_id, "cutoff_time_utc": cutoff_time_utc},
    ).mappings().first()
    return float((row or {}).get("team1_h2h_win_pct") or 0.5)


def get_xi_summary(db: Session, match_id: int, team_id: int, cutoff_time_utc: datetime) -> dict:
    row = db.execute(
        text("""
            WITH latest_xi_snapshot AS (
                SELECT
                    as_of_time_utc,
                    captured_at_utc
                FROM probable_xi
                WHERE match_id = :match_id
                  AND team_id = :team_id
                  AND captured_at_utc <= :cutoff_time_utc
                ORDER BY captured_at_utc DESC, as_of_time_utc DESC
                LIMIT 1
            ),
            xi AS (
                SELECT
                    px.player_id,
                    px.batting_order_hint
                FROM probable_xi px
                JOIN latest_xi_snapshot lxs
                  ON px.as_of_time_utc = lxs.as_of_time_utc
                 AND px.captured_at_utc = lxs.captured_at_utc
                WHERE px.match_id = :match_id
                  AND px.team_id = :team_id
            ),
            latest_player_form AS (
                SELECT DISTINCT ON (pfs.player_id)
                    pfs.player_id,
                    COALESCE(pfs.matches_used, 0) AS matches_used,
                    COALESCE(pfs.batting_form_score, 0) AS batting_form_score,
                    COALESCE(pfs.bowling_form_score, 0) AS bowling_form_score,
                    COALESCE(pfs.batting_runs_avg, 0) AS batting_runs_avg,
                    COALESCE(pfs.batting_strike_rate, 0) AS batting_strike_rate,
                    COALESCE(pfs.batting_boundary_pct, 0) AS batting_boundary_pct,
                    COALESCE(pfs.bowling_wkts_avg, 0) AS bowling_wkts_avg,
                    COALESCE(pfs.bowling_economy, 0) AS bowling_economy
                FROM player_form_snapshots pfs
                JOIN xi ON xi.player_id = pfs.player_id
                WHERE pfs.as_of_time_utc <= :cutoff_time_utc
                ORDER BY pfs.player_id, pfs.as_of_time_utc DESC
            )
            SELECT
                COUNT(*) AS xi_count,
                SUM(lpf.batting_form_score) AS xi_batting_form,
                SUM(lpf.bowling_form_score) AS xi_bowling_form,
                SUM(lpf.batting_runs_avg) AS xi_runs_avg_sum,
                AVG(lpf.batting_strike_rate) AS xi_strike_rate_avg,
                AVG(lpf.batting_boundary_pct) AS xi_boundary_pct_avg,
                SUM(lpf.matches_used) AS xi_experience,
                SUM(CASE WHEN xi.batting_order_hint <= 3 THEN lpf.batting_form_score ELSE 0 END) AS top_order_strength,
                SUM(CASE WHEN xi.batting_order_hint BETWEEN 4 AND 7 THEN lpf.batting_form_score ELSE 0 END) AS middle_order_strength,
                SUM(CASE WHEN lpf.bowling_economy > 0 THEN (10 - LEAST(lpf.bowling_economy, 10)) ELSE 0 END)
                    + 0.75 * SUM(lpf.bowling_wkts_avg) AS death_bowling_strength
            FROM xi
            LEFT JOIN latest_player_form lpf ON xi.player_id = lpf.player_id
        """),
        {"match_id": match_id, "team_id": team_id, "cutoff_time_utc": cutoff_time_utc},
    ).mappings().first()
    return _normalize_numeric_mapping(dict(row)) if row and row.get("xi_count") else _default_xi_summary()


def get_xi_players(db: Session, match_id: int, team_id: int, season: int, cutoff_time_utc: datetime) -> list[dict]:
    rows = db.execute(
        text("""
            WITH latest_xi_snapshot AS (
                SELECT
                    as_of_time_utc,
                    captured_at_utc
                FROM probable_xi
                WHERE match_id = :match_id
                  AND team_id = :team_id
                  AND captured_at_utc <= :cutoff_time_utc
                ORDER BY captured_at_utc DESC, as_of_time_utc DESC
                LIMIT 1
            ),
            xi AS (
                SELECT
                    px.player_id,
                    px.batting_order_hint,
                    px.is_captain,
                    px.is_wicketkeeper,
                    sq.role,
                    pl.full_name AS player_name,
                    pl.bowling_style
                FROM probable_xi px
                JOIN latest_xi_snapshot lxs
                  ON px.as_of_time_utc = lxs.as_of_time_utc
                 AND px.captured_at_utc = lxs.captured_at_utc
                JOIN players pl
                  ON px.player_id = pl.player_id
                LEFT JOIN squads sq
                  ON sq.season = :season
                 AND sq.team_id = :team_id
                 AND sq.player_id = px.player_id
                WHERE px.match_id = :match_id
                  AND px.team_id = :team_id
            ),
            latest_player_form AS (
                SELECT DISTINCT ON (pfs.player_id)
                    pfs.player_id,
                    COALESCE(pfs.matches_used, 0) AS matches_used,
                    COALESCE(pfs.batting_form_score, 0) AS batting_form_score,
                    COALESCE(pfs.bowling_form_score, 0) AS bowling_form_score,
                    COALESCE(pfs.batting_runs_avg, 0) AS batting_runs_avg,
                    COALESCE(pfs.batting_strike_rate, 0) AS batting_strike_rate,
                    COALESCE(pfs.batting_boundary_pct, 0) AS batting_boundary_pct,
                    COALESCE(pfs.dismissal_rate, 0) AS dismissal_rate,
                    COALESCE(pfs.bowling_wkts_avg, 0) AS bowling_wkts_avg,
                    COALESCE(pfs.bowling_economy, 0) AS bowling_economy,
                    COALESCE(pfs.bowling_strike_rate, 0) AS bowling_strike_rate
                FROM player_form_snapshots pfs
                JOIN xi
                  ON xi.player_id = pfs.player_id
                WHERE pfs.as_of_time_utc <= :cutoff_time_utc
                ORDER BY pfs.player_id, pfs.as_of_time_utc DESC
            )
            SELECT
                xi.player_id,
                xi.player_name,
                xi.role,
                xi.bowling_style,
                xi.batting_order_hint,
                xi.is_captain,
                xi.is_wicketkeeper,
                COALESCE(lpf.matches_used, 0) AS matches_used,
                COALESCE(lpf.batting_form_score, 0) AS batting_form_score,
                COALESCE(lpf.bowling_form_score, 0) AS bowling_form_score,
                COALESCE(lpf.batting_runs_avg, 0) AS batting_runs_avg,
                COALESCE(lpf.batting_strike_rate, 0) AS batting_strike_rate,
                COALESCE(lpf.batting_boundary_pct, 0) AS batting_boundary_pct,
                COALESCE(lpf.dismissal_rate, 0) AS dismissal_rate,
                COALESCE(lpf.bowling_wkts_avg, 0) AS bowling_wkts_avg,
                COALESCE(lpf.bowling_economy, 0) AS bowling_economy,
                COALESCE(lpf.bowling_strike_rate, 0) AS bowling_strike_rate
            FROM xi
            LEFT JOIN latest_player_form lpf
              ON xi.player_id = lpf.player_id
            ORDER BY COALESCE(xi.batting_order_hint, 99), xi.player_id
        """),
        {
            "match_id": match_id,
            "team_id": team_id,
            "season": season,
            "cutoff_time_utc": cutoff_time_utc,
        },
    ).mappings().all()

    players = []
    for row in rows:
        raw_player = dict(row)
        player = _normalize_numeric_mapping(raw_player)
        player["is_captain"] = bool(player.get("is_captain"))
        player["is_wicketkeeper"] = bool(player.get("is_wicketkeeper"))
        player["role"] = _normalize_role(
            raw_player.get("role"),
            is_wicketkeeper=player["is_wicketkeeper"],
            batting_form_score=_to_float(player.get("batting_form_score")),
            bowling_form_score=_to_float(player.get("bowling_form_score")),
            bowling_wkts_avg=_to_float(player.get("bowling_wkts_avg")),
            bowling_style=str(raw_player.get("bowling_style") or ""),
        )
        player["batting_power"] = _player_batting_power(player)
        player["bowling_power"] = _player_bowling_power(player)
        players.append(player)
    return players


def get_confirmed_xi_players(
    db: Session,
    match_id: int,
    team_id: int,
    season: int,
    cutoff_time_utc: datetime,
) -> list[dict]:
    rows = _latest_lineup_rows(db, "confirmed_xi", match_id, team_id, cutoff_time_utc)
    if not rows:
        return []

    order_lookup = _probable_order_lookup(db, match_id, team_id, cutoff_time_utc)
    players = []

    for row in rows:
        player_id = int(row["player_id"])
        player_meta = db.execute(
            text("""
                SELECT
                    pl.full_name AS player_name,
                    pl.bowling_style,
                    sq.role
                FROM players pl
                LEFT JOIN squads sq
                  ON sq.season = :season
                 AND sq.team_id = :team_id
                 AND sq.player_id = pl.player_id
                WHERE pl.player_id = :player_id
            """),
            {"season": season, "team_id": team_id, "player_id": player_id},
        ).mappings().first()

        form = db.execute(
            text("""
                SELECT
                    COALESCE(matches_used, 0) AS matches_used,
                    COALESCE(batting_form_score, 0) AS batting_form_score,
                    COALESCE(bowling_form_score, 0) AS bowling_form_score,
                    COALESCE(batting_runs_avg, 0) AS batting_runs_avg,
                    COALESCE(batting_strike_rate, 0) AS batting_strike_rate,
                    COALESCE(batting_boundary_pct, 0) AS batting_boundary_pct,
                    COALESCE(dismissal_rate, 0) AS dismissal_rate,
                    COALESCE(bowling_wkts_avg, 0) AS bowling_wkts_avg,
                    COALESCE(bowling_economy, 0) AS bowling_economy,
                    COALESCE(bowling_strike_rate, 0) AS bowling_strike_rate
                FROM player_form_snapshots
                WHERE player_id = :player_id
                  AND as_of_time_utc <= :cutoff_time_utc
                ORDER BY as_of_time_utc DESC
                LIMIT 1
            """),
            {"player_id": player_id, "cutoff_time_utc": cutoff_time_utc},
        ).mappings().first()

        raw_player = {
            "player_id": player_id,
            "player_name": (player_meta or {}).get("player_name"),
            "role": (player_meta or {}).get("role"),
            "bowling_style": (player_meta or {}).get("bowling_style"),
            "batting_order_hint": order_lookup.get(player_id),
            "is_captain": bool(row.get("is_captain")),
            "is_wicketkeeper": bool(row.get("is_wicketkeeper")),
        }
        raw_player.update(dict(form or {}))

        player = _normalize_numeric_mapping(raw_player)
        player["is_captain"] = bool(raw_player["is_captain"])
        player["is_wicketkeeper"] = bool(raw_player["is_wicketkeeper"])
        player["role"] = _normalize_role(
            raw_player.get("role"),
            is_wicketkeeper=player["is_wicketkeeper"],
            batting_form_score=_to_float(player.get("batting_form_score")),
            bowling_form_score=_to_float(player.get("bowling_form_score")),
            bowling_wkts_avg=_to_float(player.get("bowling_wkts_avg")),
            bowling_style=str(raw_player.get("bowling_style") or ""),
        )
        player["batting_order_hint"] = int(player.get("batting_order_hint") or 99)
        player["batting_power"] = _player_batting_power(player)
        player["bowling_power"] = _player_bowling_power(player)
        players.append(player)

    players = sorted(players, key=lambda item: int(item.get("batting_order_hint") or 99))
    return players


def get_confirmed_xi_summary(
    db: Session,
    match_id: int,
    team_id: int,
    season: int,
    cutoff_time_utc: datetime,
) -> dict:
    players = get_confirmed_xi_players(db, match_id, team_id, season, cutoff_time_utc)
    return _lineup_players_to_summary(players)


def get_bookmaker_features(db: Session, match_id: int, team1_id: int, team2_id: int, cutoff_time_utc: datetime) -> dict:
    rows = db.execute(
        text("""
            SELECT
                selection_team_id,
                COALESCE(implied_prob_norm, implied_prob_raw) AS implied_prob
            FROM odds_snapshots
            WHERE match_id = :match_id
              AND market_key = 'h2h'
              AND as_of_time_utc = (
                SELECT MAX(as_of_time_utc)
                FROM odds_snapshots
                WHERE match_id = :match_id
                  AND market_key = 'h2h'
                  AND as_of_time_utc <= :cutoff_time_utc
              )
        """),
        {"match_id": match_id, "cutoff_time_utc": cutoff_time_utc},
    ).mappings().all()

    if not rows:
        return {
            "bookmaker_prob_team1": 0.5,
            "bookmaker_prob_team2": 0.5,
            "bookmaker_prob_diff": 0.0,
            "bookmaker_market_confidence": 0.0,
        }

    team_probs = {int(row["selection_team_id"]): float(row["implied_prob"]) for row in rows if row["selection_team_id"] is not None}
    team1_prob = team_probs.get(team1_id, 0.5)
    team2_prob = team_probs.get(team2_id, 0.5)
    total = team1_prob + team2_prob
    if total > 0:
        team1_prob /= total
        team2_prob /= total
    return {
        "bookmaker_prob_team1": team1_prob,
        "bookmaker_prob_team2": team2_prob,
        "bookmaker_prob_diff": team1_prob - team2_prob,
        "bookmaker_market_confidence": abs(team1_prob - 0.5) * 2.0,
    }


def build_features(db: Session, match_id: int, stage: str, cutoff_time_utc: datetime) -> FeatureBundle:
    requested_stage = _normalize_stage(stage)
    match = db.execute(
        text("""
            SELECT
                m.match_id,
                m.season,
                m.venue_id,
                m.team1_id,
                m.team2_id,
                m.toss_winner_team_id,
                m.toss_decision,
                t1.name AS team1_name,
                t2.name AS team2_name,
                COALESCE(v.name, '') AS venue_name,
                COALESCE(v.city, '') AS venue_city
            FROM matches m
            JOIN teams t1 ON m.team1_id = t1.team_id
            JOIN teams t2 ON m.team2_id = t2.team_id
            LEFT JOIN venues v ON m.venue_id = v.venue_id
            WHERE m.match_id = :match_id
        """),
        {"match_id": match_id},
    ).mappings().first()

    if not match:
        raise ValueError(f"Match {match_id} not found")

    match_dict = dict(match)
    toss_is_available = _toss_available(match_dict)

    effective_stage = requested_stage
    if requested_stage == "post_toss" and not toss_is_available:
        effective_stage = "pre_toss"
    if requested_stage == "confirmed_xi":
        team1_confirmed_players = get_confirmed_xi_players(
            db,
            match_id,
            int(match["team1_id"]),
            int(match["season"]),
            cutoff_time_utc,
        )
        team2_confirmed_players = get_confirmed_xi_players(
            db,
            match_id,
            int(match["team2_id"]),
            int(match["season"]),
            cutoff_time_utc,
        )
        if not toss_is_available:
            effective_stage = "pre_toss"
        elif not team1_confirmed_players or not team2_confirmed_players:
            effective_stage = "post_toss"
    else:
        team1_confirmed_players = []
        team2_confirmed_players = []

    team1_snapshot = get_team_snapshot(db, int(match["team1_id"]), cutoff_time_utc)
    team2_snapshot = get_team_snapshot(db, int(match["team2_id"]), cutoff_time_utc)
    team1_recent = get_team_recent_context(db, int(match["team1_id"]), match["venue_id"], cutoff_time_utc)
    team2_recent = get_team_recent_context(db, int(match["team2_id"]), match["venue_id"], cutoff_time_utc)

    if effective_stage == "confirmed_xi":
        team1_players = team1_confirmed_players
        team2_players = team2_confirmed_players
        team1_xi = _lineup_players_to_summary(team1_players)
        team2_xi = _lineup_players_to_summary(team2_players)
        team1_lineup_source = "confirmed_xi"
        team2_lineup_source = "confirmed_xi"
    else:
        team1_players = get_xi_players(db, match_id, int(match["team1_id"]), int(match["season"]), cutoff_time_utc)
        team2_players = get_xi_players(db, match_id, int(match["team2_id"]), int(match["season"]), cutoff_time_utc)
        team1_xi = _lineup_players_to_summary(team1_players)
        team2_xi = _lineup_players_to_summary(team2_players)
        team1_lineup_source = "probable_xi" if team1_players else "historical_form"
        team2_lineup_source = "probable_xi" if team2_players else "historical_form"

    head_to_head_team1 = get_head_to_head_win_pct(db, int(match["team1_id"]), int(match["team2_id"]), cutoff_time_utc)
    bookmaker = get_bookmaker_features(db, match_id, int(match["team1_id"]), int(match["team2_id"]), cutoff_time_utc)
    pitch_type = _classify_pitch(str(match["venue_name"]), str(match["venue_city"]))
    toss = _build_toss_features(match_dict, effective_stage, db, cutoff_time_utc)

    batting_vs_bowling_form_diff = (
        (team1_xi["batting_strength"] - team2_xi["bowling_strength"])
        - (team2_xi["batting_strength"] - team1_xi["bowling_strength"])
    )
    top_order_vs_powerplay_diff = (
        (team1_xi["top_order_strength"] - team2_xi["powerplay_strength"])
        - (team2_xi["top_order_strength"] - team1_xi["powerplay_strength"])
    )
    death_matchup_diff = (
        (team1_xi["middle_order_strength"] + team1_xi["xi_strike_rate_avg"] / 100.0 - team2_xi["death_overs_strength"])
        - (team2_xi["middle_order_strength"] + team2_xi["xi_strike_rate_avg"] / 100.0 - team1_xi["death_overs_strength"])
    )

    features = {
        "elo_diff": float(team1_snapshot["elo_rating"] - team2_snapshot["elo_rating"]),
        "batting_strength_diff": float(team1_xi["batting_strength"] - team2_xi["batting_strength"]),
        "bowling_strength_diff": float(team1_xi["bowling_strength"] - team2_xi["bowling_strength"]),
        "all_rounder_balance_diff": float(team1_xi["all_rounder_balance"] - team2_xi["all_rounder_balance"]),
        "spin_strength_diff": float(team1_xi["spin_strength"] - team2_xi["spin_strength"]),
        "pace_strength_diff": float(team1_xi["pace_strength"] - team2_xi["pace_strength"]),
        "death_overs_strength_diff": float(team1_xi["death_overs_strength"] - team2_xi["death_overs_strength"]),
        "team_recent_win_pct_diff": float(team1_recent["recent_win_pct"] - team2_recent["recent_win_pct"]),
        "head_to_head_win_pct_diff": float(head_to_head_team1 - (1.0 - head_to_head_team1)),
        "venue_win_bias_diff": float(team1_recent["venue_win_bias"] - team2_recent["venue_win_bias"]),
        "rest_days_diff": float(team1_recent["rest_days"] - team2_recent["rest_days"]),
        "recent_run_rate_diff": float(team1_recent["recent_run_rate"] - team2_recent["recent_run_rate"]),
        "recent_wicket_margin_diff": float(team1_recent["recent_wicket_margin"] - team2_recent["recent_wicket_margin"]),
        "probable_xi_count_diff": float(team1_xi["xi_count"] - team2_xi["xi_count"]),
        "probable_xi_batting_form_diff": float(team1_xi["xi_batting_form"] - team2_xi["xi_batting_form"]),
        "probable_xi_bowling_form_diff": float(team1_xi["xi_bowling_form"] - team2_xi["xi_bowling_form"]),
        "probable_xi_runs_avg_diff": float(team1_xi["xi_runs_avg_sum"] - team2_xi["xi_runs_avg_sum"]),
        "probable_xi_strike_rate_diff": float(team1_xi["xi_strike_rate_avg"] - team2_xi["xi_strike_rate_avg"]),
        "probable_xi_boundary_pct_diff": float(team1_xi["xi_boundary_pct_avg"] - team2_xi["xi_boundary_pct_avg"]),
        "probable_xi_experience_diff": float(team1_xi["xi_experience"] - team2_xi["xi_experience"]),
        "top_order_strength_diff": float(team1_xi["top_order_strength"] - team2_xi["top_order_strength"]),
        "middle_order_strength_diff": float(team1_xi["middle_order_strength"] - team2_xi["middle_order_strength"]),
        "death_bowling_strength_diff": float(team1_xi["death_bowling_strength"] - team2_xi["death_bowling_strength"]),
        "batting_vs_bowling_form_diff": float(batting_vs_bowling_form_diff),
        "top_order_vs_powerplay_diff": float(top_order_vs_powerplay_diff),
        "death_matchup_diff": float(death_matchup_diff),
        "bookmaker_prob_team1": float(bookmaker["bookmaker_prob_team1"]),
        "bookmaker_prob_diff": float(bookmaker["bookmaker_prob_diff"]),
        "bookmaker_market_confidence": float(bookmaker["bookmaker_market_confidence"]),
        "team1_won_toss": float(toss["team1_won_toss"]),
        "team1_bats_first": float(toss["team1_bats_first"]),
        "team1_chasing_bias_edge": float(toss["team1_chasing_bias_edge"]),
        "venue_chasing_bias": float(toss["venue_chasing_bias"]),
    }

    for column in FEATURE_COLUMNS:
        features.setdefault(column, 0.0)

    return FeatureBundle(
        features=features,
        requested_stage=requested_stage,
        effective_stage=effective_stage,
        cutoff_time_utc=cutoff_time_utc,
        team1_name=match["team1_name"],
        team2_name=match["team2_name"],
        team1_rating=float(team1_snapshot["elo_rating"]),
        team2_rating=float(team2_snapshot["elo_rating"]),
        venue_name=str(match["venue_name"] or ""),
        venue_city=str(match["venue_city"] or ""),
        pitch_type=pitch_type,
        team1_players=team1_players,
        team2_players=team2_players,
        toss_available=toss_is_available,
        team1_lineup_source=team1_lineup_source,
        team2_lineup_source=team2_lineup_source,
    )
