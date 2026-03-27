from math import tanh
from pathlib import Path

import joblib
import numpy as np

from sqlalchemy.orm import Session

from app.services.features import build_features

MODEL_PATHS = {
    "pre_toss": Path("/app/ml/artifacts/logreg_prematch.joblib"),
    "post_toss": Path("/app/ml/artifacts/logreg_post_toss.joblib"),
    "confirmed_xi": Path("/app/ml/artifacts/logreg_confirmed_xi.joblib"),
}


def elo_prob(r1: float, r2: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r2 - r1) / 400.0))


def load_local_logreg(stage: str):
    model_path = MODEL_PATHS.get(stage, MODEL_PATHS["pre_toss"])
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def _edge_score(raw_value: float, scale: float) -> float:
    return float(tanh(raw_value / scale))


def _favored_team(edge_value: float, team1_name: str, team2_name: str, epsilon: float = 0.06) -> str:
    if abs(edge_value) < epsilon:
        return "Even"
    return team1_name if edge_value > 0 else team2_name


def _edge_strength(edge_value: float) -> str:
    magnitude = abs(edge_value)
    if magnitude >= 0.55:
        return "Strong"
    if magnitude >= 0.25:
        return "Moderate"
    if magnitude >= 0.10:
        return "Slight"
    return "Even"


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _format_player_card(player: dict, discipline: str) -> dict:
    matches_used = int(player.get("matches_used", 0) or 0)
    if discipline == "batting":
        summary = (
            f"Avg {float(player.get('batting_runs_avg', 0.0)):.1f}, "
            f"SR {float(player.get('batting_strike_rate', 0.0)):.1f}, "
            f"{matches_used} recent T20s"
        )
        power = float(player.get("batting_power", 0.0))
    else:
        summary = (
            f"{float(player.get('bowling_wkts_avg', 0.0)):.2f} wkts/match, "
            f"Econ {float(player.get('bowling_economy', 0.0)):.1f}, "
            f"{matches_used} recent T20s"
        )
        power = float(player.get("bowling_power", 0.0))

    tags = []
    if player.get("is_captain"):
        tags.append("Captain")
    if player.get("is_wicketkeeper"):
        tags.append("WK")

    return {
        "player_id": int(player.get("player_id")),
        "player_name": player.get("player_name"),
        "role": player.get("role"),
        "power": power,
        "summary": summary,
        "tags": tags,
    }


def _build_team_breakdown(team_name: str, players: list[dict]) -> dict:
    batting_ranked = sorted(players, key=lambda item: float(item.get("batting_power", 0.0)), reverse=True)
    bowling_ranked = sorted(players, key=lambda item: float(item.get("bowling_power", 0.0)), reverse=True)
    batting_power = sum(float(player.get("batting_power", 0.0)) for player in batting_ranked[:7])
    bowling_power = sum(float(player.get("bowling_power", 0.0)) for player in bowling_ranked[:6])

    return {
        "team_name": team_name,
        "batting_power": float(batting_power),
        "bowling_power": float(bowling_power),
        "top_batters": [_format_player_card(player, "batting") for player in batting_ranked[:3]],
        "top_bowlers": [_format_player_card(player, "bowling") for player in bowling_ranked[:3]],
    }


def _build_key_factors(feature_bundle, team1_name: str, team2_name: str) -> list[dict]:
    features = feature_bundle.features

    batting_raw = (
        float(features.get("probable_xi_batting_form_diff", 0.0))
        + 0.35 * float(features.get("probable_xi_runs_avg_diff", 0.0))
        + 0.10 * float(features.get("probable_xi_strike_rate_diff", 0.0))
        + 8.0 * float(features.get("probable_xi_boundary_pct_diff", 0.0))
        + 0.55 * float(features.get("top_order_strength_diff", 0.0))
        + 0.35 * float(features.get("middle_order_strength_diff", 0.0))
    )
    bowling_raw = (
        float(features.get("probable_xi_bowling_form_diff", 0.0))
        + 0.50 * float(features.get("bowling_strength_diff", 0.0))
        + 0.45 * float(features.get("death_bowling_strength_diff", 0.0))
        + 0.30 * float(features.get("spin_strength_diff", 0.0))
        + 0.30 * float(features.get("pace_strength_diff", 0.0))
        + 0.25 * float(features.get("death_overs_strength_diff", 0.0))
    )
    recent_raw = (
        float(features.get("elo_diff", 0.0)) / 18.0
        + float(features.get("team_recent_win_pct_diff", 0.0)) * 3.5
        + float(features.get("head_to_head_win_pct_diff", 0.0)) * 2.0
        + float(features.get("recent_run_rate_diff", 0.0)) * 1.2
        + float(features.get("recent_wicket_margin_diff", 0.0)) * 0.7
    )
    matchup_raw = (
        float(features.get("batting_vs_bowling_form_diff", 0.0)) / 18.0
        + float(features.get("top_order_vs_powerplay_diff", 0.0)) / 12.0
        + float(features.get("death_matchup_diff", 0.0)) / 12.0
    )
    venue_raw = float(features.get("venue_win_bias_diff", 0.0)) * 3.0
    market_raw = float(features.get("bookmaker_prob_diff", 0.0)) * 4.0
    toss_raw = (
        0.80 * float(features.get("team1_won_toss", 0.0))
        - 0.55 * float(features.get("team1_bats_first", 0.0))
        + 2.0 * float(features.get("team1_chasing_bias_edge", 0.0))
    )

    batting_edge = _edge_score(batting_raw, 24.0)
    bowling_edge = _edge_score(bowling_raw, 18.0)
    recent_edge = _edge_score(recent_raw, 4.0)
    matchup_edge = _edge_score(matchup_raw, 3.0)
    venue_edge = _edge_score(venue_raw, 1.5)
    market_edge = _edge_score(market_raw, 1.2)
    toss_edge = _edge_score(toss_raw, 1.6)

    venue_label = feature_bundle.venue_name or feature_bundle.venue_city or "Unknown venue"
    pitch_type = feature_bundle.pitch_type

    factors = [
        {
            "id": "batting",
            "title": "Batting Power",
            "edge": batting_edge,
            "favored_team": _favored_team(batting_edge, team1_name, team2_name),
            "strength": _edge_strength(batting_edge),
            "summary": "Expected XI batting form, runs base, strike rate, and boundary pressure.",
            "bullets": [
                f"Expected XI batting-form diff: {float(features.get('probable_xi_batting_form_diff', 0.0)):.2f}",
                f"Projected strike-rate diff: {float(features.get('probable_xi_strike_rate_diff', 0.0)):.1f}",
                f"Top-order edge: {float(features.get('top_order_strength_diff', 0.0)):.2f}",
            ],
        },
        {
            "id": "bowling",
            "title": "Bowling Power",
            "edge": bowling_edge,
            "favored_team": _favored_team(bowling_edge, team1_name, team2_name),
            "strength": _edge_strength(bowling_edge),
            "summary": "Bowling form, wicket threat, spin or pace balance, and death-over control.",
            "bullets": [
                f"Expected XI bowling-form diff: {float(features.get('probable_xi_bowling_form_diff', 0.0)):.2f}",
                f"Death-bowling edge: {float(features.get('death_bowling_strength_diff', 0.0)):.2f}",
                f"Spin/Pace split diff: {float(features.get('spin_strength_diff', 0.0)):.2f} / {float(features.get('pace_strength_diff', 0.0)):.2f}",
            ],
        },
        {
            "id": "recent_form",
            "title": "Recent Form",
            "edge": recent_edge,
            "favored_team": _favored_team(recent_edge, team1_name, team2_name),
            "strength": _edge_strength(recent_edge),
            "summary": "Recent results, Elo, run-rate trend, wicket margin, and recent head-to-head shape.",
            "bullets": [
                f"Elo diff: {float(features.get('elo_diff', 0.0)):.1f}",
                f"Recent win-rate diff: {float(features.get('team_recent_win_pct_diff', 0.0)):.3f}",
                f"Head-to-head diff: {float(features.get('head_to_head_win_pct_diff', 0.0)):.3f}",
            ],
        },
        {
            "id": "matchup",
            "title": "Bat vs Ball Matchup",
            "edge": matchup_edge,
            "favored_team": _favored_team(matchup_edge, team1_name, team2_name),
            "strength": _edge_strength(matchup_edge),
            "summary": "How the likely batting unit projects against the other side's bowling phases.",
            "bullets": [
                f"Batting-vs-bowling form diff: {float(features.get('batting_vs_bowling_form_diff', 0.0)):.2f}",
                f"Top-order vs powerplay diff: {float(features.get('top_order_vs_powerplay_diff', 0.0)):.2f}",
                f"Death matchup diff: {float(features.get('death_matchup_diff', 0.0)):.2f}",
            ],
        },
        {
            "id": "venue",
            "title": "Venue and Conditions",
            "edge": venue_edge,
            "favored_team": _favored_team(venue_edge, team1_name, team2_name),
            "strength": _edge_strength(venue_edge),
            "summary": f"{venue_label} projects as a {pitch_type}-leaning venue from the current venue heuristics.",
            "bullets": [
                f"Venue: {venue_label}",
                f"Pitch read: {pitch_type}",
                f"Venue bias diff: {float(features.get('venue_win_bias_diff', 0.0)):.3f}",
            ],
        },
        {
            "id": "market",
            "title": "Bookmaker Lean",
            "edge": market_edge,
            "favored_team": _favored_team(market_edge, team1_name, team2_name),
            "strength": _edge_strength(market_edge),
            "summary": "Median live bookmaker prices across the synced market snapshot.",
            "bullets": [
                f"{team1_name} bookmaker win probability: {_pct(float(features.get('bookmaker_prob_team1', 0.5)))}",
                f"Market confidence: {_pct(float(features.get('bookmaker_market_confidence', 0.0)))}",
                f"Model vs market diff: {float(features.get('bookmaker_prob_diff', 0.0)):.3f}",
            ],
        },
    ]

    if feature_bundle.effective_stage != "pre_toss":
        factors.append({
            "id": "toss",
            "title": "Toss Context",
            "edge": toss_edge,
            "favored_team": _favored_team(toss_edge, team1_name, team2_name),
            "strength": _edge_strength(toss_edge),
            "summary": "Toss winner, innings choice, and venue chasing tendency after the toss became available.",
            "bullets": [
                f"Team1 won toss: {float(features.get('team1_won_toss', 0.0)):.1f}",
                f"Team1 bats first: {float(features.get('team1_bats_first', 0.0)):.1f}",
                f"Chasing-bias edge: {float(features.get('team1_chasing_bias_edge', 0.0)):.3f}",
            ],
        })

    return sorted(factors, key=lambda item: abs(item["edge"]), reverse=True)


def _build_insights(feature_bundle) -> dict:
    team1_breakdown = _build_team_breakdown(feature_bundle.team1_name, feature_bundle.team1_players)
    team2_breakdown = _build_team_breakdown(feature_bundle.team2_name, feature_bundle.team2_players)
    key_factors = _build_key_factors(feature_bundle, feature_bundle.team1_name, feature_bundle.team2_name)

    top_titles = ", ".join(factor["title"] for factor in key_factors[:2])
    overview = (
        f"The current edge is being driven most by {top_titles.lower()}. "
        f"Player cards below use recent T20 form loaded in your local database."
    )

    source_bits = [
        "Recent player form is calculated from completed T20 matches currently loaded in this local database."
    ]
    if feature_bundle.requested_stage != feature_bundle.effective_stage:
        source_bits.append(
            f"Requested stage '{feature_bundle.requested_stage}' is not fully available yet, so the response is using '{feature_bundle.effective_stage}'."
        )
    if feature_bundle.effective_stage == "confirmed_xi":
        source_bits.append("Expected XI sections are using confirmed XI for both teams.")
    elif feature_bundle.effective_stage == "post_toss":
        source_bits.append("Toss inputs are active, but confirmed XI is still falling back to the latest probable XI snapshot.")
    else:
        source_bits.append("This is a pre-toss view using the latest probable XI or historical lineup estimate.")

    source_bits.append(
        f"Lineup sources: {feature_bundle.team1_name}={feature_bundle.team1_lineup_source}, {feature_bundle.team2_name}={feature_bundle.team2_lineup_source}."
    )

    return {
        "overview": overview,
        "source_note": " ".join(source_bits),
        "venue": {
            "venue_name": feature_bundle.venue_name,
            "venue_city": feature_bundle.venue_city,
            "pitch_type": feature_bundle.pitch_type,
        },
        "key_factors": key_factors,
        "team_breakdown": [team1_breakdown, team2_breakdown],
    }


def predict_match(db: Session, match_id: int, stage: str, cutoff_time_utc, model_uri: str | None = None) -> dict:
    feature_bundle = build_features(
        db=db,
        match_id=match_id,
        stage=stage,
        cutoff_time_utc=cutoff_time_utc,
    )

    payload = load_local_logreg(feature_bundle.effective_stage)
    if payload is not None:
        model = payload["model"]
        feature_cols = payload.get("feature_cols", [])
        X = np.array([[feature_bundle.features.get(column, 0.0) for column in feature_cols]], dtype=float)
        p1 = float(model.predict_proba(X)[0, 1])
        model_name = payload.get("model_name") or f"logreg_{feature_bundle.effective_stage}"
    else:
        p1 = float(elo_prob(feature_bundle.team1_rating, feature_bundle.team2_rating))
        model_name = "elo_baseline"

    p2 = 1.0 - p1
    confidence = abs(p1 - 0.5) * 2.0
    insights = _build_insights(feature_bundle)

    return {
        "match_id": match_id,
        "requested_stage": feature_bundle.requested_stage,
        "stage": feature_bundle.effective_stage,
        "model_name": model_name,
        "team1_name": feature_bundle.team1_name,
        "team2_name": feature_bundle.team2_name,
        "team1_rating": feature_bundle.team1_rating,
        "team2_rating": feature_bundle.team2_rating,
        "team1_win_prob": p1,
        "team2_win_prob": p2,
        "confidence_score": confidence,
        "features": feature_bundle.features,
        "insights": insights,
        "cutoff_time_utc": cutoff_time_utc.isoformat(),
        "requested_model_uri": model_uri,
    }
