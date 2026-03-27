HISTORICAL_OUTPUT_PATH = "/app/ml/artifacts/pre_match_features.csv"
UPCOMING_OUTPUT_PATH = "/app/ml/artifacts/upcoming_match_features.csv"

STAGES = ("pre_toss", "post_toss", "confirmed_xi")

BASE_FEATURE_COLUMNS = [
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
]

TOSS_FEATURE_COLUMNS = [
    "team1_won_toss",
    "team1_bats_first",
    "team1_chasing_bias_edge",
    "venue_chasing_bias",
]

FEATURE_COLUMNS_BY_STAGE = {
    "pre_toss": list(BASE_FEATURE_COLUMNS),
    "post_toss": [*BASE_FEATURE_COLUMNS, *TOSS_FEATURE_COLUMNS],
    "confirmed_xi": [*BASE_FEATURE_COLUMNS, *TOSS_FEATURE_COLUMNS],
}

FEATURE_COLUMNS = FEATURE_COLUMNS_BY_STAGE["pre_toss"]
ALL_FEATURE_COLUMNS = list(dict.fromkeys(BASE_FEATURE_COLUMNS + TOSS_FEATURE_COLUMNS))

OPTIONAL_CONTEXT_COLUMNS = [
    "weather_temperature_c",
    "weather_humidity_pct",
    "weather_wind_kph",
    "weather_rain_mm",
]

MODEL_NAME_BY_STAGE = {
    "pre_toss": "logreg_prematch",
    "post_toss": "logreg_post_toss",
    "confirmed_xi": "logreg_confirmed_xi",
}

MODEL_ARTIFACT_BY_STAGE = {
    "pre_toss": "/app/ml/artifacts/logreg_prematch.joblib",
    "post_toss": "/app/ml/artifacts/logreg_post_toss.joblib",
    "confirmed_xi": "/app/ml/artifacts/logreg_confirmed_xi.joblib",
}

HISTORICAL_OUTPUT_BY_STAGE = {
    "pre_toss": HISTORICAL_OUTPUT_PATH,
    "post_toss": "/app/ml/artifacts/post_toss_features.csv",
    "confirmed_xi": "/app/ml/artifacts/confirmed_xi_features.csv",
}

UPCOMING_OUTPUT_BY_STAGE = {
    "pre_toss": UPCOMING_OUTPUT_PATH,
    "post_toss": "/app/ml/artifacts/upcoming_post_toss_features.csv",
    "confirmed_xi": "/app/ml/artifacts/upcoming_confirmed_xi_features.csv",
}


def normalize_stage(stage: str | None) -> str:
    normalized = str(stage or "pre_toss").strip().lower()
    if normalized == "post_lineup":
        return "confirmed_xi"
    if normalized not in FEATURE_COLUMNS_BY_STAGE:
        raise ValueError(f"Unsupported stage: {stage}")
    return normalized


def feature_columns_for_stage(stage: str | None) -> list[str]:
    return list(FEATURE_COLUMNS_BY_STAGE[normalize_stage(stage)])


def model_name_for_stage(stage: str | None) -> str:
    return MODEL_NAME_BY_STAGE[normalize_stage(stage)]


def model_artifact_for_stage(stage: str | None) -> str:
    return MODEL_ARTIFACT_BY_STAGE[normalize_stage(stage)]


def historical_output_for_stage(stage: str | None) -> str:
    return HISTORICAL_OUTPUT_BY_STAGE[normalize_stage(stage)]


def upcoming_output_for_stage(stage: str | None) -> str:
    return UPCOMING_OUTPUT_BY_STAGE[normalize_stage(stage)]
