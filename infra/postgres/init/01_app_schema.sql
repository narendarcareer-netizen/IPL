-- infra/postgres/init/01_app_schema.sql
CREATE TABLE teams (
  team_id        SERIAL PRIMARY KEY,
  name           TEXT NOT NULL UNIQUE,
  short_name     TEXT
);

CREATE TABLE players (
  player_id      SERIAL PRIMARY KEY,
  full_name      TEXT NOT NULL,
  country        TEXT,
  batting_hand   TEXT,   -- e.g. "RHB", "LHB"
  bowling_style  TEXT,   -- e.g. "RF", "LF", "RM", "OS", "LS"
  UNIQUE(full_name, country)
);

CREATE TABLE venues (
  venue_id       SERIAL PRIMARY KEY,
  name           TEXT NOT NULL UNIQUE,
  city           TEXT,
  country        TEXT
);

CREATE TABLE matches (
  match_id               BIGSERIAL PRIMARY KEY,
  season                 INT NOT NULL,
  match_type             TEXT NOT NULL DEFAULT 't20',
  start_time_utc         TIMESTAMPTZ NOT NULL,
  venue_id               INT REFERENCES venues(venue_id),

  team1_id               INT NOT NULL REFERENCES teams(team_id),
  team2_id               INT NOT NULL REFERENCES teams(team_id),

  toss_time_utc          TIMESTAMPTZ,
  toss_winner_team_id    INT REFERENCES teams(team_id),
  toss_decision          TEXT,  -- "bat" | "field"

  winner_team_id         INT REFERENCES teams(team_id),
  result_type            TEXT,  -- "runs" | "wickets" | "no_result" | "tie" | etc
  win_margin             INT,
  completed              BOOLEAN NOT NULL DEFAULT FALSE,

  UNIQUE(season, start_time_utc, team1_id, team2_id)
);

CREATE TABLE squads (
  squad_id       BIGSERIAL PRIMARY KEY,
  season         INT NOT NULL,
  team_id        INT NOT NULL REFERENCES teams(team_id),
  player_id      INT NOT NULL REFERENCES players(player_id),
  role           TEXT, -- "bat", "bowl", "ar", "wk"
  UNIQUE(season, team_id, player_id)
);

-- Probable XI is a user/API assumption snapshot (pre-match)
CREATE TABLE probable_xi (
  probable_xi_id BIGSERIAL PRIMARY KEY,
  match_id       BIGINT NOT NULL REFERENCES matches(match_id),
  team_id        INT NOT NULL REFERENCES teams(team_id),
  player_id      INT NOT NULL REFERENCES players(player_id),
  is_captain     BOOLEAN NOT NULL DEFAULT FALSE,
  is_wicketkeeper BOOLEAN NOT NULL DEFAULT FALSE,
  source         TEXT NOT NULL DEFAULT 'manual',
  as_of_time_utc TIMESTAMPTZ NOT NULL,
  captured_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE(match_id, team_id, player_id, as_of_time_utc)
);

-- Confirmed XI snapshot (near toss / official lineups)
CREATE TABLE confirmed_xi (
  confirmed_xi_id BIGSERIAL PRIMARY KEY,
  match_id       BIGINT NOT NULL REFERENCES matches(match_id),
  team_id        INT NOT NULL REFERENCES teams(team_id),
  player_id      INT NOT NULL REFERENCES players(player_id),
  is_captain     BOOLEAN NOT NULL DEFAULT FALSE,
  is_wicketkeeper BOOLEAN NOT NULL DEFAULT FALSE,
  source         TEXT NOT NULL DEFAULT 'api',
  as_of_time_utc TIMESTAMPTZ NOT NULL,
  captured_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE(match_id, team_id, player_id, as_of_time_utc)
);

-- Player per-match outcomes (for historical modeling only; never used directly for same-match prediction)
CREATE TABLE player_match_stats (
  match_id       BIGINT NOT NULL REFERENCES matches(match_id),
  team_id        INT NOT NULL REFERENCES teams(team_id),
  player_id      INT NOT NULL REFERENCES players(player_id),

  runs           INT,
  balls_faced    INT,
  fours          INT,
  sixes          INT,

  overs_bowled   NUMERIC(4,1),
  runs_conceded  INT,
  wickets        INT,

  PRIMARY KEY(match_id, player_id)
);

CREATE TABLE team_match_stats (
  match_id       BIGINT NOT NULL REFERENCES matches(match_id),
  team_id        INT NOT NULL REFERENCES teams(team_id),

  runs_scored    INT,
  wickets_lost   INT,
  overs_faced    NUMERIC(4,1),

  runs_conceded  INT,
  wickets_taken  INT,
  overs_bowled   NUMERIC(4,1),

  PRIMARY KEY(match_id, team_id)
);

-- Raw ball-by-ball (large); consider partitioning later
CREATE TABLE ball_by_ball (
  match_id        BIGINT NOT NULL REFERENCES matches(match_id),
  innings         SMALLINT NOT NULL,
  over_number     SMALLINT NOT NULL,
  ball_in_over    SMALLINT NOT NULL,

  batting_team_id INT NOT NULL REFERENCES teams(team_id),
  bowling_team_id INT NOT NULL REFERENCES teams(team_id),

  striker_id      INT REFERENCES players(player_id),
  non_striker_id  INT REFERENCES players(player_id),
  bowler_id       INT REFERENCES players(player_id),

  runs_bat        SMALLINT NOT NULL DEFAULT 0,
  runs_extras     SMALLINT NOT NULL DEFAULT 0,
  extras_type     TEXT,
  is_wicket       BOOLEAN NOT NULL DEFAULT FALSE,
  wicket_type     TEXT,
  player_dismissed_id INT REFERENCES players(player_id),

  PRIMARY KEY(match_id, innings, over_number, ball_in_over)
);

-- Odds snapshots (append-only)
CREATE TABLE odds_snapshots (
  odds_snapshot_id BIGSERIAL PRIMARY KEY,
  match_id         BIGINT NOT NULL REFERENCES matches(match_id),
  provider         TEXT NOT NULL,         -- e.g. "the_odds_api"
  bookmaker        TEXT,
  market_key       TEXT NOT NULL DEFAULT 'h2h',
  selection_team_id INT REFERENCES teams(team_id),

  odds_decimal     NUMERIC(10,4) NOT NULL,
  implied_prob_raw NUMERIC(10,6),
  implied_prob_norm NUMERIC(10,6),
  overround        NUMERIC(10,6),

  captured_at_utc  TIMESTAMPTZ NOT NULL,
  as_of_time_utc   TIMESTAMPTZ NOT NULL
);

-- Rolling form snapshots as-of time (derived from completed past matches only)
CREATE TABLE player_form_snapshots (
  player_id        INT NOT NULL REFERENCES players(player_id),
  as_of_time_utc    TIMESTAMPTZ NOT NULL,
  horizon_matches   INT NOT NULL DEFAULT 5,

  batting_form_score NUMERIC(10,4),
  bowling_form_score NUMERIC(10,4),
  availability_flag  BOOLEAN NOT NULL DEFAULT TRUE,

  PRIMARY KEY(player_id, as_of_time_utc, horizon_matches)
);

CREATE TABLE team_form_snapshots (
  team_id          INT NOT NULL REFERENCES teams(team_id),
  as_of_time_utc    TIMESTAMPTZ NOT NULL,
  horizon_matches   INT NOT NULL DEFAULT 5,

  elo_rating        NUMERIC(10,2),
  powerplay_strength NUMERIC(10,4),
  middle_strength    NUMERIC(10,4),
  death_strength     NUMERIC(10,4),

  PRIMARY KEY(team_id, as_of_time_utc, horizon_matches)
);

-- Model registry linkage (pairs with MLflow)
CREATE TABLE model_versions (
  model_version_id BIGSERIAL PRIMARY KEY,
  model_name       TEXT NOT NULL,
  stage            TEXT NOT NULL, -- "pre_toss" | "post_toss" | "post_lineup"
  created_at_utc   TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  mlflow_run_id    TEXT,
  artifact_uri     TEXT,
  metrics_json     JSONB,

  UNIQUE(model_name, stage, created_at_utc)
);

-- Predictions store what you showed the user + what created it
CREATE TABLE predictions (
  prediction_id     BIGSERIAL PRIMARY KEY,
  match_id          BIGINT NOT NULL REFERENCES matches(match_id),
  model_version_id  BIGINT NOT NULL REFERENCES model_versions(model_version_id),

  stage             TEXT NOT NULL, -- "pre_toss" | "post_toss" | "post_lineup"
  cutoff_time_utc   TIMESTAMPTZ NOT NULL,
  created_at_utc    TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  team1_win_prob    NUMERIC(10,6),
  team2_win_prob    NUMERIC(10,6),

  bookmaker_prob_team1 NUMERIC(10,6),
  bookmaker_prob_team2 NUMERIC(10,6),
  model_edge_team1  NUMERIC(10,6),
  model_edge_team2  NUMERIC(10,6),

  confidence_score  NUMERIC(10,6),

  features_json     JSONB
);

CREATE TABLE explanations (
  explanation_id    BIGSERIAL PRIMARY KEY,
  prediction_id     BIGINT NOT NULL REFERENCES predictions(prediction_id),
  method            TEXT NOT NULL DEFAULT 'shap',
  base_value        NUMERIC(14,6),
  shap_values_json  JSONB,
  top_features_json JSONB
);
