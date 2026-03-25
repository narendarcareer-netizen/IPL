# 🏗️ IPL Predictor Architecture

This document explains the architecture of the **IPL Match Outcome Predictor** system, including:

- data flow
- storage design
- ML pipeline
- backend API
- frontend UI
- explainability layer
- deployment model

---

# 1. System Overview

The IPL Predictor is an end-to-end machine learning system that predicts IPL match outcomes before the toss using:

- historical match data
- team strength features
- player form features
- probable playing XI
- feature engineering pipelines
- ML models
- SHAP explainability
- REST API serving
- frontend visualization

---

# 2. High-Level Architecture

```text
                ┌──────────────────────────┐
                │   Historical IPL Data    │
                │  Fixtures / Squads Data  │
                └─────────────┬────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │     Data Ingestion       │
                │  Python ETL + SQL logic  │
                └─────────────┬────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │      PostgreSQL DB       │
                │ matches / players / XI   │
                │ predictions / SHAP       │
                └─────────────┬────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
┌────────────────┐  ┌────────────────────┐  ┌────────────────────┐
│ Feature Build  │  │ Model Training      │  │ Probable XI Build   │
│ Historical     │  │ LogReg / XGBoost    │  │ Squad-aware lineup  │
└───────┬────────┘  └──────────┬─────────┘  └──────────┬─────────┘
        │                      │                       │
        ▼                      ▼                       ▼
┌───────────────────────────────────────────────────────────────┐
│                 ML Artifacts + MLflow Tracking               │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ▼
                ┌──────────────────────────┐
                │ Upcoming Feature Builder │
                │  upcoming_match_features │
                └─────────────┬────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │ Prediction Generator     │
                │ writes to predictions    │
                └─────────────┬────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │ SHAP Explanation Layer   │
                │ writes to explanations   │
                └─────────────┬────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │ FastAPI Backend          │
                │ /matches /predictions    │
                └─────────────┬────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │ Next.js Frontend         │
                │ Dashboard + match view   │
                └──────────────────────────┘
```

---

# 3. Core Layers

# 3.1 Data Layer

The database is the central system of record.

## Main tables

### `matches`
Stores:
- season
- teams
- start time
- venue
- toss info
- winner
- completed flag
- competition

### `teams`
Stores IPL team master data.

### `players`
Stores player identities.

### `venues`
Stores venue / city / country.

### `player_match_stats`
Stores player-level match statistics such as:
- runs
- balls faced
- wickets
- overs bowled
- runs conceded

### `player_form_snapshots`
Stores rolling player form metrics such as:
- batting form score
- bowling form score
- batting strike rate
- wickets average
- bowling economy

### `team_form_snapshots`
Stores team-level rolling metrics:
- elo rating
- batting strength
- bowling strength
- spin strength
- pace strength
- death overs strength
- all-rounder balance

### `squads`
Stores team squad membership by season:
- season
- team_id
- player_id
- role
- is_overseas

### `probable_xi`
Stores estimated or generated playing XIs:
- match_id
- team_id
- player_id
- source
- as_of_time_utc
- batting_order_hint
- confidence

### `predictions`
Stores generated prediction records:
- match_id
- model version reference
- win probabilities
- confidence score
- feature payload
- timestamp

### `explanations`
Stores SHAP explanation artifacts:
- prediction_id
- method
- base_value
- shap_values_json
- top_features_json

### `model_versions`
Stores model metadata and artifact references.

---

# 3.2 Feature Engineering Layer

Feature engineering is split into two main flows:

## A. Historical feature generation
Used for model training.

Builds a row per completed match using only data available before that match.

Examples:
- `elo_diff`
- `batting_strength_diff`
- `bowling_strength_diff`
- `probable_xi_batting_form_diff`
- `top_order_strength_diff`
- `death_bowling_strength_diff`

## B. Upcoming feature generation
Used for inference on future fixtures.

Builds the same feature schema for:
- seeded future matches
- live-loaded future matches

Output file:
- `upcoming_match_features.csv`

---

# 3.3 Probable XI Layer

The probable XI layer estimates likely playing XIs before official lineups are known.

## Evolution of XI logic

### Initial version
- recent form heuristic
- best recent players by score

### Improved version
- restricted to current squad
- role-aware selection
- balanced team composition
- overseas cap
- batting order ranking

## Current XI goals
- at least one wicketkeeper
- role-balanced lineup
- form-aware
- recent-usage aware
- realistic selection from squad table

This layer strongly influences model quality.

---

# 3.4 Modeling Layer

## Model 1 — Logistic Regression
Purpose:
- stable baseline
- interpretable
- easy to debug
- easy to explain with SHAP

Current honest baseline:
- accuracy around mid-50s after leakage removal

## Model 2 — XGBoost
Purpose:
- capture nonlinear interactions
- eventually outperform logistic regression
- support richer explanation

Current status:
- underperforming baseline
- requires feature quality and hyperparameter tuning

---

# 3.5 Explainability Layer

SHAP is used to explain each prediction.

For each prediction:
- compute SHAP values for all model features
- select top impactful features
- store top drivers in `explanations`

## Example drivers
- `probable_xi_count_diff`
- `death_overs_strength_diff`
- `bowling_strength_diff`
- `elo_diff`
- `spin_strength_diff`

This layer enables explainable AI in the UI.

---

# 3.6 Backend Layer (FastAPI)

The backend exposes model outputs and explanations to the frontend.

## Main routes

### Health
```http
GET /health
```

### Upcoming matches
```http
GET /matches/upcoming
```

### Match prediction
```http
GET /predictions/match/{match_id}
```

### List predictions
```http
GET /predictions
```

### Match explanations
```http
GET /predictions/explanations/{match_id}
```

The backend also acts as the integration point between:
- database
- ML artifacts
- frontend

---

# 3.7 Frontend Layer (Next.js)

The frontend provides the product surface for the system.

## Pages

### `/`
Main page:
- shows matches
- allows prediction fetch
- displays probability + confidence
- can show explanations

### `/predictions`
Dashboard page:
- shows saved prediction records
- shows future fixtures
- shows confidence
- shows favorite
- can render SHAP explanations

## Frontend responsibilities
- consume backend JSON
- render prediction cards
- show confidence / favorite
- show explanation drivers

---

# 4. Data Flow in Detail

# 4.1 Historical Training Flow

```text
Historical Match JSON / Stats
        ↓
ETL / ingest scripts
        ↓
matches / player_match_stats / players / teams
        ↓
player_form_snapshots / team_form_snapshots
        ↓
generate_historical_xi
        ↓
historical feature builder
        ↓
pre_match_features.csv
        ↓
train_logreg / train_xgb
        ↓
model artifact + MLflow
```

---

# 4.2 Upcoming Prediction Flow

```text
Future Fixtures + Current Squads
        ↓
matches (completed=false)
squads (season=2026)
        ↓
generate_probable_xi
        ↓
build_upcoming_features
        ↓
upcoming_match_features.csv
        ↓
predict_upcoming
        ↓
predictions table
        ↓
explain_predictions
        ↓
explanations table
        ↓
FastAPI
        ↓
Next.js dashboard
```

---

# 5. Deployment Model

The system is containerized with Docker Compose.

## Services

### `postgres`
Stores all structured data.

### `backend`
Runs FastAPI and serves API endpoints.

### `frontend`
Runs Next.js UI.

### `trainer`
Runs ML scripts and feature pipelines.

### `mlflow`
Optional experiment tracking service.

---

# 6. Design Principles

## 6.1 Separation of concerns
Each layer has a clear responsibility:
- storage
- feature building
- training
- prediction
- explanation
- serving
- UI

## 6.2 Reproducibility
Artifacts are generated through scripts, not notebooks.

## 6.3 Explainability
Predictions are not only stored — they are interpretable.

## 6.4 Modularity
Individual pipelines can be rerun independently:
- retrain model
- rebuild features
- regenerate XI
- rerun explanations

## 6.5 Product thinking
The system is not only an ML pipeline — it is an end-user prediction product.

---

# 7. Known Gaps / Current Limitations

- Official live fixture ingestion still needs full parser completion
- Some future fixtures may currently be manually seeded
- XGBoost still needs tuning
- Probable XI can be further improved with stricter cricket logic
- Some feature engineering pipelines still need cleanup and consistency
- UI explanations currently use raw feature names unless mapped to friendly labels

---

# 8. Planned Improvements

## Modeling
- improve XGBoost
- hyperparameter tuning
- calibration
- time-based validation
- possible stacking / ensemble

## Feature Engineering
- richer venue features
- toss-aware features
- squad freshness
- batting/bowling interaction features

## XI Logic
- role minimum/maximum constraints
- better wicketkeeper detection
- better overseas logic
- recent injury/unavailability handling

## Explainability
- better UI labels
- grouped explanations
- comparison charts

## Product
- richer dashboard
- team pages
- historical prediction archive
- odds comparison and model edge

---

# 9. Why This Architecture Is Strong

This project demonstrates the ability to build:

- data ingestion pipelines
- structured storage systems
- feature engineering pipelines
- ML model training pipelines
- explainable prediction services
- REST APIs
- frontend analytics dashboards
- Dockerized local deployment

It is not just a model — it is a full applied ML system.

---

# 10. Summary

The IPL Predictor architecture is a **full-stack, production-style machine learning platform** that combines:

- data engineering
- sports analytics
- machine learning
- explainable AI
- API engineering
- frontend product design

This architecture makes the project highly relevant for:
- Data Engineer roles
- Machine Learning Engineer roles
- Applied Data Scientist roles
- Analytics / Product Data roles
