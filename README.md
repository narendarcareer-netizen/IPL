# 🏏 IPL Match Outcome Predictor

An end-to-end **machine learning system** for predicting **Indian Premier League (IPL)** match outcomes using:

- historical match data
- team form snapshots
- player form snapshots
- probable playing XI
- engineered pre-match features
- ML models (Logistic Regression / XGBoost)
- explainability (SHAP)
- FastAPI backend
- Next.js frontend
- PostgreSQL database
- Dockerized local deployment

This project is designed as a **production-style sports analytics platform** that combines **data engineering**, **feature engineering**, **ML modeling**, **API serving**, and **frontend visualization**.

---

# 🚀 Project Highlights

## ✅ What this system does

- Predicts **team1 win probability vs team2 win probability**
- Generates predictions for **upcoming IPL fixtures**
- Builds **probable XI** for each team before the toss
- Engineers **team + player + XI-level features**
- Stores predictions in a **PostgreSQL database**
- Exposes predictions through a **FastAPI backend**
- Displays results in a **Next.js dashboard**
- Explains model predictions using **SHAP**

---

# 🧠 Why this project matters

This project demonstrates real-world skills across:

- **Machine Learning Engineering**
- **Data Engineering**
- **Feature Engineering**
- **Backend API Development**
- **Database Design**
- **MLOps / MLflow**
- **Frontend Product Development**
- **Explainable AI**

It is intentionally built like a **portfolio-quality production system**, not just a notebook experiment.

---

# 🏗️ System Architecture

## High-Level Flow

```text
Raw IPL data / fixtures / squads
        ↓
PostgreSQL data warehouse
        ↓
Feature engineering pipelines
        ↓
Historical training dataset
        ↓
ML model training (LogReg / XGBoost)
        ↓
Model artifacts + MLflow tracking
        ↓
Upcoming match feature generation
        ↓
Prediction pipeline
        ↓
Predictions + explanations stored in DB
        ↓
FastAPI backend
        ↓
Next.js frontend dashboard
```

---

# 📂 Project Structure

```text
ipl-predictor/
│
├── api/                          # FastAPI backend
│   ├── routes/
│   │   ├── health.py
│   │   ├── matches.py
│   │   └── predictions.py
│   ├── services/
│   │   └── predict.py
│   └── ...
│
├── frontend/                     # Next.js frontend
│   ├── pages/
│   │   ├── index.tsx
│   │   └── predictions.tsx
│   ├── lib/
│   │   └── api.ts
│   └── ...
│
├── ml/                           # ML pipelines and scripts
│   ├── build_upcoming_features.py
│   ├── generate_probable_xi.py
│   ├── generate_historical_xi.py
│   ├── train_logreg.py
│   ├── train_xgb.py
│   ├── predict_upcoming.py
│   ├── explain_predictions.py
│   ├── load_official_fixtures.py
│   ├── load_current_squads.py
│   └── ...
│
├── sql/                          # SQL setup / migrations / helpers
│
├── artifacts/                    # Model artifacts / CSV outputs
│
├── docker-compose.yml
├── README.md
└── requirements.txt
```

---

# 🧱 Core Components

# 1) Database Layer (PostgreSQL)

The database stores:

- teams
- players
- matches
- team form snapshots
- player form snapshots
- probable XI
- predictions
- explanations
- model versions

### Important tables

- `matches`
- `teams`
- `players`
- `team_form_snapshots`
- `player_form_snapshots`
- `probable_xi`
- `predictions`
- `explanations`
- `model_versions`

---

# 2) Feature Engineering

The model uses **pre-match engineered features** created from historical data.

## Feature categories

### Team strength features
- `elo_diff`
- `batting_strength_diff`
- `bowling_strength_diff`
- `all_rounder_balance_diff`
- `spin_strength_diff`
- `pace_strength_diff`
- `death_overs_strength_diff`

### Team form features
- `recent_win_pct_diff`
- `venue_win_bias_diff`

### Probable XI features
- `probable_xi_count_diff`
- `probable_xi_batting_form_diff`
- `probable_xi_bowling_form_diff`
- `top_order_strength_diff`
- `middle_order_strength_diff`
- `death_bowling_strength_diff`

These features are generated for both:

- **historical completed matches** → for training
- **future fixtures** → for inference

---

# 3) Probable XI Engine

A custom pipeline estimates likely playing XIs before the toss.

## Inputs used
- recent player form
- batting order hints
- player role balance
- team squad composition

## Output
The `probable_xi` table stores:

- `match_id`
- `team_id`
- `player_id`
- `batting_order_hint`
- `confidence`
- `source`

This XI is then aggregated into match-level features.

---

# 4) Machine Learning Models

## Current models

### Logistic Regression
- baseline interpretable model
- fast and stable
- currently strongest baseline

### XGBoost
- nonlinear boosting model
- intended next-stage performance improvement
- supports SHAP very naturally

---

# 5) Explainability (SHAP)

The system generates **SHAP explanations** for predictions.

For each prediction, the system stores:

- base value
- full SHAP vector
- top 5 most important features

Stored in the `explanations` table.

Example explanation drivers:

- `probable_xi_count_diff`
- `death_overs_strength_diff`
- `bowling_strength_diff`
- `elo_diff`
- `spin_strength_diff`

This makes the model much more **transparent and interview-ready**.

---

# 6) Backend API (FastAPI)

The backend serves predictions and explanations.

## Example routes

### Health
```http
GET /health
```

### Upcoming matches
```http
GET /matches/upcoming
```

### Predict a match
```http
GET /predictions/match/{match_id}?stage=pre_toss
```

### List saved predictions
```http
GET /predictions
```

### Get SHAP explanations
```http
GET /predictions/explanations/{match_id}
```

---

# 7) Frontend (Next.js)

The frontend provides a local dashboard for viewing:

- upcoming fixtures
- win probabilities
- confidence score
- favorite team
- prediction timestamp
- SHAP explanations

## Pages

### `/`
Main homepage with upcoming matches and prediction trigger

### `/predictions`
Dashboard view of saved predictions

---

# ⚙️ Tech Stack

## Backend / Data
- Python
- FastAPI
- SQLAlchemy
- PostgreSQL
- Pandas
- NumPy

## Machine Learning
- scikit-learn
- XGBoost
- SHAP
- MLflow
- joblib

## Frontend
- Next.js
- React
- TypeScript

## Infrastructure
- Docker
- Docker Compose

---

# 🐳 Running the Project Locally

## 1) Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

---

## 2) Start services

```bash
docker compose up --build
```

This should start:

- PostgreSQL
- backend API
- frontend UI
- ML services (if configured)

---

# 🔧 Common Local URLs

## Frontend
```text
http://localhost:3000
```

## Prediction Dashboard
```text
http://localhost:3000/predictions
```

## Backend API
```text
http://localhost:8000
```

## API docs
```text
http://localhost:8000/docs
```

## MLflow (if enabled)
```text
http://localhost:5000
```

---

# 🧪 ML Pipeline Workflow

## Step 1 — Build historical training features

```bash
docker compose --profile ml run --rm trainer python -m ml.build_upcoming_features
```

> Note: this script is currently also used to generate the historical training feature CSV.

---

## Step 2 — Train Logistic Regression model

```bash
docker compose --profile ml run --rm trainer python -m ml.train_logreg
```

Outputs:
- trained model artifact
- MLflow run
- metrics

---

## Step 3 — Train XGBoost model

```bash
docker compose --profile ml run --rm trainer python -m ml.train_xgb
```

---

## Step 4 — Generate probable XI

### Historical XI
```bash
docker compose --profile ml run --rm trainer python -m ml.generate_historical_xi
```

### Upcoming XI
```bash
docker compose --profile ml run --rm trainer python -m ml.generate_probable_xi
```

---

## Step 5 — Build upcoming match features

```bash
docker compose --profile ml run --rm trainer python -m ml.build_upcoming_features
```

---

## Step 6 — Predict upcoming matches

```bash
docker compose --profile ml run --rm trainer python -m ml.predict_upcoming
```

This inserts rows into the `predictions` table.

---

## Step 7 — Generate SHAP explanations

```bash
docker compose --profile ml run --rm trainer python -m ml.explain_predictions
```

This inserts rows into the `explanations` table.

---

# 📊 Example Prediction Output

```text
Punjab Kings vs Gujarat Titans
Match #7199 • pre_toss • logreg_prematch
Starts: 2026-03-25T14:00:00+00:00
Favorite: Punjab Kings
Confidence: Medium (22.2%)

Punjab Kings: 61.1%
Gujarat Titans: 38.9%
```

---

# 📈 Current Model Status

## Logistic Regression
- strong baseline
- stable inference
- explainable
- integrated end-to-end

## XGBoost
- currently experimental
- next target for model improvement

---

# ⚠️ Known Limitations

- Current future fixtures may be partially seeded manually depending on data availability
- Some live squad / fixture ingestion pipelines are still under refinement
- Probable XI logic is heuristic and can be further improved
- Model accuracy is still being optimized toward **65%+**

---

# 🔥 Planned Improvements

## Modeling
- Improve XGBoost performance
- Add time-based cross-validation
- Calibrate probabilities
- Ensemble models

## Feature Engineering
- Better player role inference
- Better venue features
- Toss-aware post-toss model
- More robust squad freshness

## Explainability
- Better SHAP visual summaries
- Match-level explanation dashboard
- Friendly explanation labels

## Product / UI
- Better charts and confidence visuals
- Team filters and comparison tools
- Historical prediction archive
- Team-level analytics page

---

# 💼 Why This Project Is Valuable

This project demonstrates the ability to build:

- an end-to-end ML product
- production-style data pipelines
- explainable prediction systems
- full-stack analytics applications

This is highly relevant for roles in:

- Data Science
- Machine Learning Engineering
- Data Engineering
- Analytics Engineering
- Applied AI / ML Product

---

# 👤 Author

**Lalithya Kalluri**

Built as a portfolio project to demonstrate:
- ML system design
- data engineering
- feature engineering
- API development
- frontend analytics product thinking

---

# 📬 Future Extensions

Potential future add-ons:

- live odds integration
- betting edge analytics
- toss prediction branch
- score simulation engine
- fantasy cricket optimization
- player-level performance forecasting

---

# ⭐ If you found this useful

If you like this project, feel free to:

- star the repo
- fork it
- extend it
- use it as inspiration for your own ML systems
