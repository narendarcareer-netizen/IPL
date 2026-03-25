# 🛠️ IPL Predictor Runbook

This document explains how to run the IPL Predictor project locally using **Docker**, including:

- backend setup
- frontend setup
- PostgreSQL
- ML pipelines
- prediction generation
- SHAP explanation generation
- troubleshooting

---

# 📦 Services in this Project

This project uses Docker Compose to run multiple services.

## Main services

- **postgres** → PostgreSQL database
- **backend** → FastAPI API server
- **frontend** → Next.js UI
- **trainer** → ML / feature engineering / prediction jobs
- **mlflow** → ML experiment tracking (optional)

---

# 🚀 1) Start the Full App

From the project root:

```bash
docker compose up --build
```

This starts the main app stack.

---

# 🌐 Local URLs

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

## FastAPI Swagger Docs
```text
http://localhost:8000/docs
```

## MLflow (if enabled)
```text
http://localhost:5000
```

---

# 🐳 2) Docker Basics

## See running containers

```bash
docker ps
```

## See all containers

```bash
docker ps -a
```

## Stop all services

```bash
docker compose down
```

## Stop + remove volumes (careful: deletes DB data)

```bash
docker compose down -v
```

## Rebuild everything from scratch

```bash
docker compose build --no-cache
docker compose up
```

---

# 🧱 3) Start Only the App (without ML jobs)

```bash
docker compose up backend frontend postgres
```

---

# 🤖 4) Run ML Jobs

ML jobs are run using the **trainer** container.

Use this pattern:

```bash
docker compose --profile ml run --rm trainer python -m ml.<module_name>
```

Example:

```bash
docker compose --profile ml run --rm trainer python -m ml.train_logreg
```

---

# 📊 5) Historical Training Pipeline

## Step 1 — Build historical training features

```bash
docker compose --profile ml run --rm trainer python -m ml.build_upcoming_features
```

> Note: In this project, this script currently creates the historical training CSV used by training.

Output:
```text
/app/ml/artifacts/pre_match_features.csv
```

---

## Step 2 — Train Logistic Regression

```bash
docker compose --profile ml run --rm trainer python -m ml.train_logreg
```

Output:
- trained model artifact
- metrics
- MLflow run

Saved model:
```text
/app/ml/artifacts/logreg_prematch.joblib
```

---

## Step 3 — Train XGBoost

```bash
docker compose --profile ml run --rm trainer python -m ml.train_xgb
```

---

# 🏏 6) Probable XI Generation

## Historical probable XI

```bash
docker compose --profile ml run --rm trainer python -m ml.generate_historical_xi
```

This populates:

```text
probable_xi
```

for historical matches.

---

## Upcoming probable XI

```bash
docker compose --profile ml run --rm trainer python -m ml.generate_probable_xi
```

This creates probable lineups for future fixtures.

---

# 📅 7) Future Fixtures / Squads

## Load official fixtures (if scraper is configured)

```bash
docker compose --profile ml run --rm trainer python -m ml.load_official_fixtures
```

## Load current squads (if scraper is configured)

```bash
docker compose --profile ml run --rm trainer python -m ml.load_current_squads
```

If scraping is unavailable or unstable, you can instead use **manual fixture/squad seed files**.

---

# 🧮 8) Build Upcoming Match Features

```bash
docker compose --profile ml run --rm trainer python -m ml.build_upcoming_features
```

Output:
```text
/app/ml/artifacts/upcoming_match_features.csv
```

This file is used for prediction.

---

# 🎯 9) Predict Upcoming Matches

```bash
docker compose --profile ml run --rm trainer python -m ml.predict_upcoming
```

This inserts rows into:

```text
predictions
```

---

# 🔍 10) Generate SHAP Explanations

```bash
docker compose --profile ml run --rm trainer python -m ml.explain_predictions
```

This inserts rows into:

```text
explanations
```

---

# 🗃️ 11) Useful Database Commands

## Open PostgreSQL shell

```bash
docker compose exec postgres psql -U ipl_user -d ipl_db
```

---

## Count predictions

```bash
docker compose exec postgres psql -U ipl_user -d ipl_db -c "SELECT COUNT(*) FROM predictions;"
```

## Count explanations

```bash
docker compose exec postgres psql -U ipl_user -d ipl_db -c "SELECT COUNT(*) FROM explanations;"
```

## Count probable XI rows

```bash
docker compose exec postgres psql -U ipl_user -d ipl_db -c "SELECT COUNT(*) FROM probable_xi;"
```

## Show future IPL matches

```bash
docker compose exec postgres psql -U ipl_user -d ipl_db -P pager=off -c "SELECT match_id, season, start_time_utc, completed, competition FROM matches WHERE competition = 'Indian Premier League' AND completed = false ORDER BY start_time_utc ASC;"
```

## Show recent predictions

```bash
docker compose exec postgres psql -U ipl_user -d ipl_db -P pager=off -c "SELECT match_id, model_name, stage, team1_win_prob, team2_win_prob, confidence_score FROM predictions ORDER BY created_at_utc DESC LIMIT 20;"
```

## Show explanations

```bash
docker compose exec postgres psql -U ipl_user -d ipl_db -P pager=off -c "SELECT explanation_id, prediction_id, method, base_value, top_features_json FROM explanations LIMIT 20;"
```

---

# 🧪 12) Common Local Development Workflow

A normal development workflow looks like this:

## Step A — Start services

```bash
docker compose up --build
```

## Step B — Build / refresh features

```bash
docker compose --profile ml run --rm trainer python -m ml.build_upcoming_features
```

## Step C — Train model

```bash
docker compose --profile ml run --rm trainer python -m ml.train_logreg
```

## Step D — Generate XI

```bash
docker compose --profile ml run --rm trainer python -m ml.generate_probable_xi
```

## Step E — Predict matches

```bash
docker compose --profile ml run --rm trainer python -m ml.predict_upcoming
```

## Step F — Generate SHAP explanations

```bash
docker compose --profile ml run --rm trainer python -m ml.explain_predictions
```

## Step G — Open UI

```text
http://localhost:3000
http://localhost:3000/predictions
```

---

# 🧯 13) Troubleshooting

## Problem: container still using old code

### Fix
Rebuild trainer/backend/frontend:

```bash
docker compose build --no-cache trainer
docker compose build --no-cache backend
docker compose build --no-cache frontend
docker compose up
```

---

## Problem: Python module not found

Example:
```text
No module named ml.load_official_fixtures
```

### Fix
Make sure the file exists inside:

```text
ml/ml/
```

Example:
```text
ml/ml/load_official_fixtures.py
```

Then rebuild trainer:

```bash
docker compose build --no-cache trainer
```

---

## Problem: dependency missing inside Docker

Example:
```text
ModuleNotFoundError: No module named 'httpx'
```

### Fix
Add the package to:

```text
requirements.txt
```

Then rebuild trainer:

```bash
docker compose build --no-cache trainer
```

---

## Problem: PostgreSQL schema mismatch

Example:
```text
column does not exist
null value violates not-null constraint
```

### Fix
Inspect schema:

```bash
docker compose exec postgres psql -U ipl_user -d ipl_db -P pager=off -c "\d+ predictions"
```

Then align your insert SQL to actual columns.

---

## Problem: frontend not showing latest API changes

### Fix

```bash
docker compose build --no-cache frontend
docker compose up frontend
```

---

## Problem: backend route updated but UI still broken

### Fix

```bash
docker compose build --no-cache backend
docker compose up backend
```

---

## Problem: no SHAP explanations showing

### Checklist
- Did predictions get inserted?
- Did explanations get inserted?
- Is backend route reading from `top_features_json`?
- Is frontend calling `/predictions/explanations/{match_id}`?

Check DB:

```bash
docker compose exec postgres psql -U ipl_user -d ipl_db -P pager=off -c "SELECT * FROM explanations LIMIT 20;"
```

---

# 🐧 14) Linux / Docker Notes

This project is run inside Linux-based Docker containers.

So even if your local machine is Windows, your scripts run inside:

```text
/app/
```

Example local path:
```text
C:\Users\yourname\projects\ipl-predictor
```

Inside Docker becomes:
```text
/app
```

So Python paths inside scripts should look like:

```python
MODEL_PATH = "/app/ml/artifacts/logreg_prematch.joblib"
```

NOT like:

```python
MODEL_PATH = "C:\\Users\\..."
```

---

# 📁 15) Important File Locations

## Model artifacts
```text
/app/ml/artifacts/
```

## Training features
```text
/app/ml/artifacts/pre_match_features.csv
```

## Upcoming features
```text
/app/ml/artifacts/upcoming_match_features.csv
```

## ML scripts
```text
/app/ml/
```

---

# ✅ Recommended Order for Fresh Setup

If you are starting from scratch, run in this order:

```bash
docker compose up --build
docker compose --profile ml run --rm trainer python -m ml.generate_historical_xi
docker compose --profile ml run --rm trainer python -m ml.build_upcoming_features
docker compose --profile ml run --rm trainer python -m ml.train_logreg
docker compose --profile ml run --rm trainer python -m ml.generate_probable_xi
docker compose --profile ml run --rm trainer python -m ml.predict_upcoming
docker compose --profile ml run --rm trainer python -m ml.explain_predictions
```

Then open:

```text
http://localhost:3000/predictions
```

---

# 📌 Final Note

This runbook is intentionally written like a real engineering project setup guide.

It is useful for:
- recruiters
- hiring managers
- collaborators
- your future self
- interview walkthroughs
