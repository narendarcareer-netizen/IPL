# 📌 IPL Predictor — Project Status / AI Handoff

This file is the **handoff context** for continuing this project with any future AI assistant, collaborator, or future development session.

It explains:

- what this project does
- what is completed
- what is working
- what is still fake/manual
- what is pending
- what should be done next

---

# 1. Project Goal

Build a **production-style IPL Match Prediction System** that predicts IPL match outcomes before the toss using:

- historical IPL data
- team strength features
- player form features
- probable XI generation
- machine learning models
- SHAP explanations
- FastAPI backend
- Next.js frontend
- PostgreSQL storage
- Dockerized local deployment

The project is designed to look like a **real-world ML + Data Engineering system** suitable for portfolio, interviews, and recruiter demos.

---

# 2. Current Stack

## Backend
- FastAPI
- SQLAlchemy

## Frontend
- Next.js / React

## Database
- PostgreSQL

## ML / Data
- Python
- pandas
- scikit-learn
- XGBoost
- SHAP
- MLflow

## Infra / Runtime
- Docker
- Docker Compose

---

# 3. Current Architecture Summary

The system currently supports:

- historical training data generation
- model training
- model artifact saving
- prediction generation for future matches
- SHAP explanation generation
- API serving
- frontend dashboard rendering

Key docs:
- `README.md`
- `RUNBOOK.md`
- `ARCHITECTURE.md`

---

# 4. What Is Fully Working Right Now

## 4.1 Dockerized local app
Working:
- PostgreSQL container
- FastAPI backend
- Next.js frontend
- trainer container for ML jobs
- optional MLflow

## 4.2 Database pipeline
Working:
- DB connection
- historical data storage
- predictions table
- explanations table
- probable_xi table
- model_versions table

## 4.3 Historical feature engineering
Working:
- historical pre-match feature CSV generation
- leakage check added to training pipeline
- leakage issue detected and corrected

## 4.4 Logistic regression model
Working:
- trains successfully
- saved as:
  - `/app/ml/artifacts/logreg_prematch.joblib`
- logged to MLflow
- used for local predictions

## 4.5 Prediction generation
Working:
- predictions inserted into DB
- predictions shown in UI
- future seeded fixtures produce predictions

## 4.6 SHAP explanations
Working:
- SHAP explanations generated successfully
- stored in DB
- explanation rows exist for recent predictions

## 4.7 Frontend
Working:
- home page loads matches
- prediction dashboard loads saved predictions
- future 2026 manual fixtures visible
- UI shows:
  - favorite
  - confidence
  - team probabilities

---

# 5. What Is Working But Still “Manual / Placeholder”

These are functioning, but not fully production-ready.

## 5.1 Future 2026 fixtures
Status:
- currently **manual seeded fixtures**
- not yet fully sourced from live official IPL fixture ingestion

This means:
- UI shows future-looking 2026 matches
- but they are currently manually inserted / seeded

## 5.2 Current squads
Status:
- current squad loading partially works
- squad data exists in DB
- but some ingestion/scraping still needs hardening

## 5.3 Probable XI logic
Status:
- probable XI is working
- but still heuristic-based
- not yet fully cricket-intelligent

Current logic is usable but can be significantly improved.

---

# 6. What Was Fixed During Development

## 6.1 Missing model artifact
Issue:
- prediction script failed because model file was missing

Fix:
- training script updated to save:
  - `/app/ml/artifacts/logreg_prematch.joblib`

---

## 6.2 Predictions table mismatch
Issue:
- insert failed because schema and insert columns didn’t match

Fix:
- aligned insert logic to actual `predictions` schema
- inserted valid `model_version_id`

---

## 6.3 `model_versions` missing row
Issue:
- prediction pipeline failed because model metadata didn’t exist

Fix:
- inserted `model_versions` row for:
  - `logreg_prematch`

---

## 6.4 Leakage in training data
Issue:
- model showed fake-perfect metrics:
  - accuracy = 1.0
  - log loss ≈ 0

Cause:
- leakage via `recent_win_pct_diff`

Fix:
- leakage check added
- suspicious feature detected
- honest baseline recovered

---

## 6.5 Missing Docker dependencies
Issues encountered:
- `httpx` missing
- `beautifulsoup4` missing
- `lxml` version mismatch

Fix:
- dependencies added / adjusted in `requirements.txt`

---

## 6.6 SHAP explanation table mismatch
Issue:
- explanation schema expected one structure, script inserted another

Fix:
- explanation script adapted to current DB shape
- SHAP explanations now successfully stored

---

# 7. Current Model Performance (Honest Status)

## Logistic Regression
Current realistic performance is approximately:

- accuracy ≈ **55–56%**
- log loss ≈ **0.69**
- brier score ≈ **0.249**

This is the **real baseline after leakage removal**.

## XGBoost
Status:
- currently underperforming baseline
- needs tuning / better features / better XI quality

---

# 8. Current Prediction Product Status

The app currently behaves like this:

## User flow
1. open frontend
2. see upcoming/manual future matches
3. fetch predictions
4. view probabilities and confidence
5. optionally view SHAP explanation drivers

## Current dashboard shows
- match ID
- teams
- start time
- model name
- favorite
- confidence
- probabilities

This is already demo-worthy.

---

# 9. Files That Matter Most Right Now

These are the most important files for continuing development.

## Core docs
- `README.md`
- `RUNBOOK.md`
- `ARCHITECTURE.md`
- `PROJECT_STATUS.md`

## Backend
- `api/routes/health.py`
- `api/routes/matches.py`
- `api/routes/predictions.py`
- backend DB session / services files

## Frontend
- `frontend/pages/index.tsx`
- `frontend/pages/predictions.tsx`
- `frontend/lib/api.ts`

## ML
- `ml/train_logreg.py`
- `ml/train_xgb.py`
- `ml/build_upcoming_features.py`
- `ml/predict_upcoming.py`
- `ml/explain_predictions.py`
- `ml/generate_probable_xi.py`
- `ml/generate_historical_xi.py`
- `ml/load_official_fixtures.py`
- `ml/load_current_squads.py`

---

# 10. What Is Pending / Not Yet Complete

These are the main unfinished parts.

## 10.1 Reach stronger model quality
Goal:
- improve from ~55% baseline toward ~65%+ honest accuracy

Needed:
- better feature engineering
- better XI quality
- stronger validation
- XGBoost tuning
- possibly ensemble approach

---

## 10.2 Improve probable XI logic
Current probable XI is still too simple.

Needs:
- role-based team composition
- wicketkeeper logic
- overseas player cap
- bowling depth logic
- realistic batting order
- squad-aware selection only

This is one of the highest-leverage improvements.

---

## 10.3 Better live / official future fixtures
Current status:
- future matches are manually seeded

Need:
- stable official fixtures loader
- correct venue / team mapping
- automatic insertion into `matches`

---

## 10.4 Better current squad ingestion
Need:
- stable squad ingestion for all IPL teams
- reliable player-to-team mapping
- optional manual override file

---

## 10.5 Better SHAP frontend rendering
Backend SHAP generation works, but UI can be improved.

Need:
- show top drivers per match card
- map raw feature names to friendly labels
- optionally show positive vs negative impact styling

---

## 10.6 Better model serving design
Currently predictions are mostly batch-generated and stored.

Future improvements:
- cleaner on-demand inference route
- model registry handling
- model fallback logic
- prediction versioning

---

# 11. Recommended Next Priorities (Best Order)

This is the best next development order.

## Priority 1 — Improve XI logic
Reason:
- strongest impact on feature quality
- strongest path to better model performance

## Priority 2 — Improve model honestly
Reason:
- current 55–56% is acceptable baseline
- goal is 60–65%+ realistic improvement

## Priority 3 — Improve SHAP UI
Reason:
- very strong for interviews and demos

## Priority 4 — Replace manual fixtures with real official fixtures
Reason:
- makes the app feel more real / live

## Priority 5 — Add better dashboard polish
Reason:
- improves recruiter / hiring manager impact

---

# 12. Recommended “Next AI Assistant” Prompt

If continuing this project with another AI assistant, use this:

## Prompt
Read these files first:
- `README.md`
- `RUNBOOK.md`
- `ARCHITECTURE.md`
- `PROJECT_STATUS.md`

Then help me continue the IPL Predictor project.

Current status:
- Dockerized full-stack app is running
- predictions are being generated
- SHAP explanations are stored
- frontend is showing predictions
- future fixtures are currently manual
- current honest model baseline is around 55–56%
- probable XI logic still needs improvement
- goal is to improve model quality, XI realism, and explanation UI

Please continue from the pending items section and suggest the best next implementation step.

---

# 13. Honest Project Positioning

This project is currently strong enough to describe as:

> “An end-to-end explainable sports ML system with data pipelines, feature engineering, prediction serving, SHAP explanations, Dockerized deployment, FastAPI APIs, PostgreSQL storage, and a React/Next.js frontend dashboard.”

That is already very strong for:
- Data Engineer roles
- ML Engineer roles
- Data Scientist roles
- Analytics Engineering roles

---

# 14. Final Truth

## What this project is right now
A **real, working local end-to-end ML product**.

## What it is not yet
A fully live, production-grade, continuously updating sports prediction platform.

## What would make it feel “Goldman / FAANG level”
- stronger XI intelligence
- better model quality
- cleaner live data ingestion
- richer explanation UX
- cleaner system hardening

That is the next stage.
