# 🤖 AI Context — IPL Predictor

This file is the **primary AI handoff file** for this repository.

If you are an AI coding assistant (ChatGPT, Copilot, Cursor, Claude, Gemini, etc.), read this file first before making changes.

This file tells you:
- what this project is
- what is already working
- what is incomplete
- what files matter most
- what to avoid breaking
- what the next priorities are

---

# 1. Project Identity

This repo is an **end-to-end IPL Match Prediction System** built as a portfolio-quality machine learning + data engineering + product project.

It predicts IPL match outcomes using:

- historical IPL match data
- team strength features
- player form features
- probable playing XI logic
- ML models
- SHAP explanations
- FastAPI backend
- Next.js frontend
- PostgreSQL database
- Dockerized local development

This project is intended to feel like a **real production-style sports ML platform**, not just a notebook model.

---

# 2. Tech Stack

## Backend
- FastAPI
- SQLAlchemy

## Frontend
- Next.js / React / TypeScript

## Data / ML
- Python
- pandas
- scikit-learn
- XGBoost
- SHAP
- MLflow

## Database
- PostgreSQL

## Infra
- Docker
- Docker Compose

---

# 3. Project State Summary

## Fully working now
- Dockerized local app
- backend routes
- frontend UI
- PostgreSQL integration
- historical feature generation
- logistic regression training
- prediction generation for future matches
- SHAP explanation generation
- predictions displayed in frontend

## Partially working / still manual
- future 2026 fixtures are currently manual / seeded
- squad ingestion is partially automated
- probable XI is heuristic-based and needs improvement
- XGBoost exists but is underperforming
- SHAP UI rendering can be improved

---

# 4. Most Important Current Truths

## Truth 1
The project currently works end-to-end locally.

## Truth 2
The current honest model baseline is about **55–56% accuracy** after leakage was removed.

## Truth 3
A previous fake-perfect model result (accuracy = 1.0) was caused by **feature leakage** and must NOT be reintroduced.

## Truth 4
The most important next improvement is:
**probable XI intelligence + better feature quality**

## Truth 5
The app currently demonstrates a real ML product, even though some future fixtures are manually seeded.

---

# 5. Files That Matter Most

If you need project context, read these in order:

## First priority
1. `AI_CONTEXT.md`
2. `PROJECT_STATUS.md`

## Second priority
3. `ARCHITECTURE.md`
4. `RUNBOOK.md`

## Third priority
5. `README.md`

---

# 6. Most Important Code Areas

## Backend
- `api/routes/health.py`
- `api/routes/matches.py`
- `api/routes/predictions.py`
- prediction service files under backend/app/services

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

# 7. Database Tables That Matter Most

Key tables:

- `matches`
- `teams`
- `players`
- `player_match_stats`
- `player_form_snapshots`
- `team_form_snapshots`
- `squads`
- `probable_xi`
- `predictions`
- `explanations`
- `model_versions`

Do not assume schemas — inspect them before changing insert logic.

---

# 8. Things That Must NOT Be Broken

These are important invariants.

## Do not reintroduce leakage
Leakage previously made the model look fake-perfect.

If editing training / feature logic:
- ensure historical features are built only from information available **before the match**
- be especially careful with:
  - recent form
  - venue history
  - probable XI
  - anything using future outcomes

## Do not break prediction insertion
The `predictions` table schema has already caused issues before.

Before changing insert logic:
- inspect actual DB columns
- ensure `model_version_id` is handled correctly

## Do not break SHAP generation
SHAP currently works and writes explanations to DB.

If changing SHAP:
- keep DB insertion aligned with schema
- keep frontend explanation compatibility in mind

## Do not break Docker paths
All runtime code inside containers should use Linux-style paths like:

```python
/app/ml/artifacts/...
```

NOT local Windows paths.

---

# 9. Current Known Gaps

These are expected unfinished areas:

- future fixtures are still partly manual
- squad ingestion is not fully robust
- probable XI is still too simplistic
- XGBoost is not yet stronger than logistic regression
- frontend explanation UX is basic
- model serving can be made cleaner

These are not bugs unless they regress from current behavior.

---

# 10. Preferred Development Priorities

When deciding what to improve next, use this order:

## Priority 1
Improve probable XI logic

## Priority 2
Improve honest model quality

## Priority 3
Improve SHAP explanation UX

## Priority 4
Replace manual fixtures with better official ingestion

## Priority 5
Improve frontend polish

---

# 11. Preferred AI Working Style

If you are helping modify this project, please follow this style:

## Always do:
- make small safe changes
- explain exactly which file to edit
- prefer copy-paste-ready code
- preserve working code unless necessary
- keep code production-style and clean
- align changes with existing architecture

## Avoid:
- rewriting the whole project unnecessarily
- changing multiple unrelated layers at once
- inventing fake data sources or fake APIs
- introducing leakage into training features
- changing DB schemas casually without reason

---

# 12. How to Continue This Project

If asked “what next?”, the best next answer is usually:

1. improve probable XI logic
2. improve model quality honestly
3. improve SHAP display in UI

That is the highest-value continuation path.

---

# 13. Best Prompt for Future AI Use

If a user asks for help continuing this repo, respond best if prompted like this:

## Prompt
Read:
- `AI_CONTEXT.md`
- `PROJECT_STATUS.md`
- `ARCHITECTURE.md`

Then help continue this IPL Predictor project.

Focus on:
- what is already working
- what is pending
- what should be improved next
- making safe changes without breaking working functionality

---

# 14. Final Reminder for AI Tools

This project is already a real working local system.

Treat it as:
- a product
- an ML pipeline
- a data engineering system
- an interview portfolio project

Do not treat it like a toy notebook project.
