import os
import mlflow
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

DATABASE_URL = os.environ["DATABASE_URL"]
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")

def fetch_training_frame(engine) -> pd.DataFrame:
    # Placeholder: You will replace with a proper feature view that enforces cutoff_time.
    # For now we assume you materialize a "training_features" view/table:
    q = text("SELECT * FROM training_features ORDER BY match_start_time_utc ASC")
    return pd.read_sql(q, engine)

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("ipl_local_winprob")

    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    df = fetch_training_frame(engine)

    y = df["team1_won"].astype(int).values
    X = df.drop(columns=["team1_won", "match_id", "match_start_time_utc"], errors="ignore")

    # Time-aware CV skeleton
    tscv = TimeSeriesSplit(n_splits=5)

    base = XGBClassifier(
        objective="binary:logistic",
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        eval_metric="logloss",
        n_jobs=4,
    )

    # Calibrate with CV to reduce overconfidence.
    # (Ensure disjointness; in production you may do: train -> validation -> calibrate)
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=tscv)

    with mlflow.start_run():
        calibrated.fit(X, y)
        mlflow.log_params({"model": "xgb+calibration", "calibration": "sigmoid"})

        # Log as a pyfunc model for uniform loading in the API
        mlflow.sklearn.log_model(calibrated, artifact_path="model", registered_model_name="ipl_winprob_xgb")

if __name__ == "__main__":
    main()
