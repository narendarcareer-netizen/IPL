import mlflow
import mlflow.pyfunc
from functools import lru_cache
from app.core.config import settings

@lru_cache(maxsize=8)
def load_model(model_uri: str):
    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return mlflow.pyfunc.load_model(model_uri)
