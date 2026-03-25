from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    redis_url: str

    cricket_api_key: str | None = None
    odds_api_key: str | None = None
    odds_api_base_url: str | None = None

    mlflow_tracking_uri: str | None = None

    class Config:
        env_prefix = ""
        case_sensitive = False

settings = Settings()
