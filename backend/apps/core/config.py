"""Application configuration from environment variables."""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration — all values come from env vars or .env file."""

    # Database
    database_url: str = "sqlite:///./data/visionguard_v2.db"

    # Uploads
    upload_dir: str = str(Path(__file__).resolve().parent.parent.parent / "data" / "uploads")
    max_upload_size_mb: int = 20
    allowed_origins: str = "http://localhost:5173,http://localhost:3000"

    # ML model
    model_path: str = str(Path(__file__).resolve().parent.parent.parent / "models" / "model.joblib")
    model_version: str = "feature-baseline-v1"
    model_loaded: bool = False

    # Static files
    upload_url_base: str = "/uploads"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


settings = Settings()
