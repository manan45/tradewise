from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    APP_NAME: str = "TradewiseAI"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # API Settings
    API_PREFIX: str = "/api"
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # Paths
    MODEL_PATH: str = "./models/"
    SESSION_SAVE_DIR: str = "./sessions/"
    LOG_DIR: str = "./logs/"
    DATA_DIR: str = "./data/"
    
    # Trading Settings
    DEFAULT_LOOKBACK_PERIODS: int = 100
    DEFAULT_FORECAST_HORIZON: int = 24
    CONFIDENCE_THRESHOLD: float = 0.7
    RISK_REWARD_MIN: float = 2.0
    
    class Config:
        env_file = ".env"

settings = Settings()
