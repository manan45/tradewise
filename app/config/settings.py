"""Runtime settings — pydantic-settings, sourced from env / .env."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "TraderWise"
    DEBUG: bool = False

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    DB_HOST: str = "postgres-db"
    DB_PORT: int = 5432
    DB_USER: str = "traderwise"
    DB_PASSWORD: str = "traderwise"
    DB_NAME: str = "traderwise"

    REDIS_URL: str = "redis://redis:6379/0"
    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6334
    RABBITMQ_URL: str = "amqp://traderwise:traderwise@rabbitmq:5672/"

    POLYGON_API_KEY: str = ""
    APCA_API_KEY_ID: str = ""
    APCA_API_SECRET_KEY: str = ""
    ALPACA_PAPER: bool = True
    IBKR_HOST: str = "127.0.0.1"
    IBKR_PORT: int = 7497
    IBKR_CLIENT_ID: int = 1
    FRED_API_KEY: str = ""

    ANTHROPIC_API_KEY: str = ""

    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""
    TWILIO_ACCOUNT_SID: str = ""
    TWILIO_AUTH_TOKEN: str = ""
    TWILIO_FROM_NUMBER: str = ""

    NO_NETWORK: bool = False

    @property
    def database_url(self) -> str:
        return (f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}"
                f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}")

    @property
    def database_url_async(self) -> str:
        return (f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}"
                f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}")


settings = Settings()
