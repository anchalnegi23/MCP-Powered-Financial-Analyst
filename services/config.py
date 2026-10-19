"""Central configuration. Every secret enters the process here and nowhere else."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed settings shared by MCP servers, the agent and the API."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # LLM
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    # The provider is inferred from this name: 'claude-*' uses Anthropic, anything
    # else uses OpenAI. See agent/llm.py.
    llm_model: str = "claude-sonnet-5"

    # Data providers (all optional: yfinance needs no credentials)
    fmp_api_key: str | None = None
    alpha_vantage_api_key: str | None = None
    newsapi_api_key: str | None = None

    # Persistence
    database_url: str = "postgresql://analyst:change-me-locally@localhost:5432/financial_analyst"

    # Cache
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = Field(default=900, ge=0)

    # RAG
    chroma_persist_dir: str = "./data/chroma"

    # App
    api_base_url: str = "http://localhost:8000"
    log_level: str = "INFO"
    http_timeout_seconds: float = Field(default=15.0, gt=0)


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton. Call this instead of instantiating Settings()."""
    return Settings()
