"""Centralised application settings loaded from environment variables.

All tuneable constants live here; no magic numbers elsewhere in the codebase.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROMPTS_DIR = Path(__file__).parent / "prompts"


class Settings(BaseSettings):
    """Application-wide configuration."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Anthropic / LLM
    anthropic_api_key: str = Field(..., alias="ANTHROPIC_API_KEY")
    llm_model: str = Field("claude-sonnet-4-6", alias="LLM_MODEL")
    llm_temperature: float = Field(0.0, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(4096, alias="LLM_MAX_TOKENS")

    # Qdrant
    qdrant_url: str = Field("http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str = Field("", alias="QDRANT_API_KEY")

    # Embeddings
    embedding_model: str = Field("all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")

    # Retrieval
    retrieval_top_k: int = Field(10, alias="RETRIEVAL_TOP_K")
    retrieval_score_threshold: float = Field(0.5, alias="RETRIEVAL_SCORE_THRESHOLD")

    # Scoring weights
    weight_temporal: float = Field(0.35, alias="WEIGHT_TEMPORAL")
    weight_authority: float = Field(0.40, alias="WEIGHT_AUTHORITY")
    weight_graph: float = Field(0.25, alias="WEIGHT_GRAPH")

    # API
    api_host: str = Field("0.0.0.0", alias="API_HOST")
    api_port: int = Field(8000, alias="API_PORT")
    api_reload: bool = Field(True, alias="API_RELOAD")

    # Logging
    log_level: str = Field("INFO", alias="LOG_LEVEL")


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    pass
