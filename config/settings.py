"""Centralised application settings loaded from environment variables.

All tuneable constants live here; no magic numbers elsewhere in the codebase.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROMPTS_DIR = Path(__file__).parent / "prompts"


class Settings(BaseSettings):
    """Application-wide configuration."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Ollama
    ollama_base_url: str = Field("http://localhost:11434", alias="OLLAMA_BASE_URL")

    # LLM — two tiers to spread load across models
    #   heavy: synthesis, claim extraction, factual judge, baseline answer
    #   light: query analysis, conflict classification (many short calls)
    llm_model_heavy: str = Field("gemma4:e4b", alias="LLM_MODEL_HEAVY")
    llm_model_light: str = Field("gemma4:e2b", alias="LLM_MODEL_LIGHT")
    llm_temperature: float = Field(0.0, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(4096, alias="LLM_MAX_TOKENS")

    # Qdrant
    qdrant_url: str = Field("http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str = Field("", alias="QDRANT_API_KEY")

    # Embeddings
    embedding_model: str = Field("all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")

    # Retrieval
    retrieval_top_k: int = Field(5, alias="RETRIEVAL_TOP_K")
    retrieval_score_threshold: float = Field(0.5, alias="RETRIEVAL_SCORE_THRESHOLD")

    # Scoring weights
    weight_temporal: float = Field(0.35, alias="WEIGHT_TEMPORAL")
    weight_authority: float = Field(0.40, alias="WEIGHT_AUTHORITY")
    weight_graph: float = Field(0.25, alias="WEIGHT_GRAPH")

    # Web search (Tavily)
    tavily_api_key: str = Field("", alias="TAVILY_API_KEY")
    tavily_max_results: int = Field(5, alias="TAVILY_MAX_RESULTS")

    # Deep research agent
    deep_research_max_iterations: int = Field(5, alias="DEEP_RESEARCH_MAX_ITERATIONS")

    # MCP server
    mcp_server_port: int = Field(8100, alias="MCP_SERVER_PORT")

    # API
    api_host: str = Field("0.0.0.0", alias="API_HOST")
    api_port: int = Field(8000, alias="API_PORT")
    api_reload: bool = Field(True, alias="API_RELOAD")

    # Logging
    log_level: str = Field("INFO", alias="LOG_LEVEL")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
