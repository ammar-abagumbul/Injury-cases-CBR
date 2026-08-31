"""
Configuration for PSLA extraction experiments.

Uses pydantic-settings to load from environment / .env file.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Project-wide settings, sourced from environment or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Keys and credentials
    # ------------------------------------------------------------------
    OPENROUTER_API_KEY: str = ""
    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_API_VERSION: str = "2024-05-01-preview"

    # ------------------------------------------------------------------
    # Model identifiers
    # ------------------------------------------------------------------
    OPENROUTER_DEEPSEEK_MODEL: str = "deepseek/deepseek-v4-pro"
    OPENROUTER_QWEN_MODEL: str = "qwen/qwen3-32b"
    OPENROUTER_GLM_MODEL: str = "z-ai/glm-4.5"
    OPENROUTER_KIMI_MODEL: str = "moonshotai/kimi-k2"
    OPENROUTER_MINIMAX_MODEL: str = "minimax/minimax-m1"
    AZURE_OPENAI_DEPLOYMENT_NAME: str = "gpt-5-mini"


    OLLAMA_MODEL: str = ""

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    DATA_DIR: Path = Path("data")
    SAMPLE_DIR: Path = Path("reprocess")
    BENCHMARK_DIR: Path = Path("benchmark")
    # OUTPUT_DIR: Path = Path("output")
    OUTPUT_DIR: Path = Path("judgement_extractions")

    # ------------------------------------------------------------------
    # Extraction defaults
    # ------------------------------------------------------------------
    DEFAULT_TEMPERATURE: float = 0.0
    DEFAULT_MAX_TOKENS: int = 4096

    # ------------------------------------------------------------------
    # Ollama
    # ------------------------------------------------------------------
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    DEFAULT_TIMEOUT: int = 60


settings = Settings()
