"""
Enterprise RAG — Centralized Configuration

All configuration is managed through Pydantic Settings, with environment
variable overrides and sensible defaults. This ensures type safety and
makes the system easy to reconfigure for different environments.
"""

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
PROCESSED_DIR = DATA_DIR / "processed"

# Load environment variables from .env file explicitly
load_dotenv(PROJECT_ROOT / ".env")


class EmbeddingSettings(BaseSettings):
    """Configuration for embedding models used in semantic chunking."""

    # Primary: Google Gemini
    google_api_key: str = Field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""),
        description="Google API key for Gemini embeddings",
    )
    google_model: str = Field(
        default="models/gemini-embedding-2",
        description="Google embedding model name",
    )

    # Fallback: HuggingFace (local, no API key)
    hf_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="HuggingFace sentence-transformer model for local fallback",
    )

    # Which provider to try first
    provider: Literal["google", "huggingface", "auto"] = Field(
        default="auto",
        description="'auto' tries Google first, falls back to HuggingFace",
    )


class ChunkerSettings(BaseSettings):
    """Configuration for the semantic chunking pipeline."""

    # Semantic chunker params
    breakpoint_threshold_type: Literal[
        "percentile", "standard_deviation", "interquartile", "gradient"
    ] = Field(
        default="percentile",
        description="Method used to detect semantic breakpoints between sentences",
    )
    breakpoint_threshold_amount: float | None = Field(
        default=None,
        description="Threshold value for breakpoint detection (None = use library default)",
    )

    # Fallback: RecursiveCharacterTextSplitter params
    fallback_chunk_size: int = Field(
        default=1000,
        description="Chunk size in characters for fallback splitter",
    )
    fallback_chunk_overlap: int = Field(
        default=200,
        description="Overlap between chunks for fallback splitter",
    )

    # Minimum chunk length (discard very small chunks)
    min_chunk_length: int = Field(
        default=50,
        description="Discard chunks shorter than this many characters",
    )


class ParserSettings(BaseSettings):
    """Configuration for document parsing."""

    supported_extensions: list[str] = Field(
        default=[".pdf", ".docx", ".doc", ".txt", ".md", ".csv", ".html", ".pptx"],
        description="File extensions the parser will attempt to process",
    )
    strategy: Literal["auto", "fast", "hi_res", "ocr_only"] = Field(
        default="auto",
        description="Unstructured partition strategy",
    )
    include_page_breaks: bool = Field(
        default=True,
        description="Whether to include page break elements",
    )


class Settings(BaseSettings):
    """
    Root configuration for the Enterprise RAG system.

    All settings can be overridden via environment variables or a .env file.
    Nested settings are accessed as attributes: settings.embedding.google_api_key
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Sub-configs
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chunker: ChunkerSettings = Field(default_factory=ChunkerSettings)
    parser: ParserSettings = Field(default_factory=ParserSettings)

    # Global
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging verbosity level",
    )
    data_dir: Path = Field(
        default=DATA_DIR,
        description="Root directory for data files",
    )


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the global Settings singleton, creating it on first call."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
