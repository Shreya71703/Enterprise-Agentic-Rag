"""
Enterprise RAG — Embedding Service

Shared embedding model factory and batch embedding utilities.
Extracted from the chunker module so it can be reused across
the ingestion pipeline, vector store, and query layers.

Supports:
    - Google Gemini text-embedding-004 (primary)
    - HuggingFace sentence-transformers (local fallback)
    - Batch embedding with progress tracking
    - Automatic retry on transient failures
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from config.settings import EmbeddingSettings, get_settings

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding model factory
# ---------------------------------------------------------------------------
def create_embedding_model(settings: EmbeddingSettings | None = None) -> Embeddings:
    """
    Create an embedding model based on configuration.

    Resolution order (when provider='auto'):
        1. Google Gemini (if API key is set)
        2. HuggingFace sentence-transformers (local, no key needed)

    Raises:
        RuntimeError: If no embedding model could be initialized.
    """
    settings = settings or get_settings().embedding
    errors: list[str] = []

    providers_to_try = []
    if settings.provider == "auto":
        providers_to_try = ["google", "huggingface"]
    else:
        providers_to_try = [settings.provider]

    for provider in providers_to_try:
        if provider == "google":
            if not settings.google_api_key:
                errors.append("Google: GOOGLE_API_KEY not set")
                continue
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings

                model = GoogleGenerativeAIEmbeddings(
                    model=settings.google_model,
                    google_api_key=settings.google_api_key,
                )
                logger.info(f"✅ Embedding model loaded: Google Gemini ({settings.google_model})")
                return model
            except Exception as e:
                errors.append(f"Google: {e}")
                continue

        elif provider == "huggingface":
            try:
                from langchain_huggingface import HuggingFaceEmbeddings

                model = HuggingFaceEmbeddings(
                    model_name=settings.hf_model,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
                logger.info(f"✅ Embedding model loaded: HuggingFace ({settings.hf_model})")
                return model
            except Exception as e:
                errors.append(f"HuggingFace: {e}")
                continue

    error_detail = "; ".join(errors)
    raise RuntimeError(
        f"Could not initialize any embedding model. Tried: {providers_to_try}. "
        f"Errors: {error_detail}"
    )


# ---------------------------------------------------------------------------
# Batch Embedding Service
# ---------------------------------------------------------------------------
class EmbeddingService:
    """
    High-level embedding service with batch processing and progress tracking.

    Usage:
        service = EmbeddingService()
        vectors = service.embed_texts(["Hello world", "Another text"])
        dimension = service.get_dimension()
    """

    def __init__(
        self,
        model: Embeddings | None = None,
        settings: EmbeddingSettings | None = None,
    ):
        self._model = model
        self._settings = settings
        self._initialized = False
        self._dimension: int | None = None

    def _lazy_init(self) -> None:
        """Initialize embedding model on first use."""
        if self._initialized:
            return
        if self._model is None:
            self._model = create_embedding_model(self._settings)
        self._initialized = True

    @property
    def model(self) -> Embeddings:
        """Get the underlying embedding model."""
        self._lazy_init()
        return self._model  # type: ignore[return-value]

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> list[list[float]]:
        """
        Embed a list of texts in batches with progress tracking.

        Args:
            texts: List of text strings to embed.
            batch_size: Number of texts per batch.
            show_progress: Whether to log progress.

        Returns:
            List of embedding vectors (each a list of floats).
        """
        self._lazy_init()

        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_num = (i // batch_size) + 1

            if show_progress:
                logger.info(f"   🧮 Embedding batch {batch_num}/{total_batches} ({len(batch)} texts)")

            try:
                embeddings = self._model.embed_documents(batch)  # type: ignore[union-attr]
                all_embeddings.extend(embeddings)
            except Exception as e:
                # Retry once after a short delay
                logger.warning(f"   ⚠️  Batch {batch_num} failed ({e}), retrying in 2s...")
                time.sleep(2)
                embeddings = self._model.embed_documents(batch)  # type: ignore[union-attr]
                all_embeddings.extend(embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query text.

        Uses embed_query (vs embed_documents) which may apply different
        processing for some models (e.g., instruction-prefixed models).
        """
        self._lazy_init()
        return self._model.embed_query(text)  # type: ignore[union-attr]

    def get_dimension(self) -> int:
        """
        Get the dimensionality of the embedding vectors.

        Embeds a sample text to determine the dimension on first call.
        """
        if self._dimension is None:
            sample = self.embed_query("dimension probe")
            self._dimension = len(sample)
            logger.info(f"   📐 Embedding dimension: {self._dimension}")
        return self._dimension
