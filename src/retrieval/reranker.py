"""
Enterprise RAG — Re-ranking Service

Re-ranks retrieved documents relative to the query using a Cross-Encoder
or Cohere's Re-rank API. Cross-encoders examine query and document text
jointly, producing highly accurate relevance scores that are superior
to standard vector dot-product/cosine similarity scores.

Supports:
    - Local CrossEncoder (sentence-transformers)
    - Cohere Re-ranker (via API key if provided)
    - Fallback: No-op re-ranking (returns documents as-is with default scores)
"""

from __future__ import annotations

import os
import logging
from typing import TYPE_CHECKING

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Re-ranking Service
# ---------------------------------------------------------------------------
class RerankingService:
    """
    Reranks document search results to ensure top-K relevance.

    Resolution:
        1. Cohere API (if COHERE_API_KEY is found in env)
        2. Local HuggingFace Cross-Encoder (MS-MARCO MiniLM default)
        3. Fallback: No-op (maintains original ordering)
    """

    def __init__(
        self,
        local_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_local_fallback: bool = True,
    ):
        self.local_model_name = local_model_name
        self.use_local_fallback = use_local_fallback

        self._cohere_client = None
        self._local_model = None
        self._provider = "fallback"
        self._initialized = False

    def _lazy_init(self) -> None:
        """Initialize the re-ranker provider on first use."""
        if self._initialized:
            return
        self._initialized = True

        cohere_key = os.environ.get("COHERE_API_KEY", "")
        if cohere_key:
            try:
                import cohere
                self._cohere_client = cohere.Client(cohere_key)
                self._provider = "cohere"
                logger.info("✅ Re-ranker initialized: Cohere API")
                return
            except Exception as e:
                logger.warning(f"⚠️  Failed to init Cohere client ({e}). Trying local.")

        if self.use_local_fallback:
            try:
                from sentence_transformers import CrossEncoder

                # Load a lightweight Cross-Encoder model locally
                self._local_model = CrossEncoder(
                    self.local_model_name,
                    max_length=512,
                    device="cpu",
                )
                self._provider = "local"
                logger.info(f"✅ Re-ranker initialized: Local CrossEncoder ({self.local_model_name})")
                return
            except Exception as e:
                logger.warning(f"⚠️  Failed to load local CrossEncoder ({e}). Using no-op fallback.")

        logger.info("ℹ️  Using no-op fallback re-ranker (no-op)")
        self._provider = "fallback"

    @property
    def provider(self) -> str:
        self._lazy_init()
        return self._provider

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
    ) -> list[tuple[Document, float]]:
        """
        Re-rank documents relative to a query.

        Args:
            query: User's query.
            documents: Retrieved document list.
            top_k: Top results to return after re-ranking.

        Returns:
            List of (Document, rerank_score) tuples sorted by score descending.
        """
        self._lazy_init()

        if not documents:
            return []

        if self._provider == "cohere" and self._cohere_client:
            return self._rerank_cohere(query, documents, top_k)
        elif self._provider == "local" and self._local_model:
            return self._rerank_local(query, documents, top_k)
        else:
            return self._rerank_fallback(documents, top_k)

    # -------------------------------------------------------------------
    # Provider Implementations
    # -------------------------------------------------------------------
    def _rerank_cohere(
        self,
        query: str,
        documents: list[Document],
        top_k: int,
    ) -> list[tuple[Document, float]]:
        """Re-rank using Cohere's hosted API."""
        try:
            texts = [doc.page_content for doc in documents]
            # Use cohere client v5.x+ structure
            response = self._cohere_client.rerank(
                query=query,
                documents=texts,
                top_n=top_k,
                model="rerank-english-v3.0",
            )

            reranked_results = []
            for result in response.results:
                orig_doc = documents[result.index]
                reranked_results.append((orig_doc, float(result.relevance_score)))

            return reranked_results
        except Exception as e:
            logger.error(f"❌ Cohere rerank failed: {e}. Falling back to local/fallback.")
            if self._local_model is None and self.use_local_fallback:
                # Attempt to initialize local on-the-fly
                self._initialized = False
                self.use_local_fallback = True
                self._lazy_init()
                if self._local_model:
                    return self._rerank_local(query, documents, top_k)
            return self._rerank_fallback(documents, top_k)

    def _rerank_local(
        self,
        query: str,
        documents: list[Document],
        top_k: int,
    ) -> list[tuple[Document, float]]:
        """Re-rank using local Cross-Encoder sentence-transformers."""
        # Create (query, document_text) pairs
        pairs = [[query, doc.page_content] for doc in documents]

        # Predict scores (higher = more relevant)
        scores = self._local_model.predict(pairs)

        # Zip and sort descending
        scored_docs = list(zip(documents, [float(s) for s in scores]))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return scored_docs[:top_k]

    def _rerank_fallback(
        self,
        documents: list[Document],
        top_k: int,
    ) -> list[tuple[Document, float]]:
        """No-op fallback: return documents in original order with dummy scores."""
        logger.info("   ⚠️  Applying fallback re-ranking (original search ordering kept)")
        # Give decreasing dummy scores
        return [(doc, 1.0 - (idx * 0.05)) for idx, doc in enumerate(documents[:top_k])]
