"""
Enterprise RAG — Hybrid Retriever Orchestrator

Integrates BM25 Keyword Search, Vector Semantic Search, Reciprocal Rank
Fusion (RRF), and Cross-Encoder Re-ranking to deliver highly accurate context
retrieval.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document

from src.retrieval.bm25 import BM25RetrieverService
from src.retrieval.reranker import RerankingService

if TYPE_CHECKING:
    from src.embeddings import EmbeddingService
    from src.vectorstore import QdrantVectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hybrid Retriever
# ---------------------------------------------------------------------------
class HybridRetriever:
    """
    Orchestrates Hybrid Retrieval:
        1. Fetch candidates from Vector search (semantic)
        2. Fetch candidates from BM25 search (keyword)
        3. Merge and deduplicate candidates
        4. Re-rank combined candidates using a Cross-Encoder
    """

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embedding_service: EmbeddingService,
        bm25_service: BM25RetrieverService | None = None,
        reranking_service: RerankingService | None = None,
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.bm25 = bm25_service or BM25RetrieverService()
        self.reranker = reranking_service or RerankingService()

    def build_bm25_from_vector_store(self, collection_name: str | None = None) -> None:
        """
        Bootstrap the BM25 index by pulling all documents from the Qdrant collection.
        This allows keyword indexing without requiring raw files to be present on disk.
        """
        name = collection_name or self.vector_store.collection_name
        logger.info(f"📤 Bootstrapping BM25 index from Qdrant collection '{name}'...")

        try:
            # Scroll/retrieve all points from Qdrant
            # We fetch a large number (e.g. 10000) to ensure we cover full database
            scroll_result = self.vector_store._client.scroll(
                collection_name=name,
                limit=10000,
                with_payload=True,
                with_vectors=False,
            )

            points = scroll_result[0]
            docs = []
            for p in points:
                payload = p.payload or {}
                content = payload.get("page_content", "")
                meta = {k: v for k, v in payload.items() if k != "page_content"}
                docs.append(Document(page_content=content, metadata=meta))

            if not docs:
                logger.warning("⚠️  No documents found in Qdrant to build BM25 index.")
                return

            self.bm25.build_index(docs)
            logger.info(f"✅ BM25 index populated with {len(docs)} documents scrolled from Qdrant")

        except Exception as e:
            logger.error(f"❌ Failed to build BM25 from Qdrant: {e}")
            raise e

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        vector_candidates: int = 20,
        bm25_candidates: int = 20,
        filter_conditions: dict[str, Any] | None = None,
        use_reranking: bool = True,
    ) -> list[tuple[Document, float]]:
        """
        Execute hybrid search and re-ranking.

        Args:
            query: User search query.
            top_k: Number of final documents to return.
            vector_candidates: Number of candidates to fetch from vector search.
            bm25_candidates: Number of candidates to fetch from BM25 search.
            filter_conditions: Metadata filters.
            use_reranking: Whether to apply the Cross-Encoder.

        Returns:
            List of (Document, score) tuples.
        """
        logger.info(f"🔍 Hybrid search for: '{query}'")

        # 1. Fetch from Vector Search
        query_embedding = self.embedding_service.embed_query(query)
        vector_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=vector_candidates,
            filter_conditions=filter_conditions,
        )
        logger.info(f"   ✓ Vector search retrieved {len(vector_results)} candidates")

        # 2. Fetch from BM25 Search
        bm25_results = self.bm25.search(
            query=query,
            top_k=bm25_candidates,
            filter_conditions=filter_conditions,
        )
        logger.info(f"   ✓ BM25 search retrieved {len(bm25_results)} candidates")

        # 3. Merge & Deduplicate
        merged_docs: dict[str, Document] = {}

        # Add vector results
        for r in vector_results:
            # Use page content hash or unique signature as key
            doc = Document(page_content=r.content, metadata=r.metadata)
            key = doc.page_content.strip()
            merged_docs[key] = doc

        # Add BM25 results
        for doc, _score in bm25_results:
            key = doc.page_content.strip()
            if key not in merged_docs:
                merged_docs[key] = doc

        candidates = list(merged_docs.values())
        logger.info(f"   ✓ Merged into {len(candidates)} unique candidate(s)")

        if not candidates:
            return []

        # 4. Re-rank or Fallback
        if use_reranking:
            logger.info(f"   🏆 Re-ranking with {self.reranker.provider} re-ranker...")
            reranked = self.reranker.rerank(query, candidates, top_k=top_k)
            return reranked
        else:
            # Reciprocal Rank Fusion (RRF) as fallback scoring mechanism
            logger.info("   🔄 Using Reciprocal Rank Fusion (RRF) to score candidates...")
            rrf_scores = self._reciprocal_rank_fusion(
                query, vector_results, bm25_results, candidates
            )
            scored = [(doc, rrf_scores[doc.page_content.strip()]) for doc in candidates]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]

    # -------------------------------------------------------------------
    # RRF Scoring Helper
    # -------------------------------------------------------------------
    @staticmethod
    def _reciprocal_rank_fusion(
        query: str,
        vector_results: list[Any],
        bm25_results: list[tuple[Document, float]],
        candidates: list[Document],
        rrf_constant: int = 60,
    ) -> dict[str, float]:
        """
        Compute Reciprocal Rank Fusion (RRF) scores for candidates.
        RRF combines rankings from different systems without normalising scores.
        """
        rrf_scores: dict[str, float] = {doc.page_content.strip(): 0.0 for doc in candidates}

        # Build rank maps
        vector_rank = {r.content.strip(): idx + 1 for idx, r in enumerate(vector_results)}
        bm25_rank = {doc.page_content.strip(): idx + 1 for idx, (doc, _) in enumerate(bm25_results)}

        for doc in candidates:
            key = doc.page_content.strip()
            # Vector score contribution
            if key in vector_rank:
                rrf_scores[key] += 1.0 / (rrf_constant + vector_rank[key])
            # BM25 score contribution
            if key in bm25_rank:
                rrf_scores[key] += 1.0 / (rrf_constant + bm25_rank[key])

        return rrf_scores
