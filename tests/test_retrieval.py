"""
Enterprise RAG — Retrieval Tests

Unit and integration tests for BM25 search, Cross-Encoder re-ranking,
Reciprocal Rank Fusion (RRF), and the Hybrid Retriever orchestrator.
"""

from __future__ import annotations

import tempfile
import pytest
from pathlib import Path
from langchain_core.documents import Document

from src.retrieval.bm25 import BM25RetrieverService, tokenize
from src.retrieval.reranker import RerankingService
from src.retrieval import HybridRetriever
from src.embeddings import EmbeddingService
from src.vectorstore import QdrantVectorStore


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def sample_documents() -> list[Document]:
    """Sample documents for indexing."""
    return [
        Document(
            page_content="NovaTech Cortex 3.0 has multi-modal reasoning engine support.",
            metadata={"source_file": "faq.md", "element_type": "NarrativeText"},
        ),
        Document(
            page_content="Enterprise pricing: Starter tier starts at $2,500 monthly fee.",
            metadata={"source_file": "pricing.md", "element_type": "Table"},
        ),
        Document(
            page_content="R&D spends reached record $198 million in fiscal year 2024.",
            metadata={"source_file": "report.md", "element_type": "NarrativeText"},
        ),
    ]


@pytest.fixture
def bm25_service(sample_documents: list[Document]) -> BM25RetrieverService:
    """Builds and returns a BM25 index populated with sample docs."""
    service = BM25RetrieverService()
    service.build_index(sample_documents)
    return service


@pytest.fixture
def reranker() -> RerankingService:
    """Create a reranker (falls back to local Cross-Encoder or fallback)."""
    # Use a dummy local model name or default
    return RerankingService(use_local_fallback=False)  # Force no-op fallback for fast unit tests


# ===================================================================
# Tokenizer & BM25 Tests
# ===================================================================

class TestBM25Retriever:
    """Tests for the BM25RetrieverService."""

    def test_tokenize_lowercases_and_strips_punctuation(self):
        text = "Hello, World! This is a test-string."
        tokens = tokenize(text)
        assert tokens == ["hello", "world", "this", "is", "a", "test-string"]

    def test_build_index(self, bm25_service: BM25RetrieverService):
        assert bm25_service.corpus_size == 3
        assert len(bm25_service.documents) == 3

    def test_search_exact_match(self, bm25_service: BM25RetrieverService):
        results = bm25_service.search("Cortex 3.0", top_k=1)
        assert len(results) == 1
        doc, score = results[0]
        assert "Cortex 3.0" in doc.page_content
        assert score > 0.0

    def test_search_no_match(self, bm25_service: BM25RetrieverService):
        results = bm25_service.search("nonexistent keywords", top_k=5)
        assert results == []

    def test_search_with_filter(self, bm25_service: BM25RetrieverService):
        results = bm25_service.search("Cortex", top_k=5, filter_conditions={"source_file": "faq.md"})
        assert len(results) == 1
        doc, _ = results[0]
        assert doc.metadata["source_file"] == "faq.md"

    def test_save_and_load(self, bm25_service: BM25RetrieverService, tmp_path: Path):
        save_path = tmp_path / "bm25.pkl"
        bm25_service.save(save_path)
        assert save_path.exists()

        new_service = BM25RetrieverService()
        new_service.load(save_path)
        assert new_service.corpus_size == 3
        assert new_service.documents[0].page_content == bm25_service.documents[0].page_content


# ===================================================================
# Re-ranker Tests
# ===================================================================

class TestReranker:
    """Tests for the RerankingService."""

    def test_fallback_rerank(self, reranker: RerankingService, sample_documents: list[Document]):
        # Reranker has use_local_fallback=False, so it goes to fallback (no-op)
        results = reranker.rerank("What is the cost?", sample_documents, top_k=2)
        assert len(results) == 2
        assert results[0][0] == sample_documents[0]
        assert results[1][0] == sample_documents[1]
        assert results[0][1] > results[1][1]  # Dummy scores decrease


# ===================================================================
# Hybrid Retriever Tests
# ===================================================================

class TestHybridRetriever:
    """Tests for the HybridRetriever orchestration."""

    def test_rrf_scoring(self):
        # Mock class/namedtuple for SearchResult representation
        class MockResult:
            def __init__(self, content, score, metadata):
                self.content = content
                self.score = score
                self.metadata = metadata

        vector_res = [
            MockResult("Doc A", 0.9, {}),
            MockResult("Doc B", 0.8, {}),
        ]
        bm25_res = [
            (Document(page_content="Doc B", metadata={}), 15.0),
            (Document(page_content="Doc C", metadata={}), 10.0),
        ]
        candidates = [
            Document(page_content="Doc A", metadata={}),
            Document(page_content="Doc B", metadata={}),
            Document(page_content="Doc C", metadata={}),
        ]

        scores = HybridRetriever._reciprocal_rank_fusion(
            query="test",
            vector_results=vector_res,
            bm25_results=bm25_res,
            candidates=candidates,
            rrf_constant=60
        )

        assert "Doc A" in scores
        assert "Doc B" in scores
        assert "Doc C" in scores
        # Doc B is ranked #2 in vector and #1 in BM25, so it should have highest RRF score
        assert scores["Doc B"] > scores["Doc A"]
        assert scores["Doc B"] > scores["Doc C"]
