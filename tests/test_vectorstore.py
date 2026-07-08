"""
Enterprise RAG — Vector Store Tests

Tests for the embedding service, Qdrant vector store, and the
embed → store → search workflow. Uses in-memory Qdrant for fast,
isolated testing (no disk or network required).
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from src.embeddings import EmbeddingService
from src.vectorstore import QdrantVectorStore, SearchResult


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def embedding_service() -> EmbeddingService:
    """Create an EmbeddingService (uses HuggingFace fallback)."""
    return EmbeddingService()


@pytest.fixture
def in_memory_store() -> QdrantVectorStore:
    """Create an in-memory Qdrant store for testing."""
    store = QdrantVectorStore(mode="memory", collection_name="test_collection")
    return store


@pytest.fixture
def sample_documents() -> list[Document]:
    """Sample documents for testing."""
    return [
        Document(
            page_content="NovaTech achieved revenue of $847 million in fiscal year 2024.",
            metadata={"source_file": "annual_report.md", "element_type": "NarrativeText"},
        ),
        Document(
            page_content="The company has 523 enterprise clients across 30 countries.",
            metadata={"source_file": "annual_report.md", "element_type": "NarrativeText"},
        ),
        Document(
            page_content="Cortex 3.0 supports multi-modal reasoning with text, images, and audio.",
            metadata={"source_file": "faq.md", "element_type": "NarrativeText"},
        ),
        Document(
            page_content="Pricing: Starter tier is $2,500/month with 100,000 queries included.",
            metadata={"source_file": "faq.md", "element_type": "Table"},
        ),
        Document(
            page_content="R&D investment reached $198 million, representing 23.4% of revenue.",
            metadata={"source_file": "annual_report.md", "element_type": "NarrativeText"},
        ),
    ]


@pytest.fixture
def populated_store(
    in_memory_store: QdrantVectorStore,
    sample_documents: list[Document],
    embedding_service: EmbeddingService,
) -> tuple[QdrantVectorStore, EmbeddingService]:
    """Create a store populated with sample documents and embeddings."""
    texts = [doc.page_content for doc in sample_documents]
    embeddings = embedding_service.embed_texts(texts, show_progress=False)
    dimension = len(embeddings[0])

    in_memory_store.create_collection(dimension=dimension)
    in_memory_store.upsert_documents(sample_documents, embeddings)

    return in_memory_store, embedding_service


# ===================================================================
# Embedding Service Tests
# ===================================================================

class TestEmbeddingService:
    """Tests for the EmbeddingService."""

    def test_embed_single_text(self, embedding_service: EmbeddingService):
        """Embedding a single text should return a vector."""
        vector = embedding_service.embed_query("Hello world")
        assert isinstance(vector, list)
        assert len(vector) > 0
        assert all(isinstance(v, float) for v in vector)

    def test_embed_batch(self, embedding_service: EmbeddingService):
        """Batch embedding should return one vector per text."""
        texts = ["First text", "Second text", "Third text"]
        vectors = embedding_service.embed_texts(texts, show_progress=False)
        assert len(vectors) == 3
        # All vectors should have the same dimension
        dims = {len(v) for v in vectors}
        assert len(dims) == 1

    def test_embed_empty_list(self, embedding_service: EmbeddingService):
        """Embedding empty list should return empty list."""
        vectors = embedding_service.embed_texts([], show_progress=False)
        assert vectors == []

    def test_get_dimension(self, embedding_service: EmbeddingService):
        """Should detect embedding dimensionality."""
        dim = embedding_service.get_dimension()
        assert isinstance(dim, int)
        assert dim > 0

    def test_different_texts_different_embeddings(self, embedding_service: EmbeddingService):
        """Different texts should produce different embeddings."""
        v1 = embedding_service.embed_query("Machine learning is great")
        v2 = embedding_service.embed_query("The weather is sunny today")
        # They shouldn't be identical
        assert v1 != v2


# ===================================================================
# Qdrant Vector Store Tests
# ===================================================================

class TestQdrantVectorStore:
    """Tests for the QdrantVectorStore."""

    def test_create_collection(self, in_memory_store: QdrantVectorStore):
        """Should create a new collection."""
        in_memory_store.create_collection(dimension=384)
        assert in_memory_store.collection_exists()

    def test_create_collection_idempotent(self, in_memory_store: QdrantVectorStore):
        """Creating the same collection twice should not error."""
        in_memory_store.create_collection(dimension=384)
        in_memory_store.create_collection(dimension=384)  # Should not raise
        assert in_memory_store.collection_exists()

    def test_collection_exists_false(self, in_memory_store: QdrantVectorStore):
        """Non-existent collection should return False."""
        assert not in_memory_store.collection_exists("nonexistent_collection")

    def test_delete_collection(self, in_memory_store: QdrantVectorStore):
        """Should delete a collection."""
        in_memory_store.create_collection(dimension=384)
        assert in_memory_store.collection_exists()
        in_memory_store.delete_collection()
        assert not in_memory_store.collection_exists()

    def test_get_collection_info(self, in_memory_store: QdrantVectorStore):
        """Should return collection statistics."""
        in_memory_store.create_collection(dimension=384)
        info = in_memory_store.get_collection_info()
        assert info["name"] == "test_collection"
        assert info["dimension"] == 384
        assert info["points_count"] == 0

    def test_upsert_documents(
        self,
        in_memory_store: QdrantVectorStore,
        sample_documents: list[Document],
    ):
        """Should upsert documents with embeddings."""
        dim = 4  # Small dimension for fast testing
        in_memory_store.create_collection(dimension=dim)

        fake_embeddings = [[0.1, 0.2, 0.3, 0.4]] * len(sample_documents)
        count = in_memory_store.upsert_documents(sample_documents, fake_embeddings)

        assert count == len(sample_documents)
        info = in_memory_store.get_collection_info()
        assert info["points_count"] == len(sample_documents)

    def test_upsert_mismatched_lengths(
        self,
        in_memory_store: QdrantVectorStore,
        sample_documents: list[Document],
    ):
        """Should raise error if document and embedding counts don't match."""
        in_memory_store.create_collection(dimension=4)

        with pytest.raises(ValueError, match="Mismatch"):
            in_memory_store.upsert_documents(
                sample_documents,
                [[0.1, 0.2, 0.3, 0.4]],  # Only 1 embedding for 5 docs
            )

    def test_upsert_empty(self, in_memory_store: QdrantVectorStore):
        """Upserting empty list should return 0."""
        in_memory_store.create_collection(dimension=4)
        count = in_memory_store.upsert_documents([], [])
        assert count == 0

    def test_invalid_mode(self):
        """Should raise error for unknown mode."""
        with pytest.raises(ValueError, match="Unknown mode"):
            QdrantVectorStore(mode="invalid")


# ===================================================================
# Search Tests (Integration — uses real embeddings)
# ===================================================================

class TestSearch:
    """Integration tests for semantic search."""

    def test_search_returns_results(
        self,
        populated_store: tuple[QdrantVectorStore, EmbeddingService],
    ):
        """Search should return relevant results."""
        store, emb_service = populated_store

        query_vec = emb_service.embed_query("What is NovaTech's revenue?")
        results = store.search(query_embedding=query_vec, top_k=3)

        assert len(results) > 0
        assert len(results) <= 3
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_results_have_content(
        self,
        populated_store: tuple[QdrantVectorStore, EmbeddingService],
    ):
        """Each result should have content and score."""
        store, emb_service = populated_store

        query_vec = emb_service.embed_query("pricing information")
        results = store.search(query_embedding=query_vec, top_k=3)

        for r in results:
            assert r.content, "Result content should not be empty"
            assert isinstance(r.score, float)
            assert r.score > 0

    def test_search_relevance_ordering(
        self,
        populated_store: tuple[QdrantVectorStore, EmbeddingService],
    ):
        """Results should be ordered by relevance (descending score)."""
        store, emb_service = populated_store

        query_vec = emb_service.embed_query("company revenue and financials")
        results = store.search(query_embedding=query_vec, top_k=5)

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), "Results should be sorted by score descending"

    def test_search_with_metadata_filter(
        self,
        populated_store: tuple[QdrantVectorStore, EmbeddingService],
    ):
        """Metadata filtering should narrow results."""
        store, emb_service = populated_store

        query_vec = emb_service.embed_query("information")

        # Filter to only FAQ source
        results = store.search(
            query_embedding=query_vec,
            top_k=10,
            filter_conditions={"source_file": "faq.md"},
        )

        for r in results:
            assert r.metadata.get("source_file") == "faq.md"

    def test_search_with_type_filter(
        self,
        populated_store: tuple[QdrantVectorStore, EmbeddingService],
    ):
        """Should filter by element type."""
        store, emb_service = populated_store

        query_vec = emb_service.embed_query("pricing table")
        results = store.search(
            query_embedding=query_vec,
            top_k=10,
            filter_conditions={"element_type": "Table"},
        )

        for r in results:
            assert r.metadata.get("element_type") == "Table"

    def test_search_with_documents(
        self,
        populated_store: tuple[QdrantVectorStore, EmbeddingService],
    ):
        """search_with_documents should return LangChain Document objects."""
        store, emb_service = populated_store

        query_vec = emb_service.embed_query("revenue")
        results = store.search_with_documents(query_embedding=query_vec, top_k=3)

        assert len(results) > 0
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)
            assert doc.page_content, "Document should have content"

    def test_search_top_k_limit(
        self,
        populated_store: tuple[QdrantVectorStore, EmbeddingService],
    ):
        """Should respect the top_k limit."""
        store, emb_service = populated_store

        query_vec = emb_service.embed_query("anything")
        results = store.search(query_embedding=query_vec, top_k=2)

        assert len(results) <= 2


# ===================================================================
# SearchResult Tests
# ===================================================================

class TestSearchResult:
    """Tests for the SearchResult data class."""

    def test_repr(self):
        result = SearchResult(content="Short text content", score=0.95, metadata={})
        repr_str = repr(result)
        assert "0.95" in repr_str
        assert "Short text" in repr_str

    def test_metadata_default(self):
        result = SearchResult(content="test", score=0.5)
        assert result.metadata == {}
