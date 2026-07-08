"""
Enterprise RAG — Qdrant Vector Store

Production-grade vector store wrapper around Qdrant for storing and
querying document embeddings. Supports both local (in-memory / on-disk)
and cloud deployments.

Features:
    - Automatic collection creation with correct dimensionality
    - Batch upsert with progress tracking
    - Similarity search with score filtering
    - Metadata-based filtering (by source_file, element_type, etc.)
    - Collection management (stats, deletion, listing)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from config.settings import get_settings

logger = logging.getLogger(__name__)

# Default collection name
DEFAULT_COLLECTION = "enterprise_rag_docs"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class SearchResult:
    """A single search result with score and metadata."""

    content: str
    score: float
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.content[:80].replace("\n", " ")
        return f"SearchResult(score={self.score:.4f}, text='{preview}...')"


# ---------------------------------------------------------------------------
# Qdrant Vector Store
# ---------------------------------------------------------------------------
class QdrantVectorStore:
    """
    Production-grade Qdrant vector store for the Enterprise RAG system.

    Supports three deployment modes:
        - In-memory (default for testing): QdrantVectorStore(mode="memory")
        - On-disk (persistent local): QdrantVectorStore(mode="disk", path="./data/qdrant")
        - Cloud: QdrantVectorStore(mode="cloud", url="...", api_key="...")

    Usage:
        store = QdrantVectorStore(mode="disk", path="./data/qdrant")
        store.create_collection("my_docs", dimension=384)
        store.upsert_documents(documents, embeddings)
        results = store.search("What is RAG?", query_embedding, top_k=5)
    """

    def __init__(
        self,
        mode: str = "disk",
        path: str | Path | None = None,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str = DEFAULT_COLLECTION,
    ):
        self.collection_name = collection_name
        self.mode = mode

        if mode == "memory":
            self._client = QdrantClient(location=":memory:")
            logger.info("✅ Qdrant client initialized (in-memory mode)")

        elif mode == "disk":
            storage_path = str(path or (get_settings().data_dir / "qdrant_storage"))
            Path(storage_path).mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=storage_path)
            logger.info(f"✅ Qdrant client initialized (on-disk: {storage_path})")

        elif mode == "cloud":
            if not url:
                raise ValueError("Qdrant Cloud requires a 'url' parameter")
            self._client = QdrantClient(url=url, api_key=api_key)
            logger.info(f"✅ Qdrant client initialized (cloud: {url})")

        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'memory', 'disk', or 'cloud'.")

    # -------------------------------------------------------------------
    # Collection Management
    # -------------------------------------------------------------------
    def create_collection(
        self,
        collection_name: str | None = None,
        dimension: int = 384,
        distance: str = "cosine",
    ) -> None:
        """
        Create a vector collection if it doesn't exist.

        Args:
            collection_name: Name of the collection (defaults to self.collection_name).
            dimension: Dimensionality of the embedding vectors.
            distance: Distance metric ('cosine', 'euclid', 'dot').
        """
        name = collection_name or self.collection_name

        distance_map = {
            "cosine": models.Distance.COSINE,
            "euclid": models.Distance.EUCLID,
            "dot": models.Distance.DOT,
        }

        if self.collection_exists(name):
            info = self._client.get_collection(name)
            existing_dim = info.config.params.vectors.size  # type: ignore[union-attr]
            if existing_dim != dimension:
                logger.warning(
                    f"⚠️  Collection '{name}' exists with dimension {existing_dim}, "
                    f"but requested {dimension}. Recreating..."
                )
                self._client.delete_collection(name)
            else:
                logger.info(
                    f"✅ Collection '{name}' already exists "
                    f"(dim={existing_dim}, points={info.points_count})"
                )
                return

        self._client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=dimension,
                distance=distance_map.get(distance, models.Distance.COSINE),
            ),
        )
        logger.info(
            f"✅ Created collection '{name}' (dim={dimension}, distance={distance})"
        )

    def collection_exists(self, collection_name: str | None = None) -> bool:
        """Check if a collection exists."""
        name = collection_name or self.collection_name
        try:
            self._client.get_collection(name)
            return True
        except (UnexpectedResponse, Exception):
            return False

    def delete_collection(self, collection_name: str | None = None) -> None:
        """Delete a collection."""
        name = collection_name or self.collection_name
        self._client.delete_collection(name)
        logger.info(f"🗑️  Deleted collection '{name}'")

    def get_collection_info(self, collection_name: str | None = None) -> dict[str, Any]:
        """Get collection statistics."""
        name = collection_name or self.collection_name
        info = self._client.get_collection(name)
        return {
            "name": name,
            "points_count": info.points_count,
            "dimension": info.config.params.vectors.size,  # type: ignore[union-attr]
            "distance": str(info.config.params.vectors.distance),  # type: ignore[union-attr]
            "status": str(info.status),
        }

    # -------------------------------------------------------------------
    # Document Operations
    # -------------------------------------------------------------------
    def upsert_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
        collection_name: str | None = None,
        batch_size: int = 100,
    ) -> int:
        """
        Upsert documents with their embeddings into the vector store.

        Args:
            documents: LangChain Document objects with page_content and metadata.
            embeddings: Corresponding embedding vectors.
            collection_name: Target collection (defaults to self.collection_name).
            batch_size: Number of points per upsert batch.

        Returns:
            Number of points upserted.
        """
        name = collection_name or self.collection_name

        if len(documents) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(documents)} documents but {len(embeddings)} embeddings"
            )

        if not documents:
            return 0

        # Build points
        points: list[models.PointStruct] = []
        for doc, embedding in zip(documents, embeddings):
            point_id = str(uuid.uuid4())
            payload = {
                "page_content": doc.page_content,
                **{k: v for k, v in doc.metadata.items() if _is_serializable(v)},
            }
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        # Batch upsert
        total_batches = (len(points) + batch_size - 1) // batch_size
        upserted = 0

        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            batch_num = (i // batch_size) + 1

            self._client.upsert(
                collection_name=name,
                points=batch,
            )
            upserted += len(batch)
            logger.info(
                f"   📥 Upserted batch {batch_num}/{total_batches} "
                f"({len(batch)} points, total: {upserted})"
            )

        logger.info(f"✅ Upserted {upserted} points into '{name}'")
        return upserted

    # -------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------
    def search(
        self,
        query_embedding: list[float],
        collection_name: str | None = None,
        top_k: int = 5,
        score_threshold: float | None = None,
        filter_conditions: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Search for similar documents in the vector store.

        Args:
            query_embedding: The query vector to search with.
            collection_name: Collection to search (defaults to self.collection_name).
            top_k: Number of results to return.
            score_threshold: Minimum similarity score (0-1 for cosine).
            filter_conditions: Dict of metadata filters, e.g.
                {"source_file": "report.pdf", "element_type": "Table"}

        Returns:
            List of SearchResult objects sorted by relevance.
        """
        name = collection_name or self.collection_name

        # Build Qdrant filter from conditions
        query_filter = None
        if filter_conditions:
            must_conditions = []
            for key, value in filter_conditions.items():
                if isinstance(value, list):
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchAny(any=value),
                        )
                    )
                else:
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value),
                        )
                    )
            query_filter = models.Filter(must=must_conditions)

        results = self._client.query_points(
            collection_name=name,
            query=query_embedding,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=query_filter,
            with_payload=True,
        )

        search_results: list[SearchResult] = []
        for point in results.points:
            payload = point.payload or {}
            content = payload.pop("page_content", "")
            search_results.append(
                SearchResult(
                    content=content,
                    score=point.score,
                    metadata=payload,
                )
            )

        return search_results

    def search_with_documents(
        self,
        query_embedding: list[float],
        collection_name: str | None = None,
        top_k: int = 5,
        score_threshold: float | None = None,
        filter_conditions: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Search and return LangChain Document objects with scores.

        Convenience method for downstream LangChain pipeline integration.
        """
        results = self.search(
            query_embedding=query_embedding,
            collection_name=collection_name,
            top_k=top_k,
            score_threshold=score_threshold,
            filter_conditions=filter_conditions,
        )

        return [
            (
                Document(page_content=r.content, metadata=r.metadata),
                r.score,
            )
            for r in results
        ]

    # -------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------
    def close(self) -> None:
        """Close the Qdrant client connection."""
        self._client.close()
        logger.info("Qdrant client closed")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _is_serializable(value: Any) -> bool:
    """Check if a value can be serialized to JSON (for Qdrant payload)."""
    return isinstance(value, (str, int, float, bool, list, type(None)))
