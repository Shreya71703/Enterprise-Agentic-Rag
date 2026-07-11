"""
Enterprise RAG — Semantic Query Cache Service

Caches LLM synthesized answers based on semantic similarity of query embeddings.
Reduces sub-second latencies to ~5ms for matching queries, bypassing LangGraph/LLM.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from config.settings import get_settings
from src.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class SemanticCacheService:
    """
    Thread-safe SQLite based semantic cache using vector cosine similarity.
    """

    def __init__(self, db_path: Path | str | None = None, similarity_threshold: float = 0.95):
        if db_path is None:
            self.db_path = get_settings().data_dir / "enterprise_rag.db"
        else:
            self.db_path = Path(db_path)

        self.similarity_threshold = similarity_threshold
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Create the semantic cache table if it does not exist."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_text TEXT UNIQUE,
                    response_text TEXT,
                    embedding_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()
            logger.info("✅ Semantic Cache DB table initialized.")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Semantic Cache table: {e}")
        finally:
            conn.close()

    @staticmethod
    def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
        """Compute the cosine similarity between two float vectors."""
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm_v1 = sum(a * a for a in v1) ** 0.5
        norm_v2 = sum(b * b for b in v2) ** 0.5
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return dot_product / (norm_v1 * norm_v2)

    def lookup(self, query_text: str, query_embedding: list[float]) -> str | None:
        """
        Check the cache for a query with cosine similarity above threshold.
        Returns the cached response_text if found, otherwise None.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Fetch all cached queries and their serialized embeddings
            cursor.execute("SELECT query_text, response_text, embedding_json FROM semantic_cache")
            rows = cursor.fetchall()
            
            best_score = -1.0
            best_response = None
            best_query = None

            for row in rows:
                cached_query, response, emb_json = row
                try:
                    cached_emb = json.loads(emb_json)
                    score = self._cosine_similarity(query_embedding, cached_emb)
                    if score > best_score:
                        best_score = score
                        best_response = response
                        best_query = cached_query
                except Exception as parse_err:
                    logger.warning(f"Failed to parse cached embedding: {parse_err}")

            if best_score >= self.similarity_threshold:
                logger.info(
                    f"🎯 Semantic Cache HIT! Match score: {best_score:.4f} "
                    f"(Query: '{query_text}' matches cached query: '{best_query}')"
                )
                return best_response

            logger.info(f"🔎 Semantic Cache MISS. Best score: {best_score:.4f}")
            return None

        except Exception as e:
            logger.error(f"Error during semantic cache lookup: {e}")
            return None
        finally:
            conn.close()

    def store(self, query_text: str, response_text: str, query_embedding: list[float]) -> None:
        """Save a new query-response pair and its embedding in the cache database."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            embedding_json = json.dumps(query_embedding)
            cursor.execute(
                """
                INSERT OR REPLACE INTO semantic_cache (query_text, response_text, embedding_json)
                VALUES (?, ?, ?)
                """,
                (query_text, response_text, embedding_json)
            )
            conn.commit()
            logger.info(f"💾 Saved query to semantic cache: '{query_text[:50]}...'")
        except Exception as e:
            logger.error(f"Error storing query in semantic cache: {e}")
        finally:
            conn.close()
