"""
Enterprise RAG — BM25 Retriever

Keyword-based search index wrapping rank_bm25.
Used in parallel with semantic vector search to catch exact keyword matches,
product codes, numbers, and specific terms that embedding models might miss.
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simple Tokenizer for English
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25: lowercase, strip punctuation, split on spaces.
    """
    text = text.lower()
    # Replace punctuation with spaces
    text = re.sub(r"[^\w\s-]", " ", text)
    # Split and remove empty tokens
    return [t for t in text.split() if t.strip()]


# ---------------------------------------------------------------------------
# BM25 Retriever Service
# ---------------------------------------------------------------------------
class BM25RetrieverService:
    """
    Simple keyword search index using the BM25 algorithm (rank_bm25).

    Supports:
        - Building index from LangChain Documents
        - Searching docs returning scores
        - Pickle serialization for persistence
    """

    def __init__(self):
        self.documents: list[Document] = []
        self.corpus_size = 0
        self._bm25 = None

    def build_index(self, documents: list[Document]) -> None:
        """
        Build the BM25 index from a list of documents.
        """
        from rank_bm25 import BM25Okapi

        if not documents:
            logger.warning("⚠️  Empty document list passed to BM25 indexer.")
            self.documents = []
            self.corpus_size = 0
            self._bm25 = None
            return

        self.documents = documents
        self.corpus_size = len(documents)

        # Tokenize corpus
        tokenized_corpus = [tokenize(doc.page_content) for doc in documents]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"✅ Built BM25 index with {len(documents)} documents")

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_conditions: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Search the index for keyword matches.

        Args:
            query: Natural language query text.
            top_k: Max results to return.
            filter_conditions: Optional metadata filters.

        Returns:
            List of (Document, score) tuples sorted descending by score.
        """
        if not self._bm25 or not self.documents:
            return []

        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []

        # Get scores for all documents in the corpus
        scores = self._bm25.get_scores(tokenized_query)

        # Zip documents with scores and filter out score <= 0
        scored_docs = []
        for doc, score in zip(self.documents, scores):
            if score <= 0:
                continue

            # Apply metadata filters if provided
            if filter_conditions:
                matches_filters = True
                for k, v in filter_conditions.items():
                    doc_val = doc.metadata.get(k)
                    if isinstance(v, list):
                        if doc_val not in v:
                            matches_filters = False
                            break
                    else:
                        if doc_val != v:
                            matches_filters = False
                            break
                if not matches_filters:
                    continue

            scored_docs.append((doc, float(score)))

        # Sort descending by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]

    # -------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------
    def save(self, filepath: Path | str) -> None:
        """Save the index state using pickle."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "documents": self.documents,
            "bm25": self._bm25,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"💾 Saved BM25 index to {filepath}")

    def load(self, filepath: Path | str) -> None:
        """Load the index state from a pickle file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"BM25 index file not found: {filepath}")

        with open(filepath, "rb") as f:
            state = pickle.load(f)

        self.documents = state["documents"]
        self.corpus_size = len(self.documents)
        self._bm25 = state["bm25"]
        logger.info(f"✅ Loaded BM25 index from {filepath} ({self.corpus_size} docs)")
