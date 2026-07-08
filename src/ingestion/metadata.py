"""
Enterprise RAG — Metadata Enricher

Attaches rich, consistent metadata to every document chunk so that downstream
retrieval and filtering are as powerful as possible.

Each chunk receives:
    - doc_id:               SHA-256 hash of the source file (for deduplication)
    - source_file:          Original filename
    - page_number:          Page number (if available)
    - element_type:         Title / NarrativeText / Table / ListItem / etc.
    - chunk_index:          Position of this chunk within the document
    - total_chunks:         Total number of chunks from this document
    - char_count:           Character count of the chunk content
    - ingestion_timestamp:  ISO-8601 timestamp of when the chunk was processed
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class MetadataEnricher:
    """
    Enriches a list of LangChain Documents with standardized metadata.

    Usage:
        enricher = MetadataEnricher()
        enriched_docs = enricher.enrich(chunks, source_path=Path("report.pdf"))
    """

    def enrich(
        self,
        documents: list[Document],
        source_path: Path | None = None,
    ) -> list[Document]:
        """
        Enrich a list of document chunks with standardized metadata.

        Args:
            documents: List of LangChain Document objects from the chunker.
            source_path: Path to the original source file (for doc_id hashing).

        Returns:
            The same list of Documents with enriched metadata (mutated in-place).
        """
        if not documents:
            return documents

        # Generate a deterministic document ID from the source file
        doc_id = self._generate_doc_id(source_path) if source_path else "unknown"

        timestamp = datetime.now(timezone.utc).isoformat()
        total_chunks = len(documents)

        for idx, doc in enumerate(documents):
            doc.metadata.update(
                {
                    "doc_id": doc_id,
                    "chunk_index": idx,
                    "total_chunks": total_chunks,
                    "char_count": len(doc.page_content),
                    "ingestion_timestamp": timestamp,
                }
            )

            # Ensure source_file is set
            if "source_file" not in doc.metadata and source_path:
                doc.metadata["source_file"] = source_path.name

            # Ensure element_type has a default
            if "element_type" not in doc.metadata:
                doc.metadata["element_type"] = "NarrativeText"

        logger.info(
            f"   🏷️  Enriched {total_chunks} chunk(s) with metadata "
            f"(doc_id={doc_id[:12]}...)"
        )
        return documents

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------
    @staticmethod
    def _generate_doc_id(file_path: Path) -> str:
        """
        Generate a deterministic document ID from file contents.

        Uses SHA-256 hash of the file's binary content, so the same document
        always produces the same ID (useful for deduplication).
        """
        file_path = Path(file_path)

        if not file_path.exists():
            # Fall back to hashing the filename
            return hashlib.sha256(file_path.name.encode()).hexdigest()

        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                hasher.update(block)

        return hasher.hexdigest()
