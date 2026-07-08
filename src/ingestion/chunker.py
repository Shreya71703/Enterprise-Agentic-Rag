"""
Enterprise RAG — Semantic Chunker

Splits parsed document elements into semantically coherent chunks using
LangChain's SemanticChunker. Falls back to RecursiveCharacterTextSplitter
when embeddings are unavailable.

Why semantic chunking?
    Traditional chunking (fixed character windows) breaks text at arbitrary
    positions, often splitting mid-sentence or mid-concept. Semantic chunking
    uses embedding similarity between sentences to find natural "breakpoints"
    where the topic shifts — producing chunks that are more coherent and
    retrieve better.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.documents import Document

from config.settings import ChunkerSettings, EmbeddingSettings, get_settings
from src.embeddings import create_embedding_model

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

    from src.ingestion.parser import ParsedElement

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Semantic Chunker Service
# ---------------------------------------------------------------------------
class SemanticChunkerService:
    """
    Chunks parsed document elements into semantically coherent segments.

    Primary strategy: LangChain SemanticChunker (embedding-based breakpoints)
    Fallback strategy: RecursiveCharacterTextSplitter (character-based)

    Usage:
        chunker = SemanticChunkerService()
        documents = chunker.chunk_elements(parsed_elements)
    """

    def __init__(
        self,
        chunker_settings: ChunkerSettings | None = None,
        embedding_settings: EmbeddingSettings | None = None,
        embedding_model: Embeddings | None = None,
    ):
        self.settings = chunker_settings or get_settings().chunker
        self._embedding_settings = embedding_settings or get_settings().embedding
        self._embedding_model = embedding_model
        self._chunker = None
        self._using_semantic = False
        self._initialized = False

    def _lazy_init(self) -> None:
        """Initialize the chunker on first use (avoids import overhead at startup)."""
        if self._initialized:
            return
        self._initialized = True

        # Try semantic chunker first
        try:
            if self._embedding_model is None:
                self._embedding_model = create_embedding_model(self._embedding_settings)

            from langchain_experimental.text_splitter import SemanticChunker

            kwargs = {
                "embeddings": self._embedding_model,
                "breakpoint_threshold_type": self.settings.breakpoint_threshold_type,
            }
            if self.settings.breakpoint_threshold_amount is not None:
                kwargs["breakpoint_threshold_amount"] = self.settings.breakpoint_threshold_amount

            self._chunker = SemanticChunker(**kwargs)
            self._using_semantic = True
            logger.info(
                f"✅ Semantic chunker initialized "
                f"(threshold: {self.settings.breakpoint_threshold_type})"
            )

        except Exception as e:
            logger.warning(f"⚠️  Semantic chunker unavailable ({e}). Using fallback splitter.")
            self._init_fallback()

    def _init_fallback(self) -> None:
        """Initialize the RecursiveCharacterTextSplitter as fallback."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self._chunker = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.fallback_chunk_size,
            chunk_overlap=self.settings.fallback_chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        self._using_semantic = False
        logger.info(
            f"✅ Fallback chunker initialized "
            f"(chunk_size={self.settings.fallback_chunk_size}, "
            f"overlap={self.settings.fallback_chunk_overlap})"
        )

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------
    def chunk_elements(
        self,
        elements: list[ParsedElement],
        source_file: str = "unknown",
    ) -> list[Document]:
        """
        Chunk a list of parsed elements into LangChain Documents.

        Strategy:
            - Group adjacent elements of the same type into text blocks
            - Run the chunker on each block
            - Preserve element-type metadata through chunking

        Args:
            elements: Parsed elements from DocumentParser.
            source_file: Source filename for metadata.

        Returns:
            List of LangChain Document objects, each with rich metadata.
        """
        self._lazy_init()

        if not elements:
            return []

        # Group adjacent same-type elements into coherent blocks
        blocks = self._group_elements(elements)
        logger.info(
            f"   Grouped {len(elements)} elements into {len(blocks)} block(s) for chunking"
        )

        all_chunks: list[Document] = []

        for block_type, block_text, block_meta in blocks:
            # Skip very short blocks
            if len(block_text.strip()) < self.settings.min_chunk_length:
                continue

            # Tables are kept as single chunks (don't split tabular data)
            if block_type == "Table":
                doc = Document(
                    page_content=block_text,
                    metadata={
                        "element_type": "Table",
                        "source_file": source_file,
                        **block_meta,
                    },
                )
                all_chunks.append(doc)
                continue

            # Chunk the text block
            try:
                if self._using_semantic:
                    chunks = self._chunker.create_documents([block_text])
                else:
                    chunks = self._chunker.create_documents([block_text])

                for chunk in chunks:
                    if len(chunk.page_content.strip()) < self.settings.min_chunk_length:
                        continue
                    chunk.metadata.update(
                        {
                            "element_type": block_type,
                            "source_file": source_file,
                            **block_meta,
                        }
                    )
                    all_chunks.append(chunk)

            except Exception as e:
                # If semantic chunking fails on a specific block, fallback inline
                logger.warning(f"   Chunking error on block ({block_type}): {e}. Keeping as-is.")
                doc = Document(
                    page_content=block_text,
                    metadata={
                        "element_type": block_type,
                        "source_file": source_file,
                        **block_meta,
                    },
                )
                all_chunks.append(doc)

        logger.info(
            f"   ✂️  Produced {len(all_chunks)} chunk(s) "
            f"({'semantic' if self._using_semantic else 'character-based'})"
        )
        return all_chunks

    def chunk_text(self, text: str, metadata: dict | None = None) -> list[Document]:
        """
        Chunk a raw text string into Documents (convenience method).

        Args:
            text: Raw text to chunk.
            metadata: Optional metadata to attach to all chunks.

        Returns:
            List of LangChain Document objects.
        """
        self._lazy_init()

        if not text.strip():
            return []

        chunks = self._chunker.create_documents([text])
        if metadata:
            for chunk in chunks:
                chunk.metadata.update(metadata)

        return [c for c in chunks if len(c.page_content.strip()) >= self.settings.min_chunk_length]

    @property
    def is_semantic(self) -> bool:
        """Whether the chunker is using semantic (embedding-based) splitting."""
        self._lazy_init()
        return self._using_semantic

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------
    @staticmethod
    def _group_elements(
        elements: list[ParsedElement],
    ) -> list[tuple[str, str, dict]]:
        """
        Group adjacent elements of the same type into blocks.

        Returns:
            List of (element_type, concatenated_text, aggregated_metadata) tuples.
        """
        if not elements:
            return []

        groups: list[tuple[str, str, dict]] = []
        current_type = elements[0].element_type
        current_texts: list[str] = [elements[0].text]
        current_meta: dict = {**elements[0].metadata}

        for elem in elements[1:]:
            if elem.element_type == current_type and elem.element_type != "Table":
                # Same type (non-table) → merge
                current_texts.append(elem.text)
                # Keep the first page number but update if we see a new one
                if "page_number" in elem.metadata:
                    current_meta.setdefault("page_number", elem.metadata["page_number"])
                    current_meta["page_number_end"] = elem.metadata["page_number"]
            else:
                # Type boundary → flush current group
                groups.append((current_type, "\n\n".join(current_texts), current_meta))
                current_type = elem.element_type
                current_texts = [elem.text]
                current_meta = {**elem.metadata}

        # Don't forget the last group
        groups.append((current_type, "\n\n".join(current_texts), current_meta))

        return groups
