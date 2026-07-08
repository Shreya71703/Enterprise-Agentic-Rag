"""
Enterprise RAG — Ingestion Pipeline Orchestrator

Composes the Parser → Chunker → MetadataEnricher into a single, clean
pipeline. This is the main entry point for processing raw documents into
LangChain Documents ready for embedding and vector storage.

Usage:
    from src.ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    documents = pipeline.run(input_dir=Path("data/sample_docs/"))
    # → List[Document], each with semantic chunks + rich metadata
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

from src.ingestion.chunker import SemanticChunkerService
from src.ingestion.metadata import MetadataEnricher
from src.ingestion.parser import DocumentParser

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline statistics
# ---------------------------------------------------------------------------
@dataclass
class IngestionStats:
    """Tracks statistics across a pipeline run."""

    files_processed: int = 0
    files_failed: int = 0
    total_elements: int = 0
    total_chunks: int = 0
    chunk_sizes: list[int] = field(default_factory=list)

    @property
    def avg_chunk_size(self) -> float:
        return sum(self.chunk_sizes) / len(self.chunk_sizes) if self.chunk_sizes else 0

    @property
    def min_chunk_size(self) -> int:
        return min(self.chunk_sizes) if self.chunk_sizes else 0

    @property
    def max_chunk_size(self) -> int:
        return max(self.chunk_sizes) if self.chunk_sizes else 0

    def to_dict(self) -> dict:
        return {
            "files_processed": self.files_processed,
            "files_failed": self.files_failed,
            "total_elements": self.total_elements,
            "total_chunks": self.total_chunks,
            "avg_chunk_size_chars": round(self.avg_chunk_size),
            "min_chunk_size_chars": self.min_chunk_size,
            "max_chunk_size_chars": self.max_chunk_size,
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class IngestionPipeline:
    """
    End-to-end document ingestion pipeline.

    Stages:
        1. PARSE:   Extract structured elements from raw documents
        2. CHUNK:   Split elements into semantically coherent chunks
        3. ENRICH:  Attach standardized metadata to every chunk

    Each stage is pluggable — you can inject custom parser/chunker/enricher.
    """

    def __init__(
        self,
        parser: DocumentParser | None = None,
        chunker: SemanticChunkerService | None = None,
        enricher: MetadataEnricher | None = None,
    ):
        self.parser = parser or DocumentParser()
        self.chunker = chunker or SemanticChunkerService()
        self.enricher = enricher or MetadataEnricher()
        self.stats = IngestionStats()

    def run(
        self,
        input_dir: Path | None = None,
        file_paths: list[Path] | None = None,
    ) -> list[Document]:
        """
        Run the full ingestion pipeline.

        Provide either `input_dir` (process all files in directory) or
        `file_paths` (process specific files). Returns a flat list of
        LangChain Documents ready for embedding.

        Args:
            input_dir: Directory containing documents to process.
            file_paths: Explicit list of file paths to process.

        Returns:
            List of enriched LangChain Document objects.

        Raises:
            ValueError: If neither input_dir nor file_paths is provided.
        """
        if input_dir is None and file_paths is None:
            raise ValueError("Provide either input_dir or file_paths")

        self.stats = IngestionStats()  # Reset stats

        # Resolve file list
        if file_paths:
            files_to_process = [(Path(f), f.name) for f in file_paths if Path(f).exists()]
        else:
            input_dir = Path(input_dir)  # type: ignore[arg-type]
            parsed_by_file = self.parser.parse_directory(input_dir)
            return self._process_parsed_results(parsed_by_file, input_dir)

        # Process individual files
        all_documents: list[Document] = []
        for file_path, filename in files_to_process:
            try:
                docs = self._process_single_file(file_path)
                all_documents.extend(docs)
                self.stats.files_processed += 1
            except Exception as e:
                logger.error(f"❌ Pipeline failed for {filename}: {e}")
                self.stats.files_failed += 1

        self._log_summary()
        return all_documents

    # -------------------------------------------------------------------
    # Internal processing
    # -------------------------------------------------------------------
    def _process_parsed_results(
        self,
        parsed_by_file: dict[str, list],
        input_dir: Path,
    ) -> list[Document]:
        """Process pre-parsed results from directory parsing."""
        all_documents: list[Document] = []

        for filename, elements in parsed_by_file.items():
            if not elements:
                self.stats.files_failed += 1
                continue

            try:
                self.stats.total_elements += len(elements)

                # Stage 2: CHUNK
                logger.info(f"   ✂️  Chunking {filename}...")
                chunks = self.chunker.chunk_elements(elements, source_file=filename)

                # Stage 3: ENRICH
                source_path = input_dir / filename
                chunks = self.enricher.enrich(chunks, source_path=source_path)

                # Track stats
                self.stats.total_chunks += len(chunks)
                self.stats.chunk_sizes.extend(len(c.page_content) for c in chunks)
                self.stats.files_processed += 1

                all_documents.extend(chunks)

            except Exception as e:
                logger.error(f"❌ Pipeline failed for {filename}: {e}")
                self.stats.files_failed += 1

        self._log_summary()
        return all_documents

    def _process_single_file(self, file_path: Path) -> list[Document]:
        """Process a single file through all three stages."""
        # Stage 1: PARSE
        elements = self.parser.parse_file(file_path)
        self.stats.total_elements += len(elements)

        if not elements:
            logger.warning(f"   No elements extracted from {file_path.name}")
            return []

        # Stage 2: CHUNK
        chunks = self.chunker.chunk_elements(elements, source_file=file_path.name)

        # Stage 3: ENRICH
        chunks = self.enricher.enrich(chunks, source_path=file_path)

        # Track stats
        self.stats.total_chunks += len(chunks)
        self.stats.chunk_sizes.extend(len(c.page_content) for c in chunks)

        return chunks

    def _log_summary(self) -> None:
        """Log a summary of the pipeline run."""
        s = self.stats
        logger.info("\n" + "=" * 60)
        logger.info("📊 INGESTION PIPELINE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"   Files processed:  {s.files_processed}")
        logger.info(f"   Files failed:     {s.files_failed}")
        logger.info(f"   Elements parsed:  {s.total_elements}")
        logger.info(f"   Chunks produced:  {s.total_chunks}")
        if s.chunk_sizes:
            logger.info(f"   Avg chunk size:   {s.avg_chunk_size:.0f} chars")
            logger.info(f"   Min chunk size:   {s.min_chunk_size} chars")
            logger.info(f"   Max chunk size:   {s.max_chunk_size} chars")
        logger.info("=" * 60)

    # -------------------------------------------------------------------
    # Export utilities
    # -------------------------------------------------------------------
    def save_chunks(
        self,
        documents: list[Document],
        output_path: Path,
    ) -> None:
        """
        Save processed chunks to a JSON file for inspection or later use.

        Args:
            documents: List of processed Document objects.
            output_path: Path to write the JSON output.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        serialized = []
        for doc in documents:
            serialized.append(
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                }
            )

        output_path.write_text(
            json.dumps(serialized, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info(f"💾 Saved {len(documents)} chunks to {output_path}")
