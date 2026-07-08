"""
Enterprise RAG — Ingestion Pipeline Tests

Tests for the parser, chunker, metadata enricher, and full pipeline.
Uses sample text data (no external API calls) for fast, deterministic testing.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from langchain_core.documents import Document

from config.settings import ChunkerSettings, ParserSettings
from src.ingestion.metadata import MetadataEnricher
from src.ingestion.parser import DocumentParser, ParsedElement


# ===================================================================
# Fixtures
# ===================================================================

SAMPLE_TEXT = """# NovaTech Q4 Earnings Summary

NovaTech reported strong Q4 results, with revenue reaching $245 million,
up 38% year-over-year. The company's AI Platform segment was the primary
growth driver, contributing $157 million in quarterly revenue.

## Key Metrics

Operating margin improved to 16.2%, driven by economies of scale in
cloud infrastructure and a shift toward higher-margin platform subscriptions.
Customer count grew to 523 enterprise clients, with a net revenue retention
rate of 135%.

## Product Updates

The launch of Cortex 3.0 in Q2 exceeded expectations, with 89% of existing
customers upgrading within the first 90 days. The new multi-modal reasoning
engine has been particularly well-received in the healthcare and financial
services verticals.

## Forward Guidance

Management raised full-year 2025 guidance to $1.08-1.12 billion in revenue,
reflecting confidence in the pipeline and strong renewal rates. R&D investment
will increase to approximately 25% of revenue as the company accelerates
development of autonomous agent capabilities.
"""

SAMPLE_CSV = """name,role,department,salary
Alice Johnson,Senior Engineer,Engineering,185000
Bob Smith,Product Manager,Product,165000
Carol Williams,Data Scientist,AI Research,195000
"""


@pytest.fixture
def sample_text_file(tmp_path: Path) -> Path:
    """Create a temporary text file with sample content."""
    file_path = tmp_path / "sample_report.md"
    file_path.write_text(SAMPLE_TEXT, encoding="utf-8")
    return file_path


@pytest.fixture
def sample_csv_file(tmp_path: Path) -> Path:
    """Create a temporary CSV file."""
    file_path = tmp_path / "sample_data.csv"
    file_path.write_text(SAMPLE_CSV, encoding="utf-8")
    return file_path


@pytest.fixture
def sample_dir(tmp_path: Path, sample_text_file: Path, sample_csv_file: Path) -> Path:
    """Create a directory with multiple sample files."""
    return tmp_path


@pytest.fixture
def parser() -> DocumentParser:
    """Create a DocumentParser with default settings."""
    return DocumentParser()


@pytest.fixture
def enricher() -> MetadataEnricher:
    """Create a MetadataEnricher."""
    return MetadataEnricher()


# ===================================================================
# Parser Tests
# ===================================================================

class TestDocumentParser:
    """Tests for the DocumentParser."""

    def test_parse_text_file(self, parser: DocumentParser, sample_text_file: Path):
        """Parser should extract elements from a text/markdown file."""
        elements = parser.parse_file(sample_text_file)

        assert len(elements) > 0
        assert all(isinstance(e, ParsedElement) for e in elements)

    def test_parsed_elements_have_text(self, parser: DocumentParser, sample_text_file: Path):
        """Each parsed element should have non-empty text."""
        elements = parser.parse_file(sample_text_file)

        for elem in elements:
            assert elem.text.strip(), f"Element has empty text: {elem}"

    def test_parsed_elements_have_metadata(self, parser: DocumentParser, sample_text_file: Path):
        """Each parsed element should carry metadata."""
        elements = parser.parse_file(sample_text_file)

        for elem in elements:
            assert isinstance(elem.metadata, dict)
            assert elem.element_type, "Element type should not be empty"

    def test_parse_csv_file(self, parser: DocumentParser, sample_csv_file: Path):
        """Parser should handle CSV files."""
        elements = parser.parse_file(sample_csv_file)

        # CSV should produce at least some elements
        assert len(elements) > 0

    def test_parse_nonexistent_file(self, parser: DocumentParser):
        """Parser should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            parser.parse_file(Path("/nonexistent/file.txt"))

    def test_parse_unsupported_extension(self, parser: DocumentParser, tmp_path: Path):
        """Parser should skip unsupported file types."""
        file_path = tmp_path / "data.xyz"
        file_path.write_text("some content")

        elements = parser.parse_file(file_path)
        assert elements == []

    def test_parse_directory(self, parser: DocumentParser, sample_dir: Path):
        """Parser should process all supported files in a directory."""
        results = parser.parse_directory(sample_dir)

        assert isinstance(results, dict)
        assert len(results) > 0
        for filename, elements in results.items():
            assert isinstance(filename, str)
            assert isinstance(elements, list)

    def test_parse_empty_directory(self, parser: DocumentParser, tmp_path: Path):
        """Parser should return empty dict for directory with no supported files."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        results = parser.parse_directory(empty_dir)
        assert results == {}

    def test_parse_invalid_directory(self, parser: DocumentParser):
        """Parser should raise error for non-directory path."""
        with pytest.raises(NotADirectoryError):
            parser.parse_directory(Path("/nonexistent/dir"))


# ===================================================================
# Metadata Enricher Tests
# ===================================================================

class TestMetadataEnricher:
    """Tests for the MetadataEnricher."""

    def test_enrich_adds_required_fields(self, enricher: MetadataEnricher, tmp_path: Path):
        """Enricher should add all required metadata fields."""
        source = tmp_path / "test.txt"
        source.write_text("test content")

        docs = [
            Document(page_content="First chunk of text.", metadata={"element_type": "NarrativeText"}),
            Document(page_content="Second chunk of text.", metadata={"element_type": "Title"}),
        ]

        enriched = enricher.enrich(docs, source_path=source)

        for doc in enriched:
            assert "doc_id" in doc.metadata
            assert "chunk_index" in doc.metadata
            assert "total_chunks" in doc.metadata
            assert "char_count" in doc.metadata
            assert "ingestion_timestamp" in doc.metadata

    def test_enrich_chunk_indices(self, enricher: MetadataEnricher):
        """Chunk indices should be sequential starting from 0."""
        docs = [
            Document(page_content=f"Chunk {i}", metadata={})
            for i in range(5)
        ]

        enriched = enricher.enrich(docs)

        for i, doc in enumerate(enriched):
            assert doc.metadata["chunk_index"] == i
            assert doc.metadata["total_chunks"] == 5

    def test_enrich_doc_id_deterministic(self, enricher: MetadataEnricher, tmp_path: Path):
        """Same file should always produce the same doc_id."""
        source = tmp_path / "test.txt"
        source.write_text("consistent content")

        docs1 = [Document(page_content="chunk", metadata={})]
        docs2 = [Document(page_content="chunk", metadata={})]

        enricher.enrich(docs1, source_path=source)
        enricher.enrich(docs2, source_path=source)

        assert docs1[0].metadata["doc_id"] == docs2[0].metadata["doc_id"]

    def test_enrich_char_count(self, enricher: MetadataEnricher):
        """char_count should match actual content length."""
        content = "Hello, this is a test chunk with exactly this many characters."
        docs = [Document(page_content=content, metadata={})]

        enricher.enrich(docs)

        assert docs[0].metadata["char_count"] == len(content)

    def test_enrich_empty_list(self, enricher: MetadataEnricher):
        """Enricher should handle empty document list gracefully."""
        result = enricher.enrich([])
        assert result == []

    def test_enrich_preserves_existing_metadata(self, enricher: MetadataEnricher):
        """Enricher should not overwrite existing metadata."""
        docs = [
            Document(
                page_content="test",
                metadata={"element_type": "Table", "custom_field": "preserved"},
            )
        ]

        enricher.enrich(docs)

        assert docs[0].metadata["element_type"] == "Table"
        assert docs[0].metadata["custom_field"] == "preserved"


# ===================================================================
# ParsedElement Tests
# ===================================================================

class TestParsedElement:
    """Tests for the ParsedElement data class."""

    def test_page_number_property(self):
        elem = ParsedElement(
            text="test", element_type="NarrativeText", metadata={"page_number": 5}
        )
        assert elem.page_number == 5

    def test_page_number_missing(self):
        elem = ParsedElement(text="test", element_type="NarrativeText", metadata={})
        assert elem.page_number is None

    def test_source_file_property(self):
        elem = ParsedElement(
            text="test", element_type="Title", metadata={"filename": "report.pdf"}
        )
        assert elem.source_file == "report.pdf"

    def test_repr(self):
        elem = ParsedElement(text="Short text", element_type="Title", metadata={})
        repr_str = repr(elem)
        assert "Title" in repr_str
        assert "Short text" in repr_str


# ===================================================================
# Pipeline Integration Tests (lightweight, no embeddings)
# ===================================================================

class TestIngestionPipeline:
    """Integration tests for the full pipeline (using fallback chunker)."""

    def test_pipeline_runs_end_to_end(self, sample_dir: Path):
        """Pipeline should process a directory and produce Documents."""
        from src.ingestion.chunker import SemanticChunkerService
        from src.ingestion.pipeline import IngestionPipeline

        # Force fallback chunker (no embeddings needed)
        chunker = SemanticChunkerService()
        chunker._initialized = True
        chunker._using_semantic = False

        from langchain_text_splitters import RecursiveCharacterTextSplitter

        chunker._chunker = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100
        )

        pipeline = IngestionPipeline(chunker=chunker)
        documents = pipeline.run(input_dir=sample_dir)

        assert len(documents) > 0
        assert all(isinstance(d, Document) for d in documents)

    def test_pipeline_chunks_have_metadata(self, sample_dir: Path):
        """Every chunk from the pipeline should have enriched metadata."""
        from src.ingestion.chunker import SemanticChunkerService
        from src.ingestion.pipeline import IngestionPipeline

        chunker = SemanticChunkerService()
        chunker._initialized = True
        chunker._using_semantic = False

        from langchain_text_splitters import RecursiveCharacterTextSplitter

        chunker._chunker = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100
        )

        pipeline = IngestionPipeline(chunker=chunker)
        documents = pipeline.run(input_dir=sample_dir)

        for doc in documents:
            assert "doc_id" in doc.metadata
            assert "chunk_index" in doc.metadata
            assert "ingestion_timestamp" in doc.metadata

    def test_pipeline_save_chunks(self, sample_dir: Path, tmp_path: Path):
        """Pipeline should save chunks to JSON."""
        from src.ingestion.chunker import SemanticChunkerService
        from src.ingestion.pipeline import IngestionPipeline

        chunker = SemanticChunkerService()
        chunker._initialized = True
        chunker._using_semantic = False

        from langchain_text_splitters import RecursiveCharacterTextSplitter

        chunker._chunker = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100
        )

        pipeline = IngestionPipeline(chunker=chunker)
        documents = pipeline.run(input_dir=sample_dir)

        output_path = tmp_path / "output" / "chunks.json"
        pipeline.save_chunks(documents, output_path)

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert len(data) == len(documents)
        assert "page_content" in data[0]
        assert "metadata" in data[0]

    def test_pipeline_stats(self, sample_dir: Path):
        """Pipeline should track processing statistics."""
        from src.ingestion.chunker import SemanticChunkerService
        from src.ingestion.pipeline import IngestionPipeline

        chunker = SemanticChunkerService()
        chunker._initialized = True
        chunker._using_semantic = False

        from langchain_text_splitters import RecursiveCharacterTextSplitter

        chunker._chunker = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100
        )

        pipeline = IngestionPipeline(chunker=chunker)
        pipeline.run(input_dir=sample_dir)

        stats = pipeline.stats
        assert stats.files_processed > 0
        assert stats.total_chunks > 0
        assert stats.avg_chunk_size > 0

    def test_pipeline_no_input_raises(self):
        """Pipeline should raise ValueError when no input is provided."""
        from src.ingestion.pipeline import IngestionPipeline

        pipeline = IngestionPipeline()
        with pytest.raises(ValueError, match="Provide either"):
            pipeline.run()
