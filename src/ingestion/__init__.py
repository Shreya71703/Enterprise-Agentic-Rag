"""
Enterprise RAG — Data Ingestion & Chunking Pipeline

This module handles the complete document ingestion flow:
    1. Parse: Extract structured elements from raw documents (PDF, DOCX, etc.)
    2. Chunk: Split elements into semantically coherent chunks
    3. Enrich: Attach rich metadata for downstream retrieval
"""

from src.ingestion.chunker import SemanticChunkerService
from src.ingestion.metadata import MetadataEnricher
from src.ingestion.parser import DocumentParser
from src.ingestion.pipeline import IngestionPipeline

__all__ = [
    "DocumentParser",
    "SemanticChunkerService",
    "MetadataEnricher",
    "IngestionPipeline",
]
