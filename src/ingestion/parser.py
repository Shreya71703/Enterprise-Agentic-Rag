"""
Enterprise RAG — Document Parser

Extracts structured elements (text, tables, titles, images) from raw documents
using the `unstructured` library. Supports PDF, DOCX, TXT, CSV, Markdown, HTML,
and PowerPoint files.

Design decisions:
    - We use `partition_auto` which auto-detects the file type and applies the
      best extraction strategy.
    - Each extracted element is normalized into a `ParsedElement` dataclass for
      a consistent interface downstream.
    - Graceful fallback: if unstructured extras aren't installed for a file type,
      we fall back to plain text reading.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from config.settings import ParserSettings, get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model for parsed elements
# ---------------------------------------------------------------------------
@dataclass
class ParsedElement:
    """A single extracted element from a document."""

    text: str
    element_type: str  # e.g. "NarrativeText", "Title", "Table", "ListItem", "Image"
    metadata: dict = field(default_factory=dict)

    @property
    def page_number(self) -> int | None:
        return self.metadata.get("page_number")

    @property
    def source_file(self) -> str | None:
        return self.metadata.get("filename")

    def __repr__(self) -> str:
        preview = self.text[:80].replace("\n", " ")
        return f"ParsedElement(type={self.element_type}, page={self.page_number}, text='{preview}...')"


# ---------------------------------------------------------------------------
# Document Parser
# ---------------------------------------------------------------------------
class DocumentParser:
    """
    Parses documents into structured elements using unstructured.io.

    Usage:
        parser = DocumentParser()
        elements = parser.parse_file(Path("report.pdf"))
        elements = parser.parse_directory(Path("data/sample_docs/"))
    """

    def __init__(self, settings: ParserSettings | None = None):
        self.settings = settings or get_settings().parser
        self._validate_unstructured()

    def _validate_unstructured(self) -> None:
        """Check that the unstructured library is available."""
        try:
            from unstructured.partition.auto import partition  # noqa: F401

            self._partition_fn = partition
            logger.info("✅ unstructured library loaded successfully")
        except ImportError:
            logger.warning(
                "⚠️  unstructured library not found. "
                "Install with: pip install 'unstructured[all-docs]'\n"
                "Falling back to plain text extraction."
            )
            self._partition_fn = None

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------
    def parse_file(self, file_path: Path) -> list[ParsedElement]:
        """
        Parse a single file into a list of structured elements.

        Args:
            file_path: Path to the document to parse.

        Returns:
            List of ParsedElement objects with text, type, and metadata.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        if file_path.suffix.lower() not in self.settings.supported_extensions:
            logger.warning(f"Skipping unsupported file type: {file_path.suffix} ({file_path.name})")
            return []

        logger.info(f"📄 Parsing: {file_path.name}")

        if self._partition_fn is not None:
            return self._parse_with_unstructured(file_path)
        else:
            return self._parse_fallback(file_path)

    def parse_directory(self, dir_path: Path) -> dict[str, list[ParsedElement]]:
        """
        Parse all supported documents in a directory.

        Args:
            dir_path: Path to directory containing documents.

        Returns:
            Dict mapping filename → list of ParsedElement objects.
        """
        dir_path = Path(dir_path)

        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        results: dict[str, list[ParsedElement]] = {}
        files = sorted(dir_path.iterdir())

        supported_files = [
            f for f in files if f.is_file() and f.suffix.lower() in self.settings.supported_extensions
        ]

        if not supported_files:
            logger.warning(f"No supported files found in {dir_path}")
            return results

        logger.info(f"📁 Found {len(supported_files)} supported file(s) in {dir_path.name}/")

        for file_path in supported_files:
            try:
                elements = self.parse_file(file_path)
                results[file_path.name] = elements
                logger.info(f"   ✓ {file_path.name}: {len(elements)} elements extracted")
            except Exception as e:
                logger.error(f"   ✗ {file_path.name}: {e}")
                results[file_path.name] = []

        return results

    # -------------------------------------------------------------------
    # Internal: Unstructured-based parsing
    # -------------------------------------------------------------------
    def _parse_with_unstructured(self, file_path: Path) -> list[ParsedElement]:
        """Parse using unstructured.io's partition function."""
        raw_elements = self._partition_fn(
            filename=str(file_path),
            strategy=self.settings.strategy,
            include_page_breaks=self.settings.include_page_breaks,
        )

        parsed: list[ParsedElement] = []
        for elem in raw_elements:
            # Skip empty elements and page breaks
            text = str(elem).strip()
            if not text:
                continue

            # Extract element type name (e.g., "NarrativeText", "Title")
            element_type = type(elem).__name__

            # Build metadata from unstructured's element metadata
            meta = {}
            if hasattr(elem, "metadata"):
                em = elem.metadata
                meta["page_number"] = getattr(em, "page_number", None)
                meta["filename"] = getattr(em, "filename", file_path.name)
                meta["file_directory"] = getattr(em, "file_directory", str(file_path.parent))
                meta["filetype"] = getattr(em, "filetype", None)
                meta["coordinates"] = getattr(em, "coordinates", None)
                # Clean None values
                meta = {k: v for k, v in meta.items() if v is not None}
            else:
                meta["filename"] = file_path.name

            parsed.append(ParsedElement(text=text, element_type=element_type, metadata=meta))

        logger.info(
            f"   Extracted {len(parsed)} elements "
            f"({_summarize_types(parsed)})"
        )
        return parsed

    # -------------------------------------------------------------------
    # Internal: Fallback plain-text parsing
    # -------------------------------------------------------------------
    def _parse_fallback(self, file_path: Path) -> list[ParsedElement]:
        """Simple fallback parser that reads files as plain text."""
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = file_path.read_text(encoding="latin-1")

        if not text.strip():
            logger.warning(f"   Empty file: {file_path.name}")
            return []

        # Split into paragraphs for basic structure
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        elements: list[ParsedElement] = []
        for para in paragraphs:
            # Heuristic: lines starting with # are titles (Markdown)
            if para.startswith("#"):
                etype = "Title"
            elif "|" in para and "---" in para:
                etype = "Table"
            else:
                etype = "NarrativeText"

            elements.append(
                ParsedElement(
                    text=para,
                    element_type=etype,
                    metadata={"filename": file_path.name, "parser": "fallback"},
                )
            )

        logger.info(
            f"   [fallback] Extracted {len(elements)} elements "
            f"({_summarize_types(elements)})"
        )
        return elements


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _summarize_types(elements: list[ParsedElement]) -> str:
    """Create a concise summary of element types, e.g. '12 NarrativeText, 3 Title, 1 Table'."""
    from collections import Counter

    counts = Counter(e.element_type for e in elements)
    return ", ".join(f"{count} {etype}" for etype, count in counts.most_common())
