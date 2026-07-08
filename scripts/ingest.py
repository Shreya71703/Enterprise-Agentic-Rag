#!/usr/bin/env python3
"""
Enterprise RAG — Document Ingestion CLI

Command-line entry point for processing raw documents through the
ingestion pipeline. Produces chunked, metadata-enriched documents
ready for embedding and vector storage.

Usage:
    python scripts/ingest.py --input-dir data/sample_docs/
    python scripts/ingest.py --input-dir data/sample_docs/ --output data/processed/chunks.json --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with rich formatting."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def print_chunk_details(documents: list, max_preview: int = 5) -> None:
    """Pretty-print chunk details using Rich (or fallback to plain text)."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()

        # Summary table
        table = Table(title="📊 Chunk Summary", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        total = len(documents)
        sizes = [len(d.page_content) for d in documents]
        types = {}
        for d in documents:
            t = d.metadata.get("element_type", "Unknown")
            types[t] = types.get(t, 0) + 1

        table.add_row("Total Chunks", str(total))
        table.add_row("Avg Size", f"{sum(sizes) / len(sizes):.0f} chars" if sizes else "—")
        table.add_row("Min Size", f"{min(sizes)} chars" if sizes else "—")
        table.add_row("Max Size", f"{max(sizes)} chars" if sizes else "—")
        for t, count in sorted(types.items()):
            table.add_row(f"  └ {t}", str(count))

        console.print(table)

        # Preview first N chunks
        console.print(f"\n[bold]📝 Preview (first {min(max_preview, total)} chunks):[/bold]\n")
        for i, doc in enumerate(documents[:max_preview]):
            preview = doc.page_content[:200].replace("\n", " ")
            meta_str = ", ".join(
                f"{k}={v}"
                for k, v in doc.metadata.items()
                if k in ("element_type", "page_number", "chunk_index", "char_count")
            )
            console.print(
                Panel(
                    f"[dim]{meta_str}[/dim]\n\n{preview}{'...' if len(doc.page_content) > 200 else ''}",
                    title=f"Chunk {i + 1}",
                    border_style="blue",
                )
            )

    except ImportError:
        # Fallback without rich
        print(f"\n{'=' * 60}")
        print(f"📊 CHUNK SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total chunks: {len(documents)}")
        if documents:
            sizes = [len(d.page_content) for d in documents]
            print(f"Avg size: {sum(sizes) / len(sizes):.0f} chars")
            print(f"Range: {min(sizes)} — {max(sizes)} chars")
        print(f"{'=' * 60}")

        for i, doc in enumerate(documents[:max_preview]):
            preview = doc.page_content[:200].replace("\n", " ")
            print(f"\n--- Chunk {i + 1} ---")
            print(f"Type: {doc.metadata.get('element_type', '?')}")
            print(f"Preview: {preview}...")


def main() -> None:
    # Fix Windows console encoding for emoji/unicode output
    import io
    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if sys.stderr.encoding != "utf-8":
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Enterprise RAG — Document Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ingest.py --input-dir data/sample_docs/
  python scripts/ingest.py --input-dir data/sample_docs/ --output data/processed/chunks.json
  python scripts/ingest.py --input-dir data/sample_docs/ --verbose --preview 10
        """,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing documents to process",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save processed chunks as JSON (optional)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=5,
        help="Number of chunks to preview (default: 5)",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Validate input
    if not args.input_dir.exists():
        print(f"❌ Input directory not found: {args.input_dir}")
        sys.exit(1)

    # Run pipeline
    print(f"\n🚀 Enterprise RAG — Ingestion Pipeline")
    print(f"{'─' * 40}")
    print(f"📁 Input:  {args.input_dir.resolve()}")
    if args.output:
        print(f"💾 Output: {args.output.resolve()}")
    print()

    from src.ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    documents = pipeline.run(input_dir=args.input_dir)

    if not documents:
        print("⚠️  No documents were produced. Check your input files.")
        sys.exit(1)

    # Display results
    print_chunk_details(documents, max_preview=args.preview)

    # Save if requested
    if args.output:
        pipeline.save_chunks(documents, args.output)
        print(f"\n✅ Chunks saved to: {args.output.resolve()}")

    print(f"\n✅ Pipeline complete! {len(documents)} chunks ready for embedding.\n")


if __name__ == "__main__":
    main()
