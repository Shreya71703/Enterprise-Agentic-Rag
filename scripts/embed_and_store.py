#!/usr/bin/env python3
"""
Enterprise RAG — Embed & Store CLI

Loads processed chunks (from Step 1), embeds them using the configured
embedding model, and stores them in a Qdrant vector database.

Usage:
    python scripts/embed_and_store.py
    python scripts/embed_and_store.py --input data/processed/chunks.json --collection my_docs
    python scripts/embed_and_store.py --fresh --verbose
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> None:
    # Fix Windows console encoding
    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if sys.stderr.encoding != "utf-8":
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Enterprise RAG — Embed & Store Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest sample docs, embed, and store (all-in-one):
  python scripts/embed_and_store.py --ingest data/sample_docs/

  # Embed pre-processed chunks from JSON:
  python scripts/embed_and_store.py --input data/processed/chunks.json

  # Fresh start (delete existing collection first):
  python scripts/embed_and_store.py --ingest data/sample_docs/ --fresh
        """,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to chunks.json file from Step 1 ingestion",
    )
    parser.add_argument(
        "--ingest",
        type=Path,
        default=None,
        help="Directory of raw documents to ingest → embed → store (runs full pipeline)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="enterprise_rag_docs",
        help="Qdrant collection name (default: enterprise_rag_docs)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete existing collection before storing (clean start)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.input is None and args.ingest is None:
        print("❌ Provide either --input (chunks.json) or --ingest (raw docs directory)")
        sys.exit(1)

    print(f"\n🚀 Enterprise RAG — Embed & Store Pipeline")
    print(f"{'─' * 45}")

    # ---------------------------------------------------------------
    # Step A: Get documents (either from JSON or run ingestion)
    # ---------------------------------------------------------------
    from langchain_core.documents import Document

    documents: list[Document] = []

    if args.ingest:
        print(f"📁 Running ingestion pipeline on: {args.ingest.resolve()}")
        from src.ingestion.pipeline import IngestionPipeline

        pipeline = IngestionPipeline()
        documents = pipeline.run(input_dir=args.ingest)

        if not documents:
            print("⚠️  No documents produced from ingestion. Exiting.")
            sys.exit(1)

    elif args.input:
        print(f"📄 Loading chunks from: {args.input.resolve()}")

        if not args.input.exists():
            print(f"❌ File not found: {args.input}")
            sys.exit(1)

        data = json.loads(args.input.read_text(encoding="utf-8"))
        documents = [
            Document(page_content=item["page_content"], metadata=item.get("metadata", {}))
            for item in data
        ]

    print(f"   Loaded {len(documents)} document chunks\n")

    # ---------------------------------------------------------------
    # Step B: Create embeddings
    # ---------------------------------------------------------------
    print("🧮 Generating embeddings...")
    from src.embeddings import EmbeddingService

    embedding_service = EmbeddingService()
    texts = [doc.page_content for doc in documents]
    embeddings = embedding_service.embed_texts(texts, show_progress=True)
    dimension = len(embeddings[0])
    print(f"   ✅ Generated {len(embeddings)} embeddings (dim={dimension})\n")

    # ---------------------------------------------------------------
    # Step C: Store in Qdrant
    # ---------------------------------------------------------------
    print("📥 Storing in Qdrant vector database...")
    from src.vectorstore import QdrantVectorStore

    store = QdrantVectorStore(mode="disk", collection_name=args.collection)

    if args.fresh and store.collection_exists():
        store.delete_collection()
        print("   🗑️  Deleted existing collection")

    store.create_collection(dimension=dimension)
    count = store.upsert_documents(documents, embeddings)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    info = store.get_collection_info()

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="📊 Vector Store Summary", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        table.add_row("Collection", info["name"])
        table.add_row("Points Stored", str(info["points_count"]))
        table.add_row("Vector Dimension", str(info["dimension"]))
        table.add_row("Distance Metric", info["distance"])
        table.add_row("Status", info["status"])

        console.print(f"\n")
        console.print(table)

    except ImportError:
        print(f"\n{'=' * 45}")
        print(f"📊 VECTOR STORE SUMMARY")
        print(f"{'=' * 45}")
        for k, v in info.items():
            print(f"   {k}: {v}")
        print(f"{'=' * 45}")

    store.close()
    print(f"\n✅ Pipeline complete! {count} vectors stored and ready for retrieval.\n")


if __name__ == "__main__":
    main()
