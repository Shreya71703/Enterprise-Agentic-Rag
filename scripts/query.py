#!/usr/bin/env python3
"""
Enterprise RAG — Interactive Query CLI

Query the Qdrant vector store with natural language questions.
Retrieves the most relevant document chunks using semantic search.

Usage:
    python scripts/query.py "What was NovaTech's revenue in 2024?"
    python scripts/query.py "What LLMs does Cortex support?" --top-k 10
    python scripts/query.py --interactive
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def display_results(query: str, results: list, console=None) -> None:
    """Pretty-print search results."""
    if console is None:
        # Fallback plain text
        print(f"\n{'─' * 60}")
        print(f"🔍 Query: {query}")
        print(f"{'─' * 60}")
        if not results:
            print("   No results found.")
            return
        for i, r in enumerate(results, 1):
            print(f"\n--- Result {i} (score: {r.score:.4f}) ---")
            print(f"   Type: {r.metadata.get('element_type', '?')}")
            print(f"   Source: {r.metadata.get('source_file', '?')}")
            print(f"   {r.content[:300]}{'...' if len(r.content) > 300 else ''}")
        print()
        return

    from rich.panel import Panel
    from rich.text import Text

    console.print(f"\n[bold cyan]🔍 Query:[/bold cyan] {query}")
    console.print(f"[dim]{'─' * 60}[/dim]")

    if not results:
        console.print("[yellow]   No results found.[/yellow]")
        return

    for i, r in enumerate(results, 1):
        score_color = "green" if r.score > 0.7 else "yellow" if r.score > 0.5 else "red"
        meta = (
            f"[dim]score=[/dim][{score_color}]{r.score:.4f}[/{score_color}]"
            f"  [dim]type=[/dim]{r.metadata.get('element_type', '?')}"
            f"  [dim]source=[/dim]{r.metadata.get('source_file', '?')}"
        )
        content = r.content[:400] + ("..." if len(r.content) > 400 else "")

        console.print(
            Panel(
                f"{meta}\n\n{content}",
                title=f"Result {i}",
                border_style="blue",
            )
        )


def main() -> None:
    # Fix Windows console encoding
    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if sys.stderr.encoding != "utf-8":
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Enterprise RAG — Query the Vector Store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/query.py "What was NovaTech's revenue?"
  python scripts/query.py "deployment options" --top-k 10
  python scripts/query.py --interactive
  python scripts/query.py "pricing" --filter-type Table
        """,
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Search query (omit for interactive mode)",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Enter interactive query mode",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=None,
        help="Minimum similarity score threshold (0.0-1.0)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="enterprise_rag_docs",
        help="Qdrant collection name",
    )
    parser.add_argument(
        "--filter-source",
        type=str,
        default=None,
        help="Filter results by source file name",
    )
    parser.add_argument(
        "--filter-type",
        type=str,
        default=None,
        help="Filter results by element type (e.g., Table, NarrativeText)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Try to use Rich console
    console = None
    try:
        from rich.console import Console
        console = Console()
    except ImportError:
        pass

    # Initialize services
    from src.embeddings import EmbeddingService
    from src.vectorstore import QdrantVectorStore

    embedding_service = EmbeddingService()
    store = QdrantVectorStore(mode="disk", collection_name=args.collection)

    if not store.collection_exists():
        print(f"❌ Collection '{args.collection}' not found. Run embed_and_store.py first.")
        sys.exit(1)

    info = store.get_collection_info()
    print(f"\n📊 Collection: {info['name']} ({info['points_count']} vectors, dim={info['dimension']})")

    # Build metadata filters
    filters = {}
    if args.filter_source:
        filters["source_file"] = args.filter_source
    if args.filter_type:
        filters["element_type"] = args.filter_type

    def run_query(query_text: str) -> None:
        """Execute a single query and display results."""
        query_embedding = embedding_service.embed_query(query_text)
        results = store.search(
            query_embedding=query_embedding,
            top_k=args.top_k,
            score_threshold=args.threshold,
            filter_conditions=filters if filters else None,
        )
        display_results(query_text, results, console)

    # Single query mode
    if args.query and not args.interactive:
        run_query(args.query)
        store.close()
        return

    # Interactive mode
    print("\n🔮 Interactive Query Mode (type 'quit' or 'exit' to stop)\n")

    try:
        while True:
            try:
                query_text = input("❓ Enter query: ").strip()
            except EOFError:
                break

            if not query_text:
                continue
            if query_text.lower() in ("quit", "exit", "q"):
                break

            run_query(query_text)

    except KeyboardInterrupt:
        print("\n")

    store.close()
    print("👋 Goodbye!\n")


if __name__ == "__main__":
    main()
