#!/usr/bin/env python3
"""
Enterprise RAG — Hybrid Retrieval CLI

Runs queries against the search indices and compares retrieval strategies:
Vector Search vs BM25 Keyword Search vs Hybrid RRF vs Hybrid + Re-ranking.

Usage:
    python scripts/retrieve.py "revenue of NovaTech in 2024"
    python scripts/retrieve.py "Cortex 3.0" --top-k 3
    python scripts/retrieve.py "deployment requirements" --strategy compare
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


def format_doc_preview(doc, score: float, rank: int) -> str:
    meta = doc.metadata
    return (
        f"[bold blue]#{rank}[/bold blue] (score: [green]{score:.4f}[/green]) | "
        f"[dim]source=[/dim]{meta.get('source_file', '?')} | [dim]type=[/dim]{meta.get('element_type', '?')}\n"
        f"{doc.page_content[:300].strip()}{'...' if len(doc.page_content) > 300 else ''}"
    )


def run_comparison(query: str, retriever, top_k: int, console) -> None:
    """Run vector, BM25, and hybrid rerank search side-by-side to show difference."""
    from rich.columns import Columns
    from rich.panel import Panel

    # 1. Vector Search only
    query_emb = retriever.embedding_service.embed_query(query)
    vector_raw = retriever.vector_store.search_with_documents(query_emb, top_k=top_k)

    # 2. BM25 Search only
    bm25_raw = retriever.bm25.search(query, top_k=top_k)

    # 3. Hybrid + Rerank
    hybrid_rerank = retriever.retrieve(query, top_k=top_k, use_reranking=True)

    # Output using Rich Panel layout
    console.print(f"\n[bold magenta]🔎 Search Comparison for:[/bold magenta] '{query}'\n")

    def build_column_panel(title: str, results: list, is_scored_docs: bool = False) -> Panel:
        body = []
        if not results:
            body.append("[yellow]No matches found[/yellow]")
        else:
            for idx, item in enumerate(results, 1):
                if is_scored_docs:
                    # Item is (SearchResult) or similar
                    doc = Document(page_content=item.content, metadata=item.metadata)
                    score = item.score
                elif isinstance(item, tuple):
                    doc, score = item
                else:
                    doc, score = item, 1.0

                body.append(format_doc_preview(doc, score, idx))
                if idx < len(results):
                    body.append("[dim]─" * 40 + "[/dim]")

        return Panel("\n".join(body), title=title, border_style="cyan", expand=True)

    from langchain_core.documents import Document

    vector_panel = build_column_panel("📡 1. Semantic Vector Search", vector_raw)
    bm25_panel = build_column_panel("⌨️ 2. BM25 Keyword Search", bm25_raw)
    hybrid_panel = build_column_panel("🏆 3. Hybrid + Cross-Encoder Re-ranked", hybrid_rerank)

    # Render panels side-by-side or stacked depending on width
    console.print(vector_panel)
    console.print(bm25_panel)
    console.print(hybrid_panel)


def main() -> None:
    # Fix Windows console encoding
    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if sys.stderr.encoding != "utf-8":
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Enterprise RAG — Retrieval Testing CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "query",
        type=str,
        help="Search query to execute",
    )
    parser.add_argument(
        "--strategy",
        choices=["vector", "bm25", "hybrid", "hybrid-rerank", "compare"],
        default="hybrid-rerank",
        help="Search strategy to use (default: hybrid-rerank)",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=3,
        help="Number of results to return",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="enterprise_rag_docs",
        help="Qdrant collection name",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Load Rich
    console = None
    try:
        from rich.console import Console
        console = Console()
    except ImportError:
        pass

    # Initialize components
    from src.embeddings import EmbeddingService
    from src.retrieval import HybridRetriever
    from src.vectorstore import QdrantVectorStore

    embedding_service = EmbeddingService()
    vector_store = QdrantVectorStore(mode="disk", collection_name=args.collection)

    if not vector_store.collection_exists():
        print(f"❌ Qdrant collection '{args.collection}' not found. Run embed_and_store.py first.")
        sys.exit(1)

    # Initialize Hybrid retriever and bootstrap BM25 from the Qdrant DB
    retriever = HybridRetriever(vector_store, embedding_service)
    retriever.build_bm25_from_vector_store()

    if args.strategy == "compare" and console:
        run_comparison(args.query, retriever, args.top_k, console)
    else:
        # Run specific strategy
        print(f"\n🚀 Running strategy: {args.strategy}")
        results = []

        if args.strategy == "vector":
            query_emb = embedding_service.embed_query(args.query)
            results = vector_store.search_with_documents(query_emb, top_k=args.top_k)
        elif args.strategy == "bm25":
            results = retriever.bm25.search(args.query, top_k=args.top_k)
        elif args.strategy == "hybrid":
            results = retriever.retrieve(args.query, top_k=args.top_k, use_reranking=False)
        else:  # hybrid-rerank
            results = retriever.retrieve(args.query, top_k=args.top_k, use_reranking=True)

        # Print results
        if console:
            from rich.panel import Panel
            console.print(f"\n[bold magenta]🔎 Search Results for Strategy '{args.strategy}':[/bold magenta] '{args.query}'\n")
            for idx, (doc, score) in enumerate(results, 1):
                console.print(
                    Panel(
                        format_doc_preview(doc, score, idx),
                        title=f"Result {idx}",
                        border_style="blue",
                    )
                )
        else:
            print(f"\n🔎 Search Results for '{args.query}':")
            for idx, (doc, score) in enumerate(results, 1):
                print(f"  #{idx} (score={score:.4f}): {doc.page_content[:200]}...")

    vector_store.close()
    print("\n✅ Search completed.\n")


if __name__ == "__main__":
    main()
