#!/usr/bin/env python3
"""
Enterprise RAG — Interactive Chatbot CLI

Conversational agent that routes user queries to the best tool:
    - Unstructured company docs (Hybrid search)
    - SQL product metrics database (SQLite)
    - Web search fallback (DuckDuckGo)

Usage:
    python scripts/chat.py
    python scripts/chat.py --verbose
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


def main() -> None:
    # Fix Windows console encoding
    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if sys.stderr.encoding != "utf-8":
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Enterprise RAG — Chatbot Agent CLI",
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
        from rich.markdown import Markdown
        from rich.panel import Panel
        console = Console()
    except ImportError:
        pass

    print("\n🚀 Starting Enterprise RAG Agentic Chatbot...")
    print("🤖 Bootstrapping tools & SQL database...")

    from src.router.tools import bootstrap_tools
    bootstrap_tools(collection_name=args.collection)

    from src.router.agent import AgenticRouter
    router = AgenticRouter()

    # Track chat history
    from langchain_core.messages import HumanMessage, AIMessage
    chat_history = []

    print("\n✅ Chatbot ready! Ask a question. Types of queries you can try:")
    print("  1. Unstructured: 'What makes Cortex different from other AI platforms?'")
    print("  2. SQL/Tabular:   'What is the total monthly revenue across all regions?'")
    print("  3. Web Search:    'What was the closing price of AAPL stock yesterday?'")
    print("\nType 'quit', 'exit', or 'q' to stop.\n")

    try:
        while True:
            try:
                user_input = input("👤 You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                break

            print("🧠 Thinking...")
            try:
                response = router.query(user_input, history=chat_history)
                
                # Append to history (keep history buffer at max 10 messages to avoid context overflow)
                chat_history.append(HumanMessage(content=user_input))
                chat_history.append(AIMessage(content=response))
                if len(chat_history) > 10:
                    chat_history = chat_history[-10:]

                # Print response
                if console:
                    console.print(f"\n[bold green]🤖 Agent:[/bold green]")
                    console.print(Panel(Markdown(response), border_style="green"))
                    console.print()
                else:
                    print(f"\n🤖 Agent:\n{response}\n")

            except Exception as e:
                print(f"❌ Error during query execution: {e}")

    except KeyboardInterrupt:
        print("\n")

    print("👋 Goodbye!\n")


if __name__ == "__main__":
    main()
