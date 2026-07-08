#!/usr/bin/env python3
"""
Enterprise RAG — Evaluation Runner

Executes RAG evaluation on a test suite of queries, computing Faithfulness
and Context Relevance scores using the LLM-as-a-judge service. Outputs
a summary statistics table and saves results.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --verbose
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
        description="Enterprise RAG — Evaluation runner",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="enterprise_rag_docs",
        help="Qdrant collection name",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save evaluation JSON report",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Set default output
    output_path = args.output or (PROJECT_ROOT / "data" / "processed" / "evaluation_report.json")

    print("\n🚀 Starting RAG Evaluation Runner...")
    print("🤖 Bootstrapping tools, retriever, and database...")

    from src.router.tools import bootstrap_tools, search_knowledge_base
    bootstrap_tools(collection_name=args.collection)

    from src.router.agent import AgenticRouter
    router = AgenticRouter()

    from src.evaluation.evaluator import RAGEvaluator
    evaluator = RAGEvaluator(router.llm)

    # Define test suite dataset
    test_cases = [
        {
            "query": "What was NovaTech's revenue in 2024?",
            "ground_truth": "NovaTech achieved record revenue of $847 million in fiscal year 2024.",
            "expected_source": "novatech_annual_report_2024.md"
        },
        {
            "query": "What makes Cortex different from other AI platforms?",
            "ground_truth": "Cortex is differentiated by Hybrid Search Architecture, Agentic Routing, and Enterprise-Grade Security.",
            "expected_source": "novatech_cortex_faq.md"
        },
        {
            "query": "What are the hardware requirements for on-premise deployment?",
            "ground_truth": "On-premise deployment requires 8 CPU cores, 64GB RAM for control plane, and minimum 2 NVIDIA A100 or H100 GPUs.",
            "expected_source": "novatech_cortex_faq.md"
        }
    ]

    print(f"📊 Running evaluation on {len(test_cases)} test cases...")
    
    results = []
    total_faithfulness = 0.0
    total_relevance = 0.0

    for idx, tc in enumerate(test_cases, 1):
        query = tc["query"]
        print(f"\n[{idx}/{len(test_cases)}] Evaluating query: '{query}'")

        # 1. Run RAG Pipeline
        try:
            # Generate answer
            answer = router.query(query)
            
            # Retrieve supporting context documents directly to evaluate relevance
            # We call the tool helper directly or fetch documents
            contexts = []
            context_str = search_knowledge_base.invoke(query)
            if context_str and "Error" not in context_str:
                # Split clean chunks
                contexts = [c.strip() for c in context_str.split("--- Match") if c.strip()]
            else:
                # If SQL query used, get DB metrics
                from src.router.tools import _sql_db
                if _sql_db:
                    contexts = [_sql_db.get_schema()]
                else:
                    contexts = ["No context available."]

            # 2. Run Evaluator
            eval_res = evaluator.evaluate(query, contexts, answer)
            
            faith = eval_res["faithfulness"]
            rel = eval_res["context_relevance"]
            total_faithfulness += faith
            total_relevance += rel

            results.append({
                "query": query,
                "answer": answer,
                "faithfulness": faith,
                "context_relevance": rel,
                "reasoning": eval_res["reasoning"]
            })

            print(f"   ✓ Faithfulness:       {faith:.2f}")
            print(f"   ✓ Context Relevance:  {rel:.2f}")

        except Exception as e:
            print(f"   ✗ Failed to evaluate case: {e}")

    # Compute averages
    avg_faithfulness = total_faithfulness / len(results) if results else 0.0
    avg_relevance = total_relevance / len(results) if results else 0.0

    report = {
        "summary": {
            "test_cases_count": len(test_cases),
            "evaluated_count": len(results),
            "average_faithfulness": round(avg_faithfulness, 2),
            "average_context_relevance": round(avg_relevance, 2)
        },
        "results": results
    }

    # Save output report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n💾 Saved evaluation report to: {output_path.resolve()}")

    # Render summary table
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="🏆 RAG Evaluation Summary", show_header=True, header_style="bold cyan")
        table.add_column("Query", style="bold")
        table.add_column("Faithfulness Score", justify="right")
        table.add_column("Context Relevance Score", justify="right")

        for r in results:
            table.add_row(r["query"][:40] + "...", f"{r['faithfulness']:.2f}", f"{r['context_relevance']:.2f}")

        table.add_row("─" * 40, "───", "───")
        table.add_row("AVERAGE", f"[green]{avg_faithfulness:.2f}[/green]", f"[green]{avg_relevance:.2f}[/green]")
        console.print(table)

    except ImportError:
        print(f"\n{'=' * 45}")
        print(f"🏆 RAG EVALUATION SUMMARY")
        print(f"{'=' * 45}")
        for r in results:
            print(f"   Query: {r['query']}")
            print(f"     Faithfulness:       {r['faithfulness']:.2f}")
            print(f"     Context Relevance:  {r['context_relevance']:.2f}")
        print(f"{'=' * 45}")
        print(f"   AVERAGE FAITHFULNESS:      {avg_faithfulness:.2f}")
        print(f"   AVERAGE CONTEXT RELEVANCE: {avg_relevance:.2f}")
        print(f"{'=' * 45}\n")


if __name__ == "__main__":
    main()
