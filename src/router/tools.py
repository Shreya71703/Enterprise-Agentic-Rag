"""
Enterprise RAG — LLM Agent Tools

Defines the tools available to the LangGraph Agentic Router.
    - Tool A: search_knowledge_base (Step 3 Hybrid search wrapper)
    - Tool B: query_product_metrics (Structured sqlite runner)
    - Tool C: web_search (DuckDuckGo fallback search runner)
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

from src.retrieval import HybridRetriever
from src.router.sql_db import SQLDatabaseService
from src.embeddings import EmbeddingService
from src.vectorstore import QdrantVectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool instances & bootstrap helper
# ---------------------------------------------------------------------------
_retriever: HybridRetriever | None = None
_sql_db: SQLDatabaseService | None = None


def bootstrap_tools(collection_name: str = "enterprise_rag_docs") -> None:
    """Initialize singleton tool services."""
    global _retriever, _sql_db
    
    if _retriever is not None:
        logger.info("ℹ️  Retriever already initialized. Skipping bootstrap.")
        return

    # 1. Setup Hybrid retriever
    embedding_service = EmbeddingService()
    vector_store = QdrantVectorStore(mode="disk", collection_name=collection_name)
    _retriever = HybridRetriever(vector_store, embedding_service)
    _retriever.build_bm25_from_vector_store()

    # 2. Setup SQLite DB
    _sql_db = SQLDatabaseService()
    # Auto-bootstrap SQLite database if CSV is present
    from config.settings import get_settings
    csv_path = get_settings().data_dir / "sample_docs" / "novatech_product_metrics.csv"
    if csv_path.exists():
        _sql_db.bootstrap_from_csv(csv_path)


# ---------------------------------------------------------------------------
# Tool Declarations
# ---------------------------------------------------------------------------
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search unstructured company documents, manuals, and FAQs (annual reports, deployment FAQ).
    Use this tool when asking about company policies, product capabilities, release notes, or text details.
    """
    global _retriever
    if _retriever is None:
        return "Error: Retriever tool is not initialized."

    # Search top 4 documents
    results = _retriever.retrieve(query, top_k=4)
    if not results:
        return f"No results found in company knowledge base for '{query}'"

    formatted = []
    for idx, (doc, score) in enumerate(results, 1):
        src = doc.metadata.get("source_file", "unknown")
        elem_type = doc.metadata.get("element_type", "Text")
        formatted.append(
            f"--- Match {idx} [Score: {score:.4f} | Source: {src} | Type: {elem_type}] ---\n"
            f"{doc.page_content.strip()}"
        )
    return "\n\n".join(formatted)


# Retrieve SQL schema details dynamically for the tool docstring/instructions
def get_metrics_schema_help() -> str:
    """Retrieve schema formatting for query_product_metrics tool helper."""
    global _sql_db
    if _sql_db is None:
        # Fallback schema
        return (
            "Table: product_metrics\n"
            "Columns: product_id (text), product_name (text), category (text), "
            "monthly_revenue_usd (real), active_users (integer), churn_rate_pct (real), "
            "nps_score (integer), avg_response_time_ms (integer), region (text)"
        )
    return _sql_db.get_schema()


@tool
def query_product_metrics(sql_query: str) -> str:
    """
    Query the structured database containing active product metrics, revenue, active users, region data, and NPS scores.
    Use this tool for mathematical operations, averages, total sums, filters, and comparisons (e.g., 'What is the average churn rate?').

    Supported Schema:
      Table: product_metrics
        - product_id (text)
        - product_name (text)
        - category (text)
        - monthly_revenue_usd (real)
        - active_users (integer)
        - churn_rate_pct (real)
        - nps_score (integer)
        - avg_response_time_ms (text/real)
        - region (text)

    Input MUST be a valid SQL SELECT query starting with SELECT.
    """
    global _sql_db
    if _sql_db is None:
        return "Error: SQL Database tool is not initialized."

    res = _sql_db.execute_query(sql_query)
    if isinstance(res, str):
        # Query failed or was invalid
        return res
    if not res:
        return "Query returned 0 rows."

    # Return formatted table string
    import pandas as pd
    df = pd.DataFrame(res)
    return df.to_markdown(index=False)


@tool
def web_search(query: str) -> str:
    """
    Query DuckDuckGo for live internet search and recent external information.
    Use this tool ONLY if the information cannot be found in the knowledge base or product metrics (e.g., live stock price, current news).
    """
    try:
        search = DuckDuckGoSearchRun()
        return search.run(query)
    except Exception as e:
        return f"Web search error: {str(e)}"


# Export list
def get_tools_list() -> list[Any]:
    return [search_knowledge_base, query_product_metrics, web_search]
