"""
Enterprise RAG — Router & Tool Tests

Tests for SQL Database service, LLM agent tools, and the LangGraph
Agentic Router graph compilation.
"""

from __future__ import annotations

import tempfile
import pytest
from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage

from src.router.sql_db import SQLDatabaseService
from src.router.tools import query_product_metrics, search_knowledge_base, web_search
from src.router.agent import AgenticRouter


# ===================================================================
# Fixtures
# ===================================================================

SAMPLE_CSV_DATA = """product_id,product_name,category,monthly_revenue_usd,active_users,churn_rate_pct,nps_score,avg_response_time_ms,region
P-001,Alpha Engine,AI,150000,500,2.5,75,120,US
P-002,Beta Vision,Vision,80000,300,4.0,68,200,EU
"""

@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Create a temporary path for the sqlite database."""
    return tmp_path / "test_enterprise_rag.db"


@pytest.fixture
def csv_file_path(tmp_path: Path) -> Path:
    """Create a temporary CSV file representing sample metrics."""
    csv_file = tmp_path / "test_metrics.csv"
    csv_file.write_text(SAMPLE_CSV_DATA, encoding="utf-8")
    return csv_file


@pytest.fixture
def sql_service(temp_db_path: Path, csv_file_path: Path) -> SQLDatabaseService:
    """Setup and return SQLDatabaseService loaded with sample CSV data."""
    service = SQLDatabaseService(db_path=temp_db_path)
    service.bootstrap_from_csv(csv_file_path, table_name="product_metrics")
    return service


# ===================================================================
# SQL Database Tests
# ===================================================================

class TestSQLDatabaseService:
    """Tests for the SQL Database service."""

    def test_bootstrap_and_schema(self, sql_service: SQLDatabaseService):
        schema = sql_service.get_schema()
        assert "Table: product_metrics" in schema
        assert "product_id" in schema
        assert "monthly_revenue_usd" in schema

    def test_safe_read_only_queries(self, sql_service: SQLDatabaseService):
        # Read query should succeed
        res = sql_service.execute_query("SELECT SUM(monthly_revenue_usd) as total FROM product_metrics")
        assert isinstance(res, list)
        assert len(res) == 1
        assert res[0]["total"] == 230000

        # Non-SELECT query should fail/be blocked
        res_blocked = sql_service.execute_query("DELETE FROM product_metrics")
        assert "Error: Only SELECT queries are permitted" in res_blocked

    def test_query_syntax_error(self, sql_service: SQLDatabaseService):
        # Invalid query syntax
        res = sql_service.execute_query("SELECT * FROM invalid_table_name")
        assert "Error executing query" in res


# ===================================================================
# Agent Tools Tests
# ===================================================================

class TestAgentTools:
    """Tests for the registered agent tools."""

    def test_tools_return_errors_when_uninitialized(self):
        # If retriever is None, search_knowledge_base tool returns helpful error
        import src.router.tools as rt
        rt._retriever = None
        rt._sql_db = None

        res_kb = search_knowledge_base.invoke("query")
        assert "Error: Retriever tool is not initialized" in res_kb

        res_sql = query_product_metrics.invoke("SELECT * FROM product_metrics")
        assert "Error: SQL Database tool is not initialized" in res_sql

    def test_web_search_runs(self):
        # Simple test to make sure web_search runs/fails gracefully
        res = web_search.invoke("who is the current US President?")
        # Should return search output or a graceful search error string
        assert isinstance(res, str)


# ===================================================================
# Agentic Router Tests
# ===================================================================

class TestAgenticRouter:
    """Tests for the Agentic Router and Graph construction."""

    def test_router_graph_compiles(self):
        # Force dummy model to avoid hitting Google endpoint
        from langchain_core.language_models import BaseChatModel
        from langchain_core.outputs import ChatResult, ChatGeneration
        
        class DummyChatModel(BaseChatModel):
            def _generate(self, messages, stop=None, **kwargs):
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="mocked"))])
            def bind_tools(self, tools, **kwargs):
                return self
            @property
            def _llm_type(self) -> str:
                return "dummy"

        router = AgenticRouter(llm=DummyChatModel())
        assert router.graph is not None

        # Verify state graph contains expected nodes
        node_names = list(router.graph.get_graph().nodes.keys())
        assert "agent" in node_names
        assert "tools" in node_names

    def test_router_query_fallback(self):
        from langchain_core.language_models import BaseChatModel
        from langchain_core.outputs import ChatResult, ChatGeneration
        
        class DummyChatModel(BaseChatModel):
            def _generate(self, messages, stop=None, **kwargs):
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Final Mock Response"))])
            def bind_tools(self, tools, **kwargs):
                return self
            @property
            def _llm_type(self) -> str:
                return "dummy"

        router = AgenticRouter(llm=DummyChatModel())
        res = router.query("What is the product schema?")
        assert res == "Final Mock Response"
