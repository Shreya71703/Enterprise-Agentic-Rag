"""
Enterprise RAG — SQL Database Service

Sets up a local SQLite database containing structured tables (e.g., product metrics).
Used by the Agentic Router (Tool C) for exact numerical queries, aggregations,
and analysis of tabular data.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
import pandas as pd

from config.settings import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SQL Database Service
# ---------------------------------------------------------------------------
class SQLDatabaseService:
    """
    Manages SQLite database for tabular RAG querying.

    Usage:
        db = SQLDatabaseService()
        db.bootstrap_from_csv(Path("data/sample_docs/novatech_product_metrics.csv"))
        schema = db.get_schema()
        results = db.execute_query("SELECT SUM(monthly_revenue_usd) FROM product_metrics")
    """

    def __init__(self, db_path: Path | str | None = None):
        if db_path is None:
            self.db_path = get_settings().data_dir / "enterprise_rag.db"
        else:
            self.db_path = Path(db_path)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def bootstrap_from_csv(
        self,
        csv_path: Path | str,
        table_name: str = "product_metrics",
    ) -> None:
        """
        Load structured metrics from CSV into SQLite table.
        Skips if table already exists and is populated.
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found for SQL bootstrapping: {csv_path}")

        # Check if table is already populated
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            table_exists = cursor.fetchone()
            if table_exists:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                if row_count > 0:
                    logger.info(f"ℹ️ SQLite database table '{table_name}' already exists and contains {row_count} rows. Skipping CSV reload.")
                    return
        except Exception as e:
            logger.warning(f"⚠️ Failed to check database status: {e}. Reloading CSV...")
        finally:
            conn.close()

        logger.info(f"💾 Bootstrapping SQLite database table '{table_name}' from CSV: {csv_path.name}")

        df = pd.read_csv(csv_path)

        # Clean column names to make them SQL friendly
        df.columns = [c.strip().lower().replace(" ", "_").replace("(", "").replace(")", "") for c in df.columns]

        conn = sqlite3.connect(self.db_path)
        try:
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            logger.info(f"✅ Bootstrapped table '{table_name}' with {len(df)} rows")
        finally:
            conn.close()

    def execute_query(self, sql_query: str) -> list[dict] | str:
        """
        Execute a read-only SQL query and return rows as dict list.
        """
        # Simple SQL injection safety guard (readonly enforce)
        cleaned_query = sql_query.strip().lower()
        if not cleaned_query.startswith("select"):
            return "Error: Only SELECT queries are permitted on this database."

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"❌ SQL Execution failed: {e}")
            return f"Error executing query: {str(e)}"
        finally:
            conn.close()

    def get_schema(self, table_name: str = "product_metrics") -> str:
        """
        Retrieve schema and sample rows for LLM system prompting.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Fetch table columns and types
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()

            schema_lines = [f"Table: {table_name}"]
            for col in columns:
                # col[1] = name, col[2] = type
                schema_lines.append(f"  - {col[1]} ({col[2]})")

            # Fetch sample rows
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
            samples = cursor.fetchall()
            if samples:
                schema_lines.append("\nSample rows:")
                for row in samples:
                    schema_lines.append(f"  {list(row)}")

            return "\n".join(schema_lines)
        except Exception as e:
            return f"Error retrieving schema: {e}"
        finally:
            conn.close()
