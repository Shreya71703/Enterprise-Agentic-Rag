# 🧠 Enterprise RAG System

> An Advanced Enterprise-Grade Retrieval-Augmented Generation system demonstrating production-level AI engineering with hybrid search, SQL tool routing, and a modern Streamlit interface.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    🧠 Agentic Router                     │
│              (LangGraph Agent + Tool Selection)          │
├──────────┬──────────────────────┬───────────────────────┤
│ Tool A   │      Tool B          │      Tool C           │
│ Vector   │    Web Search        │    SQL Database        │
│ Database │  (Tavily/DDG)        │  (Structured Data)     │
├──────────┴──────────────────────┴───────────────────────┤
│              🔍 Hybrid Retrieval Layer                   │
│         Vector Search + BM25 + Cross-Encoder             │
├─────────────────────────────────────────────────────────┤
│              📄 Data Ingestion Pipeline                  │
│     Unstructured.io → Semantic Chunking → Metadata       │
└─────────────────────────────────────────────────────────┘
```

## Features Implemented

| Component | Description |
|-----------|-------------|
| **Document Parser** | Multi-format extraction (PDF, DOCX, TXT, CSV, MD) via `unstructured.io` with graceful fallback |
| **Semantic Chunker** | Embedding-based splitting (Gemini / HuggingFace) with `RecursiveCharacterTextSplitter` fallback |
| **Metadata Enricher** | Stamps every chunk with `doc_id`, `page_number`, `element_type`, `chunk_index`, `timestamp` |
| **Hybrid Retriever** | Combines semantic vector database search (Qdrant) and keyword search (BM25) with Cross-Encoder re-ranking |
| **SQL Database Service** | Structured SQLite database for executing analytical queries on product metrics |
| **Agentic Router** | LangGraph-based state-graph agent that dynamically routes questions to SQL, Vector DB, or Web Search |
| **Frontend Web App** | Modern Streamlit dashboard featuring an interactive agentic chat interface and search debugger |

## Quick Start

### 1. Setup the Environment
```bash
# Clone the repository
git clone https://github.com/Shreya71703/Enterprise-Agentic-Rag.git
cd Enterprise-Agentic-Rag

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -e ".[dev]"
```

### 2. Configure Credentials
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your-gemini-api-key-here
```

### 3. Run Ingestion Pipeline
```bash
python scripts/ingest.py --input-dir data/sample_docs/ --verbose
```

### 4. Run the Streamlit Dashboard
```bash
streamlit run src/frontend/app.py
```

### 5. Run Tests
```bash
python -m pytest tests/ -v
```

## Project Structure

```
enterprise-rag/
├── config/
│   └── settings.py            # Pydantic-based configuration
├── data/
│   └── sample_docs/           # Sample documents (annual report, FAQ, CSV)
├── src/
│   ├── ingestion/
│   │   ├── parser.py          # Document parsing (unstructured.io)
│   │   ├── chunker.py         # Semantic chunking (LangChain)
│   │   ├── metadata.py        # Metadata enrichment
│   │   └── pipeline.py        # Pipeline orchestrator
│   ├── retrieval/
│   │   ├── __init__.py        # Hybrid retriever orchestrator
│   │   ├── bm25.py            # Keyword search index
│   │   └── reranker.py        # Cross-Encoder re-ranker
│   ├── router/
│   │   ├── agent.py           # LangGraph router agent
│   │   ├── sql_db.py          # SQLite database service
│   │   └── tools.py           # Agent tool definitions
│   └── frontend/
│       └── app.py             # Streamlit dashboard interface
├── scripts/
│   └── ingest.py              # Ingestion CLI entry point
├── tests/
│   └── test_ingestion.py      # Unit & integration tests
└── pyproject.toml             # Dependencies & project config
```

## Roadmap

- [x] **Step 1**: Data Ingestion & Chunking Pipeline
- [x] **Step 2**: Embeddings & Vector Store (Qdrant)
- [x] **Step 3**: Hybrid Retrieval (Vector + BM25 + Re-ranking)
- [x] **Step 4**: Agentic Router (LangGraph + Tool Selection)
- [x] **Step 5**: Frontend Dashboard (Streamlit) & RAGAS Evaluation

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | Gemini 2.5 Flash / Gemini 2.0 Flash Lite |
| **Framework** | LangChain & LangGraph |
| **Embeddings** | Google `gemini-embedding-2` |
| **Document Parsing** | `unstructured.io` |
| **Vector DB** | Qdrant (Local Disk Mode) |
| **SQL DB** | SQLite |
| **Frontend** | Streamlit |

## License

MIT
