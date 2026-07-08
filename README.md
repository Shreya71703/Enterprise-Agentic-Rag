# 🧠 Enterprise RAG System

> An Advanced Enterprise-Grade Retrieval-Augmented Generation system demonstrating production-level AI engineering.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-green)
![Status](https://img.shields.io/badge/Status-Step%201%20Complete-yellow)

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
│              📄 Data Ingestion Pipeline  ← YOU ARE HERE  │
│     Unstructured.io → Semantic Chunking → Metadata       │
└─────────────────────────────────────────────────────────┘
```

## Step 1: Data Ingestion & Chunking Pipeline

### What's Implemented

| Component | Description |
|-----------|-------------|
| **Document Parser** | Multi-format extraction (PDF, DOCX, TXT, CSV, MD) via `unstructured.io` with graceful fallback |
| **Semantic Chunker** | Embedding-based splitting (Gemini / HuggingFace) with `RecursiveCharacterTextSplitter` fallback |
| **Metadata Enricher** | Stamps every chunk with `doc_id`, `page_number`, `element_type`, `chunk_index`, `timestamp` |
| **Pipeline Orchestrator** | Composes parse → chunk → enrich with stats tracking and JSON export |
| **CLI Interface** | Run ingestion via command line with rich terminal output |

### Quick Start

```bash
# 1. Clone and enter the project
cd enterprise-rag

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Set up environment
cp .env.example .env
# Edit .env with your GOOGLE_API_KEY

# 5. Run the ingestion pipeline
python scripts/ingest.py --input-dir data/sample_docs/ --verbose

# 6. Run tests
python -m pytest tests/ -v
```

### Project Structure

```
enterprise-rag/
├── config/
│   └── settings.py            # Pydantic-based configuration
├── data/
│   └── sample_docs/           # Sample documents (annual report, FAQ, CSV)
├── src/
│   └── ingestion/
│       ├── parser.py           # Document parsing (unstructured.io)
│       ├── chunker.py          # Semantic chunking (LangChain)
│       ├── metadata.py         # Metadata enrichment
│       └── pipeline.py         # Pipeline orchestrator
├── scripts/
│   └── ingest.py              # CLI entry point
├── tests/
│   └── test_ingestion.py      # Unit & integration tests
├── pyproject.toml             # Dependencies & project config
└── .env.example               # Environment variable template
```

## Roadmap

- [x] **Step 1**: Data Ingestion & Chunking Pipeline
- [ ] **Step 2**: Embeddings & Vector Store (Qdrant/Pinecone)
- [ ] **Step 3**: Hybrid Retrieval (Vector + BM25 + Re-ranking)
- [ ] **Step 4**: Agentic Router (LangGraph + Tool Selection)
- [ ] **Step 5**: Frontend (Next.js / Streamlit) & RAGAS Evaluation

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Gemini 1.5 Pro / Groq (Llama 3) |
| Framework | LangChain & LangGraph |
| Embeddings | Google `text-embedding-004` + HuggingFace `all-MiniLM-L6-v2` |
| Document Parsing | `unstructured.io` |
| Vector DB | Qdrant / Pinecone (upcoming) |
| Evaluation | RAGAS (upcoming) |
| Frontend | Next.js / Streamlit (upcoming) |

## License

MIT
