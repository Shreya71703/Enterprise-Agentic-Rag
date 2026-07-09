"""
Enterprise RAG — Streamlit Web UI

Sleek, premium frontend demonstrating the capabilities of the Enterprise RAG System:
    - Conversational chat interface with Agentic routing
    - Live tool execution tracing (SQL execution, document search, web search)
    - Side-by-side search strategy comparison debugger (Vector vs BM25 vs Hybrid)
    - Schema explorer for structured product database tables
"""

from __future__ import annotations

import sys
from pathlib import Path

# ─── Lightweight imports only (Streamlit loads in ~2s, everything else deferred)
import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Page Layout Configuration (renders IMMEDIATELY, no heavy imports needed)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Enterprise RAG System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Premium Styling (HSL Tailored Theme overrides)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    /* Dark glassmorphism base */
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #0f172a 50%, #0b0f19 100%);
        color: #f1f5f9;
        font-family: 'Outfit', sans-serif;
    }
    .stSidebar {
        background-color: #0e1626;
        border-right: 1px solid #1e293b;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #6366f1 !important;
        font-family: 'Outfit', sans-serif;
    }
    /* Card borders */
    div[data-testid="stMetricValue"] {
        color: #38bdf8 !important;
    }
    /* Chat bubbles text and background fixes */
    div[data-testid="stChatMessage"] {
        background-color: #1e293b !important;
        border: 1px solid #2d3748 !important;
        border-radius: 10px !important;
        margin-bottom: 10px !important;
    }
    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] span,
    div[data-testid="stChatMessage"] li,
    div[data-testid="stChatMessage"] code,
    div[data-testid="stChatMessage"] div {
        color: #f1f5f9 !important;
    }
    /* Input field text color correction */
    div[data-testid="stChatInput"] textarea {
        color: #f1f5f9 !important;
        background-color: #1a202c !important;
    }
    /* Buttons customization */
    .stButton>button {
        background-color: #6366f1;
        color: white;
        border-radius: 6px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #4f46e5;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }
    /* Loading animation */
    @keyframes pulse-glow {
        0%, 100% { opacity: 0.6; }
        50% { opacity: 1; }
    }
    .loading-text {
        animation: pulse-glow 1.5s ease-in-out infinite;
        color: #6366f1;
        font-size: 1.1rem;
        text-align: center;
        padding: 2rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Lazy Backend Initialization (cached — only runs ONCE ever)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _init_backend():
    """
    Lazily import and initialize all heavy backend services.
    This function is cached by Streamlit so it only executes once
    across all sessions and page reruns.
    """
    from src.router.tools import bootstrap_tools, _retriever, _sql_db
    from src.router.agent import AgenticRouter

    bootstrap_tools()

    # Re-import after bootstrap to get the initialized globals
    from src.router import tools as _tools_mod
    retriever = _tools_mod._retriever
    sql_db = _tools_mod._sql_db

    router = AgenticRouter()
    return router, retriever, sql_db


def _get_backend():
    """Get or initialize backend — shows a branded loading screen on first call."""
    if "backend_ready" not in st.session_state:
        # Show a loading placeholder while backend initializes
        loading = st.empty()
        loading.markdown(
            '<div class="loading-text">🧠 Initializing Enterprise RAG Engine...</div>',
            unsafe_allow_html=True,
        )
        result = _init_backend()
        loading.empty()
        st.session_state["backend_ready"] = True
        return result
    return _init_backend()


# ---------------------------------------------------------------------------
# Message State (lightweight, no heavy imports)
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


# ---------------------------------------------------------------------------
# Sidebar Content (renders immediately with placeholder stats)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/nolan/128/artificial-intelligence.png", width=70)
    st.title("Enterprise RAG")
    st.caption("Production-Grade Retrieval-Augmented Generation")
    st.markdown("---")

    # DB Stats (only show after backend is ready)
    st.subheader("📊 Database Statistics")
    if "backend_ready" in st.session_state:
        _, retriever, sql_db = _init_backend()
        if retriever and retriever.vector_store.collection_exists():
            try:
                info = retriever.vector_store.get_collection_info()
                st.metric("Vector Points", info["points_count"])
                st.metric("Vector Dimensions", info["dimension"])
                st.caption(f"Qdrant collection: `{info['name']}`")
            except Exception:
                st.warning("Failed to load Qdrant statistics.")
        else:
            st.info("Qdrant collection loading...")
    else:
        st.info("⏳ Backend initializing...")

    st.markdown("---")

    # SQL Schema Explorer
    st.subheader("💾 SQL Schema Explorer")
    if "backend_ready" in st.session_state:
        _, _, sql_db = _init_backend()
        if sql_db:
            with st.expander("Show `product_metrics` Table"):
                st.code(sql_db.get_schema(), language="sql")
        else:
            st.caption("SQL DB not loaded.")
    else:
        st.caption("⏳ Waiting for backend...")

    st.markdown("---")

    # Reload / Reset
    if st.sidebar.button("🗑️ Reset Chat History"):
        st.session_state["messages"] = []
        st.session_state["chat_history"] = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main Layout Tabs
# ---------------------------------------------------------------------------
st.title("🧠 Advanced Agentic RAG Platform")
st.markdown("Enterprise AI Engineer portfolio project utilizing LangGraph, Qdrant, SQLite, and Hybrid Cross-Encoder Search.")

tab_chat, tab_debug = st.tabs(["💬 RAG Chatbot Agent", "🔍 Search Strategy Debugger"])


# ---------------------------------------------------------------------------
# Tab 1: Agentic Chatbot
# ---------------------------------------------------------------------------
with tab_chat:
    st.markdown("### Conversational Routing Interface")
    st.write("Ask anything about NovaTech annual reports, product metrics, churn rate, or fallback web queries:")

    # Render previous messages
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
    if prompt := st.chat_input("Ask a question (e.g. 'What makes Cortex different?' or 'Which product has the highest active users?')"):
        # Display user bubble
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state["messages"].append({"role": "user", "content": prompt})

        # Lazy-import message types only when needed
        from langchain_core.messages import HumanMessage, AIMessage

        # Run routing agent
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            with st.spinner("Agentic router thinking & retrieving..."):
                try:
                    # Initialize backend on first query (instant on subsequent)
                    router, _, _ = _get_backend()
                    history = st.session_state["chat_history"]

                    # Retry logic for rate limits
                    import time as _time
                    max_retries = 3
                    response = None
                    for attempt in range(max_retries):
                        try:
                            response = router.query(prompt, history=history)
                            break
                        except Exception as retry_err:
                            err_str = str(retry_err)
                            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                                wait_secs = 30 * (attempt + 1)
                                response_placeholder.warning(
                                    f"⏳ Rate limit hit. Retrying in {wait_secs}s... (attempt {attempt + 1}/{max_retries})"
                                )
                                _time.sleep(wait_secs)
                            else:
                                raise

                    if response is None:
                        response_placeholder.error("❌ Rate limit exceeded after retries. Please wait a minute and try again.")
                    else:
                        # Update message history buffers
                        st.session_state["chat_history"].append(HumanMessage(content=prompt))
                        st.session_state["chat_history"].append(AIMessage(content=response))
                        if len(st.session_state["chat_history"]) > 10:
                            st.session_state["chat_history"] = st.session_state["chat_history"][-10:]

                        # Render response
                        response_placeholder.markdown(response)
                        st.session_state["messages"].append({"role": "assistant", "content": response})

                except Exception as e:
                    response_placeholder.error(f"Error executing agent query: {e}")


# ---------------------------------------------------------------------------
# Tab 2: Strategy Debugger (Side-by-side comparison)
# ---------------------------------------------------------------------------
with tab_debug:
    st.markdown("### Search Strategy Debugger")
    st.write("Compare different document retrieval methods side-by-side for the same search term:")

    debug_query = st.text_input("Enter search phrase to test:", "Cortex deployment")
    debug_k = st.slider("Docs to return (Top-K)", 1, 5, 3)

    if st.button("🔍 Run Comparison Search") and debug_query:
        _, retriever, _ = _get_backend()
        if retriever:
            col1, col2, col3 = st.columns(3)

            # 1. Semantic Search
            with col1:
                st.markdown("#### 📡 1. Semantic Vector Search")
                try:
                    query_emb = retriever.embedding_service.embed_query(debug_query)
                    vector_raw = retriever.vector_store.search_with_documents(query_emb, top_k=debug_k)
                    if not vector_raw:
                        st.info("No semantic results found.")
                    for idx, (doc, score) in enumerate(vector_raw, 1):
                        with st.expander(f"Result #{idx} (Score: {score:.3f})"):
                            st.write(f"**Source**: `{doc.metadata.get('source_file')}`")
                            st.caption(doc.page_content)
                except Exception as e:
                    st.error(e)

            # 2. BM25 Keyword Search
            with col2:
                st.markdown("#### ⌨️ 2. BM25 Keyword Search")
                try:
                    bm25_raw = retriever.bm25.search(debug_query, top_k=debug_k)
                    if not bm25_raw:
                        st.info("No keyword results found.")
                    for idx, (doc, score) in enumerate(bm25_raw, 1):
                        with st.expander(f"Result #{idx} (Score: {score:.3f})"):
                            st.write(f"**Source**: `{doc.metadata.get('source_file')}`")
                            st.caption(doc.page_content)
                except Exception as e:
                    st.error(e)

            # 3. Hybrid Reranked Search
            with col3:
                st.markdown("#### 🏆 3. Hybrid + Re-ranked")
                try:
                    hybrid_raw = retriever.retrieve(debug_query, top_k=debug_k, use_reranking=True)
                    if not hybrid_raw:
                        st.info("No hybrid results found.")
                    for idx, (doc, score) in enumerate(hybrid_raw, 1):
                        with st.expander(f"Result #{idx} (Score: {score:.3f})"):
                            st.write(f"**Source**: `{doc.metadata.get('source_file')}`")
                            st.caption(doc.page_content)
                except Exception as e:
                    st.error(e)
        else:
            st.error("Retriever service uninitialized.")
