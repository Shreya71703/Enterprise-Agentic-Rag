"""
Enterprise RAG — LangGraph Agentic Router

Defines the LangGraph workflow orchestrating tool routing:
    1. Agent Node: Analyses query and calls appropriate tools
    2. Tool Execution Node: Dynamically executes search/SQL/web tools
    3. Generator Node: Composes the final contextual answer

State structure maintains conversational history.
"""

from __future__ import annotations

import logging
import re
from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from config.settings import get_settings
from src.router.tools import get_tools_list

import threading

logger = logging.getLogger(__name__)

# Thread-local storage to track step latencies thread-safely
_thread_local = threading.local()


# ---------------------------------------------------------------------------
# State Definition
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    """
    State representing the current message history and context.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# LLM Router Factory
# ---------------------------------------------------------------------------
def create_router_llm(google_api_key: str | None = None) -> BaseChatModel:
    """
    Instantiates ChatGoogleGenerativeAI or a lightweight mock/alternative.
    """
    key = google_api_key or get_settings().embedding.google_api_key
    if not key:
        logger.warning("⚠️  GOOGLE_API_KEY not found in settings. Routing agent calls will fail.")
        # Fallback dummy chat model for testing purposes
        from langchain_core.outputs import ChatResult, ChatGeneration
        from langchain_core.messages import AIMessage
        
        class DummyChatModel(BaseChatModel):
            def _generate(self, messages: Sequence[BaseMessage], stop: list[str] | None = None, **kwargs: Any) -> ChatResult:
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="API Key Missing. Here is a mocked response."))])
            def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> BaseChatModel:
                return self
            @property
            def _llm_type(self) -> str:
                return "dummy"
                
        return DummyChatModel()

    from langchain_google_genai import ChatGoogleGenerativeAI

    # Instantiate ONLY the primary model on startup to save ~3.5 seconds.
    # Other fallback models will be instantiated lazily only if a 429 occurs.
    primary_name = "gemini-2.5-flash"
    try:
        model = ChatGoogleGenerativeAI(
            model=primary_name,
            google_api_key=key,
            temperature=0.0,
            max_retries=0,  # CRITICAL: Fail fast so we don't hang
        )
    except Exception as e:
        logger.error(f"❌ Failed to initialize primary Gemini model: {e}")
        # Fallback to lite model
        primary_name = "gemini-2.0-flash-lite"
        model = ChatGoogleGenerativeAI(
            model=primary_name,
            google_api_key=key,
            temperature=0.0,
            max_retries=0,
        )

    # Set fallback model names for lazy instantiation on the primary model
    model._fallback_model_names = ["gemini-2.0-flash-lite", "gemini-2.0-flash"]  # type: ignore[attr-defined]
    logger.info(f"✅ Primary router LLM initialized: {primary_name} (fallbacks deferred)")
    return model


# ---------------------------------------------------------------------------
# Content Extraction Helper
# ---------------------------------------------------------------------------
def _extract_text_content(content: Any) -> str:
    """
    Extract clean text from Gemini's response content.
    
    Gemini 2.5+ may return structured content blocks like:
        [{'type': 'text', 'text': '...', 'extras': {...}}]
    instead of a plain string. This normalizes all formats to plain text.
    """
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if "text" in block:
                    text_parts.append(block["text"])
                elif "content" in block:
                    text_parts.append(str(block["content"]))
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(text_parts) if text_parts else str(content)
    
    return str(content)


# ---------------------------------------------------------------------------
# Agentic Router Class
# ---------------------------------------------------------------------------
class AgenticRouter:
    """
    LangGraph-based conversational router agent.

    Usage:
        router = AgenticRouter()
        response = router.query("What is the revenue for Cortex NLP Engine?")
    """

    def __init__(self, llm: BaseChatModel | None = None):
        self.llm = llm or create_router_llm()
        self.tools = get_tools_list()

        # Bind tools to the LLM
        self.bound_llm = self.llm.bind_tools(self.tools)

        # Build the graph
        self.graph = self._build_graph()

        # Instantiate Semantic Cache
        from src.router.semantic_cache import SemanticCacheService
        self.cache = SemanticCacheService()

    def _build_graph(self) -> StateGraph:
        """Compile nodes and conditional edges into a LangGraph runner."""
        workflow = StateGraph(AgentState)

        # Define nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self._tools_node)  # Use timed custom tools node

        # Connect paths
        workflow.add_edge(START, "agent")
        
        # Add conditional path after agent call
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END,
            }
        )

        # Connect tool execution back to agent loop
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    # -------------------------------------------------------------------
    # Node logic
    # -------------------------------------------------------------------
    def _agent_node(self, state: AgentState) -> dict[str, Any]:
        """Node executing the primary LLM call with latency measurement."""
        import time
        logger.info("🤖 Routing agent node triggered")
        t0 = time.time()
        response = self.bound_llm.invoke(state["messages"])
        t1 = time.time()

        # Identify query type based on message history
        step_name = "LLM Synthesis" if len(state["messages"]) > 3 else "LLM Agentic Router"
        
        if not hasattr(_thread_local, "steps"):
            _thread_local.steps = []
        _thread_local.steps.append({
            "name": step_name,
            "latency_ms": int((t1 - t0) * 1000)
        })

        return {"messages": [response]}

    def _tools_node(self, state: AgentState) -> dict[str, Any]:
        """Custom tools execution node to measure performance latency."""
        import time
        logger.info("⚙️ Timed tools node triggered")
        t0 = time.time()
        tool_node = ToolNode(self.tools)
        res = tool_node.invoke(state)
        t1 = time.time()

        if not hasattr(_thread_local, "steps"):
            _thread_local.steps = []
        _thread_local.steps.append({
            "name": "Datastores & Tools Execution",
            "latency_ms": int((t1 - t0) * 1000)
        })

        return res

    @staticmethod
    def _should_continue(state: AgentState) -> str:
        """Conditional routing edge checking if LLM requested tools."""
        last_message = state["messages"][-1]
        
        # If the last message contains tool calls, continue to tools node
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            logger.info(f"   ↪ Routing agent requested tools: {[t['name'] for t in last_message.tool_calls]}")
            return "continue"
        
        logger.info("   ↪ Routing agent finished (final answer generated)")
        return "end"

    # -------------------------------------------------------------------
    # Public Execution API
    # -------------------------------------------------------------------
    def query(
        self,
        user_query: str,
        history: list[BaseMessage] | None = None,
        return_trace: bool = False
    ) -> str | tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Run the agent graph for a single user query.
        Automatically falls back to alternate Gemini models on rate limits.
        """
        import time
        from langchain_core.messages import HumanMessage, SystemMessage

        # Reset thread-local latency steps for this specific query run
        _thread_local.steps = []

        # ─── 1. Semantic Cache Lookup (Only run on single queries to prevent dialogue context leaks)
        query_emb = None
        if not history:
            from src.router import tools as _tools_mod
            retriever = _tools_mod._retriever
            if retriever:
                try:
                    query_emb = retriever.embedding_service.embed_query(user_query)
                    cached_response = self.cache.lookup(user_query, query_emb)
                    if cached_response is not None:
                        # Construct simulated trace & cache hit latency stats
                        trace = [
                            {"type": "call", "tool": "semantic_cache", "args": {"query": user_query}},
                            {"type": "result", "tool": "semantic_cache", "result": "⚡ Semantic Cache hit: retrieved answer instantly"}
                        ]
                        latencies = [{"name": "⚡ Semantic Cache Hit", "latency_ms": 5}]
                        
                        if return_trace:
                            return cached_response, trace, latencies
                        return cached_response
                except Exception as cache_err:
                    logger.warning(f"⚠️ Semantic Cache lookup failed: {cache_err}")

        system_prompt = (
            "You are an Advanced Enterprise RAG System Agent.\n"
            "You have access to structured product metrics tables (SQLite), "
            "unstructured knowledge bases (Annual Reports & FAQs), and a fallback Web Search.\n\n"
            "ROUTING RULES:\n"
            "- For greetings (hi, hello, hey, etc.), respond directly without calling any tool.\n"
            "- For general knowledge questions (e.g. 'what is RAG?', 'explain transformers'), "
            "answer from your own knowledge WITHOUT using any tool.\n"
            "- For questions about NovaTech, Cortex, company reports, or product details, "
            "use the search_knowledge_base tool.\n"
            "- For questions needing numbers, sums, categories, comparisons, or counts "
            "about product metrics, use the query_product_metrics tool.\n"
            "- ONLY use web_search for live/recent information that you cannot answer yourself "
            "(e.g., today's stock price, breaking news).\n\n"
            "Compile the data collected from your tools into "
            "a helpful, comprehensive final answer."
        )

        messages = [SystemMessage(content=system_prompt)]
        if history:
            messages.extend(history)
        messages.append(HumanMessage(content=user_query))

        # Try each model on rate limit failures
        fallback_names = getattr(self.llm, '_fallback_model_names', [])
        primary_name = getattr(self.llm, 'model_name', 'gemini-2.5-flash')
        
        # Build models sequence: (model_name, model_instance)
        models_to_try = [(primary_name, self.llm)]
        for name in fallback_names:
            if name != primary_name:
                models_to_try.append((name, None))

        last_error = None
        from config.settings import get_settings
        key = get_settings().embedding.google_api_key

        for model_name, model_instance in models_to_try:
            # Swap or lazily instantiate model
            if model_instance is None:
                logger.info(f"⏳ Lazily instantiating fallback model: {model_name}...")
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    model_instance = ChatGoogleGenerativeAI(
                        model=model_name,
                        google_api_key=key,
                        temperature=0.0,
                        max_retries=0,
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Failed to instantiate lazy model {model_name}: {e}")
                    continue

            try:
                # Rebuild graph with this model
                bound = model_instance.bind_tools(self.tools)
                
                # Temporarily swap the bound_llm for the graph invocation
                original_bound = self.bound_llm
                self.bound_llm = bound
                
                logger.info(f"🚀 Trying model '{model_name}' for query: '{user_query}'")
                output_state = self.graph.invoke({"messages": messages})

                # Restore and return
                self.bound_llm = original_bound
                final_msg = output_state["messages"][-1]
                response_content = _extract_text_content(final_msg.content)

                if return_trace:
                    # Construct routing trace list from state messages
                    trace = []
                    for msg in output_state["messages"]:
                        # Tool calls made by AI
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                trace.append({
                                    "type": "call",
                                    "tool": tc.get("name"),
                                    "args": tc.get("args")
                                })
                        # Tool execution results
                        elif msg.__class__.__name__ == "ToolMessage" or getattr(msg, "type", None) == "tool":
                            raw_content = str(msg.content)
                            content_preview = raw_content[:200] + "..." if len(raw_content) > 200 else raw_content
                            trace.append({
                                "type": "result",
                                "tool": getattr(msg, "name", "unknown"),
                                "result": content_preview
                            })
                    
                    # Gather timed steps from thread-local storage
                    latencies = getattr(_thread_local, "steps", [])
                    
                    # Store in semantic cache for future queries
                    if query_emb is not None:
                        try:
                            self.cache.store(user_query, response_content, query_emb)
                        except Exception as cache_store_err:
                            logger.warning(f"Failed to cache query: {cache_store_err}")
                            
                    return response_content, trace, latencies

                # Store in semantic cache for future queries
                if query_emb is not None:
                    try:
                        self.cache.store(user_query, response_content, query_emb)
                    except Exception as cache_store_err:
                        logger.warning(f"Failed to cache query: {cache_store_err}")

                return response_content

            except Exception as e:
                err_str = str(e)
                self.bound_llm = original_bound  # Restore on error
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    logger.warning(f"⚠️  Rate limit on {model_name}, trying next model...")
                    last_error = e
                    time.sleep(2)  # Brief pause before trying next model
                    continue
                else:
                    raise

        # All models exhausted — raise the last rate limit error
        raise last_error or RuntimeError("All models rate-limited")
