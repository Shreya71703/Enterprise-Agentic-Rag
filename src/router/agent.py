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
from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from config.settings import get_settings
from src.router.tools import get_tools_list

logger = logging.getLogger(__name__)


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

    # Using gemini-2.0-flash as default router model for low latency and high quality
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=key,
        temperature=0.0,
    )
    return model


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

    def _build_graph(self) -> StateGraph:
        """Compile nodes and conditional edges into a LangGraph runner."""
        workflow = StateGraph(AgentState)

        # Define nodes
        workflow.add_node("agent", self._agent_node)
        
        # Tools node
        tool_node = ToolNode(self.tools)
        workflow.add_node("tools", tool_node)

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
        """Node executing the primary LLM call."""
        logger.info("🤖 Routing agent node triggered")
        response = self.bound_llm.invoke(state["messages"])
        return {"messages": [response]}

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
    def query(self, user_query: str, history: list[BaseMessage] | None = None) -> str:
        """
        Run the agent graph for a single user query.

        Args:
            user_query: Search/Question text.
            history: Message history list.

        Returns:
            Final text response from the agent.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        system_prompt = (
            "You are an Advanced Enterprise RAG System Agent.\n"
            "You have access to structured product metrics tables (SQLite), "
            "unstructured knowledge bases (Annual Reports & FAQs), and a fallback Web Search.\n"
            "Analyze the user query and select the appropriate tool. "
            "If the query requires numbers, sums, categories, comparisons, or counts, "
            "use the SQLite DB tool.\n"
            "Always state which tool you are using. Compile the data collected from your tools into "
            "a helpful, comprehensive final answer. Do not output markdown table formatting if "
            "it is not needed, but keep data clear."
        )

        messages = [SystemMessage(content=system_prompt)]
        if history:
            messages.extend(history)
        messages.append(HumanMessage(content=user_query))

        logger.info(f"🚀 Running agent router graph for query: '{user_query}'")
        output_state = self.graph.invoke({"messages": messages})

        # Return the content of the final AI message
        final_msg = output_state["messages"][-1]
        return final_msg.content
