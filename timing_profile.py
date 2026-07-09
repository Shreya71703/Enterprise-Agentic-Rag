import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.router.tools import bootstrap_tools
bootstrap_tools()

from langchain_google_genai import ChatGoogleGenerativeAI
from src.router.agent import create_router_llm, get_tools_list
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from src.router.agent import AgentState

print("Starting step-by-step timing of AgenticRouter...")

t0 = time.time()
llm = create_router_llm()
print(f"create_router_llm completed in {time.time() - t0:.4f}s")

t1 = time.time()
tools = get_tools_list()
print(f"get_tools_list completed in {time.time() - t1:.4f}s")

t2 = time.time()
bound_llm = llm.bind_tools(tools)
print(f"llm.bind_tools completed in {time.time() - t2:.4f}s")

t3 = time.time()
workflow = StateGraph(AgentState)
print(f"StateGraph instantiation completed in {time.time() - t3:.4f}s")

t4 = time.time()
workflow.add_node("agent", lambda state: state)
print(f"add_node(agent) completed in {time.time() - t4:.4f}s")

t5 = time.time()
tool_node = ToolNode(tools)
workflow.add_node("tools", tool_node)
print(f"ToolNode and add_node(tools) completed in {time.time() - t5:.4f}s")

t6 = time.time()
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", lambda state: "end", {"continue": "tools", "end": END})
print(f"edges added in {time.time() - t6:.4f}s")

t7 = time.time()
graph = workflow.compile()
print(f"workflow.compile completed in {time.time() - t7:.4f}s")
