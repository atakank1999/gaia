from prompts import system_prompt
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import AnyMessage,SystemMessage
from langgraph.graph import StateGraph, add_messages,START,END
from langgraph.prebuilt import ToolNode,tools_condition
from typing_extensions import TypedDict, Annotated
from typing import Sequence, List
from dotenv import load_dotenv
import os


class GraphState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


# Add an instruction to the system prompt to always use tool calls in the correct format
TOOL_CALL_INSTRUCTION = (
    "If you want to use a tool, always respond with a tool call, not plain text or JSON. "
    "Use the function/tool call format expected by the system."
)

# Combine the original system prompt with the tool call instruction
combined_system_prompt = f"{system_prompt}\n\n{TOOL_CALL_INSTRUCTION}"

load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")

search_tool = TavilySearch()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(wiki_client=None))
tools = [search_tool, wikipedia]
chat = ChatOllama(model="qwen2.5:7b-instruct",base_url="http://localhost:11434")
chat_with_tools = chat.bind_tools(tools)
tool_node = ToolNode(tools)

def call_model(state: GraphState):
    """
    Call the chat model with tools and return the updated state.
    """
    messages = state["messages"]
    response = chat_with_tools.invoke(messages)
    return {"messages": [response]}

def build_graph():
    """
    Build a graph using the chat model with tools.
    """
    graph = StateGraph(GraphState)
    graph.add_node(
        "agent",
        call_model,
    )
    graph.add_node(
        "tools",
        tool_node,
    )
    graph.add_edge(START, "agent")
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges(
        "agent",
        tools_condition,
    )

    return graph.compile()


# Example: How to start the graph with the improved system prompt
from langchain_core.messages import SystemMessage

# Initialize the state with the improved system prompt as the first message
initial_state = {
    "messages": [SystemMessage(content=combined_system_prompt)]
}

graph = build_graph()
# To invoke the graph, use:
# result = graph.invoke(initial_state)
