from typing import Dict, List, Literal, Optional, TypedDict
from pydantic import BaseModel
from typing_extensions import Annotated
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    BaseMessage,
    ToolMessage,
    AIMessage,
)
from langgraph.graph import StateGraph, add_messages, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import PromptTemplate
from src.llm_provider import get_llm
from src.tools.audioTool import analyze_audio
from src.tools.spreadsheetTool import analyze_spreadsheet, query_spreadsheet
from src.tools.fetchFile import fetch_file
from src.tools.webSearchTool import news_search, academic_search, wikipedia_search
from src.tools.youtubeTranscriptTool import youtube_transcript_tool


def append_reducer(left: List[str], right: str) -> List[str]:
    return left + [right]


class GraphState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    tool_calls: Annotated[List[str], append_reducer]
    question: str
    task_id: str
    file_name: Optional[str]
    file_location: Optional[str]


class Nodes:
    """
    A class to define various nodes for the state graph.
    Each method corresponds to a specific tool or action in the graph.
    """

    def __init__(self):
        self.llm = get_llm()

    def chatbot(self, state: GraphState):
        res = self.llm.invoke(str(state))
        return {"messages": [res]}


class EdgeConditions:
    def file_fetch_condition(self, state: GraphState) -> str:
        """
        Checks if the state is in a condition to fetch a file.
        """
        return "file_fetcher" if state.get("file_name") is not None else "chatbot"

    def file_tool_decider(self, state: GraphState) -> str:
        """
        Orchestrates the flow based on the presence of file-related tools.
        """
        file_location = state.get("file_location")
        if file_location is not None:
            suffix = file_location.split(".")[-1].lower()
            if suffix in ["mp3", "wav", "flac"]:
                return "analyze_audio"
            elif suffix in ["xlsx", "csv"]:
                return "handle_spreadsheet"
        return "chatbot"

    def spreadsheet_tool_decider(self, state: GraphState) -> str:
        """
        Decides which spreadsheet tool to use based on the state.
        """
        last_message = state.get("messages")[-1]
        if isinstance(last_message, AIMessage):
            if last_message.tool_calls:
                if last_message.tool_calls[0]["name"] == "query_spreadsheet":
                    return "query_spreadsheet"
                else:
                    return "analyze_spreadsheet"
        return "chatbot"


class ToolNodes:
    def __init__(self):
        self.llm = get_llm()

    def file_fetcher_tool(self, state: GraphState):
        """
        Fetch the file based on the task_id and file_name in the state.
        """
        task_id = state.get("task_id")
        file_name = state.get("file_name")
        if task_id and file_name:
            file_location = fetch_file(task_id, file_name)
            # if file location is returned
            if len(file_location) > 0:
                message = ToolMessage(
                    content=f"File fetched and saved to {file_location}",
                    tool_call_id=len(state["tool_calls"]),
                )
                return {
                    "tool_calls": "fetch_file",
                    "file_location": file_location,
                    "messages": [message],
                }
            # if file location is empty string it means there was an error during fetching
            else:
                message = ToolMessage(
                    content="Error fetching file", tool_call_id=len(state["tool_calls"])
                )
                return {"tool_calls": "fetch_file", "messages": [message]}

        else:  # If no task_id or file_name is provided
            message = ToolMessage(
                content="No task_id or file_name provided in the state.",
                tool_call_id=len(state["tool_calls"]),
            )
            return {"tool_calls": "fetch_file", "messages": [message]}

    def analyze_audio(self, state: GraphState):
        """
        Analyzes the audio file at the given location.
        """
        file_location = state.get("file_location")
        analysis_result = analyze_audio.invoke(
            {"audio_path": file_location, "question": state.get("question")}
        )
        msg = ToolMessage(
            content=analysis_result, tool_call_id=len(state["tool_calls"])
        )
        return {"tool_calls": "analyze_audio", "messages": [msg]}

    def handle_spreadsheet(self, state: GraphState):
        """
        Analyzes the spreadsheet file at the given location.
        """
        tools = [analyze_spreadsheet, query_spreadsheet]
        llm_with_spreadsheet_tools = self.llm.bind_tools(tools)
        return {"messages": [llm_with_spreadsheet_tools.invoke(str(state))]}

    def query_spreadsheet_tool(self, state: GraphState):
        """
        Queries the spreadsheet file at the given location.
        """
        file_location = state.get("file_location")
        query_result = query_spreadsheet.invoke(
            {"file_path": file_location, "question": state.get("question")}
        )
        msg = ToolMessage(content=query_result, tool_call_id=len(state["tool_calls"]))
        return {"tool_calls": "query_spreadsheet", "messages": [msg]}

    def analyze_spreadsheet_tool(self, state: GraphState):
        """
        Analyzes the spreadsheet file at the given location.
        """
        file_location = state.get("file_location")
        analysis_result = analyze_spreadsheet.invoke(
            {"file_path": file_location, "question": state.get("question")}
        )
        msg = ToolMessage(
            content=analysis_result, tool_call_id=len(state["tool_calls"])
        )
        return {"tool_calls": "analyze_spreadsheet", "messages": [msg]}


nodes = Nodes()
tool_nodes = ToolNodes()
edge_conditions = EdgeConditions()

graph_builder = StateGraph(GraphState)
graph_builder.add_node(
    "chatbot",
    nodes.chatbot,
)
graph_builder.add_node(
    "file_fetcher",
    tool_nodes.file_fetcher_tool,
)

graph_builder.add_node(
    "analyze_audio",
    tool_nodes.analyze_audio,
)
graph_builder.add_node(
    "handle_spreadsheet",
    tool_nodes.handle_spreadsheet,
)
graph_builder.add_node(
    "analyze_spreadsheet",
    tool_nodes.analyze_spreadsheet_tool,
)
graph_builder.add_node(
    "query_spreadsheet",
    tool_nodes.query_spreadsheet_tool,
)

graph_builder.add_conditional_edges(
    START,
    edge_conditions.file_fetch_condition,
    {"file_fetcher": "file_fetcher", "chatbot": "chatbot"},
)

graph_builder.add_conditional_edges(
    "file_fetcher",
    edge_conditions.file_tool_decider,
    {
        "analyze_audio": "analyze_audio",
        "handle_spreadsheet": "handle_spreadsheet",
        "chatbot": "chatbot",
    },
)

graph_builder.add_conditional_edges(
    "handle_spreadsheet",
    edge_conditions.spreadsheet_tool_decider,
    {
        "query_spreadsheet": "query_spreadsheet",
        "analyze_spreadsheet": "analyze_spreadsheet",
        "chatbot": "chatbot",
    },
)
graph_builder.add_edge("analyze_audio", "chatbot")
graph_builder.add_edge("analyze_spreadsheet", "chatbot")
graph_builder.add_edge("query_spreadsheet", "chatbot")

graph_builder.add_edge(
    "chatbot",
    END,
)
workflow = graph_builder.compile()


def main():
    import json

    with open("questions.json", "r") as f:
        questions = f.read()
        questions = json.loads(questions)
    initial_state = GraphState(
        {
            "messages": [HumanMessage(content=questions[18]["question"])],
            "tool_calls": [],
            "question": questions[18]["question"],
            "task_id": questions[18]["task_id"],
            "file_name": questions[18]["file_name"],
            "file_location": None,
        }
    )
    result = workflow.invoke(initial_state)
    print(result)  # This will print the final state after processing the graph


if __name__ == "__main__":
    # Example of how to invoke the graph
    main()
