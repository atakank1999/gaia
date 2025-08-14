import os
from typing import Dict, List, Literal, Optional, TypedDict, cast
from langchain_tavily import TavilySearch
from pydantic import BaseModel
from typing_extensions import Annotated
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    ToolMessage,
    AIMessage,
)
from langgraph.graph import StateGraph, add_messages, START, END
from langchain_core.prompts import PromptTemplate
from src.llm_provider import get_llm
from src.tools.audioTool import analyze_audio
from src.tools.spreadsheetTool import analyze_spreadsheet, query_spreadsheet
from src.tools.fetchFile import fetch_file
from src.tools.webSearchTool import news_search, academic_search, wikipedia_search
from src.tools.youtubeTranscriptTool import youtube_transcript_tool
from prompts import system_prompt
from langchain_core.prompts import ChatPromptTemplate


def append_reducer(left: List[str], right: str) -> List[str]:
    return left + [right]


class GraphState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    tool_calls: Annotated[List[str], append_reducer]
    question: str
    task_id: str
    file_name: Optional[str]
    file_location: Optional[str]
    max_tries: int
    retries: int


class Nodes:
    """
    A class to define various nodes for the state graph.
    Each method corresponds to a specific tool or action in the graph.
    """

    def __init__(self):
        self.llm = get_llm()

    def entry(self, state: GraphState):
        return

    def generate_answer(self, state: GraphState):
        """
        Generate an answer based on the question in the state and the context.
        """
        prompt_template = PromptTemplate.from_template(
"""
You will be given two inputs question and the state of the conversation as context. To answer the question you need to ground your answer based on the context provided. Keep your answer concise and relevant. If you find the context insufficient, you should ask for clarification.

##Inputs
- QUESTION: {question}
- CONTEXT: {context}
"""
)

        prompt = prompt_template.format(
            question=state["question"],
            context=state["messages"]
        )

        res = self.llm.invoke(prompt)

        return {"messages": [res]}

    def assess_response(self, state: GraphState):
        """
        Assess the response from the LLM and decide whether to continue or end the conversation.
        """
        if state["retries"] >= state["max_tries"]:
            return {"messages": [AIMessage(content="continue")]}
        last_message = state["messages"][-1]
        prompt_template = PromptTemplate.from_template(
            """
You are the Response Assessor in an AI Agent workflow. Your job is to judge the AGENT_RESPONSE below and decide the next action. If you find the response answers the user question, you should return "continue". If not, you should return "retry".Evaluate the response based on the correctness, grounding, relevance and completeness of the response. You will be given the context, agent response and the initial user message question as inputs.
##Inputs
- AGENT_RESPONSE: {last_message}
- USER_MESSAGE: {question}
- CONTEXT: {context}
            """
        )
        prompt = prompt_template.format(
            last_message=last_message,
            question=state["question"],
            context=state["messages"][:-1],
        )

        class Response(BaseModel):
            answer: Literal["continue", "retry"]

        structured_llm = self.llm.with_structured_output(Response)
        response = cast(Response, structured_llm.invoke(prompt))
        if response.answer == "continue":
            for root, dirs, files in os.walk("./tmp"):
                for file in files:
                    os.remove(os.path.join(root, file))
            return {"messages": [AIMessage(content="continue")]}
        else:
            return {"messages": [AIMessage(content="retry")]}

    def youtube_transcript(self, state: GraphState):
        """
        Fetch the YouTube transcript based on the question in the state.
        """
        question = state.get("question")
        res = youtube_transcript_tool.invoke({"question": question})
        if res is None:
            return {
                "messages": [
                    AIMessage(content="No results found for YouTube transcript.")
                ]
            }
        return {"messages": [AIMessage(content=res)]}

    def news_search(self, state: GraphState):
        """
        Perform a news search based on the question in the state.
        """
        question = state.get("question")
        if not question:
            return {
                "messages": [AIMessage(content="No question provided for news search.")]
            }

        res = news_search.invoke({"question": question})
        return {"messages": [AIMessage(content=res)]}

    def academic_search(self, state: GraphState):
        """
        Perform an academic search based on the question in the state.
        """
        question = state.get("question")
        if not question:
            return {
                "messages": [
                    AIMessage(content="No question provided for academic search.")
                ]
            }
        prompt = PromptTemplate.from_template(
            """
Your job is to find the optimal query to search in academic sources that answers the user's question. Since querying the whole questions might lead to irrelevant results, you should focus on the most important keywords. Do not forget your answer should only include the query no unnecessary context.
##Inputs
- USER_QUESTION: {question}
            """
        )
        prompt = prompt.format(question=question)
        query = self.llm.invoke(prompt)
        docs = wikipedia_search.invoke({"query": str(query.content)})
        if docs:
            prompt = PromptTemplate.from_template(
                """
You will be given a user question and a list of documents retrieved from Academic resources. Your task is to extract the most relevant information from these documents to answer the user's question. Keep your answer concise and focused on the user's query. Don't output unnecessary context just the parts from the documents.
##Inputs
- USER_QUESTION: {query}
- DOCUMENTS RETRIEVED: {docs}
                """
            )
            prompt = prompt.format(query=query, docs=docs)
            res = self.llm.invoke(prompt)
            return {"messages": [res]}

        res = academic_search.invoke({"question": question})
        return {"messages": [AIMessage(content=res)]}

    def wikipedia_search(self, state: GraphState):
        """
        Perform a Wikipedia search based on the question in the state.
        """
        question = state.get("question")
        if not question:
            return {
                "messages": [
                    AIMessage(content="No question provided for Wikipedia search.")
                ]
            }
        prompt = PromptTemplate.from_template(
            """
Your job is to find the optimal query to search in wikipedia that answers the user's question. Since querying the whole questions might lead to irrelevant results, you should focus on the most important keywords. Do not forget your answer should only include the query no unnecessary context.
##Inputs
- USER_QUESTION: {question}
            """
        )
        prompt = prompt.format(question=question)
        query = self.llm.invoke(prompt)
        docs = wikipedia_search.invoke({"query": str(query.content)})
        if docs:
            prompt = PromptTemplate.from_template(
                """
You will be given a user question and a list of documents retrieved from Wikipedia. Your task is to extract the most relevant information from these documents to answer the user's question. Keep your answer concise and focused on the user's query. Don't output unnecessary context just the parts from the documents.
##Inputs
- USER_QUESTION: {query}
- DOCUMENTS RETRIEVED: {docs}
                """
            )
            prompt = prompt.format(query=query, docs=docs)
            res = self.llm.invoke(prompt)
            return {"messages": [res]}

    def web_search(self, state: GraphState):
        """
        Perform a web search based on the question in the state.
        """
        question = state.get("question")
        if not question:
            return {
                "messages": [AIMessage(content="No question provided for web search.")]
            }
        search = TavilySearch(search_depth="advanced")
        res = self.llm.bind_tools([search]).invoke(question)
        if res.tool_calls:
            search_results = search.invoke(res.tool_calls[0]["args"])
            formatted_search_results = "\n".join([result["content"] for result in search_results["results"]]) # type: ignore
            prompt = PromptTemplate.from_template(
                """
You are a search assistant. Based on the user's question and the search results, generate a concise summary of the most relevant information. Your summary should directly address the user's question and include key details from the search results. Keep your answer short and focused.
##Inputs
- USER_QUESTION: {question}
- SEARCH_RESULTS: {formatted_search_results}
                """
            )
            res = self.llm.invoke(prompt.format(question=question, formatted_search_results=formatted_search_results))

        return {"messages": [res]}

    def select_node(self, state: GraphState):

        routes = ["news_search", "academic_search", "wikipedia_search", "web_search", "youtube_transcript"]
        used_routes = []
        for message in state["messages"]:
            if message.content in routes:
                used_routes.append(message.content)
        if used_routes:
            routes = [route for route in routes if route not in used_routes]

        class Routes(BaseModel):
            route: Literal[*routes]

        llm = self.llm.with_structured_output(Routes)
        prompt = """
You are a router. Based on the user's question and context, determine the appropriate search route to take. The available routes are: news_search, academic_search, wikipedia_search, web_search, youtube_transcript. If a route is used before you need to choose another option.
User question: {question}
Context: {context}
"""
        formatted_prompt = prompt.format(
            question=state["question"], context=state["messages"][:-1]
        )
        response = cast(Routes, llm.invoke(formatted_prompt))
        return {"messages": [AIMessage(content=response.route)]}


class EdgeConditions:
    def __init__(self):
        self.llm = get_llm()

    def non_tool_node_condition(self, state: GraphState) -> str:
        """
        Checks if the state is in a condition to proceed without using a tool.
        """
        last_message = state.get("messages")[-1]
        return str(last_message.content)

    def file_fetch_condition(self, state: GraphState) -> str:
        """
        Checks if the state is in a condition to fetch a file.
        """
        return "file_fetcher" if state.get("file_name") else "select_node"

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
        return "select_node"

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
        return "select_node"

    def assessment_condition(self, state: GraphState) -> str:
        """
        Checks if the state is in a condition to assess the response.
        """
        last_message = state.get("messages")[-1]
        if isinstance(last_message, AIMessage) and last_message.content == "retry":
            return "entry"
        return "END"


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

        class SpreadSheetTools(BaseModel):
            tool: Literal["analyze_spreadsheet", "query_spreadsheet"]

        structured_llm = self.llm.with_structured_output(SpreadSheetTools)
        prompt = PromptTemplate.from_template(
            """
    You need to decide which tool to use for the spreadsheet: analyze_spreadsheet or query_spreadsheet.
    You can find the detailed explanations of each tool below:
    - analyze_spreadsheet: This tool is used for analyzing high-level metadata informations of the spreadsheet. An LLM is used to extract insights from the spreadsheet without directly interacting with the data.
    - query_spreadsheet: This tool is used to gather more explicit information from the spreadsheet. If you need to run python code on the spreadsheet data, use this tool. This is the only tool that lets you interact with the spreadsheet data directly.

    Make your decision based on the current context and user input.

    Message History: {messages}
"""
        )
        formatted_prompt = prompt.format(messages=state["messages"])
        response = structured_llm.invoke(formatted_prompt)
        response = cast(SpreadSheetTools, response)
        message = AIMessage(
            content=f"Choosing {response.tool}",
            tool_calls=[
                {"name": response.tool, "args": {}, "id": str(len(state["tool_calls"]))}
            ],
        )
        return {"tool_calls": "handle_spreadsheet", "messages": [message]}

    def query_spreadsheet_tool(self, state: GraphState):
        """
        Queries the spreadsheet file at the given location.
        """
        file_location = state.get("file_location")
        query_result = query_spreadsheet.invoke(
            {"file_path": file_location, "query": state.get("question")}
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
    "entry",
    nodes.entry,
)
graph_builder.add_node(
    "generate_answer",
    nodes.generate_answer
)

graph_builder.add_node(
    "assessment",
    nodes.assess_response,
)
graph_builder.add_node("youtube_transcript", nodes.youtube_transcript)
graph_builder.add_node(
    "academic_search",
    nodes.academic_search,
)
graph_builder.add_node(
    "wikipedia_search",
    nodes.wikipedia_search,
)
graph_builder.add_node(
    "web_search",
    nodes.web_search,
)
graph_builder.add_node(
    "news_search",
    nodes.news_search,
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
graph_builder.add_node(
    "select_node",
    nodes.select_node,
)

graph_builder.add_conditional_edges(
    "entry",
    edge_conditions.file_fetch_condition,
    {"file_fetcher": "file_fetcher", "select_node": "select_node"},
)

graph_builder.add_conditional_edges(
    "file_fetcher",
    edge_conditions.file_tool_decider,
    {
        "analyze_audio": "analyze_audio",
        "handle_spreadsheet": "handle_spreadsheet",
        "select_node": "select_node",
    },
)
graph_builder.add_conditional_edges(
    "handle_spreadsheet",
    edge_conditions.spreadsheet_tool_decider,
    {
        "query_spreadsheet": "query_spreadsheet",
        "analyze_spreadsheet": "analyze_spreadsheet",
        "handle_spreadsheet": "handle_spreadsheet",
    },
)
graph_builder.add_conditional_edges(
    "select_node",
    edge_conditions.non_tool_node_condition,
    {
        "news_search": "news_search",
        "academic_search": "academic_search",
        "wikipedia_search": "wikipedia_search",
        "web_search": "web_search",
        "youtube_transcript": "youtube_transcript",
    },
)
graph_builder.add_edge(START, "entry")
graph_builder.add_edge("analyze_audio", "generate_answer")
graph_builder.add_edge("analyze_spreadsheet", "generate_answer")
graph_builder.add_edge("query_spreadsheet", "generate_answer")
graph_builder.add_edge("web_search", "generate_answer")
graph_builder.add_edge("wikipedia_search", "generate_answer")
graph_builder.add_edge("youtube_transcript", "generate_answer")
graph_builder.add_edge("academic_search", "generate_answer")
graph_builder.add_edge("news_search", "generate_answer")
graph_builder.add_edge("generate_answer", "assessment")

graph_builder.add_conditional_edges(
    "assessment", edge_conditions.assessment_condition, {"entry": "entry", "END": END}
)
workflow = graph_builder.compile()


def main():
    import json

    with open("questions.json", "r") as f:
        questions = f.read()
        questions = json.loads(questions)
    initial_state = GraphState(
        {
            "messages": [HumanMessage(content=questions[6]["question"])],
            "tool_calls": [],
            "question": questions[6]["question"],
            "task_id": questions[6]["task_id"],
            "file_name": questions[6]["file_name"],
            "file_location": None,
            "max_tries": 3,
            "retries": 0,
        }
    )
    result = workflow.invoke(initial_state)
    #result = workflow.invoke(initial_state)
    for message in result["messages"]:
        print(message.pretty_print())


if __name__ == "__main__":
    # Example of how to invoke the graph
    main()
