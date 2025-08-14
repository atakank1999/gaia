from langchain_core.tools import tool
import os
from typing import List, Literal, Optional, Sequence
from dotenv import load_dotenv
from sympy import arg
from tavily import TavilyClient
from langchain_community.document_loaders import WikipediaLoader
from langchain_tavily import TavilySearch

from src.llm_provider import get_llm
load_dotenv()
class WebSearchTool:
    def __init__(self):
        self.llm = get_llm()
        self.tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    def _format_search_results(self, results: List[dict]) -> str:
        if results:
            titles = [result.get("title", "No title") for result in results]
            contents = [result.get("content", "No content") for result in results]
            formatted_results = "\n\n".join(
                f"Title: {title}\nContent: {content}" for title, content in zip(titles, contents)
            )
            return formatted_results
        return "No results found."

    def search_academic(self, query: str):
        """
        Search academic databases for the given query and return the summary.
        
        Args:
            query (str): The search query.
        
        Returns:
            str: Summary of the academic search results.
        """
        include_domains = ["arxiv.org", "researchgate.net", "jstor.org"]
        exclude_domains = ["wikipedia.org"]

        response = self.tavily_client.search(
            query=query,
            max_results=5,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            search_depth="advanced",
            chunks_per_source=3,
        )
        return self._format_search_results(response.get("results", []))

    def search_wikipedia(self, query: str) -> str:
        """
        Search Wikipedia for the given query and return the summary.
        
        Args:
            query (str): The search query.
        
        Returns:
            str: Summary of the Wikipedia page.
        """
        loader = WikipediaLoader(query=query, load_max_docs=1, doc_content_chars_max=50000)
        documents = loader.load()
        if documents:
            return "\n\n".join(f"Document {i}:\n{doc.page_content}" for i, doc in enumerate(documents))
        else:
            return "No results found on Wikipedia."

    def news_search(self, query: str, start_date: str, end_date: str) -> str:
        """
        Search news articles for the given query and return the summary.
        
        Args:
            query (str): The search query.
        
        Returns:
            str: Summary of the news articles.
        """
        response = self.tavily_client.search(
            query=query,
            max_results=5,
            topic="news",
            start_date=start_date,
            end_date=end_date,
        )
        return self._format_search_results(response.get("results", []))

my_tool = WebSearchTool()

@tool
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia for comprehensive information on any topic. Returns detailed page content including summaries, facts, and relevant information from Wikipedia articles.
    
    Args:
        query (str): The search query - can be a person, place, concept, or any topic
    
    Returns:
        str: Comprehensive summary of the Wikipedia page content
    """
    return my_tool.search_wikipedia(query)

@tool
def academic_search(query: str) -> str:
    """
    Search academic databases (arxiv, researchgate, and jstor) for scholarly research papers and publications on a specific topic.

    Args:
        query (str): The academic search query - can be research topics, paper titles, authors, or scientific concepts

    Returns:
        str: Summary of academic search results including titles, abstracts, and key findings from scholarly sources
    """
    return my_tool.search_academic(query)
@tool
def news_search(query: str, start_date: str, end_date: str) -> str:
    """
    Search for recent news articles within a specific date range to get current information and developments on any topic.

    Args:
        query (str): The news search query - topics, events, companies, or current affairs
        start_date (str): Start date for the news search in YYYY-MM-DD format (e.g., "2023-01-01")
        end_date (str): End date for the news search in YYYY-MM-DD format (e.g., "2023-12-31")

    Returns:
        str: Summary of relevant news articles including headlines, publication dates, and key information from the specified time period
    """
    return my_tool.news_search(query, start_date, end_date)

def main():
    """
    Main function to demonstrate the usage of the web search tool.
    """
    print("Schema:", news_search.name)
    print("Description:", news_search.description)
    print("Input Schema:", news_search.args)

    # Example usage
    res = news_search.invoke({
        "query": "openai",
        "start_date": "2023-01-01",
        "end_date": "2025-07-01"
    })
    print(res)


