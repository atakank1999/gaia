import re
from typing import Optional
import numpy as np
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from src.llm_provider import get_llm


class SpreadsheetTool:
    def __init__(self) -> None:
        self.llm = get_llm()

    def _get_spreadsheet(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Reads a spreadsheet file and returns its content as a DataFrame.

        Args:
            file_path (str): The path to the spreadsheet file.
        Returns:
            pd.DataFrame: The content of the spreadsheet.
        Raises:
            ValueError: If the file format is unsupported or if there is an error reading the file
        """
        ext = file_path.split(".")[-1].lower()
        try:
            if ext in ["xls", "xlsx"]:
                df = pd.read_excel(file_path)
            elif ext in ["csv"]:
                df = pd.read_csv(file_path)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
        except Exception as e:
            print(f"Error reading spreadsheet: {e}")
            return None
        return df

    def _extract_metadata(self, df: pd.DataFrame) -> str:
        """
        Extracts metadata from the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to extract metadata from.

        Returns:
            str: A string containing the metadata.
        """
        metadata = {
            "columns": df.columns.tolist(),
            "shape": df.shape,
            "dtypes": df.dtypes.to_dict(),
            "sample_data": df.head().to_dict("records"),
        }
        return str(metadata)

    def query_spreadsheet(self, file_path: str, query: str) -> Optional[str]:

        df = self._get_spreadsheet(file_path)
        if df is None:
            return "Error: Could not load spreadsheet file"

        metadata = self._extract_metadata(df)

        code_prompt = PromptTemplate.from_template(
            template="""
You are a software engineer who is an expert in data analysis and manipulation.
You are provided with a spreadsheet metadata and a query to answer based on the data in that file.

Metadata: {metadata}
Query: {query}
File Path: {file_path}

Your task is to generate code to answer the following query based on the provided spreadsheet data.
Generate Python code that uses pandas to answer the query. Provide only the code, no explanations.
Your code must follow these rules:

- Always filter the DataFrame first, then select the specific column(s) needed **before** applying aggregations.
- Never call aggregation functions (mean, sum, min, max, etc.) on the entire DataFrame if it contains non-numeric columns.
- Always use bracket notation for column access (`df["Column"]`), never dot notation (`df.Column`).
- Always assign the final computed answer to a variable named `_result`.
- The code must be executable without imports or comments, and must use the existing `df` variable.

Make sure to use the DataFrame `df` that is already loaded from the file.
Here is the rest of the program that your answer will be used which will be assigned to the code variable:
```python
import pandas as pd
import numpy as np:
df = self._get_spreadsheet(file_path)
code = self.llm.invoke(code_prompt).content
safe = {
        "df": df,
        "pd": pd,
        "np": np,
        "print": print,
        "len": len,
        "str": str,
        "int": int,
        "float": float,
        "list": list,
        "dict": dict,
        "sum": sum,
        "min": min,
        "max": max,
        "round": round
    }
exec(str(code), safe)
result = safe.get("_result", None)
return result
```
        """
        )

        formatted_prompt = code_prompt.format(
            query=query, metadata=metadata, file_path=file_path
        )

        code = self.llm.invoke(formatted_prompt).content

        safe_env = {
            "df": df,
            "pd": pd,
            "np": np,
            "print": print,
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "list": list,
            "dict": dict,
            "sum": sum,
            "min": min,
            "max": max,
            "round": round,
        }
        if "```python" in code or "```" in code:
            # Extract code between ```python and ``` or just between ```
            pattern = r"```(?:python)?\s*(.*?)\s*```"
            match = re.search(pattern, str(code), re.DOTALL)
            if match:
                code = match.group(1).strip()
        try:
            exec(str(code), safe_env)
        except Exception as e:
            return f"Error executing code: {e}"
        result = str(safe_env.get("_result", None))
        return result

    def analyze_spreadsheet(self, df: pd.DataFrame, question: str) -> str:
        """
        Analyze a spreadsheet DataFrame based on a query.

        Args:
            df (pd.DataFrame): The DataFrame to analyze.
            question (str): The question to answer.

        Returns:
            str: The result of the analysis.
        """
        if df.empty:
            return "The DataFrame is empty."

        metadata = self._extract_metadata(df)
        prompt_template = PromptTemplate.from_template(
            template="""
You are a data analyst tasked with analyzing a spreadsheet.
You are provided with the metadata of the spreadsheet and a question to answer based on the data in that file.
Metadata: {metadata}
Question: {question}
Your task is to provide helpful insights about the spreadsheet based on the provided metadata and question.
            """
        )
        prompt = prompt_template.format(metadata=metadata, question=question)
        response = self.llm.invoke(prompt)
        if response:
            return str(response.content)
        else:
            return "No response from the LLM."


spreadsheetTool = SpreadsheetTool()


@tool
def analyze_spreadsheet(df: pd.DataFrame, query: str) -> Optional[str]:
    """Analyze a spreadsheet DataFrame based on a query.
    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        query (str): The query to answer.
    Returns:
        pd.DataFrame: The result of the analysis.
    """
    return spreadsheetTool.analyze_spreadsheet(df, query)


@tool
def query_spreadsheet(file_path: str, query: str) -> Optional[str]:
    """Query a spreadsheet file based on a query.
    Args:
        file_path (str): The path to the spreadsheet file.
        query (str): The query to answer.
    Returns:
        Optional[str]: The result of the query as a string.
    """
    return spreadsheetTool.query_spreadsheet(file_path, query)


def main():
    file_path = "./files/7bd855d8-463d-4ed5-93ca-5fe35145f733.xlsx"
    query = "Query spreadsheet to find the average value in Pinebrook?"
    tool = SpreadsheetTool()
    df = tool._get_spreadsheet(file_path)
    if df is not None:
        res = tool.query_spreadsheet(file_path=file_path, query=query)
        if res is not None:
            print(res)
        else:
            print("No results found for the query.")
    else:
        print("Failed to read the spreadsheet.")
