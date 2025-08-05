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
        """
        ext = file_path.split('.')[-1].lower()
        if ext in ['xls', 'xlsx']:
            df = pd.read_excel(file_path)
        elif ext in ['csv']:
            df = pd.read_csv(file_path)
        else:
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
            'columns': df.columns.tolist(),
            'shape': df.shape,
            'dtypes': df.dtypes.to_dict(),
            'sample_data': df.head().to_dict('records')
        }
        return str(metadata)
    
    def query_spreadsheet(self, file_path: str, query: str) -> Optional[pd.DataFrame]:

        df = self._get_spreadsheet(file_path)
        if df is None:
            raise ValueError("Unsupported file format or file not found.")
        metadata = self._extract_metadata(df)

        code_prompt = PromptTemplate.from_template(
            template="""
You are software engineer who is an expert in data analysis and manipulation.
You are provided with a spreadsheet metadata and a query to answer based on the data in that file.

Metadata: {metadata}
Query: {query}
File Path: {file_path}

Your task is to generate code to answer the following query based on the provided spreadsheet data.
Generate Python code that uses pandas to answer the query. Provide only the code, no explanations.
Always return the result in a variable named '_result'.
            """
        )

        code_prompt = code_prompt.format(
            query=query,
            data=df.to_dict(orient='records')
        )
        code = self.llm.invoke(code_prompt).content
        safe = {
                'df': df,
                'pd': pd,
                'np': np,
                'print': print,
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'list': list,
                'dict': dict,
                'sum': sum,
                'min': min,
                'max': max,
                'round': round
            }
        exec(str(code), safe)
        result = safe.get('_result', None)
        return result

    def analyze_spreadsheet(self, df: pd.DataFrame, query: str) -> Optional[pd.DataFrame]:
        """
        Analyze a spreadsheet DataFrame based on a query.
        
        Args:
            df (pd.DataFrame): The DataFrame to analyze.
            query (str): The query to answer.
        
        Returns:
            pd.DataFrame: The result of the analysis.
        """
        if df.empty:
            print("The DataFrame is empty.")
            return None
        
        metadata = self._extract_metadata(df)
        prompt_template = PromptTemplate.from_template(
            template="""
You are a data analyst tasked with analyzing a spreadsheet.
You are provided with the metadata of the spreadsheet and a query to answer based on the data in that file.
Metadata: {metadata}
Query: {query}
Your task is to provide helpful insights about the spreadsheet based on the provided metadata and query.
            """
        )
        prompt = prompt_template.format(metadata=metadata, query=query)
        response = self.llm.invoke(prompt)
        if response:
            return pd.DataFrame(response.content)
        else:
            print("No response from the LLM.")
            return None



spreadsheetTool = SpreadsheetTool()

@tool
def analyze_spreadsheet(df: pd.DataFrame, query: str) -> Optional[pd.DataFrame]:
    return spreadsheetTool.analyze_spreadsheet(df, query)

@tool
def query_spreadsheet(file_path: str, query: str) -> Optional[pd.DataFrame]:
    return spreadsheetTool.query_spreadsheet(file_path, query)

