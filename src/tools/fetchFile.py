import requests
import os
from langchain_core.tools import tool

class FileFetcher:
    def __init__(self):
        self.default_api_url = "https://agents-course-unit4-scoring.hf.space"
        self.folder_path = "./tmp/"

    def fetch(self,task_id:str,file_name:str) -> str:
        try:
            if not os.path.exists(self.folder_path):
                os.makedirs(self.folder_path)
            file_path = os.path.join(self.folder_path, file_name)
            if os.path.exists(file_path):
                print(f"File {file_name} already exists, skipping download.")
                return file_path
            
            file_url = f"{self.default_api_url}/files/{task_id}"
            response = requests.get(file_url)
            response.raise_for_status()  # Raise an error for HTTP errors
            
            with open(file_path, "wb") as file:
                file.write(response.content)
            print(f"File {file_name} downloaded successfully. Saved to {file_path}")
            return file_path
        except requests.RequestException as e:
            print(f"Error downloading file for task {task_id}: {e}")
            return ""
        
fetcher = FileFetcher()
@tool
def fetch_file(task_id: str, file_name: str) -> str:
    """
    Fetch a file from the server based on task ID and file name.
    
    Args:
        task_id (str): The ID of the task associated with the file.
        file_name (str): The name of the file to fetch.
    
    Returns:
        str: The path to the downloaded file or an error message.
    """
    return fetcher.fetch(task_id, file_name)
