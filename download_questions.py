DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
questions_url = f"{DEFAULT_API_URL}/questions"
import json
import os
import requests
try:
    response = requests.get(questions_url)
    response.raise_for_status()  # Raise an error for HTTP errors
    questions = response.json()
except requests.RequestException as e:
    print(f"Error fetching questions: {e}")
    questions = []

with open("questions.json", "w") as f:
    json.dump(questions, f, indent=2)

for question in questions:
    if question.get("file_name"):
        file_url = f"{DEFAULT_API_URL}/files/{question['task_id']}"
        try:
            if not os.path.exists("./files/"):
                os.makedirs("./files/")
            if os.path.exists(f"./files/{question['file_name']}"):
                print(f"File {question['file_name']} already exists, skipping download.")
                continue
            file_response = requests.get(file_url)
            file_response.raise_for_status()  # Raise an error for HTTP errors
            with open(f"./files/{question['file_name']}", "wb") as file:
                file.write(file_response.content)
        except requests.RequestException as e:
            print(f"Error downloading file for question {question['task_id']}: {e}")
