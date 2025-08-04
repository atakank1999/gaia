from langchain_ollama import ChatOllama

def get_llm(temperature: float = 0.0) -> ChatOllama:
    llm = ChatOllama(model="qwen2.5:7b-instruct", base_url="http://localhost:11434", temperature=temperature)
    return llm

