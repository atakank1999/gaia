import whisper
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from src.llm_provider import get_llm

class AudioTool:
    def __init__(self, model_name: str = "base"):
        self.model = whisper.load_model(model_name)
        self.llm = get_llm(temperature=0.0)
        self.prompt_template = PromptTemplate.from_template(
            """
            You are a tool that analyzes audio files. The transcription of the audio is provided below.
            Question: {question}
            Transcription: {transcription}
            Instructions:
            1. You are given a question about an audio file and the transcription of that audio.
            2. Carefully analyze the transcription and answer the questions based on the content.
            3. Give direct and concise answers.
            4. Keep the response focused on the questions asked.
            """
            )

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio from the given path using the Whisper model.
        """
        result = self.model.transcribe(audio_path)
        if isinstance(result['text'], list):
            return "\n".join(str(item) for item in result['text'])
        else:
            return str(result['text'])

    def analyze(self, audio_path: str) -> str:
        """
        Analyze audio from the given path using the Whisper model and LLM.
        """

        transcription = self.transcribe(audio_path)
        if not transcription:
            return "No transcription available."
        question = "Hi, I was out sick from my classes on Friday, so I'm trying to figure out what I need to study for my Calculus mid-term next week. My friend from class sent me an audio recording of Professor Willowbrook giving out the recommended reading for the test, but my headphones are broken :(\n\nCould you please listen to the recording for me and tell me the page numbers I'm supposed to go over? I've attached a file called Homework.mp3 that has the recording. Please provide just the page numbers as a comma-delimited list. And please provide the list in ascending order."
        prompt = self.prompt_template.format(question=question, transcription=transcription)
        analysis = self.llm.invoke(prompt)
        if not analysis:
            return "No analysis available."
        else:
            return str(analysis.content)

AudioAnalyzeTool = AudioTool()
@tool
def analyze_audio(audio_path: str) -> str:
    """
    Analyze an audio file and return the transcription and analysis.
    """
    return AudioAnalyzeTool.analyze(audio_path)

def main():
    """
    Main function to run the audio transcription tool.
    """

    analysis = AudioAnalyzeTool.analyze("./files/1f975693-876d-457b-a649-393859e79bf3.mp3")
    print("Analysis Result:")
    print(analysis)
