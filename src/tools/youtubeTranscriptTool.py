import re
from youtube_transcript_api import YouTubeTranscriptApi, FetchedTranscript
from ..llm_provider import get_llm
from typing import Optional
from langchain_core.prompts import PromptTemplate


class YouTubeTranscriptTool:
    def __init__(self):
        self.llm = get_llm()
        self.youtube_transcript_api = YouTubeTranscriptApi()
        self.prompt_template = PromptTemplate.from_template(
            template="""
You are a tool that analyzes YouTube videos and provides their transcripts.
Question: {question}
Transcript: {transcript}
Instructions:
1. You are given a question about a YouTube video. And the transcript of that video with timestamps.
2. Carefully analyze the transcript and answer the questions based on the content.
3. Give direct and concise answers.
4. Keep the response focused on the questions asked.
"""
        )

    def _extract_video_id(self, url: str) -> Optional[str]:
        """
        Extract the video ID from a YouTube URL.
        """
        pattern = r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})"
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        else:
            return None

    def _get_transcript(self, video_id: str) -> Optional[FetchedTranscript]:
        """
        Get the transcript for a YouTube video.
        """
        try:
            transcript = self.youtube_transcript_api.fetch(video_id=video_id)
            return transcript
        except Exception as e:
            print(f"Error fetching transcript: {e}")
            return None

    def _format_transcript(self, transcript: FetchedTranscript) -> str:
        """
        Format the transcript into a readable string.
        """
        formatted_transcript = []
        for snippet in transcript:
            start = snippet.start
            duration = snippet.duration
            text = snippet.text
            formatted_transcript.append(
                f"[{start:.2f}s - {start + duration:.2f}s] {text}"
            )
        return "\n".join(formatted_transcript)

    def analyze_youtube_video(self, question: str) -> Optional[str]:
        """
        Analyze a YouTube video by URL and return its transcript.
        """
        video_id = self._extract_video_id(question)
        if not video_id:
            print("Invalid YouTube URL.")
            return None

        transcript = self._get_transcript(video_id)
        if not transcript:
            print("No transcript available.")
            return None
        formatted_transcript = self._format_transcript(transcript)
        if not formatted_transcript:
            print("Transcript is empty or not available.")
            return None
        prompt = self.prompt_template.format(
            video_id=video_id, question=question, transcript=formatted_transcript
        )
        response = self.llm.invoke(prompt)
        if response:
            return str(response.content)
        else:
            print("No response from the LLM.")
            return None


def main():
    """
    Main function to run the YouTubeTranscriptTool.
    """
    tool = YouTubeTranscriptTool()
    question = "Examine the video at https://www.youtube.com/watch?v=1htKBjuUWec.\n\nWhat does Teal'c say in response to the question \"Isn't that hot?\""
    video_id = tool._extract_video_id(question)

    if not video_id:
        print("Invalid YouTube URL.")
        return

    tool_response = tool.analyze_youtube_video(question)
    if tool_response:
        print(tool_response)
    else:
        print("No response from the tool.")
