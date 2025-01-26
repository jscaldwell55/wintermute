import tiktoken
import logging
from typing import List, Tuple, Optional
from api.utils.config import get_settings
from langchain.text_splitter import RecursiveCharacterTextSplitter

class ContextWindow:
    def __init__(self, max_tokens: Optional[int] = None):
        settings = get_settings()
        self.max_tokens = max_tokens or settings.get_setting("MAX_TOKENS_PER_CONTEXT_WINDOW")
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.window: List[Tuple[str, str]] = []
        self.current_token_count = 0
        self.window_id = 0

    def add_q_r_pair(self, query: str, response: str) -> bool:
        """Adds a query and response pair to the context window."""
        if not isinstance(query, str) or not isinstance(response, str):
            raise ValueError("Query and Response must be strings.")

        tokens = len(self.encoding.encode(query)) + len(self.encoding.encode(response))
        if tokens > self.max_tokens:
            return False

        self.window.append((query, response))
        self.current_token_count += tokens
        return True

    def is_full(self) -> bool:
        """Checks if the context window is full."""
        return self.current_token_count >= self.max_tokens

    async def generate_window_summary(self) -> str:
        """Generates a summary of the current context window."""
        if not self.window:
            return ""

        text = " ".join(f"Query: {q}\nResponse: {r}" for q, r in self.window)

        # Assuming summarize is an async method, await it
        summary = await self.summarize(text)
        return summary

    def reset_context(self):
        """Resets the context window."""
        self.window = []
        self.current_token_count = 0
        self.window_id += 1

    async def summarize(self, text: str) -> str:
        """
        Summarizes the given text using a summarization pipeline.

        Args:
            text: The text to summarize.

        Returns:
            The summary of the text.
        """
        from transformers import pipeline

        # Initialize the summarization pipeline
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

        # Split the text into chunks that fit within the model's maximum token limit
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,  # Adjust based on the model's max token limit
            chunk_overlap=64,
            length_function=len,
        )

        chunks = text_splitter.split_text(text)

        # Summarize each chunk
        summaries = []
        for chunk in chunks:
            try:
                summary = summarizer(
                    chunk,
                    max_length=150,  # Adjust based on your requirements
                    min_length=50,   # Adjust based on your requirements
                    do_sample=False
                )[0]['summary_text']
                summaries.append(summary)
            except Exception as e:
                logging.error(f"Error during summarization: {e}")
                # Handle the error as appropriate for your application
                # For example, you might want to return a partial summary or a default message

        # Combine the summaries into a single summary
        combined_summary = " ".join(summaries)

        return combined_summary