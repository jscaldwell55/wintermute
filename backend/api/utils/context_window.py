import logging
import uuid
from typing import List
from gpt3_tokenizer import Tokenizer

import api.utils.config as config
from api.utils.task_queue import task_queue
import api.utils.llm_service as llm_service
from api.core.memory.models import Memory  # Import Memory

logger = logging.getLogger(__name__)

class ContextWindow:
    """Manages the context window for the active conversation."""

    def __init__(self, max_tokens: int = 4096):
        """Initialize the context window with a maximum token limit."""
        self.max_tokens = max_tokens
        self.current_token_count = 0
        self.available_tokens = max_tokens
        self.tokenizer = Tokenizer()

    def get_token_count(self, text: str) -> int:
        """Returns the number of tokens in the given text."""
        return len(self.tokenizer.encode(text))

    def add_q_r_pair(self, query: str, response: str) -> bool:
        """Adds a Q/R pair to the context window, checking token count."""
        q_r_text = f"Q: {query}\nA: {response}"
        q_r_tokens = self.get_token_count(q_r_text)

        print(f"DEBUG: Adding Q/R pair: {q_r_tokens} tokens")
        print(f"DEBUG: Current token count: {self.current_token_count}")
        print(f"DEBUG: Available tokens: {self.available_tokens}")

        if self.current_token_count + q_r_tokens <= self.available_tokens:
            self.current_token_count += q_r_tokens
            return True
        else:
            return False

    async def generate_window_summary(
        self, memories: List[Memory], window_id: str
    ) -> str:
        """
        Generates a summary of the memories in the current context window using an LLM.

        Args:
            memories: A list of Memory objects representing the memories in the window.
            window_id: ID of the window being summarized.

        Returns:
            A string containing the generated summary.
        """
        logger.info(f"Generating summary for window: {window_id}")

        if not memories:
            logger.info("No memories to summarize.")
            return ""

        memory_contents = [mem.content for mem in memories]
        prompt = f"""
        Summarize the following conversation excerpts:

        {' '.join(memory_contents)}

        Summary:
        """
        logger.info(f"Summary prompt: {prompt}")

        try:
            summary = await llm_service.generate_gpt_response_async(
                prompt=prompt,
                model=config.SUMMARY_LLM_MODEL_ID,
                temperature=0.5,
                max_tokens=config.SUMMARY_MAX_LENGTH,
            )
            logger.info(f"Generated summary: {summary}")

            # Update the summary of the current `ContextWindow`
            self.summary = summary.strip()

            return self.summary

        except Exception as e:
            logger.error(f"Error generating window summary: {e}")
            return ""

    def is_full(self) -> bool:
        """Checks if the context window is full."""
        logger.debug(
            f"Checking if full. Current count: {self.current_token_count}, Available: {self.available_tokens}"
        )
        return self.current_token_count >= self.available_tokens

    @staticmethod
    def reset_context(context_window: "ContextWindow") -> "ContextWindow":
        """
        Resets the provided context window, generating a new window ID and creating a new
        ContextWindow instance.
        """
        new_window_id = ContextWindow.generate_window_id()
        logger.info(
            f"Resetting context window from {context_window.window_id} to {new_window_id}"
        )
        return ContextWindow(
            window_id=new_window_id,
            max_tokens=context_window.max_tokens,
            reserved_tokens=context_window.reserved_tokens,
            template=context_window.template,
        )

    @staticmethod
    def generate_window_id():
        """Generates a unique window ID."""
        return str(uuid.uuid4())