import logging
import uuid
from typing import List
from transformers import GPT2Tokenizer

import api.utils.config as config
from api.utils.task_queue import task_queue
import api.utils.llm_service as llm_service
from api.core.memory.models import Memory  # Import Memory

logger = logging.getLogger(__name__)

class ContextWindow:
    """Manages the context window for the active conversation."""

    def __init__(
        self,
        window_id: str = None,
        max_tokens: int = 4096,
        reserved_tokens: int = 500,
        template: str = "",
        summary: str = "",
        interactions: List[Tuple[str, str]] = None,
        memories: List[Memory] = None,
    ):
        """
        Initializes the context window.

        Args:
            window_id: A unique identifier for this window.
            max_tokens: The maximum number of tokens allowed in the window.
            reserved_tokens: The number of tokens reserved for the template and other fixed content.
            template: The initial template for this window.
            summary: A summary of the conversation so far.
            interactions: A list of tuples, each containing a query and its response.
            memories: A list of Memory objects associated with this window.
        """
        self.window_id = window_id or str(uuid.uuid4())
        self.max_tokens = max_tokens
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.reserved_tokens = reserved_tokens
        self.available_tokens = max_tokens - reserved_tokens
        self.current_token_count = 0
        self.template = template
        self.summary = summary
        self.interactions = interactions if interactions is not None else []
        self.memories = memories if memories is not None else []

    def get_token_count(self, text: str) -> int:
        """Returns the number of tokens in the given text."""
        return len(self.tokenizer.encode(text))

    def add_q_r_pair(self, query: str, response: str) -> bool:
        """
        Adds a Q/R pair to the context window, checking token count.

        Args:
            query: The user's query.
            response: The system's response.

        Returns:
            True if the Q/R pair was added successfully, False otherwise.
        """
        q_r_text = f"Q: {query}\nA: {response}"
        q_r_tokens = self.get_token_count(q_r_text)

        logger.debug(f"Adding Q/R pair: {q_r_tokens} tokens")
        logger.debug(f"Current token count: {self.current_token_count}")
        logger.debug(f"Available tokens: {self.available_tokens}")

        if self.current_token_count + q_r_tokens <= self.available_tokens:
            self.interactions.append((query, response))
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