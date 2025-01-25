import tiktoken
import logging
import uuid
import project_root.backend.api.utils.config as config
from api.utils.task_queue import task_queue
import api.utils.llm_service as llm_service

logger = logging.getLogger(__name__)

class ContextWindow:
    """Manages the context window for the active conversation."""

    def __init__(self, total_tokens: int = config.MAX_TOKENS_PER_CONTEXT_WINDOW, reserved_tokens: int = 512):
        self.total_tokens = total_tokens
        self.reserved_tokens = reserved_tokens
        self.available_tokens = self.total_tokens - self.reserved_tokens
        self.current_token_count = 0
        self.encoding = tiktoken.encoding_for_model(config.LLM_MODEL_NAME)
        self.template = ""
        self.window_id = str(uuid.uuid4())

    def get_token_count(self, text: str) -> int:
        """Returns the number of tokens in the given text."""
        return len(self.encoding.encode(text))

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

    async def generate_window_summary(self, memories: list, window_id: str) -> str:
        """
        Generates a summary of the memories in the current context window using an LLM.

        Args:
            memories: A list of Memory objects representing the memories in the window.

        Returns:
            A string containing the generated summary.
        """
        if not memories:
            return ""

        # Build a prompt for the LLM to summarize the memories
        memory_contents = [mem.content for mem in memories]
        prompt = f"""
        Summarize the following conversation excerpts:

        {' '.join(memory_contents)}

        Summary:
        """

        try:
            summary = await llm_service.generate_gpt_response_async(
                prompt=prompt,
                model=config.SUMMARY_LLM_MODEL_NAME,
                temperature=0.5,  # Adjust temperature for summarization
                max_tokens=256,  # Adjust max_tokens for summary length
            )
            # Update the template of the new `ContextWindow`
            self.generate_template(summary.strip())
            return summary.strip()

        except Exception as e:
            logger.error(f"Error generating window summary: {e}")
            return ""  # Return an empty string on error

    def generate_template(self, summary: str) -> str:
        """Generates a template for the next window, including a summary."""
        self.template = f"Summary of recent conversation:\n{summary}\n\n"
        template_tokens = self.get_token_count(self.template)

        if template_tokens > self.reserved_tokens:
            logger.warning(
                f"Template exceeds reserved tokens: {template_tokens} > {self.reserved_tokens}"
            )

        return self.template

    def is_full(self):
        """Checks if the context window is full."""
        print(f"DEBUG: Checking if full. Current count: {self.current_token_count}, Available: {self.available_tokens}")
        return self.current_token_count >= self.available_tokens

    def reset_window(self):
        """Resets the context window."""
        self.current_token_count = 0
        self.template = ""
        self.window_id = str(uuid.uuid4())