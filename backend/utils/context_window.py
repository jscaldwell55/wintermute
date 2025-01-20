import tiktoken
import logging

logger = logging.getLogger(__name__)

class ContextWindow:
    """Manages the context window for the active conversation."""

    def __init__(self, total_tokens: int = 2048, reserved_tokens: int = 512):
        self.total_tokens = total_tokens
        self.reserved_tokens = reserved_tokens
        self.available_tokens = self.total_tokens - self.reserved_tokens
        self.current_token_count = 0
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.template = ""
        self.window_id = None

    def get_token_count(self, text: str) -> int:
        """Returns the number of tokens in the given text."""
        return len(self.encoding.encode(text))

    def add_q_r_pair(self, query: str, response: str) -> bool:
        """Adds a Q/R pair to the context window, checking token count."""
        q_r_text = f"Q: {query}\nA: {response}"
        q_r_tokens = self.get_token_count(q_r_text)

        if self.current_token_count + q_r_tokens <= self.available_tokens:
            self.current_token_count += q_r_tokens
            return True
        else:
            return False

    def generate_template(self, summary: str) -> str:
        """Generates a template for the next window, including a summary."""
        self.template = f"Summary of recent conversation:\n{summary}\n\n"
        template_tokens = self.get_token_count(self.template)

        if template_tokens > self.reserved_tokens:
            logger.warning(
                f"Template exceeds reserved tokens: {template_tokens} > {self.reserved_tokens}"
            )

        return self.template

    def reset_window(self):
        """Resets the context window."""
        self.current_token_count = 0
        self.template = ""