import logging

# Configure logging
logger = logging.getLogger(__name__)

MASTER_TEMPLATE = """
You are a helpful AI assistant named Wintermute, an expert in conversation and memory.

Below are some snippets of context that may be relevant to the user's query:
{retrieved_context}

Here is the user's query:
{user_query}

Respond to the user's query, incorporating the context above if it's relevant.
"""

def format_prompt(template, retrieved_context, user_query):
    """
    Formats the prompt template with the provided context, query, and metadata.

    Args:
        template: The prompt template string.
        retrieved_context: The retrieved context string.
        user_query: The user query string.

    Returns:
        A formatted prompt string.
    """
    prompt = template.format(
        retrieved_context=retrieved_context,
        user_query=user_query,
    )
    logger.info(f"Formatted prompt: {prompt}")
    return prompt