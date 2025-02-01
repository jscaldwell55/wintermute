import os
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio
from openai import AsyncOpenAI, OpenAIError
from typing import Optional
from api.utils.config import get_settings  # Import get_settings
from api.utils.task_queue import task_queue
from tenacity import retry, stop_after_attempt, wait_fixed

logger = logging.getLogger(__name__)

# Initialize OpenAI client
_client = None

def _get_client():
    global _client
    settings = get_settings()  # Use get_settings() to access settings
    if _client is None:
        api_key = settings.openai_api_key
        if not api_key:
            logger.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
            raise ValueError("OpenAI API key not configured.")
        _client = AsyncOpenAI(api_key=api_key)
    return _client

async def generate_gpt_response_async(prompt: str, model: str = None, temperature: float = 0.7, max_tokens: int = 500) -> str:
    """
    Asynchronous wrapper for the generate_gpt_response function.

    Args:
        prompt: The prompt to send to the GPT model.
        model: The name of the GPT model to use.
        temperature: Sampling temperature for response randomness.
        max_tokens: Maximum number of tokens in the generated response.

    Returns:
        The generated response from the GPT model as a string.
    """
    logger.info("Starting async GPT response generation...")

    # Assign default model if not provided
    settings = get_settings()
    if model is None:
        model = settings.llm_model_id

    try:
        client = _get_client()
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )

        if hasattr(response.choices[0].message, 'content'):
            generated_text = response.choices[0].message.content
            logger.info(f"Generated GPT response for prompt: {prompt[:50]}...")
            return generated_text
        else:
            logger.error(f"Invalid response structure from OpenAI: {response}")
            return ""

    except OpenAIError as oe:
        logger.error(f"OpenAI API error during async generation: {oe}")
        raise RuntimeError(f"OpenAI API error during async generation: {oe}")
    except Exception as e:
        logger.error(f"Unexpected error during async GPT response generation: {e}", exc_info=True)
        raise