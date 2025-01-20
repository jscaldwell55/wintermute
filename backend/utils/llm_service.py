import os
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio
from openai import OpenAI, OpenAIError
from typing import Optional

logger = logging.getLogger(__name__)

# Initialize OpenAI client
_client = None

def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
            raise ValueError("OpenAI API key not configured.")
        _client = OpenAI(api_key=api_key)
    return _client

def generate_gpt_response(prompt: str, model: str = "gpt-3.5-turbo", temperature: float = 0.7, max_tokens: int = 500) -> str:
    """
    Generates a GPT response using the OpenAI ChatCompletion API.

    Args:
        prompt: The prompt to send to the GPT model.
        model: The name of the GPT model to use.
        temperature: Sampling temperature for response randomness.
        max_tokens: Maximum number of tokens in the generated response.

    Returns:
        The generated response from the GPT model as a string.
    """
    if not prompt.strip():
        logger.error("Prompt cannot be empty.")
        raise ValueError("Prompt cannot be empty.")

    try:
        client = _get_client()
        response = client.chat.completions.create(
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
        logger.error(f"OpenAI API error: {oe}")
        raise RuntimeError(f"OpenAI API error: {oe}")
    except Exception as e:
        logger.error(f"Unexpected error generating GPT response: {e}")
        raise

async def generate_gpt_response_async(prompt: str, model: str = "gpt-3.5-turbo", temperature: float = 0.7, max_tokens: int = 500) -> str:
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
    loop = asyncio.get_event_loop()

    try:
        with ThreadPoolExecutor() as executor:
            response = await loop.run_in_executor(
                executor,
                lambda: generate_gpt_response(
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            )
            logger.info("Async GPT response generated successfully.")
            return response

    except OpenAIError as oe:
        logger.error(f"OpenAI API error during async generation: {oe}")
        raise RuntimeError(f"OpenAI API error during async generation: {oe}")
    except Exception as e:
        logger.error(f"Unexpected error during async GPT response generation: {e}", exc_info=True)
        raise