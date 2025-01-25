import pytest
from backend.core.memory.models import Memory
from backend.utils.context_window import ContextWindow
from backend.utils.llm_service import generate_gpt_response_async
from unittest.mock import patch
import asyncio
import backend.config as config

# Mock the generate_gpt_response_async to return a predefined summary
@pytest.fixture
def mock_generate_summary():
    async def mock_summary(*args, **kwargs):
        return "This is a mock summary."
    return mock_summary

@pytest.mark.asyncio
async def test_add_q_r_pair_within_limit():
    context_window = ContextWindow(total_tokens=500, reserved_tokens=100)
    # Assuming an average of 5 tokens per word for estimation
    query = "What is the weather like today?"  # ~7 tokens
    response = "The weather today is sunny and warm."  # ~8 tokens
    # Total tokens for Q/R pair = 15, which is within the limit
    result = context_window.add_q_r_pair(query, response)
    assert result is True
    assert context_window.current_token_count > 0

@pytest.mark.asyncio
async def test_add_q_r_pair_exceeds_limit():
    context_window = ContextWindow(total_tokens=50, reserved_tokens=20)
    # Long query and response to exceed the limit
    query = "What is the weather like today in a very long and detailed description that will exceed the token limit?"
    response = "The weather today is sunny and warm with a lot of extra details to make this response exceed the token limit."
    # Estimate that the total tokens exceed the limit of 30 available tokens
    result = context_window.add_q_r_pair(query, response)
    assert result is False
    assert context_window.current_token_count == 0

@pytest.mark.asyncio
async def test_generate_template_within_limit(mock_generate_summary):
    with patch('backend.utils.llm_service.generate_gpt_response_async', mock_generate_summary):
        context_window = ContextWindow(total_tokens=500, reserved_tokens=100)
        summary = await context_window.generate_window_summary([], "window_id")
        # The summary should be within the reserved token limit
        assert context_window.get_token_count(summary) <= context_window.reserved_tokens

@pytest.mark.asyncio
async def test_generate_template_exceeds_limit(mock_generate_summary):
    with patch('backend.utils.llm_service.generate_gpt_response_async', mock_generate_summary):
        context_window = ContextWindow(total_tokens=500, reserved_tokens=20)
        summary = await context_window.generate_window_summary([], "window_id")
        # The summary should be within the reserved token limit
        assert context_window.get_token_count(summary) <= context_window.reserved_tokens

@pytest.mark.asyncio
async def test_reset_window():
    context_window = ContextWindow(total_tokens=500, reserved_tokens=100)
    # Add something to the context to simulate usage
    context_window.add_q_r_pair("What's up?", "Not much.")
    context_window.reset_window()
    # After reset, token count should be 0 and template should be empty
    assert context_window.current_token_count == 0
    assert context_window.template == ""

@pytest.mark.asyncio
async def test_generate_window_summary_empty_memories():
    context_window = ContextWindow()
    summary = await context_window.generate_window_summary([], "window_id")
    assert summary == ""

@pytest.mark.asyncio
async def test_is_full():
    context_window = ContextWindow(total_tokens=150, reserved_tokens=20)
    # Use up some tokens
    context_window.add_q_r_pair("Small talk", "Yes, indeed.")
    # Should not be full yet
    assert context_window.is_full() is False

    # Exceed the limit
    query = "What is the weather like today in a very long and detailed description that will exceed the token limit?"
    response = "The weather today is sunny and warm with a lot of extra details to make this response exceed the token limit."
    context_window.add_q_r_pair(query, response)
    # Now it should be full
    assert context_window.is_full() is True