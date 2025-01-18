import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from core.memory.memory import MemorySystem
from utils.pinecone_service import PineconeService
from utils.emotional_analysis import EmotionalAnalysis, EmotionalPolarity
from utils.vector_operations import VectorOperations
import openai

@pytest.fixture
def mock_pinecone_service():
    mock = MagicMock(spec=PineconeService)
    mock.upsert_memory = AsyncMock()
    mock.query_memory = AsyncMock(return_value={"matches": []})  # Default empty result
    return mock

@pytest.fixture
def mock_emotional_analysis():
    mock = MagicMock(spec=EmotionalAnalysis)
    mock.analyze_polarity = AsyncMock(return_value=(EmotionalPolarity.POSITIVE, 0.8))  # Mock return value
    return mock

@pytest.fixture
@patch('utils.pinecone_service.PineconeService.__init__', new_callable=MagicMock)
@patch('openai.embeddings.create')  # Patch the OpenAI API call
def memory_system(mock_openai_create, mock_pinecone_init, mock_pinecone_service, mock_emotional_analysis):
    mock_pinecone_init.return_value = None  # Mock the __init__ method to avoid actual Pinecone initialization
    mock_openai_create.return_value = {
        "data": [{"embedding": [0.1] * 1536}]
    }  # Mock the embedding with a list of 1536 floats
    vector_operations = VectorOperations()
    memory_system = MemorySystem(
        "dummy_api_key", "dummy_env", vector_operations, mock_emotional_analysis
    )
    memory_system.pinecone_service = mock_pinecone_service
    return memory_system

@pytest.mark.asyncio
@patch('os.getenv', side_effect=lambda k: {'PINECONE_API_KEY': 'dummy', 'PINECONE_ENVIRONMENT': 'dummy', 'INDEX_NAME': 'test-index', 'OPENAI_API_KEY': 'sk-fake'}.get(k))
@patch('utils.vector_operations.VectorOperations.generate_embedding')
@patch('uuid.uuid4')  # Patch uuid.uuid4
async def test_add_memory(mock_uuid4, mock_generate_embedding, mock_getenv, memory_system, mock_pinecone_service, mock_emotional_analysis):
    text = "This is a happy moment."
    metadata = {"id": "test_memory"}

    # Mock the return value of generate_embedding
    mock_generate_embedding.return_value = [0.1] * 1536

    # Mock uuid.uuid4 to return a consistent value
    mock_uuid4.return_value = "consistent-uuid"

    # Mock the return value of analyze_polarity for this specific call
    mock_emotional_analysis.analyze_polarity.return_value = (EmotionalPolarity.POSITIVE, 0.8)

    memory_id = await memory_system.add_memory(text, metadata)

    # Assert that upsert_memory was called with the correct arguments
    mock_pinecone_service.upsert_memory.assert_called_once()
    call_args = mock_pinecone_service.upsert_memory.call_args
    upserted_memory_id, upserted_vector, upserted_metadata = call_args[1].values()

    # Use the mocked UUID in the assertion
    assert upserted_memory_id == "mem_consistent-uuid"
    assert isinstance(upserted_vector, list)  # Should be a list
    assert len(upserted_vector) == 1536  # Check vector dimension
    assert upserted_metadata["content"] == text
    assert upserted_metadata["polarity"] == EmotionalPolarity.POSITIVE.value
    assert upserted_metadata["confidence"] == 0.8

# Example of how to mock the query_memory return value for more specific retrieval tests:
@pytest.mark.asyncio
@patch('os.getenv', side_effect=lambda k: {'PINECONE_API_KEY': 'dummy', 'PINECONE_ENVIRONMENT': 'dummy', 'INDEX_NAME': 'test-index', 'OPENAI_API_KEY': 'sk-fake'}.get(k))
async def test_retrieve_memory(mock_getenv, memory_system, mock_pinecone_service):
    mock_pinecone_service.query_memory.return_value = {
        "matches": [
            {
                "id": "test_memory",
                "score": 0.9,
                "values": [],
                "metadata": {
                    "content": "This is a happy moment.",
                    "polarity": EmotionalPolarity.POSITIVE.value,
                    "confidence": 0.8,
                }
            }
        ]
    }