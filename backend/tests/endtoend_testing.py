import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from core.memory.memory import MemorySystem
from utils.pinecone_service import PineconeService
from utils.emotional_analysis import EmotionalAnalysis, EmotionalPolarity
from utils.vector_operations import VectorOperations
import uuid

@pytest.fixture
def mock_pinecone_service():
    mock = MagicMock(spec=PineconeService)
    mock.upsert_memory = AsyncMock()
    mock.query_memory = AsyncMock(return_value={"matches": []})  # Default empty result
    mock.delete_memory = AsyncMock()  # Add a mock for delete_memory
    return mock

@pytest.fixture
def mock_emotional_analysis():
    mock = MagicMock(spec=EmotionalAnalysis)
    # Mock to return predefined polarity and confidence
    async def mock_analyze_polarity(text):
        if "positive" in text.lower() or "happy" in text.lower() or "joyful" in text.lower():
            return EmotionalPolarity.POSITIVE, 0.9
        elif "negative" in text.lower() or "frustrating" in text.lower():
            return EmotionalPolarity.NEGATIVE, 0.8
        else:
            return EmotionalPolarity.NEUTRAL, 0.5
    mock.analyze_polarity = AsyncMock(side_effect=mock_analyze_polarity)
    return mock

@pytest.fixture
@patch('utils.pinecone_service.PineconeService.__init__', new_callable=MagicMock)
@patch('openai.embeddings.create')
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
async def test_add_memory(mock_generate_embedding, mock_getenv, memory_system, mock_pinecone_service, mock_emotional_analysis):
    text = "This is a happy moment."
    metadata = {"id": "test_memory"}

    mock_generate_embedding.return_value = [0.1] * 1536
    mock_emotional_analysis.analyze_polarity.return_value = (EmotionalPolarity.POSITIVE, 0.9)

    with patch('uuid.uuid4', return_value=uuid.UUID('12345678-1234-5678-1234-567812345678')):
        memory_id = await memory_system.add_memory(text, metadata)

        mock_pinecone_service.upsert_memory.assert_called_once()
        call_args = mock_pinecone_service.upsert_memory.call_args
        upserted_memory_id, upserted_vector, upserted_metadata = call_args[1].values()

        assert upserted_memory_id == "mem_12345678-1234-5678-1234-567812345678"
        assert isinstance(upserted_vector, list)
        assert len(upserted_vector) == 1536
        assert upserted_metadata["content"] == text
        assert upserted_metadata["polarity"] == EmotionalPolarity.POSITIVE.value
        assert upserted_metadata["confidence"] == 0.9

@pytest.mark.asyncio
@patch('os.getenv', side_effect=lambda k: {'PINECONE_API_KEY': 'dummy', 'PINECONE_ENVIRONMENT': 'dummy', 'INDEX_NAME': 'test-index', 'OPENAI_API_KEY': 'sk-fake'}.get(k))
@patch('utils.vector_operations.VectorOperations.generate_embedding')
async def test_memory_retrieval(mock_generate_embedding, mock_getenv, memory_system, mock_pinecone_service, mock_emotional_analysis):
    mock_generate_embedding.return_value = [0.1] * 1536

    # Define test memories and add them to the system
    test_memories = [
        {"content": "This is a joyful experience.", "metadata": {"id": "positive_memory", "memory_type": "TEMPORAL"}},
        {"content": "This is a frustrating problem.", "metadata": {"id": "negative_memory", "memory_type": "TEMPORAL"}}
    ]
    for memory in test_memories:
        await memory_system.add_memory(memory["content"], memory["metadata"])

    # Directly mock the query result to return a list of dictionaries
    mock_pinecone_service.query_memory.return_value = [
        {
            "id": "positive_memory",
            "score": 0.9,
            "values": [],
            "metadata": {
                "content": "This is a joyful experience.",
                "polarity": EmotionalPolarity.POSITIVE.value,
                "confidence": 0.9,
            }
        },
        {
            "id": "negative_memory",
            "score": 0.7,
            "values": [],
            "metadata": {
                "content": "This is a frustrating problem.",
                "polarity": EmotionalPolarity.NEGATIVE.value,
                "confidence": 0.8,
            }
        }
    ]

    query_vector = np.array([0.1] * 1536)
    results = await memory_system.search_memories(query_vector, top_k=2)

    # Correctly unpack results into memories and scores
    memories, scores = zip(*results)

    # Assert memories were returned
    assert len(memories) == 2, "Expected two memories to be returned"

    # Collect the IDs from the returned Memory objects
    memory_ids = {mem.id for mem in memories}

    # Assert that both expected memory IDs are present
    assert "positive_memory" in memory_ids, "Expected positive memory to be returned"
    assert "negative_memory" in memory_ids, "Expected negative memory to be returned"