import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from core.memory.memory import MemorySystem
from utils.pinecone_service import PineconeService
from utils.vector_operations import VectorOperations
import uuid
import time

@pytest.fixture
def mock_pinecone_service():
    mock = MagicMock(spec=PineconeService)
    mock.upsert_memory = AsyncMock()
    mock.query_memory = AsyncMock()
    mock.delete_memory = AsyncMock()
    return mock

@pytest.fixture
def mock_openai_response():
    return {
        "data": [{"embedding": [0.1] * 1536}],
        "model": "text-embedding-ada-002",
        "usage": {"prompt_tokens": 1, "total_tokens": 1}
    }

@pytest.fixture
def vector_operations(mock_openai_response):
    with patch('openai.AsyncClient') as mock_client:
        mock_client_instance = AsyncMock()
        mock_client_instance.embeddings = AsyncMock()
        mock_client_instance.embeddings.create = AsyncMock(
            return_value=mock_openai_response
        )
        mock_client.return_value = mock_client_instance
        
        vector_ops = VectorOperations()
        vector_ops.client = mock_client_instance
        return vector_ops

@pytest.fixture
def memory_system(mock_pinecone_service, vector_operations):
    with patch('utils.pinecone_service.PineconeService.__init__', return_value=None):
        memory_system = MemorySystem(
            pinecone_api_key="dummy_api_key",
            pinecone_environment="dummy_env",
            vector_operations=vector_operations
        )
        memory_system.pinecone_service = mock_pinecone_service
        return memory_system

@pytest.mark.asyncio
async def test_add_memory(memory_system, mock_pinecone_service):
    text = "This is a happy moment."
    metadata = {"id": "test_memory"}

    with patch('uuid.uuid4', return_value=uuid.UUID('12345678-1234-5678-1234-567812345678')):
        memory_id = await memory_system.add_memory(text, metadata)

        mock_pinecone_service.upsert_memory.assert_called_once()
        call_args = mock_pinecone_service.upsert_memory.call_args
        upserted_memory_id, upserted_vector, upserted_metadata = call_args[1].values()

        assert upserted_memory_id == "mem_12345678-1234-5678-1234-567812345678"
        assert isinstance(upserted_vector, list)
        assert len(upserted_vector) == 1536
        assert upserted_metadata["content"] == text

@pytest.mark.asyncio
async def test_memory_retrieval(memory_system, mock_pinecone_service):
    # Set up the mock response
    mock_pinecone_service.query_memory.return_value = {
        "matches": [
            {
                "id": "positive_memory",
                "score": 0.9,
                "values": [0.1] * 1536,
                "metadata": {
                    "content": "This is a joyful experience.",
                    "memory_type": "TEMPORAL",
                    "created_at": "2024-01-17T00:00:00"
                }
            },
            {
                "id": "negative_memory",
                "score": 0.7,
                "values": [0.1] * 1536,
                "metadata": {
                    "content": "This is a frustrating problem.",
                    "memory_type": "TEMPORAL",
                    "created_at": "2024-01-17T00:00:00"
                }
            }
        ],
        "namespace": ""  # Optional: Include if your Pinecone service expects it
    }

    # Perform the search
    query_vector = np.array([0.1] * 1536)
    results = await memory_system.search_memories(query_vector, top_k=2)

    # Verify results
    assert len(results) == 2, "Expected two results"
    memories = [mem for mem, _ in results]
    
    memory_ids = {mem.id for mem in memories}
    assert "positive_memory" in memory_ids
    assert "negative_memory" in memory_ids

    memory_contents = {mem.content for mem in memories}
    assert "This is a joyful experience." in memory_contents
    assert "This is a frustrating problem." in memory_contents

@pytest.mark.asyncio
async def test_performance(memory_system, mock_pinecone_service):
    # Set up mock response for search first
    mock_pinecone_service.query_memory.return_value = {
        "matches": [
            {
                "id": "test_id",
                "score": 0.9,
                "values": [0.1] * 1536,
                "metadata": {
                    "content": "Test content",
                    "memory_type": "TEMPORAL",
                    "created_at": "2024-01-17T00:00:00"
                }
            }
        ],
        "namespace": ""
    }

    text = "This is a sample memory for performance testing."
    metadata = {"id": "performance_test"}
    
    # Test memory addition
    start_time = time.time()
    await memory_system.add_memory(text, metadata)
    end_time = time.time()
    upsertion_time = end_time - start_time
    print(f"Upsertion time: {upsertion_time:.4f} seconds")
    assert upsertion_time < 1.0, "Upsertion took longer than 1 second"

    # Test memory retrieval
    query_vector = np.array([0.1] * 1536)
    start_time = time.time()
    results = await memory_system.search_memories(query_vector, top_k=5)
    end_time = time.time()
    retrieval_time = end_time - start_time
    print(f"Retrieval time: {retrieval_time:.4f} seconds")
    
    # Verify the results
    assert len(results) == 1, "Expected one result"
    memory, score = results[0]
    assert memory.id == "test_id"
    assert memory.content == "Test content"
    assert score == 0.9

    assert retrieval_time < 1.0, "Retrieval took longer than 1 second"