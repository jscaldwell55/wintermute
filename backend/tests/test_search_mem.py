import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from core.memory.memory import MemorySystem
from utils.pinecone_service import PineconeService
from utils.vector_operations import VectorOperations
import uuid
import time
import datetime  # Correct import for datetime
import pytest_asyncio

@pytest.fixture
def mock_pinecone_service():
    mock = MagicMock(spec=PineconeService)
    mock.upsert_memory = AsyncMock()
    # Updated to return a list directly, not a dictionary with "matches"
    mock.query_memory = AsyncMock(return_value=[
        {
            "id": "positive_memory",
            "score": 0.9,
            "values": [],
            "metadata": {
                "content": "This is a joyful experience."
            }
        },
        {
            "id": "negative_memory",
            "score": 0.7,
            "values": [],
            "metadata": {
                "content": "This is a frustrating problem."
            }
        }
    ])
    mock.delete_memory = AsyncMock()
    return mock

@pytest.fixture
def vector_operations():
    return VectorOperations()

@pytest_asyncio.fixture
async def memory_system(mock_pinecone_service, vector_operations):
    with patch('utils.pinecone_service.PineconeService.__init__', return_value=None):
        memory_system = MemorySystem("dummy_api_key", "dummy_env", vector_operations)
        memory_system.pinecone_service = mock_pinecone_service
        return memory_system

@pytest.mark.asyncio
@patch("utils.vector_operations.VectorOperations.create_semantic_vector", AsyncMock(return_value=np.random.rand(1536)))
async def test_add_memory(memory_system, mock_pinecone_service):
    with patch('os.getenv', side_effect=lambda k: {
        'PINECONE_API_KEY': 'dummy',
        'PINECONE_ENVIRONMENT': 'dummy',
        'INDEX_NAME': 'test-index',
        'OPENAI_API_KEY': 'sk-fake'
    }.get(k)), \
    patch('uuid.uuid4', return_value=uuid.UUID('12345678-1234-5678-1234-567812345678')):
        
        text = "This is a happy moment."
        metadata = {"id": "test_memory"}
        
        memory_id = await memory_system.add_memory(text, metadata)

        mock_pinecone_service.upsert_memory.assert_called_once()
        call_args = mock_pinecone_service.upsert_memory.call_args
        upserted_memory_id, upserted_vector, upserted_metadata = call_args[1].values()

        assert upserted_memory_id == "mem_12345678-1234-5678-1234-567812345678"
        assert isinstance(upserted_vector, list)
        assert len(upserted_vector) == 1536
        assert upserted_metadata["content"] == text

@pytest.mark.asyncio
@patch("utils.vector_operations.VectorOperations.create_semantic_vector", AsyncMock(return_value=np.random.rand(1536)))
async def test_memory_retrieval(memory_system, mock_pinecone_service):
    with patch('os.getenv', side_effect=lambda k: {
        'PINECONE_API_KEY': 'dummy',
        'PINECONE_ENVIRONMENT': 'dummy',
        'INDEX_NAME': 'test-index',
        'OPENAI_API_KEY': 'sk-fake'
    }.get(k)):
        # Create an ISO format timestamp for testing
        current_time = datetime.datetime.now().isoformat()
        
        # Mock Pinecone query response with created_at timestamps as ISO strings
        mock_pinecone_service.query_memory.return_value = [
            {
                "id": "positive_memory",
                "score": 0.9,
                "values": [],
                "metadata": {
                    "content": "This is a joyful experience.",
                    "id": "positive_memory",
                    "created_at": current_time,
                    "memory_type": "TEMPORAL"
                }
            },
            {
                "id": "negative_memory",
                "score": 0.7,
                "values": [],
                "metadata": {
                    "content": "This is a frustrating problem.",
                    "id": "negative_memory",
                    "created_at": current_time,
                    "memory_type": "TEMPORAL"
                }
            }
        ]

        # Mock the decay weight calculation to return 1.0 (no decay)
        with patch.object(memory_system, 'calculate_decay_weight', return_value=1.0):
            query_vector = np.array([0.1] * 1536)
            results = await memory_system.search_memories(query_vector, top_k=2)

            # Debug print
            print("Debug - Raw results:", results)
            
            # Validate returned memories
            memories = [mem for mem, _ in results]
            assert len(memories) == 2, "Expected two memories to be returned"

            memory_ids = {mem.id for mem in memories}
            assert "positive_memory" in memory_ids, "Expected positive memory to be returned"
            assert "negative_memory" in memory_ids, "Expected negative memory to be returned"

            # Validate scores with detailed error message
            scores = [score for _, score in results]
            expected_scores = [0.9, 0.7]
            
            # Print detailed debug information
            print("Debug - Extracted scores:", scores)
            print("Debug - Expected scores:", expected_scores)
            
            # Compare scores with tolerance
            np.testing.assert_array_almost_equal(
                scores, 
                expected_scores, 
                decimal=6, 
                err_msg="Scores don't match within tolerance"
            )


@pytest.mark.asyncio
@patch("utils.vector_operations.VectorOperations.create_semantic_vector", AsyncMock(return_value=np.random.rand(1536)))
async def test_performance(memory_system, mock_pinecone_service):
    with patch('os.getenv', side_effect=lambda k: {
        'PINECONE_API_KEY': 'dummy',
        'PINECONE_ENVIRONMENT': 'dummy',
        'INDEX_NAME': 'test-index',
        'OPENAI_API_KEY': 'sk-fake'
    }.get(k)):
        # Measure upsertion time
        start_time = time.time()
        text = "This is a sample memory for performance testing."
        metadata = {"id": "performance_test"}
        await memory_system.add_memory(text, metadata)
        end_time = time.time()
        upsertion_time = end_time - start_time
        print(f"Upsertion time: {upsertion_time:.4f} seconds")
        assert upsertion_time < 0.1, "Upsertion took longer than 100ms"

        # Measure retrieval time
        query_vector = np.array([0.1] * 1536)
        start_time = time.time()
        await memory_system.search_memories(query_vector, top_k=5)
        end_time = time.time()
        retrieval_time = end_time - start_time
        print(f"Retrieval time: {retrieval_time:.4f} seconds")
        assert retrieval_time < 0.05, "Retrieval took longer than 5ms"
