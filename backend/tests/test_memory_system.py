import pytest
from unittest.mock import MagicMock
import numpy as np
from core.memory.memory import MemorySystem
from core.memory.models import MemoryType, Memory
from utils.vector_operations import VectorOperations
from utils.pinecone_service import PineconeService

@pytest.fixture
def memory_system():
    mock_vector_operations = MagicMock(spec=VectorOperations)
    mock_vector_operations.create_semantic_vector = MagicMock(return_value=np.random.rand(1536))

    mock_pinecone_service = MagicMock(spec=PineconeService)
    mock_pinecone_service.upsert_memory = MagicMock()
    mock_pinecone_service.query_memory = MagicMock(return_value=[
        {
            "id": "mock_id_1",
            "values": np.random.rand(1536).tolist(),
            "metadata": {"content": "Mock memory 1", "created_at": "2024-01-01T00:00:00", "memory_type": "TEMPORAL"},
            "score": 0.95,
        }
    ])
    mock_pinecone_service.delete_memory = MagicMock()

    memory_system = MemorySystem(
        pinecone_api_key="fake_key",
        pinecone_environment="test_env",
        index_name="test_index",
        vector_operations=mock_vector_operations,
    )
    memory_system.pinecone_service = mock_pinecone_service
    return memory_system


@pytest.mark.asyncio
async def test_consolidate_memories_no_memories(memory_system):
    async def mock_get_old_memories(days):
        return []

    memory_system._get_old_memories = mock_get_old_memories
    memory_system.pinecone_service.delete_memory = MagicMock()

    await memory_system.consolidate_memories()
    memory_system.pinecone_service.delete_memory.assert_not_called()
    print("Test Case Passed: Consolidate Memories - No Memories")


@pytest.mark.asyncio
async def test_add_memory_long_content(memory_system):
    long_content = "This is a very long memory content. " * 1000
    metadata = {"category": "test"}

    memory_id = await memory_system.add_memory(long_content, metadata)
    memory_system.vector_operations.create_semantic_vector.assert_called_once_with(long_content)
    memory_system.pinecone_service.upsert_memory.assert_called_once()
    assert memory_id.startswith("mem_")
    print("Test Case Passed: Add Memory - Long Content")


@pytest.mark.asyncio
async def test_add_memory_special_characters_in_metadata(memory_system):
    content = "This is a test memory with special characters in metadata."
    metadata = {"category": "test", "special_chars": "!@#$%^&*()_+|~`<>?/[]{}"}

    memory_id = await memory_system.add_memory(content, metadata)
    memory_system.vector_operations.create_semantic_vector.assert_called_once_with(content)
    memory_system.pinecone_service.upsert_memory.assert_called_once()
    assert memory_id.startswith("mem_")
    print("Test Case Passed: Add Memory - Special Characters in Metadata")


@pytest.mark.asyncio
async def test_pinecone_error_handling(memory_system):
    content = "This is a test memory."
    metadata = {"category": "test"}

    memory_system.pinecone_service.upsert_memory = MagicMock(side_effect=Exception("Pinecone service error"))

    with pytest.raises(Exception, match="Pinecone service error"):
        await memory_system.add_memory(content, metadata)
    print("Test Case Passed: Pinecone Error Handling")
