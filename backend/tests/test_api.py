import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from main import app
import numpy as np

# Mock response for OpenAI embeddings
class MockEmbeddingResponse:
    def __init__(self):
        self.data = [
            MagicMock(embedding=[0.1] * 1536)  # Create a mock embedding of correct size
        ]

class MockOpenAI:
    def __init__(self):
        self.embeddings = MagicMock()
        self.embeddings.create = MagicMock(return_value=MockEmbeddingResponse())

# Create mock classes for Pinecone dependencies
class MockPineconeIndex:
    def __init__(self):
        self.upsert = MagicMock()
        self.query = MagicMock(return_value={'matches': []})
        self.delete = MagicMock()

class MockPinecone:
    def __init__(self):
        self.list_indexes = MagicMock(return_value=MagicMock(names=lambda: ['wintermute']))
        self.create_index = MagicMock()

@pytest.fixture(autouse=True)
def mock_dependencies():
    # Mock both OpenAI and Pinecone
    with patch('openai.OpenAI', return_value=MockOpenAI()), \
         patch('utils.pinecone_service.Pinecone', return_value=MockPinecone()), \
         patch('utils.pinecone_service.Index', return_value=MockPineconeIndex()), \
         patch.dict('os.environ', {'OPENAI_API_KEY': 'mock-key', 'PINECONE_API_KEY': 'mock-key'}):
        yield

@pytest.fixture
def mock_memory_system():
    mock_system = AsyncMock()
    mock_system.add_memory = AsyncMock(return_value="mock_id")
    mock_system.vector_operations = AsyncMock()
    # Create a real numpy array for the vector
    mock_system.vector_operations.create_semantic_vector = AsyncMock(
        return_value=np.array([0.1] * 1536)
    )
    mock_system.search_memories = AsyncMock(return_value=[
        (AsyncMock(id="mock_id", content="Mock content", metadata={"category": "test"}), 0.95)
    ])
    mock_system.consolidate_memories = AsyncMock(return_value=None)
    return mock_system

@pytest.fixture
def test_client(mock_memory_system):
    # Mock both the memory system and its initialization
    with patch('main.memory_system', mock_memory_system), \
         patch('main.initialize_memory_system', AsyncMock(return_value=mock_memory_system)):
        with TestClient(app) as client:
            yield client

def test_add_memory(test_client, mock_memory_system):
    response = test_client.post(
        "/memories",
        json={"content": "This is a test memory.", "metadata": {"category": "test"}},
    )
    assert response.status_code == 200
    assert response.json()["id"] == "mock_id"
    mock_memory_system.add_memory.assert_called_once_with(
        "This is a test memory.",
        {"category": "test"}
    )

def test_query_memory(test_client, mock_memory_system):
    response = test_client.post(
        "/query",
        json={"prompt": "Mock content", "top_k": 1},
    )
    assert response.status_code == 200
    memories = response.json()["memories"]
    assert len(memories) > 0
    assert memories[0]["id"] == "mock_id"
    mock_memory_system.vector_operations.create_semantic_vector.assert_called_once()
    mock_memory_system.search_memories.assert_called_once()

def test_consolidate_memories(test_client, mock_memory_system):
    response = test_client.post("/consolidate")
    assert response.status_code == 200
    assert response.json()["message"] == "Memory consolidation completed"
    mock_memory_system.consolidate_memories.assert_called_once()