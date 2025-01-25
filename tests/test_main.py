import pytest
from fastapi.testclient import TestClient
from backend.main import app, initialize_components
from unittest.mock import patch, MagicMock
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from backend.core.memory.models import Memory, MemoryType
from httpx import AsyncClient
from typing import List, Dict, Optional

# Mock Classes
class MockContextWindow:
    def __init__(self):
        self.window_id = "test_window_id"
        self.template = ""
        self.current_token_count = 0
        self.total_tokens = 50
        self.reserved_tokens = 20
        self.reset_window = MagicMock()

    def is_full(self):
        available = self.total_tokens - self.reserved_tokens
        return self.current_token_count >= available

    def add_q_r_pair(self, query, response):
        self.current_token_count += (len(query.split()) + len(response.split())) * 2
        return True

    def reset_window(self):
        self.window_id = f"window_{uuid.uuid4()}"
        self.current_token_count = 0
        self.template = ""

    def generate_template(self, summary):
        self.template = summary

    async def generate_window_summary(self, memories, window_id):
        return "Mock summary for window " + window_id

class MockMemorySystem:
    def __init__(self):
        self.memories = []
        self.window_memories = {}

    async def add_memory(self, content, memory_type, metadata=None, window_id=None):
        return "mock_memory_id"

    async def add_interaction_memory(self, user_query: str, gpt_response: str, window_id: Optional[str] = None):
        return "mock_memory_id"

    async def query_memory(self, query_vector, query_types, top_k=5, window_id=None):
        # Always return at least 2 memories
        memory1 = Memory(
            id="mock_memory_1",
            content="Test memory 1",
            created_at=datetime.now().isoformat(),
            metadata={},
            memory_type=MemoryType.EPISODIC,
            semantic_vector=[0.1] * 1536
        )
        memory2 = Memory(
            id="mock_memory_2",
            content="Test memory 2",
            created_at=datetime.now().isoformat(),
            metadata={},
            memory_type=MemoryType.EPISODIC,
            semantic_vector=[0.1] * 1536
        )
        return [(memory1, 0.9), (memory2, 0.8)]

    async def get_memories_by_window_id(self, window_id):
        return [
            Memory(
                id=f"mock_memory_{i}",
                content=f"Window memory {i}",
                created_at=datetime.now().isoformat(),
                metadata={"window_id": window_id},
                memory_type=MemoryType.EPISODIC,
                semantic_vector=[0.1] * 1536
            ) for i in range(2)
        ]

class MockVectorOperations:
    async def create_semantic_vector(self, text: str) -> List[float]:
        return [0.1] * 1536

class MockLLMService:
    async def generate_gpt_response_async(self, prompt: str, model: str = "gpt-3.5-turbo", 
                                        temperature: float = 0.7, max_tokens: int = 150) -> str:
        return "Mocked GPT response"

# Fixtures
@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def client():
    # Mock the initialize_components function
    with patch('backend.main.initialize_components') as mock_initialize:
        async def mock_init():
            global memory_system, context_window, vector_operations
            memory_system = MockMemorySystem()
            context_window = MockContextWindow()
            vector_operations = MockVectorOperations()

        mock_initialize.side_effect = mock_init
        await mock_initialize()  # Ensure initialize_components is awaited

        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client

@pytest.mark.asyncio
async def test_root(client):
    response = await client.get("/")
    assert response.status_code == 200
    response_json = response.json()
    assert response_json == {"message": "Backend is running. Use the appropriate API endpoints."}

@pytest.mark.asyncio
async def test_add_memory(client):
    memory_data = {
        "content": "Test memory content",
        "metadata": {},
        "memory_type": "episodic"
    }
    response = await client.post("/memories", json=memory_data)
    assert response.status_code == 200
    response_json = response.json()
    assert "message" in response_json
    assert "id" in response_json

@pytest.mark.asyncio
async def test_query_memory_with_full_window(client):
    mock_context_window = MockContextWindow()
    mock_context_window.current_token_count = 35  # Make sure it's full

    with patch('backend.main.context_window', mock_context_window), \
         patch('backend.main.memory_system', MockMemorySystem()), \
         patch('backend.main.vector_operations', MockVectorOperations()), \
         patch('backend.utils.llm_service.generate_gpt_response_async', return_value="Mocked GPT response"):

        query_data = {"prompt": "Test query", "top_k": 5}
        response = await client.post("/query", json=query_data)
        
        assert response.status_code == 200
        response_json = response.json()
        assert "query" in response_json
        assert "response" in response_json

@pytest.mark.asyncio
async def test_query_memory_with_empty_window(client):
    mock_context_window = MockContextWindow()
    mock_context_window.current_token_count = 25  # Make sure it's not full

    with patch('backend.main.context_window', mock_context_window), \
         patch('backend.main.memory_system', MockMemorySystem()), \
         patch('backend.main.vector_operations', MockVectorOperations()), \
         patch('backend.utils.llm_service.generate_gpt_response_async', return_value="Mocked GPT response"):

        query_data = {"prompt": "Test query", "top_k": 5}
        response = await client.post("/query", json=query_data)
        
        assert response.status_code == 200
        response_json = response.json()
        assert "query" in response_json
        assert "response" in response_json