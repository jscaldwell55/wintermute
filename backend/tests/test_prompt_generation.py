import pytest
import numpy as np
from unittest.mock import AsyncMock
from core.memory.memory import MemorySystem
from utils.vector_operations import VectorOperations
from utils.pinecone_service import PineconeService

@pytest.mark.asyncio
async def test_generate_prompt():
    # Mock dependencies
    mock_vector_operations = VectorOperations()
    mock_vector_operations.create_semantic_vector = AsyncMock(return_value=np.random.rand(1536))

    mock_pinecone_service = PineconeService("dummy_key", "dummy_env")
    mock_pinecone_service.query_memory = AsyncMock(return_value=[
        {
            "id": "test_memory_1",
            "metadata": {
                "content": "First relevant memory content.",
                "created_at": "2025-01-10T12:00:00",
                "memory_type": "TEMPORAL"
            },
            "values": [0.1] * 1536,
            "score": 0.9
        },
        {
            "id": "test_memory_2",
            "metadata": {
                "content": "Second relevant memory content.",
                "created_at": "2025-01-12T12:00:00",
                "memory_type": "TEMPORAL"
            },
            "values": [0.2] * 1536,
            "score": 0.8
        }
    ])

    # Create MemorySystem instance with mocked dependencies
    memory_system = MemorySystem(
        pinecone_api_key="dummy_key",
        pinecone_environment="dummy_env",
        vector_operations=mock_vector_operations
    )
    memory_system.pinecone_service = mock_pinecone_service

    # Define test query and metadata
    user_query = "What insights can you provide about recent projects?"
    metadata = {
        "project_phase": "Discovery",
        "research_domain": "AI",
        "methodology_focus": "Neural networks"
    }

    # Call the function to generate the prompt
    combined_prompt = await memory_system.generate_prompt(
        query=user_query,
        template_type="research_context",
        metadata=metadata
    )

    # Validate the output
    assert "Context:" in combined_prompt
    assert "User Query:" in combined_prompt
    assert "What insights can you provide about recent projects?" in combined_prompt
    assert "First relevant memory content." in combined_prompt
    assert "Second relevant memory content." in combined_prompt
    assert "Neural networks" in combined_prompt
