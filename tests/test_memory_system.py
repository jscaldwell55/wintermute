import pytest
from backend.core.memory.memory import MemorySystem, MemoryType
from backend.core.memory.models import Memory
from backend.utils.vector_operations import VectorOperations
from backend.utils.pinecone_service import PineconeService
from datetime import datetime
import numpy as np
import os

# Mock PineconeService and VectorOperations for testing purposes
class MockPineconeService:
    def __init__(self, api_key, environment):
        self.memories = {}

    async def upsert_memory(self, memory_id, vector, metadata):
        self.memories[memory_id] = {
            "id": memory_id,
            "values": vector,
            "metadata": metadata,
        }

    async def query_memory(self, query_vector, top_k, filters=None, include_metadata=True):
        results = []
        for mem_id, mem_data in self.memories.items():
            if filters and not self._matches_filters(mem_data["metadata"], filters):
                continue
            similarity = np.dot(query_vector, mem_data["values"])
            results.append(
                {
                    "id": mem_id,
                    "score": similarity,
                    "metadata": mem_data["metadata"],
                }
            )
        results.sort(key=lambda x: x["score"], reverse=True)
        return {"matches": results[:top_k]}

    def _matches_filters(self, metadata, filters):
        for key, value in filters.items():
            if key == "memory_type":
                if metadata.get(key) not in value["$in"]:
                    return False
            elif key == "window_id":
                if metadata.get(key) != value:
                    return False
            elif metadata.get(key) != value:
                return False
        return True

    async def get_all_memories_with_metadata(self):
        return [
            {"id": mem_id, "values": mem_data["values"], "metadata": mem_data["metadata"]}
            for mem_id, mem_data in self.memories.items()
        ]

    async def upsert_batch_memories(self, memories):
        for memory in memories:
            await self.upsert_memory(memory["id"], memory["values"], memory["metadata"])
    
    async def delete_memory(self, memory_id: str):
        if memory_id in self.memories:
            del self.memories[memory_id]

class MockVectorOperations:
    async def create_semantic_vector(self, text):
        # Generate a deterministic vector for testing
        np.random.seed(len(text))
        return np.random.rand(1536).tolist()  # Ensure this is 1536-dimensional

    async def create_combined_q_r_vector(self, query, response):
        return await self.create_semantic_vector(query + response)

@pytest.fixture
def memory_system():
    pinecone_api_key = "dummy_api_key"
    pinecone_environment = "dummy_environment"
    pinecone_service = MockPineconeService(pinecone_api_key, pinecone_environment)
    vector_operations = MockVectorOperations()
    return MemorySystem(pinecone_service, vector_operations)

@pytest.mark.asyncio
async def test_add_episodic_memory(memory_system):
    content = "Test episodic memory content"
    memory_id = await memory_system.add_memory(content, MemoryType.EPISODIC)
    assert memory_id is not None

@pytest.mark.asyncio
async def test_add_semantic_memory(memory_system):
    content = "Test semantic memory content"
    memory_id = await memory_system.add_memory(content, MemoryType.SEMANTIC)
    assert memory_id is not None

@pytest.mark.asyncio
async def test_query_memory(memory_system):
    # Add some memories
    await memory_system.add_memory("Memory 1", MemoryType.EPISODIC, metadata={"window_id": "window1"})
    await memory_system.add_memory("Memory 2", MemoryType.EPISODIC, metadata={"window_id": "window1"})
    await memory_system.add_memory("Memory 3", MemoryType.SEMANTIC)

    # Create a query vector
    vector_operations = MockVectorOperations()
    query_vector = await vector_operations.create_semantic_vector("query")

    # Query episodic memories
    episodic_results = await memory_system.query_memory(
        query_vector, [MemoryType.EPISODIC], window_id="window1"
    )
    assert len(episodic_results) == 2

    # Query semantic memories
    semantic_results = await memory_system.query_memory(
        query_vector, [MemoryType.SEMANTIC]
    )
    assert len(semantic_results) == 1

@pytest.mark.asyncio
async def test_batch_add_memories(memory_system):
    memories = [
        {"id": "batch_mem_1", "content": "Batch memory 1", "metadata": {}},
        {"id": "batch_mem_2", "content": "Batch memory 2", "metadata": {}},
    ]

    # Batch add memories
    await memory_system.batch_embed_and_store(memories)

    # Verify they are added
    vector_operations = MockVectorOperations()
    query_vector = await vector_operations.create_semantic_vector("Batch")
    results = await memory_system.query_memory(query_vector, [MemoryType.EPISODIC, MemoryType.SEMANTIC], top_k=5)

    # Check if any results were returned
    assert len(results) > 0, "No results returned from query_memory"

    # Now, verify that each expected memory ID is present in the results
    retrieved_memory_ids = {memory_tuple[0].id for memory_tuple in results}  # Extract IDs from Memory objects
    for memory in memories:
        assert memory["id"] in retrieved_memory_ids, f"Memory ID {memory['id']} not found in query results"

@pytest.mark.asyncio
async def test_get_memories_by_window_id(memory_system):
    # Add some memories with different window IDs
    await memory_system.add_memory("Memory 1", MemoryType.EPISODIC, metadata={"window_id": "window1"})
    await memory_system.add_memory("Memory 2", MemoryType.EPISODIC, metadata={"window_id": "window1"})
    await memory_system.add_memory("Memory 3", MemoryType.EPISODIC, metadata={"window_id": "window2"})

    # Get memories by window ID
    memories = await memory_system.get_memories_by_window_id("window1")
    assert len(memories) == 2
    assert all(mem.metadata["window_id"] == "window1" for mem in memories)