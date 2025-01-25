import pytest
from backend.core.memory.memory import MemorySystem, MemoryType
from backend.core.memory.models import Memory
from backend.utils.vector_operations import VectorOperations
from backend.utils.pinecone_service import PineconeService
from backend.utils.dream_sequence import DreamSequence
from backend.core.evaluation import MemoryEvaluation
from datetime import datetime, timedelta
import numpy as np
import os
import asyncio
from unittest.mock import patch, MagicMock

# Mock PineconeService, VectorOperations, and MemoryEvaluation for testing purposes
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
    
    async def delete_memory(self, memory_id: str):
        if memory_id in self.memories:
            del self.memories[memory_id]

class MockVectorOperations:
    async def create_semantic_vector(self, text):
        # Generate a deterministic vector for testing based on the content
        vector = self.generate_vector(text)
        print(f"Generated vector for '{text}': {vector}")
        return vector

    async def create_combined_q_r_vector(self, query, response):
        combined_text = query + response
        vector = self.generate_vector(combined_text)
        print(f"Generated vector for combined Q/R: {vector}")
        return vector

    def generate_vector(self, text):
        # A simple hash function to generate a deterministic vector based on text content
        hashed = abs(hash(text)) % (10 ** 8)  # Simple hash to a number
        vector = [(hashed / 10 ** 8) + (i / 1536) for i in range(1536)]  # Scale and distribute
        return vector
    
    def cosine_similarity(self, vector1, vector2):
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if magnitude == 0:
            return 0.0
        return dot_product / magnitude

class MockMemoryEvaluation:
    async def evaluate_semantic_memory(self, semantic_memory, source_episodic_memories):
        return {
            "relevance": 0.9,
            "coherence": 4.5,
            "conciseness": 10,
            "consistency": 0.8,
            "novelty": 0.7,
            "utility": 0.6
        }

@pytest.fixture
def dream_sequence():
    pinecone_api_key = "dummy_api_key"
    pinecone_environment = "dummy_environment"
    pinecone_service = MockPineconeService(pinecone_api_key, pinecone_environment)
    vector_operations = MockVectorOperations()
    memory_system = MemorySystem(pinecone_service, vector_operations)
    evaluation_module = MockMemoryEvaluation()
    return DreamSequence(memory_system, vector_operations, evaluation_module)

@pytest.mark.asyncio
async def test_get_recent_episodic_memories(dream_sequence):
    # Add some episodic memories with different timestamps
    memory_system = dream_sequence.memory_system
    await memory_system.add_memory("Old memory 1", MemoryType.EPISODIC, metadata={"created_at": (datetime.now() - timedelta(days=2)).isoformat()})
    await memory_system.add_memory("Recent memory 1", MemoryType.EPISODIC, metadata={"created_at": datetime.now().isoformat()})
    await memory_system.add_memory("Recent memory 2", MemoryType.EPISODIC, metadata={"created_at": datetime.now().isoformat()})

    # Update last dream time to yesterday
    dream_sequence._update_last_dream_time(datetime.now() - timedelta(days=1))

    # Get recent episodic memories
    recent_memories = await dream_sequence._get_recent_episodic_memories()

    # Assert that only recent memories are retrieved
    assert len(recent_memories) == 2
    assert all(mem.content.startswith("Recent") for mem in recent_memories)

@pytest.mark.asyncio
async def test_identify_semantic_patterns(dream_sequence):
    # Add some episodic memories
    memory_system = dream_sequence.memory_system
    # Ensure we have enough diverse memories to form at least 2 clusters
    await memory_system.add_memory("This is about topic A", MemoryType.EPISODIC, metadata={"created_at": datetime.now().isoformat()})
    await memory_system.add_memory("Another memory about topic A", MemoryType.EPISODIC, metadata={"created_at": datetime.now().isoformat()})
    await memory_system.add_memory("Yet another memory about topic A", MemoryType.EPISODIC, metadata={"created_at": datetime.now().isoformat()})
    await memory_system.add_memory("This is about topic B", MemoryType.EPISODIC, metadata={"created_at": datetime.now().isoformat()})
    await memory_system.add_memory("More on topic B", MemoryType.EPISODIC, metadata={"created_at": datetime.now().isoformat()})
    await memory_system.add_memory("And yet more on topic B", MemoryType.EPISODIC, metadata={"created_at": datetime.now().isoformat()})
    await memory_system.add_memory("Something different about topic C", MemoryType.EPISODIC, metadata={"created_at": datetime.now().isoformat()})
    await memory_system.add_memory("And another thing about topic C", MemoryType.EPISODIC, metadata={"created_at": datetime.now().isoformat()})
    await memory_system.add_memory("Topic D is interesting", MemoryType.EPISODIC, metadata={"created_at": datetime.now().isoformat()})
    await memory_system.add_memory("Additional details on topic D", MemoryType.EPISODIC, metadata={"created_at": datetime.now().isoformat()})
    await memory_system.add_memory("Final thoughts on topic D", MemoryType.EPISODIC, metadata={"created_at": datetime.now().isoformat()})
    await memory_system.add_memory("A new perspective on topic E", MemoryType.EPISODIC, metadata={"created_at": datetime.now().isoformat()})
    await memory_system.add_memory("Elaboration on topic E", MemoryType.EPISODIC, metadata={"created_at": datetime.now().isoformat()})
    await memory_system.add_memory("Further insights into topic E", MemoryType.EPISODIC, metadata={"created_at": datetime.now().isoformat()})
    await memory_system.add_memory("Reflections on topic F", MemoryType.EPISODIC, metadata={"created_at": datetime.now().isoformat()})
    await memory_system.add_memory("A deep dive into topic F", MemoryType.EPISODIC, metadata={"created_at": datetime.now().isoformat()})

    # Get recent episodic memories
    recent_memories = await dream_sequence._get_recent_episodic_memories()

    # Identify patterns
    patterns = await dream_sequence._identify_semantic_patterns(recent_memories)

    # Assert that patterns are correctly identified
    assert len(patterns) >= 2  # Assuming at least two clusters are formed

@pytest.mark.asyncio
async def test_create_semantic_memory(dream_sequence):
    # Add some episodic memories
    memory_system = dream_sequence.memory_system
    await memory_system.add_memory("This is episodic memory 1", MemoryType.EPISODIC, metadata={"created_at": datetime.now().isoformat()})
    await memory_system.add_memory("This is episodic memory 2", MemoryType.EPISODIC, metadata={"created_at": datetime.now().isoformat()})

    # Get recent episodic memories
    recent_memories = await dream_sequence._get_recent_episodic_memories()

    # Create a pattern from the memories
    pattern = {"cluster_memories": recent_memories}

    # Create a semantic memory from the pattern
    semantic_memory = await dream_sequence._create_semantic_memory(pattern)

    # Assert that the semantic memory is created correctly
    assert semantic_memory.memory_type == MemoryType.SEMANTIC
    assert semantic_memory.content is not None  # Assuming content is generated
    assert semantic_memory.metadata["source_episodic_memories"] == [mem.id for mem in recent_memories]

@pytest.mark.asyncio
async def test_consolidate_memories(dream_sequence):
    # Add some old episodic memories
    memory_system = dream_sequence.memory_system
    old_memories = []
    for i in range(5):
        memory_id = await memory_system.add_memory(
            f"Old episodic memory {i}",
            MemoryType.EPISODIC,
            metadata={"created_at": (datetime.now() - timedelta(days=8)).isoformat()}
        )
        old_memories.append(memory_id)

    # Add a new semantic memory to avoid an error
    await memory_system.add_memory(
        "new semantic memory",
        MemoryType.SEMANTIC,
        metadata={"created_at": datetime.now().isoformat()}
    )

    # Call consolidate_memories
    await dream_sequence.consolidate_memories()

    # Verify that old episodic memories are deleted
    all_memories = await memory_system.pinecone_service.get_all_memories_with_metadata()
    all_memory_ids = {mem["id"] for mem in all_memories}
    for mem_id in old_memories:
        assert mem_id not in all_memory_ids, f"Memory {mem_id} should have been deleted"

# Mock the pipeline used in DreamSequence for summarization
class MockSummarizationPipeline:
    def __call__(self, text, max_length, min_length, do_sample):
        # Simulate summarization by returning a predefined summary
        return [{"summary_text": "This is a mock summary."}]

@pytest.mark.asyncio
async def test_generate_semantic_content(dream_sequence):
    # Mock the summarization pipeline
    mock_pipeline = MockSummarizationPipeline()
    dream_sequence.summarization_pipeline = mock_pipeline

    # Create some test memories
    memories = [
        Memory(id="1", content="Test memory content 1.", memory_type=MemoryType.EPISODIC, semantic_vector=[0.1]*1536, metadata={}, created_at="2025-01-24T21:15:56.357048"),
        Memory(id="2", content="Test memory content 2.", memory_type=MemoryType.EPISODIC, semantic_vector=[0.2]*1536, metadata={}, created_at="2025-01-24T21:15:56.357048"),
        Memory(id="3", content="Test memory content 3.", memory_type=MemoryType.EPISODIC, semantic_vector=[0.3]*1536, metadata={}, created_at="2025-01-24T21:15:56.357048")
    ]

    # Call the _generate_semantic_content method
    summary = await dream_sequence._generate_semantic_content(memories)

    # Assert that the summary is correct
    assert summary == "This is a mock summary."