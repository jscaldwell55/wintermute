"""
Core memory system implementing the direct-to-Pinecone approach.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from api.core.memory.models import Memory, MemoryType
from api.core.vector.vector_operations import VectorOperations
from api.utils.pinecone_service import PineconeService
from api.core.memory.summary import EpisodicSummarizer
from api.core.memory.cache import LRUCacheManager
import logging
import numpy as np
import uuid
import asyncio
from api.utils.config import get_settings
from api.utils.task_queue import task_queue

logger = logging.getLogger(__name__)

class MemorySystem:
    """Core memory system using Pinecone for storage."""

    def __init__(
        self,
        pinecone_service: PineconeService,
        vector_operations: VectorOperations,
        context_window,
        episodic_summarizer: EpisodicSummarizer,
        cache_manager: LRUCacheManager,
    ):
        self.settings = get_settings()
        self.pinecone_service = pinecone_service
        self.vector_operations = vector_operations
        self.context_window = context_window
        self.episodic_summarizer = episodic_summarizer
        self.cache_manager = cache_manager
        self.retention_window = timedelta(days=self.settings.delete_threshold_days)
        self.last_dream_time = datetime.utcnow() - timedelta(days=1)

    async def add_memory(
        self,
        content: str,
        memory_type: MemoryType,
        metadata: Optional[Dict[str, Any]] = None,
        summary: Optional[str] = None,
        summary_embedding: Optional[List[float]] = None,
    ) -> str:
        """
        Adds a memory to the system, storing it directly in Pinecone.

        Args:
            content: The content of the memory.
            memory_type: The type of memory (EPISODIC or SEMANTIC).
            metadata: Optional metadata for the memory.
            summary: Optional summary of the memory.
            summary_embedding: Optional embedding of the summary.

        Returns:
            The unique ID of the added memory.
        """
        try:
            if not content.strip():
                raise ValueError("Memory content cannot be empty.")
            if metadata is not None and not isinstance(metadata, dict):
                raise ValueError("Metadata must be a dictionary.")

            # Generate semantic vector
            semantic_vector = await self.vector_operations.create_semantic_vector(content)
            if len(semantic_vector) != 1536:
                raise ValueError("Generated semantic vector has incorrect dimensionality.")

            # Generate a unique memory ID
            memory_id = f"mem_{uuid.uuid4()}"

            # Prepare metadata
            metadata = metadata or {}
            current_time = datetime.now().isoformat()

            # Update metadata with required fields
            full_metadata = {
                "content": content,
                "summary": summary,  # Add the summary to the metadata
                "summary_embedding": summary_embedding,  # Add the summary embedding
                "created_at": current_time,
                "memory_type": memory_type.value,
                **(metadata),
            }

            # Create memory object
            memory = Memory(
                id=memory_id,
                memory_type=memory_type,
                content=content,
                semantic_vector=semantic_vector,
                metadata=full_metadata,
                created_at=current_time,
            )

            # Log and upsert
            logger.info(f"Upserting memory with ID: {memory_id} and metadata: {full_metadata}")

            # Upsert to Pinecone
            await self.pinecone_service.upsert_memory(
                memory_id=memory.id,
                vector=semantic_vector,
                metadata=memory.metadata,
            )

            return memory.id

        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            raise RuntimeError(f"Error adding memory: {e}")

    async def add_interaction_memory(self, user_query: str, gpt_response: str):
        """
        Adds a user query and its corresponding GPT response as an interaction memory (episodic).
        """
        try:
            # Create combined vector for Q/R pair
            combined_vector = await self.vector_operations.create_combined_q_r_vector(
                user_query, gpt_response
            )

            # Generate a unique task ID for the memory
            task_id = f"mem_{uuid.uuid4()}"

            # Define a helper function to add memory
            async def add_memory_task(content, memory_type, metadata, window_id, combined_vector):
                print("add_memory_task called with:", content, memory_type, metadata, window_id, combined_vector)
                task_id = f"mem_{uuid.uuid4()}"
                memory = Memory(
                    id=task_id,
                    memory_type=memory_type,
                    content=content,
                    metadata=metadata,
                    created_at=datetime.now().isoformat(),
                    window_id=window_id,
                    semantic_vector=combined_vector
                )
                await self.add_memory(
                    content=content,
                    memory_type=memory_type,
                    metadata=metadata,
                    window_id=window_id,
                    task_id=task_id
            )

            # Enqueue the memory addition task with retries
            task_queue.enqueue(
                add_memory_task,
                content=f"Q: {user_query}\nA: {gpt_response}",
                memory_type=MemoryType.EPISODIC,
                metadata={"interaction": True, "window_id": window_id},
                window_id=window_id,
                combined_vector=combined_vector,
                retries=self.settings.summary_retries,
                retry_delay=self.settings.summary_retry_delay,
            )

            logger.info(
                f"Interaction memory added to queue for window {window_id} with task ID {task_id}."
            )

        except Exception as e:
            logger.error(f"Error adding interaction memory: {e}")
            raise RuntimeError(f"Error adding interaction memory: {e}")

    async def query_memory(
        self,
        query_vector: List[float],
        top_k: int = 5,
        window_id: Optional[str] = None,
        query_types: Optional[List[MemoryType]] = None,
    ) -> List[Tuple[Memory, float]]:
        """
        Queries the memory system for memories of specified types.

        Args:
            query_vector: The query vector or text query.
            top_k: The number of top results to return.
            window_id: Optional window ID to filter episodic memories.
            query_types: Optional list of MemoryTypes to query.

        Returns:
            A list of tuples, where each tuple contains a Memory object and its similarity score.
        """
        try:
            # Convert query text to vector if needed
            if isinstance(query_vector, str):
                query_vector = await self.vector_operations.create_semantic_vector(query_vector)

            # Validate query_vector
            if not isinstance(query_vector, list) or not all(isinstance(x, float) for x in query_vector):
                raise ValueError("query_vector must be a list of floats.")
            if len(query_vector) != 1536:
                raise ValueError("Query vector dimensionality must be 1536.")

            # Validate query_types
            if query_types is not None:
                if not all(isinstance(qt, MemoryType) for qt in query_types):
                    raise ValueError("query_types must be a list of MemoryType enums or None.")

            # Validate top_k
            if not isinstance(top_k, int) or top_k <= 0:
                raise ValueError("top_k must be a positive integer.")

            # Build metadata filter
            filters = self.build_metadata_filter(query_types=query_types, window_id=window_id)

            # First, check the LRU cache
            cached_summaries = self.cache_manager.get_relevant_memories(query_vector)
            if cached_summaries:
                logger.info(f"Found {len(cached_summaries)} relevant summaries in LRU cache.")
                # Retrieve full memories from Pinecone based on cached summaries
                full_memories = await asyncio.gather(
                    *[self.pinecone_service.get_memory_by_id(summary['memory_id']) for summary in cached_summaries]
                )
                # Combine memories with their scores from the cache
                memories_with_scores = [
                    (memory, summary['score']) for memory, summary in zip(full_memories, cached_summaries) if memory is not None
                ]
                return memories_with_scores

            # If not in cache, query Pinecone
            results = await self.pinecone_service.query_memory(query_vector, top_k=top_k, filter=filters)

            # Process results into Memory objects with scores
            memories_with_scores = []
            for result in results:
                try:
                    memory = self._create_memory_from_result(result)

                    # Add the memory to the cache
                    self.cache_manager.store_memory(memory)

                    memories_with_scores.append((memory, result["score"]))
                except (ValueError, KeyError) as e:
                    logger.warning(f"Error creating memory from result {result}: {e}")

            return memories_with_scores

        except Exception as e:
            logger.error(f"Error querying memory: {e}")
            raise

    async def batch_embed_and_store(self, memories: List[Dict[str, Any]]):
        """
        Batch embeds and stores memories in Pinecone.

        Args:
            memories: A list of dictionaries, each representing a memory with 'content' and optional 'metadata'.
        """
        try:
            # Extract the IDs and content from each memory for embedding and upsertion
            memory_ids = [mem["id"] for mem in memories]
            memory_contents = [mem["content"] for mem in memories]
            memory_metadatas = [mem.get("metadata", {}) for mem in memories]

            # Create semantic vectors for all memories in the batch
            semantic_vectors = await asyncio.gather(
                *[
                    self.vector_operations.create_semantic_vector(content)
                    for content in memory_contents
                ]
            )

            # Prepare the vectors for upsertion
            upsert_data = [
                {
                    "id": memory_id,
                    "values": vector,
                    "metadata": metadata,
                }
                for memory_id, vector, metadata in zip(
                    memory_ids, semantic_vectors, memory_metadatas
                )
            ]

            # Upsert the batch of memories into Pinecone
            await self.pinecone_service.upsert_batch_memories(upsert_data)

            logger.info(f"Batch upserted {len(memories)} memories to Pinecone.")

        except Exception as e:
            logger.error(f"Error in batch embedding and storing memories: {e}")
            raise RuntimeError(f"Error in batch embedding and storing memories: {e}")

    async def generate_prompt(
        self, query: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Retrieves memories, formats them into a prompt template, and returns the full prompt.

        Args:
            query: The user's query.
            template_type: Type of prompt template (e.g., 'RESEARCH_CONTEXT_TEMPLATE').
            metadata: Additional metadata required by the template.

        Returns:
            A formatted prompt.
        """
        query_vector = await self.vector_operations.create_semantic_vector(query)

        # Query both episodic and semantic memories
        episodic_memories = await self.query_memory(
            query_vector=query_vector, query_types=[MemoryType.EPISODIC], top_k=5
        )
        semantic_memories = await self.query_memory(
            query_vector=query_vector, query_types=[MemoryType.SEMANTIC], top_k=5
        )

        # Combine and rank the memories
        all_memories = episodic_memories + semantic_memories
        ranked_memories = self.rank_and_filter_results(all_memories, top_k=10)

        # Format the memories for the prompt template
        retrieved_context = "\n".join(
            [f"- {memory.content}" for memory, _ in ranked_memories]
        )

        # Format the prompt using only MASTER_TEMPLATE
        prompt = format_prompt(
            MASTER_TEMPLATE,
            retrieved_context=retrieved_context,
            user_query=query,
            **(metadata or {}),
        )

        return prompt

    def build_metadata_filter(
        self, query_types: List[MemoryType], window_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Builds a metadata filter for Pinecone queries based on memory type and optionally window ID.

        Args:
            query_types: A list of MemoryTypes to include in the query.
            window_id: Optional window ID to filter episodic memories.

        Returns:
            A dictionary representing the metadata filter for Pinecone.
        """
        filter_dict = {}

        # Memory type filter
        if query_types:
            filter_dict["memory_type"] = {"$in": [mt.value for mt in query_types]}

        # Window ID filter (for episodic memories)
        if window_id:
            filter_dict["window_id"] = window_id

        return filter_dict

    def rank_and_filter_results(
        self, results: List[Tuple[Memory, float]], top_k: int
    ) -> List[Tuple[Memory, float]]:
        """Ranks and filters memory query results."""
        return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

    async def _get_old_memories(self, days: int) -> List[Memory]:
        """
        Retrieves memories older than a specified number of days.

        Args:
            days: The number of days to look back.

        Returns:
            A list of Memory objects that are older than the specified number of days.
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        try:
            all_memories = await self.pinecone_service.get_all_memories_with_metadata()
            old_memories = [
                self._create_memory_from_result(mem)
                for mem in all_memories
                if "created_at" in mem["metadata"]
                and datetime.fromisoformat(mem["metadata"]["created_at"]) < cutoff_time
            ]
            return old_memories
        except Exception as e:
            logger.error(f"Error fetching old memories: {e}")
            return []

    def merge_metadata(self, memories: List[Memory]) -> Dict:
        """Merges metadata from multiple memories (not used for now)."""
        merged_metadata = {"content": " ".join([mem.content for mem in memories])}
        for memory in memories:
            for key, value in memory.metadata.items():
                if key in merged_metadata:
                    merged_metadata[key] += (
                        "; " + value if isinstance(value, str) else ""
                    )
                else:
                    merged_metadata[key] = value
        return merged_metadata

    def _create_memory_from_result(self, result: Dict[str, Any]) -> Memory:
        """Creates a Memory object from a Pinecone query result."""
        metadata = result.get("metadata", {})
        memory_type_str = metadata.get("memory_type")

        # Fallback to a default MemoryType if the value is not recognized
        try:
            memory_type = MemoryType(memory_type_str)
        except ValueError:
            logger.warning(
                f"Unrecognized memory type '{memory_type_str}'. Using default type."
            )
            memory_type = MemoryType.EPISODIC  # Or another default value

        return Memory(
            id=result.get("id", ""),
            content=metadata.get("content", ""),
            created_at=metadata.get("created_at", datetime.now().isoformat()),
            memory_type=memory_type,
            semantic_vector=result.get("values", []),
            metadata=metadata,
            window_id=metadata.get('window_id')
        )

    async def get_memories_by_window_id(self, window_id: str) -> List[Memory]:
        """Retrieves episodic memories associated with a specific window ID."""
        try:
            # Query Pinecone using the window_id in the metadata filter
            episodic_results = await self.query_memory(
                query_vector=np.zeros(1536).tolist(),  # Dummy vector, not used in filtering
                query_types=[MemoryType.EPISODIC],  # Filter for episodic memories
                top_k=100,
                window_id=window_id,
            )
            # Convert results to Memory objects
            memories = [memory for memory, _ in episodic_results]
            return memories

        except Exception as e:
            logger.error(f"Error getting memories by window ID: {e}")
            return []

    def _update_last_dream_time(self, timestamp: datetime):
        """Updates the last dream time."""
        self.last_dream_time = timestamp

    async def shutdown(self):
        """Handles any cleanup tasks when shutting down."""
        logger.info("Shutting down MemorySystem...")
        # Add any cleanup logic here, like closing connections
        if task_queue:
            task_queue.shutdown()
        if self.pinecone_service:
            await self.pinecone_service.close_connections()

# Example usage (you'll need to adapt this to your specific setup):
# from api.utils.config import settings
# from api.core.vector.vector_operations import VectorOperations
# from api.utils.pinecone_service import PineconeService

# Load settings and initialize components
# settings = get_settings()
# vector_operations = VectorOperations()
# pinecone_service = PineconeService(api_key=settings.PINECONE_API_KEY, environment=settings.PINECONE_ENVIRONMENT, index_name=settings.INDEX_NAME)
# memory_system = MemorySystem(pinecone_service, vector_operations)

# Example usage
# async def main():
#     await pinecone_service.initialize_index()  # Initialize Pinecone index
#     memory_id = await memory_system.add_memory("This is a test memory.", MemoryType.EPISODIC)
#     print(f"Memory added with ID: {memory_id}")

#     query_vector = [0.1, 0.2, 0.3, ...]  # Replace with actual query vector
#     similar_memories = await memory_system.query_memory(query_vector)
#     print(f"Similar memories: {similar_memories}")

#     await memory_system.shutdown()  # Close connections when done

# if __name__ == "__main__":
#     asyncio.run(main())