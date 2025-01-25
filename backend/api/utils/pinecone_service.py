from core.memory.models import MemoryType, Memory
import os
import logging
from pinecone import Pinecone, ServerlessSpec
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import project_root.backend.api.utils.config as config
from tenacity import retry, stop_after_attempt, wait_fixed

logger = logging.getLogger(__name__)

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def initialize_pinecone(api_key: str, environment: str, index_name: str, dimension: int = 1536):
    """
    Initializes Pinecone and ensures the index exists.

    Args:
        api_key (str): Pinecone API key.
        environment (str): Environment for Pinecone (e.g., "us-west1-gcp").
        index_name (str): Name of the Pinecone index to use or create.
        dimension (int): Dimension of the index vectors. Default is 1536.

    Returns:
        Pinecone.Index: The initialized Pinecone index.
    """
    try:
        # Create a Pinecone instance
        pinecone = Pinecone(api_key=api_key)

        # Check if the index exists
        if index_name not in pinecone.list_indexes().names():
            logger.info(f"Creating index: {index_name}")
            pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",  # Use cosine similarity (or 'euclidean' if preferred)
                spec=ServerlessSpec(cloud="aws", region=environment)
            )

        # Return the initialized index
        index = pinecone.Index(index_name)
        logger.info(f"Pinecone index '{index_name}' initialized successfully.")
        return index

    except Exception as e:
        logger.error(f"Error initializing Pinecone: {e}")
        raise RuntimeError(f"Error initializing Pinecone: {e}")

class PineconeService:
    def __init__(self, api_key: str, environment: str):
        self.api_key = api_key
        self.environment = environment
        self.index_name = config.PINECONE_INDEX_NAME
        if not self.index_name:
            raise RuntimeError("INDEX_NAME environment variable is not set.")
        self.index = None  # Initialize index as None

    async def initialize_index(self):
        """Initializes the Pinecone index asynchronously."""
        self.index = await initialize_pinecone(api_key=self.api_key, environment=self.environment, index_name=self.index_name)

    async def upsert_memory(self, memory_id: str, vector: list, metadata: dict):
        """
        Upserts a memory vector into Pinecone.
        """
        try:
            if not self.index:
                await self.initialize_index()
            if not memory_id or not isinstance(memory_id, str):
                raise ValueError("memory_id must be a non-empty string")
            if not isinstance(vector, (list, np.ndarray)) or len(vector) != 1536:
                raise ValueError("vector must be a list or numpy array with 1536 dimensions")
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be a dictionary")

            vector_list = vector.tolist() if isinstance(vector, np.ndarray) else vector
            upsert_data = {
                "id": memory_id,
                "values": vector_list,
                "metadata": metadata,
            }

            logger.info(f"Upserting memory ID {memory_id} with metadata: {metadata}")
            response = self.index.upsert(vectors=[upsert_data])
            logger.debug(f"Successfully upserted memory ID {memory_id}")
        except Exception as e:
            logger.error(f"Error upserting memory: {str(e)}")
            raise RuntimeError(f"Error upserting memory: {str(e)}")

    async def upsert_batch_memories(self, memories: List[Dict[str, Any]]):
        """
        Upserts a batch of memory vectors into Pinecone.

        Args:
            memories: A list of dictionaries, each representing a memory with 'id', 'values', and 'metadata'.
        """
        try:
            if not self.index:
                await self.initialize_index()

            # Validate the batch of memories
            for memory in memories:
                if not isinstance(memory, dict):
                    raise ValueError("Each memory in the batch must be a dictionary.")
                if not memory.get("id") or not isinstance(memory["id"], str):
                    raise ValueError("Each memory must have an 'id' which is a non-empty string.")
                if (
                    not memory.get("values")
                    or not isinstance(memory["values"], (list, np.ndarray))
                    or len(memory["values"]) != 1536
                ):
                    raise ValueError(
                        "Each memory must have 'values' which is a list or numpy array with 1536 dimensions."
                    )
                if not isinstance(memory.get("metadata", {}), dict):
                    raise ValueError("Each memory's 'metadata' must be a dictionary.")

            # Convert numpy arrays to lists if necessary
            upsert_data = [
                {
                    "id": memory["id"],
                    "values": memory["values"].tolist()
                    if isinstance(memory["values"], np.ndarray)
                    else memory["values"],
                    "metadata": memory["metadata"],
                }
                for memory in memories
            ]

            logger.info(f"Upserting batch of {len(upsert_data)} memories to Pinecone.")
            response = self.index.upsert(vectors=upsert_data)
            logger.debug(f"Successfully upserted batch of memories.")

        except Exception as e:
            logger.error(f"Error upserting batch memories: {str(e)}")
            raise RuntimeError(f"Error upserting batch memories: {str(e)}")

    async def query_memory(
        self,
        query_vector: List[float],
        query_types: List[MemoryType],
        top_k: int = 5,
        window_id: Optional[str] = None,
    ) -> List[Tuple[Memory, float]]:
        """
        Queries the memory system with optional metadata filtering.

        Args:
            query_vector: The semantic vector of the query.
            query_types: The type of memory to prioritize (EPISODIC or SEMANTIC).
            top_k: Number of top results to return.

        Returns:
            List[Tuple[Memory, float]]: Retrieved memories with similarity scores.
        """
        try:
            if not self.index:
                await self.initialize_index()

            # Convert MemoryType enums to their string values
            query_type_values = [
                query_type.value for query_type in query_types
            ]

            # Build metadata filter
            filters = {}
            if query_type_values:
                filters["memory_type"] = {"$in": query_type_values}
            if window_id:
                filters["window_id"] = window_id

            # Perform the query
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                filter=filters,
                include_metadata=True,
            )

            # Convert results to Memory objects with scores
            memories_with_scores = [
                (self._create_memory_from_result(result), result["score"])
                for result in results.get("matches", [])
            ]

            return memories_with_scores

        except Exception as e:
            logger.error(f"Error querying memory: {e}")
            return []

    async def delete_memory(self, memory_id: str):
        """Deletes a memory from Pinecone."""
        try:
            if not self.index:
                await self.initialize_index()
            await self.index.delete(ids=[memory_id])
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            raise RuntimeError(f"Error deleting memory: {e}")

    async def get_all_memories_with_metadata(self) -> List[Dict]:
        """
        Retrieves all memories from the index along with their metadata.
        """
        try:
            if not self.index:
                await self.initialize_index()
            all_memories = []
            top_k = 1000
            query_vector = [0.0] * 1536
            start_cursor = None

            while True:
                results = self.index.query(
                    vector=query_vector,
                    top_k=top_k,
                    include_values=True,
                    include_metadata=True,
                )
                if results.matches:
                    all_memories.extend([
                        {
                            "id": item.id,
                            "values": item.values,
                            "metadata": item.metadata
                        } for item in results.matches
                    ])
                if not results.get("next"):
                    break
                start_cursor = results["next"]

            logger.info(f"Fetched {len(all_memories)} memories with metadata.")
            return all_memories
        except Exception as e:
            logger.error(f"Error fetching memories with metadata: {e}")
            raise RuntimeError(f"Error fetching memories with metadata: {e}")

    def _create_memory_from_result(self, result: Dict[str, Any]) -> Memory:
        """Creates a Memory object from a Pinecone query result."""
        metadata = result.get("metadata", {})
        memory_type_str = metadata.get("memory_type")

        # Use get method with default value to avoid KeyError
        memory_type_str = metadata.get("memory_type")

        # Convert the string to a MemoryType enum member, defaulting to EPISODIC if not found
        try:
            memory_type = MemoryType(memory_type_str)
        except ValueError:
            logger.warning(
                f"Unrecognized memory type '{memory_type_str}'. Using default type."
            )
            memory_type = MemoryType.EPISODIC  # Use EPISODIC as the default type

        return Memory(
            id=result.get("id", ""),
            content=metadata.get("content", ""),
            created_at=metadata.get("created_at", datetime.now().isoformat()),
            memory_type=memory_type,
            semantic_vector=result.get("values", []),
            metadata=metadata,
            window_id=metadata.get("window_id"),
        )

    async def get_memory_by_id(self, memory_id: str) -> Optional[Memory]:
        """Retrieves a memory from Pinecone by its ID."""
        try:
            if not self.index:
                await self.initialize_index()

            fetch_response = self.index.fetch(ids=[memory_id])
            vectors = fetch_response.get("vectors")
            if vectors and memory_id in vectors:
                result = vectors[memory_id]
                return self._create_memory_from_result(result)
            else:
                logger.warning(f"Memory with ID '{memory_id}' not found.")
                return None

        except Exception as e:
            logger.error(f"Error fetching memory by ID: {e}")
            return None