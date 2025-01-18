from core.memory.models import MemoryType, Memory
import os
import logging
from pinecone import Pinecone, ServerlessSpec
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

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
        self.index_name = os.getenv("INDEX_NAME")
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

    async def query_memory(self, query_vector: np.ndarray, query_type: MemoryType, k: int = 5) -> List[Tuple[Memory, float]]:
        """
        Queries the memory system with optional metadata filtering.

        Args:
            query_vector: The semantic vector of the query.
            query_type: The type of memory to prioritize (TEMPORAL or GROUNDING).
            k: Number of top results to return.

        Returns:
            List[Tuple[Memory, float]]: Retrieved memories with similarity scores.
        """
        try:
            # Build metadata filter based on query type (e.g., TEMPORAL or GROUNDING)
            filters = self.build_metadata_filter(query_type)

            # Query Pinecone with filters
            results = await self.pinecone_service.query_memory(
                query_vector=query_vector.tolist(),
                top_k=k,
                filters=filters
            )

            # Process results
            memories_with_scores = [
                (self._create_memory_from_result(result), result["score"])
                for result in results
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
                results = await self.index.query(
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