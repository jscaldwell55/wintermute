from api.core.memory.models import MemoryType, Memory
import os
import logging
from datetime import datetime
from pinecone import Pinecone, ServerlessSpec
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from tenacity import retry, stop_after_attempt, wait_fixed

logger = logging.getLogger(__name__)

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def initialize_pinecone(api_key: str, environment: str, index_name: str, dimension: int = 1536):
    """Initialize Pinecone and ensure index exists."""
    try:
        pinecone = Pinecone(api_key=api_key)

        if index_name not in pinecone.list_indexes().names():
            logger.info(f"Creating index: {index_name}")
            pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=environment)
            )

        index = pinecone.Index(index_name)
        logger.info(f"Pinecone index '{index_name}' initialized successfully.")
        return index

    except Exception as e:
        logger.error(f"Error initializing Pinecone: {e}")
        raise RuntimeError(f"Error initializing Pinecone: {e}")

class PineconeService:
    def __init__(self, api_key: str, environment: str, index_name: str):
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.index = None
        self.pc = None

    async def initialize_index(self):
        """Initialize the Pinecone index if not already initialized."""
        if not self.index:
            self.pc = Pinecone(api_key=self.api_key)
            self.index = await initialize_pinecone(
                api_key=self.api_key,
                environment=self.environment,
                index_name=self.index_name
            )
        return self.index

    async def upsert_memory(self, memory_id: str, vector: list, metadata: dict):
        """Upsert a single memory vector."""
        if not self.index:
            await self.initialize_index()

        try:
            if not isinstance(vector, (list, np.ndarray)) or len(vector) != 1536:
                raise ValueError("Vector must be a list or numpy array with 1536 dimensions")

            vector_list = vector.tolist() if isinstance(vector, np.ndarray) else vector
            await self.index.upsert(vectors=[{
                "id": memory_id,
                "values": vector_list,
                "metadata": metadata
            }])

        except Exception as e:
            logger.error(f"Error upserting memory: {str(e)}")
            raise

    async def upsert_batch_memories(self, memories: List[Dict[str, Any]]):
        """Upsert multiple memory vectors in batch."""
        if not self.index:
            await self.initialize_index()

        try:
            # Validate memories and convert to proper format
            upsert_data = []
            for memory in memories:
                if not all(k in memory for k in ('id', 'values', 'metadata')):
                    raise ValueError("Each memory must have 'id', 'values', and 'metadata'")

                values = memory['values']
                if isinstance(values, np.ndarray):
                    values = values.tolist()

                upsert_data.append({
                    "id": memory['id'],
                    "values": values,
                    "metadata": memory['metadata']
                })

            await self.index.upsert(vectors=upsert_data)

        except Exception as e:
            logger.error(f"Error batch upserting memories: {str(e)}")
            raise

    async def query_memory(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Memory, float]]:
        """Query memories with vector similarity search."""
        if not self.index:
            await self.initialize_index()

        try:
            results = await self.index.query(
                vector=query_vector,
                top_k=top_k,
                filter=filter,
                include_metadata=True
            )

            return [
                (self._create_memory_from_result(match), match['score'])
                for match in results['matches']
            ]

        except Exception as e:
            logger.error(f"Error querying memory: {str(e)}")
            raise

    async def get_all_memories_with_metadata(self) -> List[Dict]:
        """Retrieve all memories with metadata."""
        if not self.index:
            await self.initialize_index()

        try:
            results = await self.index.query(
                vector=[0.0] * 1536,  # Dummy vector for fetching all
                top_k=10000,  # Adjust as needed to get all memories
                include_metadata=True,
                include_values=True
            )

            return [
                self._create_memory_from_result(match)
                for match in results['matches']
            ]

        except Exception as e:
            logger.error(f"Error fetching all memories: {str(e)}")
            raise

    async def get_memory_by_id(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a specific memory by ID."""
        if not self.index:
            await self.initialize_index()

        try:
            response = await self.index.fetch(ids=[memory_id])
            vectors = response.get('vectors', {})
            if memory_id in vectors:
                return self._create_memory_from_result(vectors[memory_id])
            return None

        except Exception as e:
            logger.error(f"Error fetching memory by ID: {str(e)}")
            raise

    async def delete_memory(self, memory_id: str):
        """Delete a memory by ID."""
        if not self.index:
            await self.initialize_index()

        try:
            await self.index.delete(ids=[memory_id])
        except Exception as e:
            logger.error(f"Error deleting memory: {str(e)}")
            raise

    def _create_memory_from_result(self, result: Dict[str, Any]) -> Memory:
        """Create Memory object from Pinecone result."""
        metadata = result.get("metadata", {})
        memory_type_str = metadata.get("memory_type", "EPISODIC")

        try:
            memory_type = MemoryType[memory_type_str.upper()]
        except (KeyError, ValueError):
            logger.warning(f"Invalid memory type '{memory_type_str}', using EPISODIC")
            memory_type = MemoryType.EPISODIC

        return Memory(
            id=result.get("id", ""),
            content=metadata.get("content", ""),
            created_at=metadata.get("created_at", datetime.now().isoformat()),
            memory_type=memory_type,
            semantic_vector=result.get("values", []),
            metadata=metadata,
            window_id=metadata.get("window_id")
        )

    async def close_connections(self):
        """Close Pinecone connections."""
        self.index = None
        self.pc = None