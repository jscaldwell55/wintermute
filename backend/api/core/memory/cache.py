from cachetools import LRUCache
from api.utils.config import get_settings
from typing import Dict, Any, List, Optional
from api.core.vector.vector_operations import VectorOperations, cosine_similarity
import logging

logger = logging.getLogger(__name__)

class LRUCacheManager:
    def __init__(self, config=None):
        self.settings = get_settings() if config is None else config
        self.cache = LRUCache(maxsize=self.settings.lru_cache_size)
        self.vector_operations = VectorOperations()

    def store_summary(self, memory_id: str, summary: Dict[str, Any]):
        """
        Stores a memory summary in the LRU cache.

        Args:
            memory_id (str): The ID of the memory associated with the summary.
            summary (Dict[str, Any]): The summary data to store.
        """
        logger.info(f"Storing summary in LRU cache for memory_id: {memory_id}")
        self.cache[memory_id] = summary

    def get_summary(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a memory summary from the LRU cache.

        Args:
            memory_id (str): The ID of the memory whose summary is to be retrieved.

        Returns:
            Optional[Dict[str, Any]]: The summary if found, None otherwise.
        """
        summary = self.cache.get(memory_id)
        if summary:
            logger.info(f"Cache hit for memory_id: {memory_id}")
            return summary
        else:
            logger.info(f"Cache miss for memory_id: {memory_id}")
            return None

    def remove_summary(self, memory_id: str):
        """
        Removes a memory summary from the LRU cache.

        Args:
            memory_id (str): The ID of the memory whose summary is to be removed.
        """
        if memory_id in self.cache:
            logger.info(f"Removing summary from LRU cache for memory_id: {memory_id}")
            del self.cache[memory_id]

    def get_relevant_memories(self, query_vector: List[float], threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Retrieves memories from the cache that are relevant to the given query vector,
        based on cosine similarity between the query vector and the summary embeddings.

        Args:
            query_vector (List[float]): The query vector.
            threshold (float): The similarity threshold.

        Returns:
            List[Dict[str, Any]]: A list of relevant memories.
        """
        logger.info("Retrieving relevant memories from LRU cache.")
        relevant_memories = []
        for memory_id, summary in self.cache.items():
            try:
                if "embedding" in summary:
                    similarity = cosine_similarity(query_vector, summary["embedding"])
                    if similarity >= threshold:
                        relevant_memories.append(summary)
            except Exception as e:
                logger.error(f"Error calculating similarity for memory {memory_id}: {e}")

        logger.info(f"Found {len(relevant_memories)} relevant memories in LRU cache.")
        return relevant_memories