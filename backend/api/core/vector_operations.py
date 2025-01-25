import numpy as np
from typing import Dict, Any, List, Optional
from scipy.spatial.distance import cosine, euclidean
import logging
import asyncio
from .memory.models import Memory, MemoryType
import os
from openai import OpenAI
from api.utils import config

logger = logging.getLogger(__name__)

class VectorOperations:
    """Vector operations for memory processing and analysis."""

    def __init__(self):
      self.client = OpenAI(api_key=config.LLM_API_KEY)

    async def create_semantic_vector(self, text: str) -> List[float]:
        """
        Create a semantic vector from text using OpenAI's embedding model.
        """
        try:
            embedding = await self.generate_embedding(text)
            logger.info(f"Length of embedding list: {len(embedding)}")
            vector = np.array(embedding)
            logger.info(f"Created semantic vector with dimensions: {vector.shape}")
            return embedding
        except Exception as e:
            logger.error(f"Error creating semantic vector: {e}")
            raise

    async def generate_embedding(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """
        Generates an embedding vector for the given text using OpenAI API.
        """
        try:
            response = self.client.embeddings.create(input=text, model=model)
            embedding = response.data[0].embedding
            if len(embedding) != 1536:
                logger.warning(f"Unexpected embedding dimensionality: {len(embedding)}")
            return embedding
        except Exception as e:
            logger.error(f"Error in generate_embedding: {e}")
            raise

    async def create_combined_q_r_vector(self, query: str, response: str) -> List[float]:
        """
        Creates a single vector from a Q/R pair.
        """
        combined_text = f"Q: {query}\nA: {response}"
        return await self.create_semantic_vector(combined_text)

    def average_vectors(self, vectors: List[np.ndarray]) -> np.ndarray:
        """Calculates the average of a list of vectors."""
        if not vectors:
            logger.warning("No vectors provided for averaging.")
            return np.zeros(1536)
        try:
            return np.mean(vectors, axis=0)
        except Exception as e:
            logger.error(f"Error averaging vectors: {e}", exc_info=True)
            raise

    @staticmethod
    def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        if vector1.shape != vector2.shape:
            raise ValueError("Vectors must have the same dimensions for cosine similarity calculation.")

        dot_product = np.dot(vector1, vector2)
        magnitude_vector1 = np.linalg.norm(vector1)
        magnitude_vector2 = np.linalg.norm(vector2)

        if magnitude_vector1 == 0 or magnitude_vector2 == 0:
            return 0.0

        similarity = dot_product / (magnitude_vector1 * magnitude_vector2)
        return similarity

    @staticmethod
    def normalize_vector(vector: np.ndarray) -> np.ndarray:
        """Normalize a vector to unit length."""
        try:
            norm = np.linalg.norm(vector)
            normalized_vector = vector / norm if norm > 0 else vector
            logger.debug("Vector normalized.")
            return normalized_vector
        except Exception as e:
            logger.error(f"Error normalizing vector: {e}", exc_info=True)
            raise
    
    def decay_vector(self, vector, created_at, decay_factor=config.MEMORY_DECAY_FACTOR, hours_since_decay=config.HOURS_SINCE_DECAY):
        """
        Decays a memory vector based on time.

        Args:
            vector: The memory vector (numpy array).
            created_at: The timestamp when the memory was created.
            decay_factor: The decay factor (applied per time unit).
            hours_since_decay: The number of hours that represent a time unit for decay.

        Returns:
            The decayed vector.
        """
        now = time.time()
        time_diff = now - created_at
        hours_passed = time_diff / 3600  # Convert seconds to hours
        decay_multiplier = decay_factor ** (hours_passed / hours_since_decay)
        decayed_vector = vector * decay_multiplier
        return decayed_vector