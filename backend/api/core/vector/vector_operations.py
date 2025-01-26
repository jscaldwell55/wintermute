import numpy as np
from typing import List
from scipy.spatial.distance import cosine
import logging
from openai import OpenAI, AsyncOpenAI
from api.utils.config import get_settings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

class VectorOperations:
    """Vector operations for memory processing and analysis."""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=get_settings().llm_api_key)
        self.embedding_dim = 1536
        self.default_model = "text-embedding-ada-002"

    async def create_semantic_vector(self, text: str) -> List[float]:
        """
        Create a semantic vector from text using OpenAI's embedding model.
        Returns a 1536-dimensional zero vector if an error occurs or the embedding is empty.
        """
        logger.info(f"Creating semantic vector for text: '{text}'")
        try:
            embedding = await self.generate_embedding(text)
            if not embedding:
                logger.error("Generated embedding is empty.")
                return [0.0] * self.embedding_dim
            if len(embedding) != self.embedding_dim:
                logger.error(f"Generated embedding has incorrect dimensionality: {len(embedding)}")
                return [0.0] * self.embedding_dim
            logger.info(f"Embedding generated with length: {len(embedding)}")
            logger.debug(f"Embedding: {embedding}")
            return embedding
        except Exception as e:
            logger.error(f"Error creating semantic vector: {e}")
            return [0.0] * self.embedding_dim

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
    )
    async def generate_embedding(self, text: str, model: str = None) -> List[float]:
        """Generate embeddings using OpenAI's API with retries."""
        model = model or self.default_model
        if not text:
            logger.warning("Received empty text for embedding generation.")
            return [0.0] * self.embedding_dim

        try:
            response = self.client.embeddings.create(input=text, model=model)
            logger.debug(f"OpenAI Response: {response}")
            embedding = response.data[0].embedding
            if not embedding:
                logger.error("Received empty embedding from OpenAI API.")
                return [0.0] * self.embedding_dim
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
            return np.zeros(self.embedding_dim)
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