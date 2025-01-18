import numpy as np
from typing import Dict, Any, List, Optional
from scipy.spatial.distance import cosine, euclidean
import logging
import openai
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class VectorOperations:
    """Vector operations for memory processing and analysis."""

    async def create_semantic_vector(self, text: str) -> np.ndarray:
        """
        Create a semantic vector from text using OpenAI's embedding model.
        This function now utilizes generate_embedding internally.
        """
        try:
            embedding = await self.generate_embedding(text)
            logger.info(f"Length of embedding list: {len(embedding)}")
            vector = np.array(embedding)
            logger.info(f"Created semantic vector with dimensions: {vector.shape}")
            return vector
        except Exception as e:
            logger.error(f"Error creating semantic vector: {e}")
            raise

    async def generate_embedding(self, text: str, model: str = "text-embedding-ada-002") -> list:
        """
        Generates an embedding vector for the given text using OpenAI API.
        """
        try:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            if not openai.api_key:
                raise ValueError("OpenAI API key not configured.")

            response = openai.embeddings.create(input=text, model=model)
            embedding = response.data[0].embedding
            if len(embedding) != 1536:
                logger.warning(f"Unexpected embedding dimensionality: {len(embedding)}")
            return embedding
        except openai.OpenAIError as e:
            logger.error(f"OpenAI API error in generate_embedding: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in generate_embedding: {e}")
            raise

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
    def decay_vector(vector: np.ndarray, decay_factor: float) -> np.ndarray:
        """Apply decay to a vector based on time."""
        try:
            magnitude = np.linalg.norm(vector)
            decayed_magnitude = magnitude * decay_factor
            normalized = vector / magnitude if magnitude > 0 else vector
            decayed_vector = normalized * decayed_magnitude
            logger.debug(f"Vector decayed by factor {decay_factor}.")
            return decayed_vector
        except Exception as e:
            logger.error(f"Error applying decay to vector: {e}", exc_info=True)
            raise

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