import os
from typing import List
import numpy as np
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from api.utils.config import get_settings
import logging

logger = logging.getLogger(__name__)

class VectorOperations:
    """
    A class to handle various vector operations using OpenAI's embeddings.
    """

    def __init__(self):
        """
        Initializes the VectorOperations class with an OpenAI client.
        """
        self.settings = get_settings()
        self.client = AsyncOpenAI(api_key=self.settings.llm_api_key)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    async def generate_embedding(self, text: str, model: str = None) -> List[float]:
        """
        Generates an embedding for a given text using the specified OpenAI model.

        Args:
            text: The text to generate an embedding for.
            model: The ID of the OpenAI model to use for generating the embedding.

        Returns:
            A list of floats representing the embedding.
        """
        model = model or self.settings.vector_model_id
        text = text.replace("\n", " ")
        try:
            response = await self.client.embeddings.create(input=[text], model=model)
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    async def create_semantic_vector(self, text: str) -> List[float]:
        """
        Creates a semantic vector (embedding) for a given text using OpenAI's API.

        Args:
            text: The text to create a vector for.

        Returns:
            A list of floats representing the semantic vector.
        """
        return await self.generate_embedding(text)  # Directly use generate_embedding

    def average_vectors(self, vectors: List[List[float]]) -> List[float]:
        """
        Calculates the average of a list of vectors.

        Args:
            vectors: A list of vectors, where each vector is a list of floats.

        Returns:
            A list of floats representing the average vector.
        """
        num_vectors = len(vectors)
        if num_vectors == 0:
            return []

        vector_length = len(vectors[0])
        avg_vector = [0.0] * vector_length

        for vector in vectors:
            for i in range(vector_length):
                avg_vector[i] += vector[i] / num_vectors

        return avg_vector

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculates the cosine similarity between two vectors.

        Args:
            vec1: The first vector.
            vec2: The second vector.

        Returns:
            float: The cosine similarity between the two vectors.
        """
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension for cosine similarity.")

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude_vec1 = np.sqrt(sum(a * a for a in vec1))
        magnitude_vec2 = np.sqrt(sum(b * b for b in vec2))

        if magnitude_vec1 == 0 or magnitude_vec2 == 0:
            return 0.0  # Handle cases where one or both vectors have zero magnitude

        return dot_product / (magnitude_vec1 * magnitude_vec2)

    def normalize_vector(self, vector: List[float]) -> List[float]:
        """
        Normalizes a vector to unit length.

        Args:
            vector: The vector to normalize.

        Returns:
            List[float]: The normalized vector.
        """
        magnitude = np.sqrt(sum(a * a for a in vector))
        if magnitude == 0:
            return vector  # Avoid division by zero for zero vectors
        return [a / magnitude for a in vector]