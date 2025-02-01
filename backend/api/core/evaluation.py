import logging
from typing import List
from api.core.memory.models import Memory  # Import Memory class
from api.core.vector.vector_operations import VectorOperations
from api.utils.config import get_settings

logger = logging.getLogger(__name__)

class MemoryEvaluator:
    """
    Evaluates the quality of memories, particularly semantic memories, based on
    relevance and coherence metrics.
    """

    def __init__(self, vector_operations: VectorOperations, memory_system):
        self.settings = get_settings()
        self.vector_operations = vector_operations
        self.memory_system = memory_system

    async def evaluate_relevance(self, query: str, memory_content: str) -> float:
        """
        Evaluates the relevance of a memory to a given query using vector similarity.

        Args:
            query (str): The query string.
            memory_content (str): The content of the memory to evaluate.

        Returns:
            float: A score between 0 and 1 representing the relevance of the memory to the query.
                   Higher scores indicate greater relevance.
        """
        try:
            query_vector = await self.vector_operations.create_semantic_vector(query)
            memory_vector = await self.vector_operations.create_semantic_vector(memory_content)
            relevance_score = self.vector_operations.cosine_similarity(query_vector, memory_vector)
            logger.info(f"Relevance score for query '{query}' and memory content '{memory_content[:50]}...': {relevance_score}")
            return relevance_score
        except Exception as e:
            logger.error(f"Error evaluating memory relevance: {e}")
            return 0.0

    async def evaluate_coherence(self, memory_content: str) -> float:
        """
        Evaluates the internal coherence of a memory.

        This is a placeholder for more complex logic that could involve:
        - Analyzing the semantic similarity between sentences in the memory.
        - Checking for logical consistency and flow within the memory content.

        Args:
            memory_content (str): The content of the memory to evaluate.

        Returns:
            float: A score between 0 and 5 representing the coherence of the memory.
                   Higher scores indicate greater coherence.
        """
        try:
            # Placeholder for coherence evaluation logic
            # Example: Calculate average similarity between consecutive sentences
            sentences = memory_content.split(". ")  # Simple sentence splitting
            if len(sentences) < 2:
                return 5.0  # Single sentence is considered coherent

            embeddings = [await self.vector_operations.create_semantic_vector(sentence) for sentence in sentences]
            similarities = []
            for i in range(len(embeddings) - 1):
                similarity = self.vector_operations.cosine_similarity(embeddings[i], embeddings[i + 1])
                similarities.append(similarity)

            coherence_score = sum(similarities) / len(similarities) * 5  # Scale to 0-5 range

            logger.info(f"Coherence score for memory content '{memory_content[:50]}...': {coherence_score}")
            return coherence_score
        except Exception as e:
            logger.error(f"Error evaluating memory coherence: {e}")
            return 0.0
        
    async def evaluate_semantic_memory(self, semantic_memory: Memory, source_memories: List[Memory]) -> Dict:
        """
        Evaluates the quality of a semantic memory based on its relevance to source memories and its internal coherence.

        Args:
            semantic_memory (Memory): The semantic memory to evaluate.
            source_memories (List[Memory]): The episodic memories from which the semantic memory was derived.

        Returns:
            Dict: A dictionary containing the quality metrics (relevance, coherence, etc.) of the semantic memory.
        """
        try:
            # Evaluate relevance as the average similarity to source memories
            relevance_scores = [
                await self.evaluate_relevance(semantic_memory.content, mem.content)
                for mem in source_memories
            ]
            average_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0

            # Evaluate coherence
            coherence_score = await self.evaluate_coherence(semantic_memory.content)

            return {
                "relevance": average_relevance,
                "coherence": coherence_score,
            }
        except Exception as e:
            logger.error(f"Error evaluating semantic memory: {e}")
            return {
                "relevance": 0.0,
                "coherence": 0.0,
            }