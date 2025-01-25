import numpy as np
from core.memory.models import Memory, MemoryType
from api.core.vector_operations import VectorOperations
from typing import Dict, List
import logging
import api.utils.llm_service as llm_service
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class MemoryEvaluation:
    """
    Provides methods for evaluating the quality of semantic memories.
    """

    def __init__(self, vector_operations: VectorOperations, memory_system):
        self.vector_operations = vector_operations
        self.memory_system = memory_system

    async def evaluate_semantic_memory(self, semantic_memory: Memory, source_episodic_memories: List[Memory]) -> Dict:
        """
        Evaluates the quality of a generated semantic memory.

        Args:
            semantic_memory: The generated semantic memory.
            source_episodic_memories: The episodic memories used to create the semantic memory.

        Returns:
            A dictionary containing quality metrics.
        """
        metrics = {}

        # 1. Relevance (from new version - using source episodic memories)
        if source_episodic_memories:
            source_vectors = np.array([mem.semantic_vector for mem in source_episodic_memories])
            avg_similarity = np.mean(self.vector_operations.cosine_similarity(semantic_memory.semantic_vector, source_vectors))
            metrics["relevance"] = avg_similarity
        else:
            metrics["relevance"] = 0.0
            logger.warning("No source episodic memories provided for relevance calculation.")

        # 2. Conciseness (from new version - simple token count)
        metrics["conciseness"] = len(semantic_memory.content.split())

        # --- Metrics from the original version ---
        # 3. Coherence (modified from original to use source episodic memories if available)
        metrics["coherence"] = await self._measure_coherence(semantic_memory, source_episodic_memories)

        # 4. Consistency (original)
        metrics["consistency"] = await self._measure_consistency(semantic_memory)

        # 5. Novelty (original)
        metrics["novelty"] = await self._measure_novelty(semantic_memory)

        # 6. Utility (original - placeholder)
        metrics["utility"] = await self._measure_utility(semantic_memory)  # Placeholder for a more complex metric
      
        return metrics

    async def _measure_coherence(self, memory: Memory, source_episodic_memories: Optional[List[Memory]] = None) -> float:
        """
        Measures the internal coherence of a semantic memory.

        Uses source episodic memories if provided, otherwise falls back to using source memory IDs from metadata.
        """
        if source_episodic_memories:
            source_vectors = [
                np.array(mem.semantic_vector)
                for mem in source_episodic_memories
                if mem.semantic_vector is not None
            ]
        elif memory.metadata.get("source_episodic_memories"):
            source_memory_ids = memory.metadata["source_episodic_memories"]

            source_memories = [
                await self.memory_system.get_memory_by_id(mem_id)
                for mem_id in source_memory_ids
                if await self.memory_system.get_memory_by_id(mem_id) is not None
            ]

            if not source_memories:
                logger.warning(
                    "Source memories not found for coherence measurement."
                )
                return 0.0
            source_vectors = [
                np.array(mem.semantic_vector)
                for mem in source_memories
                if mem.semantic_vector is not None
            ]

        else:
            logger.warning(
                "Cannot measure coherence without source memories or metadata."
            )
            return 0.0

        if not source_vectors:
            logger.warning(
                "No valid semantic vectors found in source memories for coherence measurement."
            )
            return 0.0

        # Calculate average pairwise cosine similarity
        similarities = []
        for i in range(len(source_vectors)):
            for j in range(i + 1, len(source_vectors)):
                similarity = self.vector_operations.cosine_similarity(
                    source_vectors[i], source_vectors[j]
                )
                similarities.append(similarity)

        return np.mean(similarities) if similarities else 0.0

    async def _measure_consistency(self, memory: Memory) -> float:
        """
        Measures the consistency of a semantic memory with existing knowledge.

        Compares the semantic memory's vector with those of recent episodic memories and existing semantic memories.
        """
        consistency_scores = []

        # 1. Compare with recent episodic memories:
        try:
            recent_episodic_memories = await self.memory_system._get_recent_episodic_memories()
            episodic_vectors = [
                np.array(mem.semantic_vector) for mem in recent_episodic_memories if mem.semantic_vector is not None
            ]
            for episodic_vector in episodic_vectors:
                similarity = self.vector_operations.cosine_similarity(
                    np.array(memory.semantic_vector), episodic_vector
                )
                consistency_scores.append(similarity)
        except Exception as e:
            logger.error(f"Error comparing with recent episodic memories: {e}")

        # 2. Compare with existing semantic memories:
        try:
            existing_semantic_memories = await self.memory_system.query_memory(
                np.array(memory.semantic_vector),  # Use the new semantic memory's vector
                query_types=[MemoryType.SEMANTIC],
                top_k=5  # Get top 5 most similar semantic memories
            )
            semantic_vectors = [
                np.array(mem.semantic_vector) for mem, _ in existing_semantic_memories if mem.semantic_vector is not None
            ]
            for semantic_vector in semantic_vectors:
                similarity = self.vector_operations.cosine_similarity(
                    np.array(memory.semantic_vector), semantic_vector
                )
                consistency_scores.append(similarity)
        except Exception as e:
            logger.error(f"Error comparing with existing semantic memories: {e}")

        # Combine the similarity scores (e.g., using average)
        avg_consistency_score = np.mean(consistency_scores) if consistency_scores else 0.0

        return avg_consistency_score

    async def _measure_novelty(self, memory: Memory) -> float:
        """
        Measures the novelty of a semantic memory compared to existing knowledge.

        Calculates 1 - the average similarity to the top K most similar memories.
        """
        top_k = 5
        try:
            similar_memories = await self.memory_system.query_memory(
                np.array(memory.semantic_vector),
                query_types=[MemoryType.SEMANTIC],
                top_k=top_k,
            )
            similarities = [similarity for _, similarity in similar_memories]

            if not similarities:
                return 1.0  # Maximum novelty if no similar memories are found

            avg_similarity = np.mean(similarities)
            novelty_score = 1.0 - avg_similarity
            return novelty_score
        except Exception as e:
            logger.error(f"Error measuring novelty: {e}")
            return 0.5  # Return a default value indicating moderate novelty

    async def _measure_utility(self, memory: Memory) -> float:
        """
        Placeholder for measuring the utility of a semantic memory in downstream tasks.

        This would ideally involve evaluating the memory system's performance on specific tasks
        with and without the semantic memory.
        """
        # TODO: Implement task-specific evaluation or use a proxy metric
        logger.warning("Utility measurement not yet implemented.")
        return 0.5  # Placeholder value

    async def _llm_evaluation(self, memory: Memory) -> float:
        """
        Uses an LLM to evaluate the quality of the semantic memory.

        Could be used to assess coherence, meaningfulness, or other qualitative aspects.
        """
        prompt = f"""
        Evaluate the following semantic memory on a scale of 0 to 1, where 1 is high quality:

        Memory: {memory.content}

        Evaluation (0.0-1.0):
        """
        try:
            response = await llm_service.generate_gpt_response_async(
                prompt=prompt,
                model=config.LLM_MODEL_NAME,
                temperature=0.2,
                max_tokens=50,
            )
            # Extract the evaluation score from the response (you might need to adjust the parsing)
            score = float(response.strip())
            return score
        except Exception as e:
            logger.error(f"Error during LLM evaluation: {e}")
            return 0.5  # Default value