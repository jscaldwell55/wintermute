import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict
from core.memory.models import Memory, MemoryType
from core.evaluation import MemoryEvaluation
from utils.vector_operations import VectorOperations
from sklearn.cluster import KMeans
import numpy as np
import os
import config
from core.utils.task_queue import task_queue
import utils.llm_service as llm_service

logger = logging.getLogger(__name__)

class DreamSequence:
    def __init__(self, memory_system, vector_operations: VectorOperations, evaluation_module: MemoryEvaluation):
        self.memory_system = memory_system
        self.vector_operations = vector_operations
        self.evaluation_module = evaluation_module
        self.last_dream_time_file = "last_dream_time.txt"

    def run_dream_sequence_task(self):
        """
        Wraps run_dream_sequence for task queue execution with retries.
        """
        task_queue.enqueue(
            self.run_dream_sequence,
            retries=config.SUMMARY_RETRIES,  # Use a config setting for retries if applicable
            retry_delay=config.SUMMARY_RETRY_DELAY,  # Use a config setting for retry delay if applicable
        )

    async def run_dream_sequence(self):
        """
        Performs the daily dream sequence for memory consolidation and pattern recognition.
        """
        try:
            logger.info("Starting dream sequence...")

            # 1. Select memories for processing
            episodic_memories = await self._get_recent_episodic_memories()

            # 2. Identify patterns
            patterns = await self._identify_semantic_patterns(episodic_memories)

            # 3. Evaluate patterns and create semantic memories
            new_semantic_memories = []
            for pattern in patterns:
                significance = await self._evaluate_pattern_significance(pattern)
                if significance > config.SIGNIFICANCE_THRESHOLD:
                    semantic_memory = await self._create_semantic_memory(pattern)
                    quality_metrics = await self.evaluation_module.evaluate_semantic_memory(
                        semantic_memory, pattern["cluster_memories"]
                    )
                    semantic_memory.quality_metrics = quality_metrics

                    # Check against thresholds defined in config.py
                    if (
                        quality_metrics["relevance"] > config.SEMANTIC_MEMORY_RELEVANCE_THRESHOLD
                        and quality_metrics["coherence"] > config.SEMANTIC_MEMORY_COHERENCE_THRESHOLD
                    ):
                        new_semantic_memories.append(semantic_memory)

            # 4. Integrate new semantic memories
            for semantic_memory in new_semantic_memories:
                await self.memory_system.add_memory(
                    content=semantic_memory.content,
                    memory_type=MemoryType.SEMANTIC,
                    metadata={
                        "quality_metrics": semantic_memory.quality_metrics,
                        "source_episodic_memories": semantic_memory.metadata["source_episodic_memories"],
                        "source_pattern": semantic_memory.metadata.get("source_pattern", ""),  # Ensure this key exists
                    },
                )

            # 5. Consolidate memories
            await self.consolidate_memories()

            # 6. Update last dream time
            self._update_last_dream_time(datetime.now())

            logger.info("Dream sequence completed.")

        except Exception as e:
            logger.error(f"An error occurred during the dream sequence: {e}")

    async def _get_recent_episodic_memories(self) -> List[Memory]:
        """Retrieves episodic memories added since the last dream sequence."""
        last_dream_time = self._get_last_dream_time()
        recent_episodic_memories = []
        try:
            all_memories = await self.memory_system.pinecone_service.get_all_memories_with_metadata()
            for mem in all_memories:
                created_at_str = mem["metadata"].get("created_at")
                if created_at_str:
                    created_at = datetime.fromisoformat(created_at_str)
                    if (
                        mem["metadata"].get("memory_type") == MemoryType.EPISODIC.value
                        and created_at >= last_dream_time
                    ):
                        memory_obj = self.memory_system._create_memory_from_result(mem)
                        recent_episodic_memories.append(memory_obj)
        except Exception as e:
            logger.error(f"Error fetching recent episodic memories: {e}")

        return recent_episodic_memories

    def _get_last_dream_time(self) -> datetime:
        """Retrieves the timestamp of the last dream sequence."""
        try:
            with open(self.last_dream_time_file, "r") as f:
                timestamp_str = f.read().strip()
                return datetime.fromisoformat(timestamp_str)
        except FileNotFoundError:
            # Default to yesterday if no record is found
            return datetime.now() - timedelta(days=1)

    def _update_last_dream_time(self, timestamp: datetime):
        """Updates the timestamp of the last dream sequence."""
        with open(self.last_dream_time_file, "w") as f:
            f.write(timestamp.isoformat())

    async def _identify_semantic_patterns(self, memories: List[Memory]) -> List[Dict]:
        """Identifies potential semantic patterns using clustering (frequency-based approach)."""
        if not memories:
            return []

        # 1. Get semantic vectors
        vectors = [np.array(mem.semantic_vector) for mem in memories if mem.semantic_vector is not None]
        if not vectors:
            return []

        # 2. Cluster the vectors
        try:
            best_score = -1
            best_clusters = None
            for n_clusters in range(config.KMEANS_N_CLUSTERS_MIN, config.KMEANS_N_CLUSTERS_MAX):
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    init=config.KMEANS_INIT,
                    max_iter=config.KMEANS_MAX_ITER,
                    n_init=config.KMEANS_N_INIT,
                    random_state=0
                )
                labels = kmeans.fit_predict(vectors)
                score = silhouette_score(vectors, labels)
                
                if score > best_score:
                    best_score = score
                    best_clusters = labels
        except Exception as e:
            logger.error(f"Error during clustering: {e}")
            return []

        # 3. Identify frequent clusters
        cluster_counts = {}
        for label in best_clusters:
            cluster_counts[label] = cluster_counts.get(label, 0) + 1

        frequent_clusters = [
            label for label, count in cluster_counts.items() if count >= 3
        ]  # Example: At least 3 memories in a cluster

        # 4. Prepare patterns
        patterns = []
        for cluster_label in frequent_clusters:
            cluster_indices = [i for i, label in enumerate(best_clusters) if label == cluster_label]
            cluster_memories = [memories[i] for i in cluster_indices]
            patterns.append({"cluster_memories": cluster_memories})

        return patterns

    async def _create_semantic_memory(self, pattern: Dict) -> Memory:
      """Creates a semantic memory from a pattern (cluster of episodic memories)."""
      cluster_memories = pattern["cluster_memories"]

      # 1. Generate content (using LLM for summarization)
      content = await self._generate_semantic_content(cluster_memories)

      # 2. Generate semantic vector
      semantic_vector = await self.vector_operations.create_semantic_vector(content)

      # 3. Create memory object
      memory_id = f"sem_{uuid.uuid4()}"
      full_metadata = {
          "content": content,
          "created_at": datetime.now().isoformat(),
          "memory_type": MemoryType.SEMANTIC.value,
          "source_episodic_memories": [mem.id for mem in cluster_memories],  # Link to source memories
      }

      memory = Memory(
          id=memory_id,
          memory_type=MemoryType.SEMANTIC,
          content=content,
          semantic_vector=semantic_vector,
          metadata=full_metadata,
          created_at=datetime.now().isoformat(),
      )

      return memory

    async def _generate_semantic_content(self, memories: List[Memory]) -> str:
        """Generates a concise description of the semantic content using an LLM."""
        if not memories:
            return ""

        # Build a prompt for the LLM
        memory_contents = [mem.content for mem in memories]
        prompt = f"""
        Identify the common theme or knowledge from these memories:

        {' '.join(memory_contents)}

        Summary:
        """

        # Use the LLM to generate the content
        try:
            response = await llm_service.generate_gpt_response_async(
                prompt=prompt,
                model=config.SUMMARY_LLM_MODEL_NAME,
                temperature=0.2,
                max_tokens=100,
            )
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating semantic content with LLM: {e}")
            # Fallback: Use a simple concatenation of memory content
            return " ".join([mem.content for mem in memories])

    async def _evaluate_pattern_significance(self, pattern: Dict) -> float:
      """Evaluates the significance of a pattern before creating a semantic memory."""
      # Placeholder for now, you can replace this with more complex logic later.
      # For example, you can check the frequency of the pattern,
      # the diversity of the memories in the pattern, etc.
      return 1.0  # Every pattern is considered significant for now

    async def consolidate_memories(self):
        """
        Consolidates old episodic memories into semantic memories and removes outdated episodic memories.
        """
        logger.info("Starting memory consolidation...")
        try:
            # Define the age threshold for old episodic memories (e.g., 7 days)
            age_threshold = datetime.now() - timedelta(days=7)

            # Find episodic memories older than the age threshold
            old_episodic_memories = [
                mem
                for mem in await self.memory_system.get_all_memories()
                if mem.memory_type == MemoryType.EPISODIC
                and datetime.fromisoformat(mem.created_at) < age_threshold
            ]

            if not old_episodic_memories:
                logger.info("No old episodic memories to consolidate.")
                return

            # Group similar episodic memories for consolidation
            patterns = await self._identify_semantic_patterns(old_episodic_memories)

            # Create semantic memories from patterns
            for pattern in patterns:
                await self._create_semantic_memory(pattern)

            # Remove old episodic memories
            for memory in old_episodic_memories:
                await self.memory_system.pinecone_service.delete_memory(memory.id)

            logger.info(f"Consolidated {len(old_episodic_memories)} old episodic memories.")

        except Exception as e:
            logger.error(f"Error during memory consolidation: {e}")

# Constants (moved to config.py)