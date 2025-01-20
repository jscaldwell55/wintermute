import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict
from core.memory.models import Memory, MemoryType
from core.evaluation import MemoryEvaluation
from utils.vector_operations import VectorOperations
from sklearn.cluster import KMeans
import numpy as np

logger = logging.getLogger(__name__)

class DreamSequence:
    def __init__(self, memory_system, vector_operations: VectorOperations, evaluation_module: MemoryEvaluation):
        self.memory_system = memory_system
        self.vector_operations = vector_operations
        self.evaluation_module = evaluation_module

    async def run_dream_sequence(self):
        """
        Performs the daily dream sequence for memory consolidation and pattern recognition.
        """
        logger.info("Starting dream sequence...")

        # 1. Select memories for processing
        episodic_memories = await self._get_recent_episodic_memories()

        # 2. Identify patterns
        patterns = await self._identify_semantic_patterns(episodic_memories)

        # 3. Evaluate patterns and create semantic memories
        new_semantic_memories = []
        for pattern in patterns:
            significance = await self._evaluate_pattern_significance(pattern)
            if significance > SIGNIFICANCE_THRESHOLD:
                semantic_memory = await self._create_semantic_memory(pattern)
                quality_metrics = await self.evaluation_module.evaluate_semantic_memory(
                    semantic_memory
                )
                if all(
                    score > QUALITY_THRESHOLD for score in quality_metrics.values()
                ):
                    new_semantic_memories.append(semantic_memory)

        # 4. Integrate new semantic memories
        for semantic_memory in new_semantic_memories:
            await self.memory_system.add_memory(
                content=semantic_memory.content,
                memory_type=MemoryType.SEMANTIC,
                metadata={
                    "quality_metrics": quality_metrics,
                    "source_pattern": pattern,  # Link to the source pattern (if applicable)
                },
            )

        # 5. Perform other consolidation tasks (optional)
        # ... e.g., refine existing semantic memories, update knowledge graph ...

        logger.info("Dream sequence completed.")

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
        # Simple file-based storage (replace with a more robust solution if needed)
        try:
            with open("last_dream_time.txt", "r") as f:
                timestamp_str = f.read().strip()
                return datetime.fromisoformat(timestamp_str)
        except FileNotFoundError:
            # Default to yesterday if no record is found
            return datetime.now() - timedelta(days=1)

    def _update_last_dream_time(self, timestamp: datetime):
        """Updates the timestamp of the last dream sequence."""
        # Simple file-based storage (replace with a more robust solution if needed)
        with open("last_dream_time.txt", "w") as f:
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
            n_clusters = min(len(vectors), 5)  # Example: Use up to 5 clusters
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(vectors)
            labels = kmeans.labels_
        except Exception as e:
            logger.error(f"Error during clustering: {e}")
            return []

        # 3. Identify frequent clusters
        cluster_counts = {}
        for label in labels:
            cluster_counts[label] = cluster_counts.get(label, 0) + 1

        frequent_clusters = [
            label for label, count in cluster_counts.items() if count >= 3
        ]  # Example: At least 3 memories in a cluster

        # 4. Prepare patterns
        patterns = []
        for cluster_label in frequent_clusters:
            cluster_indices = [i for i, label in enumerate(labels) if label == cluster_label]
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
      prompt = "Identify the common theme or knowledge from these memories:\n"
      for mem in memories:
          prompt += f"- {mem.content}\n"
      prompt += "Summary:"

      # Use the LLM to generate the content
      try:
          response = await llm_service.generate_gpt_response_async(
              prompt=prompt,
              model="gpt-3.5-turbo",  # Or another suitable model
              temperature=0.2,  # Relatively low temperature for focused summarization
              max_tokens=100,  # Adjust as needed
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

# Constants (Ideally, these should be configurable, e.g., from a config file)
SIGNIFICANCE_THRESHOLD = 0.5  # Example threshold
QUALITY_THRESHOLD = 0.6  # Example threshold