import logging
import uuid
from sklearn.metrics import silhouette_score
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from api.core.memory.models import Memory, MemoryType
from api.core.evaluation import MemoryEvaluation
from project_root.backend.api.core.vector_operations import VectorOperations
from sklearn.cluster import KMeans
import numpy as np
import os
import project_root.backend.api.utils.config as config
from api.utils.task_queue import task_queue
import api.utils.llm_service as llm_service
from transformers import pipeline

logger = logging.getLogger(__name__)

class DreamSequence:
    def __init__(
        self,
        memory_system,
        vector_operations: VectorOperations,
        evaluation_module: MemoryEvaluation,
    ):
        self.memory_system = memory_system
        self.vector_operations = vector_operations
        self.evaluation_module = evaluation_module
        self.last_dream_time_file = "last_dream_time.txt"
        self.summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn") # facebook/bart-large-cnn is a good default for summarization tasks
    
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
                    quality_metrics = (
                        await self.evaluation_module.evaluate_semantic_memory(
                            semantic_memory, pattern["cluster_memories"]
                        )
                    )
                    semantic_memory.quality_metrics = quality_metrics

                    # Check against thresholds defined in config.py
                    if (
                        quality_metrics["relevance"]
                        > config.SEMANTIC_MEMORY_RELEVANCE_THRESHOLD
                        and quality_metrics["coherence"]
                        > config.SEMANTIC_MEMORY_COHERENCE_THRESHOLD
                    ):
                        new_semantic_memories.append(semantic_memory)

            # 4. Integrate new semantic memories
            for semantic_memory in new_semantic_memories:
                await self.memory_system.add_memory(
                    content=semantic_memory.content,
                    memory_type=MemoryType.SEMANTIC,
                    metadata={
                        "quality_metrics": semantic_memory.quality_metrics,
                        "source_episodic_memories": semantic_memory.metadata[
                            "source_episodic_memories"
                        ],
                        "source_pattern": semantic_memory.metadata.get(
                            "source_pattern", ""
                        ),  # Ensure this key exists
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
            all_memories = (
                await self.memory_system.pinecone_service.get_all_memories_with_metadata()
            )
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
            logger.info("No memories provided for pattern identification")
            return []

        # 1. Get and validate semantic vectors
        vectors = []
        for mem in memories:
            if mem.semantic_vector is None:
                logger.warning(f"Memory {mem.id} has no semantic vector, skipping")
                continue
            
            try:
                vector = np.array(mem.semantic_vector)
                if np.any(np.isnan(vector)) or np.any(np.isinf(vector)):
                    logger.warning(f"Memory {mem.id} has invalid vector values, skipping")
                    continue
                vectors.append(vector)
            except Exception as e:
                logger.error(f"Error processing vector for memory {mem.id}: {e}")
                continue

        if not vectors:
            logger.warning("No valid vectors available for clustering")
            return []

        logger.info(f"Processing {len(vectors)} valid vectors for clustering")

        # 2. Cluster the vectors
        try:
            best_score = -1
            best_clusters = None
            best_n_clusters = None
            vectors_array = np.vstack(vectors)  # Convert list to numpy array

            # Ensure the number of clusters does not exceed the number of memories
            max_clusters = min(config.KMEANS_N_CLUSTERS_MAX, len(vectors))

            # Initialize default clustering in case no better solution is found
            kmeans_default = KMeans(
                n_clusters=config.KMEANS_N_CLUSTERS_MIN,  # Start with minimum of 2 clusters
                init=config.KMEANS_INIT,
                max_iter=config.KMEANS_MAX_ITER,
                n_init=config.KMEANS_N_INIT,
                random_state=0,
            )
            best_clusters = kmeans_default.fit_predict(vectors_array)

            # Try different numbers of clusters
            for n_clusters in range(config.KMEANS_N_CLUSTERS_MIN, max_clusters + 1):
                if len(vectors) < n_clusters:
                    logger.warning(f"Skipping clustering with {n_clusters} clusters due to insufficient samples.")
                    continue

                logger.debug(f"Attempting clustering with {n_clusters} clusters")
                
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    init=config.KMEANS_INIT,
                    max_iter=config.KMEANS_MAX_ITER,
                    n_init=config.KMEANS_N_INIT,
                    random_state=0,
                )
                labels = kmeans.fit_predict(vectors_array)
                
                # Calculate silhouette score for more than one cluster and less than the number of samples
                if 1 < n_clusters < len(vectors):
                    score = silhouette_score(vectors_array, labels)
                    logger.debug(f"Silhouette score for {n_clusters} clusters: {score}")

                    if score > best_score:
                        best_score = score
                        best_clusters = labels
                        best_n_clusters = n_clusters
                        logger.info(f"New best clustering found: {n_clusters} clusters with score {score}")
                else:
                    best_clusters = labels

            if best_clusters is None:
                logger.warning("No valid clustering found, using default clustering")
                best_clusters = kmeans_default.fit_predict(vectors_array)

            # 3. Identify frequent clusters with detailed logging
            cluster_counts = {}
            for label in best_clusters:
                cluster_counts[label] = cluster_counts.get(label, 0) + 1
            
            logger.debug(f"Cluster distribution: {cluster_counts}")
            
            # Define minimum cluster size based on total memories
            min_cluster_size = max(3, int(len(memories) * 0.1))  # At least 3 or 10% of total
            frequent_clusters = [
                label for label, count in cluster_counts.items() 
                if count >= min_cluster_size
            ]
            
            logger.info(f"Found {len(frequent_clusters)} frequent clusters with minimum size {min_cluster_size}")

            # 4. Prepare patterns with validation
            patterns = []
            for cluster_label in frequent_clusters:
                cluster_indices = [
                    i for i, label in enumerate(best_clusters) if label == cluster_label
                ]
                
                if not cluster_indices:
                    logger.warning(f"No memories found for cluster {cluster_label}, skipping")
                    continue
                    
                cluster_memories = [memories[i] for i in cluster_indices]
                
                # Validate cluster memories
                if len(cluster_memories) >= min_cluster_size:
                    patterns.append({
                        "cluster_memories": cluster_memories,
                        "cluster_size": len(cluster_memories),
                        "cluster_label": cluster_label
                    })
                    logger.debug(f"Added pattern for cluster {cluster_label} with {len(cluster_memories)} memories")

            logger.info(f"Final number of patterns identified: {len(patterns)}")
            return patterns

        except Exception as e:
            logger.error(f"Error during pattern identification: {str(e)}", exc_info=True)
            return []

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
        """Generates a concise description of the semantic content using a summarization model."""
        if not memories:
            return ""

        # Build a prompt for the summarization model
        memory_contents = [mem.content for mem in memories]
        text = " ".join(memory_contents)

        try:
            # Use the summarization pipeline
            summary = self.summarization_pipeline(text, max_length=130, min_length=30, do_sample=False)
            return summary[0]['summary_text'].strip()
        except Exception as e:
            logger.error(f"Error generating semantic content with summarization model: {e}")
            # Fallback: Use a simple concatenation of memory content
            return " ".join([mem.content for mem in memories])

    async def _evaluate_pattern_significance(self, pattern: Dict) -> float:
        """Evaluates the significance of a pattern before creating a semantic memory."""
        # Placeholder for now, you can replace this with more complex logic later.
        # For example, you can check the frequency of the pattern,
        # the diversity of the memories in the pattern, etc.
        return 1.0  # Every pattern is considered significant for now

    async def consolidate_memories(self):
        logger.info("Starting memory consolidation...")
        try:
            # Define the age threshold for old episodic memories (e.g., 7 days)
            age_threshold = datetime.now() - timedelta(days=7)

            # Find episodic memories older than the age threshold
            old_episodic_memories = [
                self.memory_system._create_memory_from_result(mem)
                for mem in await self.memory_system.pinecone_service.get_all_memories_with_metadata()
                if mem.get("metadata", {}).get("memory_type") == MemoryType.EPISODIC.value
                and datetime.fromisoformat(mem["metadata"]["created_at"]) < age_threshold
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