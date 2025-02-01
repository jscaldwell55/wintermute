import logging
from typing import List, Tuple, Optional
from api.utils.config import get_settings
from collections import deque

logger = logging.getLogger(__name__)

class ContextWindowManager:
    def __init__(self, memory_system, max_tokens: Optional[int] = None):
        self.settings = get_settings()
        self.memory_system = memory_system
        self.max_tokens = max_tokens or self.settings.max_tokens_per_context_window
        self.current_token_count = 0
        self.recent_interactions = deque()  # Store recent interactions
        self.retain_percentage = 0.3  # Keep 30% of window for recent interactions
        self.active_window = deque()

    def add_interaction(self, interaction: Tuple[str, str]):
        """Adds a new interaction (query, response) to the context window."""
        interaction_tokens = self.count_tokens(str(interaction))
        if self.current_token_count + interaction_tokens > self.max_tokens:
            self.manage_window_size()

        self.recent_interactions.append(interaction)
        self.active_window.append(interaction)
        self.current_token_count += interaction_tokens
        logger.info("Added interaction to context window.")

    async def add_memory(self, memory):
        """Adds a memory to the context window and manages its size."""
        memory_tokens = self.count_tokens(memory.content)

        if self.current_token_count + memory_tokens > self.max_tokens:
          await self.manage_window_size()

        self.active_window.append(memory)
        self.current_token_count += memory_tokens
        logger.info(f"Added memory to context window: {memory.id}")

    async def manage_window_size(self):
        """Manages the size of the context window by paging out memories when it's full."""
        logger.info("Managing context window size.")
        retained_interactions = []
        retained_tokens_count = 0

        # Preserve a portion of recent interactions
        for interaction in reversed(list(self.recent_interactions)):
            interaction_tokens = self.count_tokens(str(interaction))
            if retained_tokens_count + interaction_tokens <= self.max_tokens * self.retain_percentage:
                retained_interactions.insert(0, interaction)
                retained_tokens_count += interaction_tokens
            else:
                break
        
        # Identify memories to page out (excluding recent interactions)
        memories_to_page_out = [mem for mem in self.active_window if mem not in retained_interactions]

        # Page out the selected memories
        await self.page_out_memories(memories_to_page_out)

        # Update the active window
        self.active_window = deque(retained_interactions)
        self.current_token_count = retained_tokens_count
        logger.info("Context window size managed.")

    async def page_out_memories(self, memories_to_page_out):
        """Pages out memories, generates summaries, and stores them in the LRU cache."""
        for memory in memories_to_page_out:
            # Generate a summary and store it in the LRU cache
            summary = await self.memory_system.episodic_summarizer.generate_summary(memory.content)
            self.memory_system.cache_manager.store_summary(memory.id, summary)

            # Store the full memory in long-term storage (Pinecone)
            await self.memory_system.add_memory(
                content=memory.content,
                memory_type=memory.memory_type,
                metadata=memory.metadata,
                summary=summary['summary_text'],
                summary_embedding=summary['embedding']
            )

    def count_tokens(self, text):
        """Counts the number of tokens in the given text."""
        # Placeholder for your actual token counting logic
        return len(text.split())