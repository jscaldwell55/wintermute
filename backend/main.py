import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Core components
from api.core.memory.memory import MemorySystem, MemoryType
from api.core.memory.summary import EpisodicSummarizer  # New
from api.core.memory.cache import LRUCacheManager  # New
from api.core.vector.vector_operations import VectorOperations
from api.utils.context_window import ContextWindowManager  # Updated
from api.core.scheduling.dream_sequence import DreamSequence
from api.utils.pinecone_service import PineconeService
from api.core.evaluation import MemoryEvaluator
import api.utils.llm_service as llm_service
from api.utils.task_queue import task_queue
from api.utils.config import get_settings

settings = get_settings()

logging.basicConfig(level=logging.INFO if settings.is_production else logging.DEBUG)

# Prompt templates
from api.utils.prompt_templates import (
    MASTER_TEMPLATE,
    format_prompt,
)

# Configure logging
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class Query(BaseModel):
    query: str
    input: str = ""
    top_k: int = 5

class DeleteRequest(BaseModel):
    ids: Optional[List[str]] = None
    delete_all: Optional[bool] = False
    filter: Optional[Dict] = None

class DeleteResponse(BaseModel):
    success: bool

class AddMemoryRequest(BaseModel):
    content: str
    metadata: Optional[Dict] = {}
    memory_type: MemoryType = MemoryType.EPISODIC

class SystemComponents:
    """
    Encapsulates the initialization and management of system components.
    """

    def __init__(self):
        self._initialized = False
        self._lock = asyncio.Lock()
        self.vector_operations = None
        self.pinecone_service = None
        self.memory_system = None
        self.context_window = None
        self.evaluator = None
        self.dream_sequence = None
        self.llm_service = None
        self.episodic_summarizer = None  # New
        self.cache_manager = None  # New

    async def initialize(self):
        """Initializes all system components."""
        async with self._lock:
            if not self._initialized:
                self.vector_operations = VectorOperations()
                self.pinecone_service = PineconeService(
                    api_key=settings.pinecone_api_key,
                    environment=settings.pinecone_environment,
                    index_name=settings.index_name,
                )
                await self.pinecone_service.initialize_index()
                self.llm_service = llm_service.LLMService()
                self.episodic_summarizer = EpisodicSummarizer() # New
                self.cache_manager = LRUCacheManager() # New
                self.context_window = ContextWindowManager(
                    memory_system=None,  # Set after creating MemorySystem
                    max_tokens=settings.max_tokens_per_context_window
                )
                self.memory_system = MemorySystem(
                    pinecone_service=self.pinecone_service,
                    vector_operations=self.vector_operations,
                    context_window=self.context_window, # Pass context window
                    episodic_summarizer=self.episodic_summarizer, # New
                    cache_manager=self.cache_manager, # New
                )
                self.context_window.memory_system = self.memory_system
                self.evaluator = MemoryEvaluation(self.vector_operations, self.memory_system)
                self.dream_sequence = DreamSequence(
                    memory_system=self.memory_system,
                    vector_operations=self.vector_operations,
                    evaluation_module=self.evaluator,
                )

                self._initialized = True

    async def get_memory_system(self):
        if not self._initialized:
            await self.initialize()
        return self.memory_system

    async def get_context_window(self):
        if not self._initialized:
            await self.initialize()
        return self.context_window

    async def get_vector_operations(self):
        if not self._initialized:
            await self.initialize()
        return self.vector_operations

    async def get_evaluator(self):
        if not self._initialized:
            await self.initialize()
        return self.evaluator

    async def get_dream_sequence(self):
        if not self._initialized:
            await self.initialize()
        return self.dream_sequence

    async def get_episodic_summarizer(self):  # New
        if not self._initialized:
            await self.initialize()
        return self.episodic_summarizer

    async def get_cache_manager(self):  # New
        if not self._initialized:
            await self.initialize()
        return self.cache_manager

# Global instance of SystemComponents
components = SystemComponents()

# Dependency Injection Functions
async def get_memory_system():
    return await components.get_memory_system()

async def get_context_window():
    return await components.get_context_window()

async def get_vector_operations():
    return await components.get_vector_operations()

async def get_evaluation_module():
    return await components.get_evaluator()

async def get_dream_sequence():
    return await components.get_dream_sequence()

async def get_episodic_summarizer():
    return await components.get_episodic_summarizer()

async def get_cache_manager():
    return await components.get_cache_manager()

async def run_dream_sequence_async():
    """Execute the dream sequence and update the last dream time."""
    memory_system = await components.get_memory_system()
    dream_sequence = await components.get_dream_sequence()
    logger.info("Initiating dream sequence.")
    try:
        await dream_sequence.run_dream_sequence()
        memory_system._update_last_dream_time(datetime.utcnow())
        logger.info("Dream sequence completed successfully.")
    except Exception as e:
        logger.error(f"Dream sequence failed: {e}")

async def schedule_dream_sequence():
    """Schedule and manage dream sequence execution."""
    memory_system = await components.get_memory_system()
    while True:
        now = datetime.utcnow()
        next_run = memory_system.last_dream_time + timedelta(seconds=settings.dream_time_seconds)
        if now >= next_run:
            asyncio.create_task(run_dream_sequence_async())
            memory_system._update_last_dream_time(now)
        sleep_time = (next_run - now).total_seconds() if next_run > now else 0
        await asyncio.sleep(sleep_time)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application."""
    try:
        await components.initialize()
        asyncio.create_task(schedule_dream_sequence())
        yield
    finally:
        if task_queue:
            task_queue.shutdown()
        if components.memory_system:
            await components.memory_system.shutdown()
        if components.pinecone_service:
            await components.pinecone_service.close_connections()

app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log all requests and responses."""
    logger.info(f"Received request: {request.method} {request.url}")
    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(f"Request failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    logger.info(f"Responded with status: {response.status_code}")
    return response

@app.post("/memories", dependencies=[Depends(get_memory_system)])
async def add_memory(request: AddMemoryRequest, memory_system: MemorySystem = Depends(get_memory_system)):
    """Adds a memory to the system."""
    try:
        memory_id = await memory_system.add_memory(
            content=request.content,
            memory_type=request.memory_type,
            metadata=request.metadata
        )
        return {"message": "Memory added", "id": memory_id}
    except Exception as e:
        logger.error(f"Error in /memories: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", dependencies=[Depends(get_memory_system), Depends(get_context_window), Depends(get_vector_operations)])
async def query_memory(query: Query, memory_system: MemorySystem = Depends(get_memory_system), context_window: ContextWindowManager = Depends(get_context_window), vector_operations: VectorOperations = Depends(get_vector_operations)):
    """Process a query and return a response."""
    logger.info(f"Received query: {query.prompt}")

    try:
        # Vectorize the query using the updated VectorOperations method
        query_vector = await vector_operations.create_semantic_vector(query.prompt)

        # Query memories, passing lists of MemoryType enums
        episodic_memories = await memory_system.query_memory(
            query_vector, top_k=query.top_k, query_types=[MemoryType.EPISODIC]
        )
        semantic_memories = await memory_system.query_memory(
            query_vector, top_k=query.top_k, query_types=[MemoryType.SEMANTIC]
        )

        logger.info(f"Retrieved memories - Episodic: {len(episodic_memories)}, Semantic: {len(semantic_memories)}")

        # Combine and rank the memories
        all_memories = episodic_memories + semantic_memories

        # Format the memories for the prompt template
        retrieved_context = "\n".join(
            [f"- {memory.content}" for memory, _ in all_memories]
        )

        # Format prompt and generate response
        formatted_prompt = format_prompt(MASTER_TEMPLATE, retrieved_context, query.prompt)
        response = await llm_service.generate_gpt_response_async(formatted_prompt, settings.llm_model_id)

        # Add interaction to context window
        await context_window.add_interaction(query.prompt, response)

        # Add new interaction memory
        await memory_system.add_interaction_memory(query.prompt, response)

        logger.info(f"Added interaction memory for query: {query.prompt}")

        return {"response": response}

    except Exception as e:
        logger.error(f"Error during query processing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {e}")
    finally:
        await asyncio.sleep(settings.sleep_time_after_query)

@app.post("/delete")
async def delete_memories(delete_request: DeleteRequest, memory_system: MemorySystem = Depends(get_memory_system)):
    """Delete memories based on provided criteria."""
    try:
        if delete_request.delete_all:
            await memory_system.delete_all_memories()
            return DeleteResponse(success=True)
        
        if not delete_request.ids and not delete_request.filter:
            raise HTTPException(
                status_code=400,
                detail="Must provide ids, a filter, or set delete_all to True",
            )
            
        await memory_system.delete_memories(
            ids=delete_request.ids,
            delete_all=delete_request.delete_all,
            filter=delete_request.filter,
        )
        return DeleteResponse(success=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "environment": settings.environment,
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Server is running!"}