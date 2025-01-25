import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

# Core components - Now relative to 'api'
from api.core.memory.memory import MemorySystem, MemoryType
# from api.core.vector.vector_model import VectorModel # Removed incorrect import
from api.utils.vector.vector_operations import MockVectorOperations, VectorOperations
from api.utils.context.context_window import ContextWindow
from api.utils.dream_sequence import DreamSequence
from api.utils.pinecone_service import PineconeService
from api.core.evaluation.evaluator import MemoryEvaluation
import api.utils.llm_service as llm_service
from api.utils.task_queue import task_queue
from api.utils.config import (
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    INDEX_NAME,
    ENVIRONMENT,
    LLM_MODEL_ID,
    VECTOR_MODEL_ID,
    HUGGINGFACEHUB_API_TOKEN,
    SLEEP_TIME_AFTER_QUERY,
    DREAM_TIME_SECONDS,
    SUMMARY_RETRY_DELAY,
    SUMMARY_RETRIES
)

# Prompt templates
from api.utils.prompt_templates import (
    MASTER_TEMPLATE,
    format_prompt,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MemorySystem
memory_system: MemorySystem = None

# Initialize ContextWindow
context_window: ContextWindow = None

# Initialize VectorOperations
vector_operations: VectorOperations = None

# Initialize MemoryEvaluation
evaluator: MemoryEvaluation = None

# Initialize DreamSequence
dream_sequence: DreamSequence = None

# --- Pydantic Models ---
class Query(BaseModel):
    prompt: str
    top_k: int = 5

class DeleteRequest(BaseModel):
    ids: List[str] = None
    delete_all: bool = False
    filter: dict = None

class DeleteResponse(BaseModel):
    success: bool

class AddMemoryRequest(BaseModel):
    content: str
    metadata: Dict = {}
    memory_type: MemoryType = MemoryType.EPISODIC

async def initialize_components():
    """Initializes necessary components for the application."""
    global memory_system, context_window, vector_operations, evaluator, dream_sequence

    logger.info("Initializing components...")
    
    # Initialize VectorModel
    vector_model = VectorModel(VECTOR_MODEL_ID, normalize=True)

    # Choose the vector operations implementation based on the environment
    if ENVIRONMENT == "test":
        vector_operations = MockVectorOperations(vector_model)
    else:
        vector_operations = VectorOperations(vector_model)

    # Initialize PineconeService
    pinecone_service = PineconeService(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENVIRONMENT,
        index_name=INDEX_NAME
    )

    # Initialize MemorySystem
    memory_system = MemorySystem(
        pinecone_service=pinecone_service,
        vector_operations=vector_operations,
        environment=ENVIRONMENT
    )

    # Initialize ContextWindow
    context_window = ContextWindow(
        window_id=ContextWindow.generate_window_id(),
        max_tokens=4096,
        template=MASTER_TEMPLATE,
        summary="",
        memories=[],
        interactions=[]
    )

    # Initialize MemoryEvaluation
    evaluator = MemoryEvaluation(vector_operations, memory_system)

    # Initialize DreamSequence
    dream_sequence = DreamSequence(
        memory_system,
        vector_operations,
        evaluator,
        HUGGINGFACEHUB_API_TOKEN
    )

    logger.info("Components initialized successfully.")

async def run_dream_sequence_async():
    global memory_system
    logger.info("Initiating dream sequence.")
    try:
        await dream_sequence.run_dream_sequence()
        memory_system._update_last_dream_time(datetime.utcnow())
        logger.info("Dream sequence completed successfully.")
    except Exception as e:
        logger.error(f"Dream sequence failed: {e}")

async def schedule_dream_sequence():
    while True:
        now = datetime.utcnow()
        next_run = memory_system.last_dream_time + timedelta(seconds=DREAM_TIME_SECONDS)
        if now >= next_run:
            asyncio.create_task(run_dream_sequence_async())
            memory_system._update_last_dream_time(now)
        sleep_time = (next_run - now).total_seconds() if next_run > now else 0
        await asyncio.sleep(sleep_time)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application."""
    try:
        await initialize_components()
        asyncio.create_task(schedule_dream_sequence())
        yield
    finally:
        task_queue.shutdown()

app = FastAPI(lifespan=lifespan)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Received request: {request.method} {request.url}")
    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(f"Request failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    logger.info(f"Responded with status: {response.status_code}")
    return response

@app.post("/memories")
async def add_memory(request: AddMemoryRequest):
    """
    Adds a memory to the system.
    """
    try:
        memory_id = await memory_system.add_memory(
            content=request.content,
            memory_type=request.memory_type,
            metadata=request.metadata,
            window_id=context_window.window_id,
        )
        return {"message": "Memory added", "id": memory_id}
    except Exception as e:
        logger.error(f"Error in /memories: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_memory(query: Query):
    global context_window
    logger.info(f"Received query: {query.prompt}")

    try:
        # Vectorize the query
        query_vector = await vector_operations.embed_query(query.prompt)

        # Query episodic memories
        episodic_memories = await memory_system.query_memory(
            query_vector, query.top_k, MemoryType.EPISODIC
        )
        logger.info(f"Episodic memories retrieved: {len(episodic_memories)}")

        # Query semantic memories
        semantic_memories = await memory_system.query_memory(
            query_vector, query.top_k, MemoryType.SEMANTIC
        )
        logger.info(f"Semantic memories retrieved: {len(semantic_memories)}")

        # Format the prompt with the retrieved memories
        formatted_prompt = format_prompt(MASTER_TEMPLATE, query.prompt, episodic_memories, semantic_memories)

        # Generate a response using the formatted prompt
        response = await llm_service.generate_gpt_response_async(formatted_prompt, LLM_MODEL_ID)
        logger.info(f"Generated response: {response}")

        # Update the context window and determine if summarization is needed
        is_window_full = context_window.is_full()
        if is_window_full:
            logger.info("Context window is full. Triggering summarization.")
            # Add memories to the task queue for async processing with retries
            task_queue.enqueue(
                context_window.generate_window_summary,
                context_window.interactions,
                context_window.window_id,
                retries=SUMMARY_RETRIES,
                retry_delay=SUMMARY_RETRY_DELAY
            )
            context_window = ContextWindow.reset_context(context_window)

        # Add new episodic memory
        await memory_system.add_interaction_memory(query.prompt, response, context_window.window_id)
        logger.info(f"Episodic memory added for window ID: {context_window.window_id}")

        return {"response": response}

    except Exception as e:
        logger.error(f"Error during query processing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {e}")
    finally:
        await asyncio.sleep(SLEEP_TIME_AFTER_QUERY)

@app.post("/delete")
async def delete_memories(delete_request: DeleteRequest):
    try:
        if delete_request.delete_all:
            await memory_system.delete_all_memories()
            return DeleteResponse(success=True)
        else:
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

@app.get("/")
async def root():
    return {"message": "Server is running!"}