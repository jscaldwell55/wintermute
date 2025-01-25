from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import asyncio
import uuid
import os
from datetime import timedelta
from pydantic import BaseModel

# Core components
from backend.core.memory.memory import MemorySystem, MemoryType
from backend.core.memory.models import Memory
from backend.utils.vector_operations import VectorOperations
from backend.utils.pinecone_service import PineconeService
from backend.core.evaluation import MemoryEvaluation
from backend.utils.dream_sequence import DreamSequence
from backend.utils.context_window import ContextWindow
import backend.utils.llm_service as llm_service
from backend.utils.task_queue import task_queue
import backend.config as config
from backend.core.prompts.prompt_templates import (
    MASTER_TEMPLATE,
    format_prompt,
)

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize global variables
memory_system: MemorySystem = None
context_window: ContextWindow = None
vector_operations: VectorOperations = None
evaluator: MemoryEvaluation = None
dream_sequence: DreamSequence = None

# Pydantic models for API inputs
class AddMemoryRequest(BaseModel):
    content: str
    metadata: dict = {}
    memory_type: MemoryType = MemoryType.EPISODIC

class QueryMemoryRequest(BaseModel):
    prompt: str
    top_k: int = 5

class PromptRequest(BaseModel):
    query: str
    metadata: dict

async def initialize_components():
    """Initializes necessary components for the application."""
    global memory_system, context_window, vector_operations, evaluator, dream_sequence

    logger.info("Initializing components...")
    pinecone_api_key = config.PINECONE_API_KEY
    pinecone_environment = config.PINECONE_ENVIRONMENT
    openai_api_key = config.LLM_API_KEY

    vector_operations = VectorOperations()
    pinecone_service = PineconeService(pinecone_api_key, pinecone_environment)
    memory_system = MemorySystem(pinecone_service, vector_operations)
    context_window = ContextWindow()
    evaluator = MemoryEvaluation(vector_operations, memory_system)
    dream_sequence = DreamSequence(memory_system, vector_operations, evaluator)

    logger.info("Components initialized successfully.")

async def run_dream_sequence_task():
    """Schedules the dream sequence to run periodically."""
    while True:
        # Example: Run every day at 3:00 AM
        now = datetime.now()
        target_time = now.replace(hour=3, minute=0, second=0, microsecond=0)
        if now > target_time:
            target_time += timedelta(days=1)
        sleep_duration = (target_time - now).total_seconds()
        await asyncio.sleep(sleep_duration)

        await dream_sequence.run_dream_sequence()
        memory_system._update_last_dream_time(datetime.now())

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application."""
    try:
        await initialize_components()
        asyncio.create_task(run_dream_sequence_task())
        yield
    finally:
        task_queue.shutdown()

app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {"message": "Backend is running. Use the appropriate API endpoints."}

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
async def query_memory(request: QueryMemoryRequest):
    global context_window

    try:
        # 1. Check if the current context window is full
        if context_window.is_full():
            # Window is full:
            # a. Get memories from the current window
            window_memories = await memory_system.get_memories_by_window_id(
                context_window.window_id
            )

            # b. Add memories to the task queue for async processing with retries
            task_queue.enqueue(
                context_window.generate_window_summary,
                window_memories,
                context_window.window_id,
                retries=config.SUMMARY_RETRIES,
                retry_delay=config.SUMMARY_RETRY_DELAY
            )
            logger.info("Summary task added to queue.")

            # c. Reset the window
            context_window.reset_window()

        # 2. Create combined vector for query
        query_vector = await vector_operations.create_semantic_vector(request.prompt)

        # 3. Retrieve relevant memories (both episodic and semantic)
        relevant_memories = await memory_system.query_memory(
            query_vector,
            query_types=[MemoryType.EPISODIC, MemoryType.SEMANTIC],
            top_k=request.top_k,
        )

        # 4. Format the memories for the prompt
        retrieved_context = "\n".join(
            [f"- {memory.content}" for memory, _ in relevant_memories]
        )

        # 5. Generate the prompt using the master template
        prompt = format_prompt(
            MASTER_TEMPLATE,
            retrieved_context=retrieved_context,
            user_query=request.prompt,
        )

        # Add the current template to the prompt if available
        if context_window.template:
            prompt = context_window.template + "\n" + prompt

        # 6. Get GPT-3 response
        gpt_response = await llm_service.generate_gpt_response_async(
            prompt=prompt,
            model=config.LLM_MODEL_NAME,
            temperature=0.7,
            max_tokens=500,
        )

        # 7. Add the Q/R pair to the context window
        context_window.add_q_r_pair(request.prompt, gpt_response)

        # 8. Store interaction memory (as episodic)
        await memory_system.add_interaction_memory(
            user_query=request.prompt,
            gpt_response=gpt_response,
            window_id=context_window.window_id
        )

        logger.info(f"Returning response: {gpt_response}")
        return {"query": request.prompt, "response": gpt_response}

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request.",
        )