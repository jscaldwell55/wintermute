from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import logging
import asyncio
import uuid
import os
from datetime import timedelta

# Core components
from core.memory.memory import MemorySystem, MemoryType
from core.memory.models import AddMemoryRequest, QueryMemoryRequest
from utils.vector_operations import VectorOperations
from utils.pinecone_service import PineconeService
from core.evaluation import MemoryEvaluation
from utils.dream_sequence import run_dream_sequence
from utils.context_window import ContextWindow
import utils.llm_service as llm_service
from core.prompts.prompt_templates import MASTER_TEMPLATE, format_prompt

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
summary_queue: asyncio.Queue = None

async def initialize_components():
    """Initializes necessary components for the application."""
    global memory_system, context_window, vector_operations, evaluator, summary_queue

    logger.info("Initializing components...")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    vector_operations = VectorOperations()
    pinecone_service = PineconeService(pinecone_api_key, pinecone_environment)
    memory_system = MemorySystem(pinecone_service, vector_operations)
    context_window = ContextWindow()
    evaluator = MemoryEvaluation(vector_operations, memory_system)
    summary_queue = asyncio.Queue()

    logger.info("Components initialized successfully.")

async def process_summary_queue():
    """Processes the summary queue to generate window summaries asynchronously."""
    global context_window
    while True:
        memories = await summary_queue.get()
        if memories:
            logger.info("Generating window summary...")
            summary = await generate_window_summary(memories)
            context_window.generate_template(summary)
            logger.info("Window summary generated.")
        summary_queue.task_done()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application."""
    try:
        await initialize_components()
        asyncio.create_task(schedule_dream_sequence())
        asyncio.create_task(process_summary_queue())  # Start processing the summary queue
        yield
    finally:
        pass  # Cleanup if needed

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
            memory_type=MemoryType.EPISODIC,
            metadata=request.metadata,
            window_id=context_window.window_id,
        )
        return {"message": "Memory added", "id": memory_id}
    except Exception as e:
        logger.error(f"Error in /memories: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_memory(request: QueryMemoryRequest):
    global context_window, summary_queue

    try:
        # 1. Check if the current context window is full
        if not context_window.add_q_r_pair(request.prompt, ""):
            # Window is full:
            # a. Get memories from the current window
            window_memories = await memory_system.get_memories_by_window_id(
                context_window.window_id
            )

            # b. Add memories to the summary queue for async processing
            await summary_queue.put(window_memories)

            # c. Generate a template for the next window (will be updated by process_summary_queue)
            context_window.generate_template("Generating summary...")

            # d. Reset the window
            context_window.reset_window()
            context_window.window_id = str(uuid.uuid4())

        # 2. Create combined vector for query
        query_vector = await vector_operations.create_semantic_vector(request.prompt)

        # 3. Retrieve relevant memories (both episodic and semantic)
        relevant_memories = await memory_system.query_memory(
            query_vector,
            query_types=[MemoryType.EPISODIC, MemoryType.SEMANTIC],
            k=request.top_k,
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
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=500,
        )

        # 7. Add the Q/R pair to the context window
        context_window.add_q_r_pair(request.prompt, gpt_response)

        # 8. Store interaction memory (as episodic)
        await memory_system.add_memory(
            content=f"Q: {request.prompt}\nA: {gpt_response}",
            memory_type=MemoryType.EPISODIC,
            metadata={"interaction": True},
            window_id=context_window.window_id,
        )

        logger.info(f"Returning response: {gpt_response}")
        return {"query": request.prompt, "response": gpt_response}

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request.",
        )

async def generate_window_summary(memories: List[Memory]) -> str:
    """
    Generates a summary of the memories in the current context window using an LLM.

    Args:
        memories: A list of Memory objects representing the memories in the window.

    Returns:
        A string containing the generated summary.
    """
    if not memories:
        return ""

    # Build a prompt for the LLM to summarize the memories
    memory_contents = [mem.content for mem in memories]
    prompt = f"""
    Summarize the following conversation excerpts:

    {' '.join(memory_contents)}

    Summary:
    """

    try:
        summary = await llm_service.generate_gpt_response_async(
            prompt=prompt,
            model="gpt-3.5-turbo",  # Or another suitable model
            temperature=0.5,  # Adjust temperature for summarization
            max_tokens=256,  # Adjust max_tokens for summary length
        )
        return summary.strip()
    except Exception as e:
        logger.error(f"Error generating window summary: {e}")
        return ""  # Return an empty string on error

async def schedule_dream_sequence():
    """Schedules the dream sequence to run at a specific time or interval."""
    while True:
        # Example: Run every day at 3:00 AM
        now = datetime.now()
        target_time = now.replace(hour=3, minute=0, second=0, microsecond=0)
        if now > target_time:
            target_time += timedelta(days=1)
        sleep_duration = (target_time - now).total_seconds()
        await asyncio.sleep(sleep_duration)

        await run_dream_sequence(memory_system, vector_operations, evaluator)
        memory_system._update_last_dream_time(datetime.now())