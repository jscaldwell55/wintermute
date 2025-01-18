from fastapi import FastAPI, HTTPException
from core.memory.models import MemoryType
from pydantic import BaseModel
from core.memory.memory import MemorySystem
from utils.vector_operations import VectorOperations
from utils.pinecone_service import PineconeService
from core.prompts.prompt_templates import (
    RESEARCH_CONTEXT_TEMPLATE,
    PERSONAL_ASSISTANT_TEMPLATE,
    format_prompt
)
from dotenv import load_dotenv
import os
import logging
from contextlib import asynccontextmanager
import utils.llm_service as llm_service
import sys
import asyncio
import openai
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()
print("INDEX_NAME:", os.getenv("INDEX_NAME"))
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("OPENAI_API_KEY is not configured properly.")
else:
    print("OPENAI_API_KEY loaded successfully.")

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Pydantic models for API inputs
class AddMemoryRequest(BaseModel):
    content: str
    metadata: dict = {}

class QueryMemoryRequest(BaseModel):
    prompt: str
    top_k: int = 5

class PromptRequest(BaseModel):
    query: str
    template_type: str
    metadata: dict

# Global variable for MemorySystem
memory_system = None

async def initialize_memory_system() -> MemorySystem:
    """
    Initialize the memory system with Pinecone and vector operations.
    """
    logger.info("Initializing memory system...")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
    vector_operations = VectorOperations()
    memory_system = MemorySystem(
        pinecone_api_key=pinecone_api_key,
        pinecone_environment=pinecone_environment,
        vector_operations=vector_operations,
    )
    logger.info("Memory system initialized successfully.")
    return memory_system

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    """
    global memory_system
    try:
        if memory_system is None:
            memory_system = await initialize_memory_system()
        yield
    finally:
        # Cleanup code here if needed
        pass

# Initialize the FastAPI app with the lifespan context
app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {"message": "Backend is running. Use the appropriate API endpoints."}

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://wintermute-backend.vercel.app/"],  # Replace "*" with specific domains like ["https://weweb.io"] for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/query")
async def query_endpoint(data: dict):
    query = data.get("query", "")
    return {"response": f"Received query: {query}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

@app.post("/memories")
async def add_memory(request: AddMemoryRequest):
    """
    Add a memory to the system.
    """
    try:
        memory_id = await memory_system.add_memory(request.content, request.metadata)
        return {"message": "Memory added", "id": memory_id}
    except Exception as e:
        logger.error(f"Error in /memories: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_memory(request: QueryMemoryRequest):
    """
    Handles memory retrieval and generates a GPT response based on the query.
    """
    try:
        # Construct combined prompt
        combined_prompt = await memory_system.generate_combined_prompt(
            request.prompt, top_k=request.top_k
        )

        # Generate GPT response
        gpt_response = await llm_service.generate_gpt_response_async(
            prompt=combined_prompt,
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=500,
        )

        # Save the interaction as a new memory using add_interaction_memory
        await memory_system.add_interaction_memory(request.prompt, gpt_response)

        return {"query": request.prompt, "response": gpt_response}

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")

@app.post("/consolidate")
async def consolidate():
    """
    Consolidate old memories to optimize storage.
    """
    try:
        await memory_system.consolidate_memories()
        return {"message": "Memory consolidation completed"}
    except Exception as e:
        logger.error(f"Error during memory consolidation: {e}")
        raise HTTPException(status_code=500, detail="Failed to consolidate memories.")

@app.post("/generate_prompt")
async def generate_prompt_endpoint(request: PromptRequest):
    try:
        prompt = await memory_system.generate_prompt(
            query=request.query,
            template_type=request.template_type,
            metadata=request.metadata,
        )
        return {"prompt": prompt}
    except Exception as e:
        logger.error(f"Error generating prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))