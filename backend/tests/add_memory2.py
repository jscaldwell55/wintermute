import os
import asyncio
from core.memory.memory import MemorySystem
from utils.vector_operations import VectorOperations

# Set required environment variables
os.environ["INDEX_NAME"] = "wintermute"  # Replace with your Pinecone index name
os.environ["PINECONE_API_KEY"] = "pcsk_hzCox_SLS1Yi92BDA7QNvqsKV1pbPZhuzP3rxCU9jcHuDkSCYnCbCuP3YQdsPUQs5ZYbp"  # Replace with your Pinecone API key
os.environ["PINECONE_ENVIRONMENT"] = "us-east-1"  # Replace with your Pinecone environment

async def test_add_memory():
    # Initialize required dependencies
    vector_operations = VectorOperations()

    # Initialize MemorySystem (without emotional analysis)
    memory_system = MemorySystem(
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        pinecone_environment=os.getenv("PINECONE_ENVIRONMENT"),
        vector_operations=vector_operations,
    )

    # Define memory content and metadata
    content = "This is a sample memory with semantic embeddings only."
    metadata = {"category": "test"}

    # Add the memory
    try:
        memory_id = await memory_system.add_memory(content, metadata)
        print(f"Memory added successfully with ID: {memory_id}")
    except Exception as e:
        print(f"Error adding memory: {e}")

# Run the test
if __name__ == "__main__":
    asyncio.run(test_add_memory())
