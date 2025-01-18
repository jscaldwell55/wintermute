import uuid
import sys
import os
import pinecone
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(backend, "../.env"))

from utils.pinecone_service import initialize_pinecone

# Assuming you have environment variables set for these:
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = os.getenv("INDEX_NAME")  # Make sure this is set in your environment

def add_memories(memories, index_name=INDEX_NAME):
    """
    Adds test memories to the Pinecone index.

    Args:
        memories (list): A list of dictionaries, each representing a memory.
        index_name (str): The name of the Pinecone index.
    """

    if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
        raise ValueError("Pinecone API key and environment must be set in environment variables.")

    # Initialize Pinecone service
    pinecone_service = PineconeService(
        api_key=PINECONE_API_KEY, 
        environment=PINECONE_ENVIRONMENT
    )

    # Initialize vector operations
    vector_operations = VectorOperations()

    # Get the index
    index = pinecone_service.pinecone.Index(index_name)

    for memory in memories:
        try:
            # Generate embedding for the memory content
            memory_vector = vector_operations.generate_embedding(memory["content"])

            # Prepare the vector for upsertion
            vector = {
                "id": memory["id"],
                "values": memory_vector,
                "metadata": memory["metadata"],
            }

            # Upsert the memory into Pinecone
            index.upsert(vectors=[vector])
            print(f"Memory {memory['id']} added successfully!")

        except Exception as e:
            print(f"Error adding memory {memory.get('id', 'N/A')}: {e}")

if __name__ == "__main__":
    test_memories = [
        {
            "id": str(uuid.uuid4()),
            "content": "This is a new test memory 1.",
            "metadata": {"category": "test", "memory_type": "TEMPORAL"},
        },
        {
            "id": str(uuid.uuid4()),
            "content": "This is another test memory 2.",
            "metadata": {"category": "test", "memory_type": "TEMPORAL"},
        },
    ]

    add_memories(test_memories)