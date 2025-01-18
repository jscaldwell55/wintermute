import json
import os
from pinecone import Pinecone, ServerlessSpec

# Set Pinecone API details
os.environ["PINECONE_API_KEY"] = "pcsk_hzCox_SLS1Yi92BDA7QNvqsKV1pbPZhuzP3rxCU9jcHuDkSCYnCbCuP3YQdsPUQs5ZYbp"  # Replace with your Pinecone API Key
os.environ["PINECONE_ENVIRONMENT"] = "us-east-1"       # Replace with your Pinecone environment
os.environ["INDEX_NAME"] = "wintermute"                # Replace with your Pinecone index name

# Initialize Pinecone
def initialize_pinecone():
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
    INDEX_NAME = os.getenv("INDEX_NAME")

    if not all([PINECONE_API_KEY, PINECONE_ENVIRONMENT, INDEX_NAME]):
        raise ValueError("Pinecone environment variables are not properly set.")

    # Create Pinecone instance
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Create or get index
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,  # Assuming embeddings are 1536-dimensional
            metric="cosine",  # Adjust based on your use case (e.g., "euclidean")
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
        )
    return pc.Index(INDEX_NAME)

# Read JSON File
def load_data_from_json(json_file_path):
    try:
        with open(json_file_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise ValueError(f"Error reading JSON file: {e}")

# Upsert Data to Pinecone
def upsert_to_pinecone(index, data):
    upsert_batch = []
    for memory in data:
        memory_id = memory.get("id")
        embedding = memory.get("embedding")  # Assuming 'embedding' key contains the 1536-d vector
        metadata = memory.get("metadata", {})  # Metadata field

        if memory_id and embedding:
            upsert_batch.append({
                "id": memory_id,
                "values": embedding,
                "metadata": metadata
            })
        else:
            print(f"Skipping memory due to missing ID or embedding: {memory}")

    # Upsert in batches to Pinecone
    BATCH_SIZE = 100  # Adjust batch size as needed
    for i in range(0, len(upsert_batch), BATCH_SIZE):
        batch = upsert_batch[i:i + BATCH_SIZE]
        try:
            index.upsert(vectors=batch)
            print(f"Upserted batch {i // BATCH_SIZE + 1} of {len(upsert_batch) // BATCH_SIZE + 1}")
        except Exception as e:
            print(f"Error upserting batch {i // BATCH_SIZE + 1}: {e}")

if __name__ == "__main__":
    # Set path to your JSON file
    json_file_path = "/Users/jaycaldwell/Desktop/embedded_memories.json"  # Replace with actual path

    # Initialize Pinecone index
    try:
        pinecone_index = initialize_pinecone()
        print("Pinecone index initialized successfully.")
    except Exception as e:
        print(f"Error initializing Pinecone: {e}")
        exit(1)

    # Load data from JSON
    try:
        embedded_data = load_data_from_json(json_file_path)
        print(f"Loaded {len(embedded_data)} memories from JSON file.")
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

    # Upsert data to Pinecone
    try:
        upsert_to_pinecone(pinecone_index, embedded_data)
        print("All data has been upserted to Pinecone!")
    except Exception as e:
        print(f"Error upserting data to Pinecone: {e}")
        exit(1)
