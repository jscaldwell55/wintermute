import json
import os
from openai import OpenAI

def load_synthetic_memories(file_path):
    """Load synthetic memories from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)

def embed_memories(memories, model="text-embedding-ada-002"):
    """Generate embeddings for memories using OpenAI's embedding model."""
    client = OpenAI()  # Initialize the OpenAI client
    embedded_memories = []
    
    for memory in memories:
        try:
            # Generate embedding for the content using new API format
            response = client.embeddings.create(
                input=memory["content"],
                model=model
            )
            embedding = response.data[0].embedding  # New response format

            # Append the embedding to the memory
            embedded_memory = {
                "id": memory["id"],
                "content": memory["content"],
                "metadata": memory["metadata"],
                "embedding": embedding
            }
            embedded_memories.append(embedded_memory)

        except Exception as e:
            print(f"Error embedding memory {memory['id']}: {e}")

    return embedded_memories

def save_embedded_memories(embedded_memories, output_path):
    """Save embedded memories to a JSON file."""
    with open(output_path, "w") as f:
        json.dump(embedded_memories, f, indent=4)
    print(f"Embedded memories saved to: {output_path}")

if __name__ == "__main__":
    # Path to synthetic memories JSON file (on Desktop)
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    input_file = os.path.join(desktop, "synthetic_memories.json")
    output_file = os.path.join(desktop, "embedded_memories.json")

    # Load memories
    memories = load_synthetic_memories(input_file)

    # Generate embeddings
    print("Generating embeddings...")
    embedded_memories = embed_memories(memories)

    # Save embedded memories
    save_embedded_memories(embedded_memories, output_file)