import json
import os
from faker import Faker

def generate_synthetic_memories(num_memories=1000):
    """Generate synthetic memories using Faker and save them as JSON to the desktop."""
    fake = Faker()
    memories = []

    for _ in range(num_memories):
        memory = {
            "id": fake.uuid4(),
            "content": fake.text(max_nb_chars=200),
            "metadata": {
                "created_at": fake.date_time_this_year().isoformat(),
                "memory_type": fake.random_element(elements=["TEMPORAL", "GROUNDING"]),
                "tags": [fake.word() for _ in range(fake.random_int(1, 5))]
            }
        }
        memories.append(memory)

    # Save to JSON on Desktop
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    file_path = os.path.join(desktop, "synthetic_memories.json")

    with open(file_path, "w") as f:
        json.dump(memories, f, indent=4)
    
    print(f"Synthetic memories saved to: {file_path}")

# Generate 1000 synthetic memories
generate_synthetic_memories()
