# config.py
import os

# --- LLM Service ---
LLM_API_KEY = os.environ.get("LLM_API_KEY")
LLM_MODEL_NAME = "gpt-3.5-turbo"
LLM_API_BASE_URL = "https://api.openai.com/v1" #For OpenAI models
LLM_API_RATE_LIMIT_PER_MINUTE = 30
SUMMARY_LLM_MODEL_NAME = "gpt-3.5-turbo"  # Can be the same or a different model

# --- Pinecone Service ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "pcsk_hzCox_SLS1Yi92BDA7QNvqsKV1pbPZhuzP3rxCU9jcHuDkSCYnCbCuP3YQdsPUQs5ZYbp")
PINECONE_ENVIRONMENT = "us-east-1"  # Replace with your environment
PINECONE_INDEX_NAME = "wintermute"
PINECONE_API_RATE_LIMIT_PER_SECOND = 3

# --- Memory ---
MEMORY_DECAY_FACTOR = 0.99
HOURS_SINCE_DECAY = 24
MAX_TOKENS_PER_CONTEXT_WINDOW = 4096 # Adjust this to fit for your chosen LLM
SUMMARY_RETRY_DELAY = 5
SUMMARY_RETRIES = 3
DREAM_SEQUENCE_INTERVAL_HOURS = 24 # How often to run the dream sequence

# --- Task Queue ---
TASK_QUEUE_NUM_WORKERS = 4

# --- Clustering (Dream Sequence) ---
KMEANS_N_CLUSTERS_MIN = 2
KMEANS_N_CLUSTERS_MAX = 10
KMEANS_INIT = 'k-means++'
KMEANS_MAX_ITER = 300
KMEANS_N_INIT = 10

# --- Semantic Memory Quality Thresholds ---
SEMANTIC_MEMORY_RELEVANCE_THRESHOLD = 0.7
SEMANTIC_MEMORY_COHERENCE_THRESHOLD = 3