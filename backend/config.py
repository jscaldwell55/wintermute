# config.py
import os

# --- LLM Service ---
LLM_API_KEY = os.environ.get("OPENAI_API_KEY")  # Changed to use the correct env variable name
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "gpt-3.5-turbo")  # Use an environment variable for model ID
LLM_API_BASE_URL = os.environ.get("LLM_API_BASE_URL", "https://api.openai.com/v1")  # Allow overriding the base URL
LLM_API_RATE_LIMIT_PER_MINUTE = int(os.environ.get("LLM_API_RATE_LIMIT_PER_MINUTE", 30))
SUMMARY_LLM_MODEL_ID = os.environ.get("SUMMARY_LLM_MODEL_ID", "gpt-3.5-turbo")  # Can be the same or a different model

# --- Pinecone Service ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", "us-east-1")
INDEX_NAME = os.environ.get("INDEX_NAME", "wintermute")  # Changed to use environment variable
PINECONE_API_RATE_LIMIT_PER_SECOND = int(os.environ.get("PINECONE_API_RATE_LIMIT_PER_SECOND", 3))

# --- Memory ---
MEMORY_DECAY_FACTOR = float(os.environ.get("MEMORY_DECAY_FACTOR", 0.99))
HOURS_SINCE_DECAY = float(os.environ.get("HOURS_SINCE_DECAY", 24))
MAX_TOKENS_PER_CONTEXT_WINDOW = int(os.environ.get("MAX_TOKENS_PER_CONTEXT_WINDOW", 4096))
SUMMARY_RETRY_DELAY = int(os.environ.get("SUMMARY_RETRY_DELAY", 5))
SUMMARY_RETRIES = int(os.environ.get("SUMMARY_RETRIES", 3))
DREAM_SEQUENCE_INTERVAL_HOURS = int(os.environ.get("DREAM_SEQUENCE_INTERVAL_HOURS", 24))
DELETE_THRESHOLD_DAYS = int(os.environ.get("DELETE_THRESHOLD_DAYS", 7)) # Threshold for deleting old memories

# --- Task Queue ---
TASK_QUEUE_NUM_WORKERS = int(os.environ.get("TASK_QUEUE_NUM_WORKERS", 4))

# --- Clustering (Dream Sequence) ---
KMEANS_N_CLUSTERS_MIN = int(os.environ.get("KMEANS_N_CLUSTERS_MIN", 2))
KMEANS_N_CLUSTERS_MAX = int(os.environ.get("KMEANS_N_CLUSTERS_MAX", 10))
KMEANS_INIT = os.environ.get("KMEANS_INIT", 'k-means++')
KMEANS_MAX_ITER = int(os.environ.get("KMEANS_MAX_ITER", 300))
KMEANS_N_INIT = int(os.environ.get("KMEANS_N_INIT", 10))

# --- Semantic Memory Quality Thresholds ---
SEMANTIC_MEMORY_RELEVANCE_THRESHOLD = float(os.environ.get("SEMANTIC_MEMORY_RELEVANCE_THRESHOLD", 0.7))
SEMANTIC_MEMORY_COHERENCE_THRESHOLD = float(os.environ.get("SEMANTIC_MEMORY_COHERENCE_THRESHOLD", 3))

# --- Hugging Face (Summarization) ---
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
SUMMARY_MODEL_NAME = os.environ.get("SUMMARY_MODEL_NAME", "facebook/bart-large-cnn") # Example default
SUMMARY_MAX_LENGTH = int(os.environ.get("SUMMARY_MAX_LENGTH", 150)) # Example default
SUMMARY_MIN_LENGTH = int(os.environ.get("SUMMARY_MIN_LENGTH", 50)) # Example default

# --- Dream Sequence Settings ---
EVALUATION_RETRY_ATTEMPTS = int(os.environ.get("EVALUATION_RETRY_ATTEMPTS", 3))
EVALUATION_RETRY_DELAY = int(os.environ.get("EVALUATION_RETRY_DELAY", 5))
SEMANTIC_MEMORY_THRESHOLD = float(os.environ.get("SEMANTIC_MEMORY_THRESHOLD", 0.85))
SUMMARY_RETRY_ATTEMPTS = int(os.environ.get("SUMMARY_RETRY_ATTEMPTS", 3))
SUMMARY_RETRY_DELAY = int(os.environ.get("SUMMARY_RETRY_DELAY", 5))

# --- General ---
ENVIRONMENT = os.environ.get("ENVIRONMENT", "dev") # Environment (dev, staging, production)

# --- Debugging and Logging ---
DEBUG_MODE = os.environ.get("DEBUG_MODE", "False").lower() == "true"  # Set to "True" or "False" string in env

# --- main.py config ---
SLEEP_TIME_AFTER_QUERY = float(os.environ.get("SLEEP_TIME_AFTER_QUERY", 1.0))
DREAM_TIME_SECONDS = int(os.environ.get("DREAM_TIME_SECONDS", 1800))