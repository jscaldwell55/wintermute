# config.py
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # Model configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="allow"
    )

    # --- LLM Service ---
    LLM_API_KEY: str = Field(default=None, alias="OPENAI_API_KEY")
    LLM_MODEL_ID: str = Field(default="gpt-3.5-turbo")
    LLM_API_BASE_URL: str = Field(default="https://api.openai.com/v1")
    LLM_API_RATE_LIMIT_PER_MINUTE: int = Field(default=30)
    SUMMARY_LLM_MODEL_ID: str = Field(default="gpt-3.5-turbo")

    # --- Pinecone Service ---
    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: str = Field(default="us-east-1")
    INDEX_NAME: str = Field(default="wintermute")
    PINECONE_API_RATE_LIMIT_PER_SECOND: int = Field(default=3)

    # --- Memory ---
    MEMORY_DECAY_FACTOR: float = Field(default=0.99)
    HOURS_SINCE_DECAY: float = Field(default=24.0)
    MAX_TOKENS_PER_CONTEXT_WINDOW: int = Field(default=4096)
    SUMMARY_RETRY_DELAY: int = Field(default=5)
    SUMMARY_RETRIES: int = Field(default=3)
    DREAM_SEQUENCE_INTERVAL_HOURS: int = Field(default=24)
    DELETE_THRESHOLD_DAYS: int = Field(default=7)

    # --- Task Queue ---
    TASK_QUEUE_NUM_WORKERS: int = Field(default=4)

    # --- Clustering (Dream Sequence) ---
    KMEANS_N_CLUSTERS_MIN: int = Field(default=2)
    KMEANS_N_CLUSTERS_MAX: int = Field(default=10)
    KMEANS_INIT: str = Field(default='k-means++')
    KMEANS_MAX_ITER: int = Field(default=300)
    KMEANS_N_INIT: int = Field(default=10)

    # --- Semantic Memory Quality Thresholds ---
    SEMANTIC_MEMORY_RELEVANCE_THRESHOLD: float = Field(default=0.7)
    SEMANTIC_MEMORY_COHERENCE_THRESHOLD: float = Field(default=3.0)
    SEMANTIC_MEMORY_THRESHOLD: float = Field(default=0.85)

    # --- Hugging Face (Summarization) ---
    HUGGINGFACEHUB_API_TOKEN: str | None = None
    SUMMARY_MODEL_NAME: str = Field(default="facebook/bart-large-cnn")
    SUMMARY_MAX_LENGTH: int = Field(default=150)
    SUMMARY_MIN_LENGTH: int = Field(default=50)

    # --- Dream Sequence Settings ---
    EVALUATION_RETRY_ATTEMPTS: int = Field(default=3)
    EVALUATION_RETRY_DELAY: int = Field(default=5)
    SUMMARY_RETRY_ATTEMPTS: int = Field(default=3)
    SUMMARY_RETRY_DELAY: int = Field(default=5)

    # --- General ---
    ENVIRONMENT: str = Field(default="dev")
    DEBUG_MODE: bool = Field(default=False)

    # --- main.py config ---
    SLEEP_TIME_AFTER_QUERY: float = Field(default=1.0)
    DREAM_TIME_SECONDS: int = Field(default=1800)

    # --- Vector Model ---
    VECTOR_MODEL_ID: str = Field(default="text-embedding-ada-002")

    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.ENVIRONMENT.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.ENVIRONMENT.lower() in ["dev", "development"]

# Create global settings instance
settings = Settings()

# Export ALL settings as module-level variables for backward compatibility
# --- LLM Service ---
LLM_API_KEY = settings.LLM_API_KEY
LLM_MODEL_ID = settings.LLM_MODEL_ID
LLM_API_BASE_URL = settings.LLM_API_BASE_URL
LLM_API_RATE_LIMIT_PER_MINUTE = settings.LLM_API_RATE_LIMIT_PER_MINUTE
SUMMARY_LLM_MODEL_ID = settings.SUMMARY_LLM_MODEL_ID

# --- Pinecone Service ---
PINECONE_API_KEY = settings.PINECONE_API_KEY
PINECONE_ENVIRONMENT = settings.PINECONE_ENVIRONMENT
INDEX_NAME = settings.INDEX_NAME
PINECONE_API_RATE_LIMIT_PER_SECOND = settings.PINECONE_API_RATE_LIMIT_PER_SECOND

# --- Memory ---
MEMORY_DECAY_FACTOR = settings.MEMORY_DECAY_FACTOR
HOURS_SINCE_DECAY = settings.HOURS_SINCE_DECAY
MAX_TOKENS_PER_CONTEXT_WINDOW = settings.MAX_TOKENS_PER_CONTEXT_WINDOW
SUMMARY_RETRY_DELAY = settings.SUMMARY_RETRY_DELAY
SUMMARY_RETRIES = settings.SUMMARY_RETRIES
DREAM_SEQUENCE_INTERVAL_HOURS = settings.DREAM_SEQUENCE_INTERVAL_HOURS
DELETE_THRESHOLD_DAYS = settings.DELETE_THRESHOLD_DAYS

# --- Task Queue ---
TASK_QUEUE_NUM_WORKERS = settings.TASK_QUEUE_NUM_WORKERS

# --- Clustering (Dream Sequence) ---
KMEANS_N_CLUSTERS_MIN = settings.KMEANS_N_CLUSTERS_MIN
KMEANS_N_CLUSTERS_MAX = settings.KMEANS_N_CLUSTERS_MAX
KMEANS_INIT = settings.KMEANS_INIT
KMEANS_MAX_ITER = settings.KMEANS_MAX_ITER
KMEANS_N_INIT = settings.KMEANS_N_INIT

# --- Semantic Memory Quality Thresholds ---
SEMANTIC_MEMORY_RELEVANCE_THRESHOLD = settings.SEMANTIC_MEMORY_RELEVANCE_THRESHOLD
SEMANTIC_MEMORY_COHERENCE_THRESHOLD = settings.SEMANTIC_MEMORY_COHERENCE_THRESHOLD
SEMANTIC_MEMORY_THRESHOLD = settings.SEMANTIC_MEMORY_THRESHOLD

# --- Hugging Face (Summarization) ---
HUGGINGFACEHUB_API_TOKEN = settings.HUGGINGFACEHUB_API_TOKEN
SUMMARY_MODEL_NAME = settings.SUMMARY_MODEL_NAME
SUMMARY_MAX_LENGTH = settings.SUMMARY_MAX_LENGTH
SUMMARY_MIN_LENGTH = settings.SUMMARY_MIN_LENGTH

# --- Dream Sequence Settings ---
EVALUATION_RETRY_ATTEMPTS = settings.EVALUATION_RETRY_ATTEMPTS
EVALUATION_RETRY_DELAY = settings.EVALUATION_RETRY_DELAY
SUMMARY_RETRY_ATTEMPTS = settings.SUMMARY_RETRY_ATTEMPTS
SUMMARY_RETRY_DELAY = settings.SUMMARY_RETRY_DELAY

# --- General ---
ENVIRONMENT = settings.ENVIRONMENT
DEBUG_MODE = settings.DEBUG_MODE

# --- main.py config ---
SLEEP_TIME_AFTER_QUERY = settings.SLEEP_TIME_AFTER_QUERY
DREAM_TIME_SECONDS = settings.DREAM_TIME_SECONDS

# --- Vector Model ---
VECTOR_MODEL_ID = settings.VECTOR_MODEL_ID

# Validate settings on import if in production
if settings.is_production:
    required_keys = [
        settings.LLM_API_KEY,
        settings.PINECONE_API_KEY,
        settings.HUGGINGFACEHUB_API_TOKEN
    ]
    if not all(required_keys):
        raise ValueError("Missing required API keys in production environment")