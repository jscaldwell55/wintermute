from typing import Optional
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    # --- LLM Service ---
    LLM_API_KEY: str = Field(None, env='OPENAI_API_KEY')
    LLM_MODEL_ID: str = Field("gpt-3.5-turbo")
    VECTOR_MODEL_ID: str = Field("text-embedding-ada-002")  # OpenAI's text embedding model
    LLM_MODEL_NAME: str = Field(None)  # Will be set to LLM_MODEL_ID in __init__
    LLM_API_BASE_URL: str = Field("https://api.openai.com/v1")
    LLM_API_RATE_LIMIT_PER_MINUTE: int = Field(30)
    SUMMARY_LLM_MODEL_ID: str = Field("gpt-3.5-turbo")

    # --- Pinecone Service ---
    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: str = Field("us-east-1")
    INDEX_NAME: str = Field("wintermute")
    PINECONE_API_RATE_LIMIT_PER_SECOND: int = Field(3)

    # --- Memory ---
    MEMORY_DECAY_FACTOR: float = Field(0.99)
    HOURS_SINCE_DECAY: float = Field(24.0)
    MAX_TOKENS_PER_CONTEXT_WINDOW: int = Field(4096)
    SUMMARY_RETRY_DELAY: int = Field(5)
    SUMMARY_RETRIES: int = Field(3)
    DREAM_SEQUENCE_INTERVAL_HOURS: int = Field(24)
    DELETE_THRESHOLD_DAYS: int = Field(7)

    # --- Task Queue ---
    TASK_QUEUE_NUM_WORKERS: int = Field(4)

    # --- Clustering (Dream Sequence) ---
    KMEANS_N_CLUSTERS_MIN: int = Field(2)
    KMEANS_N_CLUSTERS_MAX: int = Field(10)
    KMEANS_INIT: str = Field('k-means++')
    KMEANS_MAX_ITER: int = Field(300)
    KMEANS_N_INIT: int = Field(10)

    # --- Semantic Memory Quality Thresholds ---
    SEMANTIC_MEMORY_RELEVANCE_THRESHOLD: float = Field(0.7)
    SEMANTIC_MEMORY_COHERENCE_THRESHOLD: float = Field(3.0)
    SEMANTIC_MEMORY_THRESHOLD: float = Field(0.85)

    # --- Hugging Face (Summarization) ---
    HUGGINGFACEHUB_API_TOKEN: Optional[str] = None
    SUMMARY_MODEL_NAME: str = Field("facebook/bart-large-cnn")
    SUMMARY_MAX_LENGTH: int = Field(150)
    SUMMARY_MIN_LENGTH: int = Field(50)

    # --- Dream Sequence Settings ---
    EVALUATION_RETRY_ATTEMPTS: int = Field(3)
    EVALUATION_RETRY_DELAY: int = Field(5)
    SUMMARY_RETRY_ATTEMPTS: int = Field(3)
    SUMMARY_RETRY_DELAY: int = Field(5)

    # --- General ---
    ENVIRONMENT: str = Field("dev")
    DEBUG_MODE: bool = Field(False)

    # --- main.py config ---
    SLEEP_TIME_AFTER_QUERY: float = Field(1.0)
    DREAM_TIME_SECONDS: int = Field(1800)

    class Config:
        env_file = ".env"
        case_sensitive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set LLM_MODEL_NAME to LLM_MODEL_ID for backward compatibility
        if not self.LLM_MODEL_NAME:
            self.LLM_MODEL_NAME = self.LLM_MODEL_ID

    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.ENVIRONMENT.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.ENVIRONMENT.lower() in ["dev", "development"]

    def validate_api_keys(self) -> bool:
        """Validate that required API keys are present"""
        if self.is_production:
            return all([
                self.LLM_API_KEY,
                self.PINECONE_API_KEY,
                self.HUGGINGFACEHUB_API_TOKEN
            ])
        return True

# Create global settings instance
settings = Settings()

# For backward compatibility
LLM_API_KEY = settings.LLM_API_KEY
LLM_MODEL_ID = settings.LLM_MODEL_ID
LLM_MODEL_NAME = settings.LLM_MODEL_NAME
LLM_API_BASE_URL = settings.LLM_API_BASE_URL
PINECONE_API_KEY = settings.PINECONE_API_KEY
PINECONE_ENVIRONMENT = settings.PINECONE_ENVIRONMENT
INDEX_NAME = settings.INDEX_NAME
# ... and so on for any variables that need to be accessible directly

# Validate settings on import
if settings.is_production and not settings.validate_api_keys():
    raise ValueError("Missing required API keys in production environment")