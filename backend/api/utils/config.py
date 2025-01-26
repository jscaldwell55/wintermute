import os
from typing import Optional, Literal, Any
from pydantic import Field, model_validator, validator, AnyHttpUrl, conint, confloat
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from dotenv import load_dotenv
import logging
import re
from urllib.parse import urlparse

# Custom Exceptions
class ConfigurationError(Exception):
    """Base exception for configuration errors."""
    pass

class MissingRequiredSettingError(ConfigurationError):
    """Raised when a required setting is missing."""
    pass

class InvalidSettingValueError(ConfigurationError):
    """Raised when a setting has an invalid value."""
    pass

# Initialize structured logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(handler)

# Load environment variables from .env file
try:
    load_dotenv()
except Exception as e:
    logger.error(f"Error loading .env file: {e}")

class Settings(BaseSettings):
    """Configuration settings for the application."""

    # --- OpenAI Service ---
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    llm_model_id: str = Field(default="gpt-3.5-turbo", alias="LLM_MODEL_ID")
    llm_api_base_url: AnyHttpUrl = Field(
        default="https://api.openai.com/v1", 
        alias="LLM_API_BASE_URL"
    )
    llm_api_rate_limit_per_minute: conint(ge=1, le=100) = Field(
        default=30, 
        alias="LLM_API_RATE_LIMIT_PER_MINUTE"
    )
    summary_llm_model_id: str = Field(default="gpt-3.5-turbo", alias="SUMMARY_LLM_MODEL_ID")

    # --- Pinecone Service ---
    pinecone_api_key: Optional[str] = Field(default=None, alias="PINECONE_API_KEY")
    pinecone_environment: str = Field(default="us-east-1", alias="PINECONE_ENVIRONMENT")
    index_name: str = Field(default="wintermute", alias="INDEX_NAME")
    pinecone_api_rate_limit_per_second: conint(ge=1, le=10) = Field(
        default=3, 
        alias="PINECONE_API_RATE_LIMIT_PER_SECOND"
    )

    # --- Memory ---
    memory_decay_factor: confloat(ge=0, le=1) = Field(
        default=0.99, 
        alias="MEMORY_DECAY_FACTOR"
    )
    hours_since_decay: confloat(ge=0) = Field(default=24.0, alias="HOURS_SINCE_DECAY")
    max_tokens_per_context_window: conint(ge=1) = Field(
        default=4096, 
        alias="MAX_TOKENS_PER_CONTEXT_WINDOW"
    )
    summary_retry_delay: conint(ge=1) = Field(default=5, alias="SUMMARY_RETRY_DELAY")
    summary_retries: conint(ge=1) = Field(default=3, alias="SUMMARY_RETRIES")
    dream_sequence_interval_hours: conint(ge=1) = Field(
        default=24, 
        alias="DREAM_SEQUENCE_INTERVAL_HOURS"
    )
    delete_threshold_days: conint(ge=1) = Field(default=7, alias="DELETE_THRESHOLD_DAYS")

    # --- Task Queue ---
    task_queue_num_workers: conint(ge=1, le=32) = Field(
        default=4, 
        alias="TASK_QUEUE_NUM_WORKERS"
    )

    # --- Clustering ---
    kmeans_n_clusters_min: conint(ge=2) = Field(default=2, alias="KMEANS_N_CLUSTERS_MIN")
    kmeans_n_clusters_max: conint(ge=2) = Field(default=10, alias="KMEANS_N_CLUSTERS_MAX")
    kmeans_init: str = Field(default="k-means++", alias="KMEANS_INIT")
    kmeans_max_iter: conint(ge=1) = Field(default=300, alias="KMEANS_MAX_ITER")
    kmeans_n_init: conint(ge=1) = Field(default=10, alias="KMEANS_N_INIT")

    # --- Thresholds ---
    semantic_memory_relevance_threshold: confloat(ge=0, le=1) = Field(
        default=0.7, 
        alias="SEMANTIC_MEMORY_RELEVANCE_THRESHOLD"
    )
    semantic_memory_coherence_threshold: confloat(ge=0) = Field(
        default=3.0, 
        alias="SEMANTIC_MEMORY_COHERENCE_THRESHOLD"
    )
    semantic_memory_threshold: confloat(ge=0, le=1) = Field(
        default=0.85, 
        alias="SEMANTIC_MEMORY_THRESHOLD"
    )

    # --- Hugging Face ---
    huggingfacehub_api_token: Optional[str] = Field(
        default=None, 
        alias="HUGGINGFACEHUB_API_TOKEN"
    )
    summary_model_name: str = Field(
        default="facebook/bart-large-cnn", 
        alias="SUMMARY_MODEL_NAME"
    )
    summary_max_length: conint(ge=1) = Field(default=150, alias="SUMMARY_MAX_LENGTH")
    summary_min_length: conint(ge=1) = Field(default=50, alias="SUMMARY_MIN_LENGTH")

    # --- Dream Sequence ---
    evaluation_retry_attempts: conint(ge=1) = Field(
        default=3, 
        alias="EVALUATION_RETRY_ATTEMPTS"
    )
    evaluation_retry_delay: conint(ge=1) = Field(
        default=5, 
        alias="EVALUATION_RETRY_DELAY"
    )
    summary_retry_attempts: conint(ge=1) = Field(
        default=3, 
        alias="SUMMARY_RETRY_ATTEMPTS"
    )

    # --- General ---
    environment: Literal["dev", "test", "production"] = Field(
        default="dev", 
        alias="ENVIRONMENT"
    )
    debug_mode: bool = Field(default=False, alias="DEBUG_MODE")
    sleep_time_after_query: confloat(ge=0) = Field(
        default=1.0, 
        alias="SLEEP_TIME_AFTER_QUERY"
    )
    dream_time_seconds: conint(ge=1) = Field(
        default=1800, 
        alias="DREAM_TIME_SECONDS"
    )
    vector_model_id: str = Field(
        default="text-embedding-ada-002", 
        alias="VECTOR_MODEL_ID"
    )

    # Validators
    @validator("openai_api_key")
    def validate_openai_api_key(cls, v: Optional[str]) -> Optional[str]:
        if v and not v.startswith(("sk-", "test-")):
            raise InvalidSettingValueError("Invalid OpenAI API key format")
        return v

    @validator("pinecone_api_key")
    def validate_pinecone_api_key(cls, v: Optional[str]) -> Optional[str]:
        if v and len(v) < 8:  # Basic validation
            raise InvalidSettingValueError("Invalid Pinecone API key format")
        return v

    @validator("llm_api_base_url")
    def validate_api_url(cls, v: str) -> str:
        try:
            result = urlparse(v)
            if not all([result.scheme, result.netloc]):
                raise InvalidSettingValueError("Invalid API URL format")
        except Exception as e:
            raise InvalidSettingValueError(f"Invalid API URL: {e}")
        return v

    @model_validator(mode="after")
    def validate_production_settings(cls, values):
        """Validate production environment requirements."""
        if values.environment == "production":
            required_fields = [
                "openai_api_key",
                "pinecone_api_key",
                "pinecone_environment",
                "index_name",
            ]
            missing = [
                field for field in required_fields 
                if not getattr(values, field, None)
            ]
            if missing:
                raise MissingRequiredSettingError(
                    f"Production environment requires: {', '.join(missing)}"
                )
            
            # Additional production validations
            if values.debug_mode:
                logger.warning("Debug mode is enabled in production environment")
            
            if values.task_queue_num_workers < 2:
                raise InvalidSettingValueError(
                    "Production requires at least 2 task queue workers"
                )
        return values

    @property
    def llm_api_key(self) -> Optional[str]:
        """Backward compatibility for llm_api_key."""
        return self.openai_api_key

    @property
    def is_production(self) -> bool:
        """Check if the environment is production."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if the environment is development."""
        return self.environment == "dev"

    def get_setting(self, setting_name: str, default_value: Any = None) -> Any:
        """Get a setting value with fallback."""
        try:
            return getattr(self, setting_name.lower(), 
                         os.getenv(setting_name.upper(), default_value))
        except AttributeError:
            logger.warning(f"Attempted to access undefined setting: {setting_name}")
            return default_value

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
        validate_assignment=True,
        case_sensitive=False,
        populate_by_name=True,
    )

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance."""
    try:
        settings = Settings()
        if settings.debug_mode:
            # Log non-sensitive settings in debug mode
            safe_settings = settings.model_dump(exclude={
                'openai_api_key', 
                'pinecone_api_key', 
                'huggingfacehub_api_token'
            })
            logger.debug(f"Loaded settings: {safe_settings}")
        return settings
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        raise ConfigurationError(f"Configuration error: {e}")

__all__ = ["get_settings", "Settings", "ConfigurationError", 
           "MissingRequiredSettingError", "InvalidSettingValueError"]