import os
from typing import Optional, Literal, Any
from pydantic import Field, field_validator, model_validator
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

    # --- LLM Service ---
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    llm_model_id: str = Field(default="gpt-3.5-turbo", alias="LLM_MODEL_ID")
    llm_api_base_url: str = Field(default="https://api.openai.com/v1", alias="LLM_API_BASE_URL")
    llm_api_rate_limit_per_minute: int = Field(default=30, alias="LLM_API_RATE_LIMIT_PER_MINUTE")
    summary_llm_model_id: str = Field(default="gpt-3.5-turbo", alias="SUMMARY_LLM_MODEL_ID")

    # --- Pinecone Service ---
    pinecone_api_key: Optional[str] = Field(default=None, alias="PINECONE_API_KEY")
    pinecone_environment: str = Field(default="us-east-1", alias="PINECONE_ENVIRONMENT")
    index_name: str = Field(default="wintermute", alias="INDEX_NAME")
    pinecone_api_rate_limit_per_second: int = Field(default=3, alias="PINECONE_API_RATE_LIMIT_PER_SECOND")

    # --- Memory ---
    memory_decay_factor: float = Field(default=0.99, alias="MEMORY_DECAY_FACTOR")
    hours_since_decay: float = Field(default=24.0, alias="HOURS_SINCE_DECAY")
    max_tokens_per_context_window: int = Field(default=4096, alias="MAX_TOKENS_PER_CONTEXT_WINDOW")
    summary_retry_delay: int = Field(default=5, alias="SUMMARY_RETRY_DELAY")
    summary_retries: int = Field(default=3, alias="SUMMARY_RETRIES")
    dream_sequence_interval_hours: int = Field(default=24, alias="DREAM_SEQUENCE_INTERVAL_HOURS")
    delete_threshold_days: int = Field(default=7, alias="DELETE_THRESHOLD_DAYS")

    # --- Task Queue ---
    task_queue_num_workers: int = Field(default=4, alias="TASK_QUEUE_NUM_WORKERS")

    # --- Clustering (Dream Sequence) ---
    kmeans_n_clusters_min: int = Field(default=2, alias="KMEANS_N_CLUSTERS_MIN")
    kmeans_n_clusters_max: int = Field(default=10, alias="KMEANS_N_CLUSTERS_MAX")
    kmeans_init: str = Field(default="k-means++", alias="KMEANS_INIT")
    kmeans_max_iter: int = Field(default=300, alias="KMEANS_MAX_ITER")
    kmeans_n_init: int = Field(default=10, alias="KMEANS_N_INIT")

    # --- Semantic Memory Quality Thresholds ---
    semantic_memory_relevance_threshold: float = Field(default=0.7, alias="SEMANTIC_MEMORY_RELEVANCE_THRESHOLD")
    semantic_memory_coherence_threshold: float = Field(default=3.0, alias="SEMANTIC_MEMORY_COHERENCE_THRESHOLD")
    semantic_memory_threshold: float = Field(default=0.85, alias="SEMANTIC_MEMORY_THRESHOLD")

    # --- Hugging Face (Summarization) ---
    huggingfacehub_api_token: Optional[str] = Field(default=None, alias="HUGGINGFACEHUB_API_TOKEN")
    summary_model_name: str = Field(default="facebook/bart-large-cnn", alias="SUMMARY_MODEL_NAME")
    summary_max_length: int = Field(default=150, alias="SUMMARY_MAX_LENGTH")
    summary_min_length: int = Field(default=50, alias="SUMMARY_MIN_LENGTH")

    # --- Dream Sequence Settings ---
    evaluation_retry_attempts: int = Field(default=3, alias="EVALUATION_RETRY_ATTEMPTS")
    evaluation_retry_delay: int = Field(default=5, alias="EVALUATION_RETRY_DELAY")
    summary_retry_attempts: int = Field(default=3, alias="SUMMARY_RETRY_ATTEMPTS")

    # --- General ---
    environment: Literal['dev', 'test', 'production'] = Field(default="dev", alias="ENVIRONMENT")
    debug_mode: bool = Field(default=False, alias="DEBUG_MODE")

    # --- Application Config ---
    sleep_time_after_query: float = Field(default=1.0, alias="SLEEP_TIME_AFTER_QUERY")
    dream_time_seconds: int = Field(default=1800, alias="DREAM_TIME_SECONDS")

    # --- Vector Model ---
    vector_model_id: str = Field(default="text-embedding-ada-002", alias="VECTOR_MODEL_ID")

    # --- AWS ---
    aws_region: str = Field(default="us-west-1", alias="AWS_REGION")

    # --- LRU Cache ---
    lru_cache_size: int = Field(default=1000, alias="LRU_CACHE_SIZE")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
        validate_assignment=True,
        case_sensitive=False,
        populate_by_name=True
    )

    @field_validator('llm_api_base_url', mode='before')
    def check_and_format_url(cls, value):
        """
        Validates that the provided URL is a valid HTTP or HTTPS URL.

        Args:
            value: The URL to validate.

        Returns:
            The validated URL.

        Raises:
            ValueError: If the URL is not valid.
        """
        if not isinstance(value, str):
            raise ValueError('URL must be a string')
        
        if not re.match(r"^https?://", value):
            raise ValueError('URL must start with http:// or https://')

        try:
            result = urlparse(value)
            if all([result.scheme, result.netloc]):
                return value
            else:
                raise ValueError('Invalid URL format')
        except ValueError:
            raise ValueError('Invalid URL format')

    @field_validator("*")
    def check_for_invalid_characters(cls, value, field):
        """
        Validates that string values do not contain invalid characters.

        Args:
            value: The value to validate.
            field: The field being validated.

        Returns:
            The validated value.

        Raises:
            ValueError: If the value contains invalid characters.
        """
        if isinstance(value, str) and bool(re.search(r"[\"']", value)):
            raise ValueError(f"Invalid characters found in field '{field.name}'.")
        return value

    @field_validator('openai_api_key', 'pinecone_api_key', 'huggingfacehub_api_token')
    def check_api_key_format(cls, value, field):
        """
        Validates the format of API keys.

        Args:
            value: The API key value.
            field: The field being validated.

        Returns:
            The validated API key.

        Raises:
            ValueError: If the API key format is invalid.
        """
        if value and not re.match(r"^[a-zA-Z0-9_-]+$", value):
            raise ValueError(f"Invalid format for field '{field.name}'. It should only contain alphanumeric characters, hyphens, or underscores.")
        return value

    @model_validator(mode='after')
    def validate_required_settings(cls, values):
        """
        Validate that required settings are provided.

        This checks both for the presence of required environment variables and enforces additional
        validation for production environments.
        """
        required_settings = {
            "OPENAI_API_KEY",
            "PINECONE_API_KEY",
            "PINECONE_ENVIRONMENT",
            "INDEX_NAME"
        }

        for field, info in values.model_fields.items():
            env_var_name = info.alias or field.upper()

            if env_var_name in required_settings and env_var_name not in os.environ:
                raise MissingRequiredSettingError(f"Missing required setting: {env_var_name}")
            
        # Additional checks for production
        if values.environment == 'production':
            if not values.llm_api_key:
                raise MissingRequiredSettingError("Production environment requires LLM_API_KEY to be set.")
            if not values.pinecone_api_key:
                raise MissingRequiredSettingError("Production environment requires PINECONE_API_KEY to be set.")
            if not values.pinecone_environment:
                raise MissingRequiredSettingError("Production environment requires PINECONE_ENVIRONMENT to be set.")
            if not values.index_name:
                raise MissingRequiredSettingError("Production environment requires INDEX_NAME to be set.")

        return values

    @property
    def is_production(self) -> bool:
        """
        Check if the environment is set to production.

        Returns:
            bool: True if the environment is production, False otherwise.
        """
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """
        Check if the environment is set to development.

        Returns:
            bool: True if the environment is development, False otherwise.
        """
        return self.environment == "dev"
    
    def get_setting(self, setting_name: str, default_value: Any = None) -> Any:
        """
        Get the value of a setting.

        Args:
            setting_name (str): The name of the setting to retrieve.
            default_value (Any, optional): The default value to return if the setting is not found. Defaults to None.

        Returns:
            Any: The value of the setting, or the default value if the setting is not found.
        """
        try:
            # Try to get the value from the settings object
            value = getattr(self, setting_name.lower())
            if value is not None:
                return value
        except AttributeError:
            pass  # Setting not found in the object, proceed to check environment

        # Try to get the value from environment variables
        env_value = os.environ.get(setting_name.upper())
        if env_value is not None:
            return env_value

        # Return the default value if the setting is not found
        return default_value

# Global settings instance
_settings: Optional[Settings] = None

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get the settings for the application.

    This function uses lru_cache to cache the settings object, so it will only be created once.

    Returns:
        Settings: The settings for the application.
    """
    global _settings
    if _settings is None:
        try:
            _settings = Settings()
            logger.info(f"Loaded settings: {_settings.model_dump_json(indent=2)}")
        except ConfigurationError as e:
            logger.error(f"Configuration error: {e}")
            raise
    return _settings

# Make settings accessible on import
settings = get_settings()

__all__ = [
    "get_settings",
    "Settings",
    "ConfigurationError",
    "MissingRequiredSettingError",
    "InvalidSettingValueError",
    "settings",
]