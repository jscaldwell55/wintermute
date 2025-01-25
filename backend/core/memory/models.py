from typing import List, Dict, Optional, Any
from datetime import datetime
import numpy as np
from enum import Enum
import uuid
from pydantic import BaseModel, Field, field_validator
import logging

logger = logging.getLogger(__name__)

class MemoryType(str, Enum):
    """Enum for memory types with allowed values for validation"""
    EPISODIC = "episodic"  # Updated to reflect episodic and semantic
    SEMANTIC = "semantic"

class Memory(BaseModel):
    id: str = Field(default_factory=lambda: f"mem_{uuid.uuid4()}")
    memory_type: MemoryType  # Use the Enum here
    semantic_vector: List[float]
    content: str
    metadata: Dict[str, Any] = {} # Add metadata to pydantic model
    created_at: str
    window_id: Optional[str] = None  # Add window_id
    state: Optional[str] = "new" # Add state tracking
    quality_metrics: Optional[Dict[str, Any]] = None # Add quality metrics

    @field_validator("semantic_vector")
    def validate_vector(cls, v):
        expected_dim = 1536
        if len(v) != expected_dim:
            raise ValueError(f"Semantic vector must be {expected_dim}-dimensional (got {len(v)}).")
        return v