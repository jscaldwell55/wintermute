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
    TEMPORAL = "temporal"
    GROUNDING = "grounding"

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List
import numpy as np

class Memory(BaseModel):
    id: str = Field(default_factory=lambda: f"mem_{uuid.uuid4()}")
    memory_type: str
    semantic_vector: Optional[List[float]] = Field(None)
    content: str
    metadata: dict
    created_at: str

    @staticmethod
    def validate_vector(v: List[float], memory_type: str):
        expected_dim = 1536 if memory_type == "TEMPORAL" else 200
        if len(v) != expected_dim:
            raise ValueError(f"Semantic vector must be {expected_dim}-dimensional (got {len(v)}).")
        return v

    # Custom serialization to handle numpy arrays
    @classmethod
    def from_numpy(cls, memory_type: str, vector: np.ndarray):
        return cls(memory_type=memory_type, semantic_vector=vector.tolist())

    def to_numpy(self):
        return np.array(self.semantic_vector) if self.semantic_vector else None
