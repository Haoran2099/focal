"""Abstract base class for LLM backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LLMResponse:
    """Standardized response from an LLM generate call."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    latency_ms: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingResponse:
    """Standardized response from an embedding call."""

    vector: List[float] = field(default_factory=list)
    prompt_tokens: int = 0
    model: str = ""
    latency_ms: float = 0.0


class LLMBackend(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def generate(
        self,
        model: str,
        prompt: str,
        *,
        images: Optional[List[bytes]] = None,
        format: Optional[str] = None,
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
        think: Optional[bool] = None,
    ) -> LLMResponse:
        """Generate a completion from the model."""

    @abstractmethod
    def embed(self, model: str, prompt: str) -> EmbeddingResponse:
        """Get an embedding vector for the given text."""
