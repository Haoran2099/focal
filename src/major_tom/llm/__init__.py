"""LLM abstraction layer."""

from major_tom.llm.base import EmbeddingResponse, LLMBackend, LLMResponse
from major_tom.llm.ollama_backend import OllamaBackend
# Import OpenAIBackend conditionally inside usage or here with try/except
# But to keep init simple we can export if available or let users import directly.
# Let's keep it simple and just export base classes as before, but add specific backends if we want easy access.
# Actually, the original file only exported base classes.
# I will just leave it as is, or add the new backends if needed for convenience.
# Let's export the backends for easier access in run_all_experiments.py

__all__ = ["LLMBackend", "LLMResponse", "EmbeddingResponse"]
