"""
Core data structures and interfaces for window-based and session-based baselines.
"""
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class GUIEvent:
    """
    Represents a single GUI event in the event stream.
    """
    timestamp: float
    app: str
    title: str
    duration: float
    kpm: float
    cpm: float
    screenshot_path: Optional[str] = None
    
    def to_text_summary(self) -> str:
        return f"[{self.app}] {self.title} (Duration: {self.duration:.1f}s, KPM: {self.kpm:.1f}, CPM: {self.cpm:.1f})"


@dataclass
class SessionAction:
    """
    Represents one observable action in a session stream.

    This structure intentionally excludes GT-only fields such as task ids,
    action narration/type, and handcrafted behavioral metrics.
    """

    session_id: str
    stream_index: int
    app: str
    title: str
    screenshot_path: Optional[str] = None
    source_action_idx: Optional[int] = None
    raw_timestamp: Optional[str] = None

    def to_text_summary(self) -> str:
        base = f"[{self.app}] {self.title}"
        if self.source_action_idx is not None:
            base += f" (source_action_idx={self.source_action_idx})"
        return base

class LLMBackendInterface:
    """
    Interface for the abstraction of LLM and VLM calls so baselines can be decoupled.
    """
    def generate_text_summary(self, prompt: str, context: str, role: str = "brain") -> str:
        """Generate summary based solely on textual context."""
        raise NotImplementedError("Must implement generate_text_summary")
        
    def generate_multimodal_summary(self, prompt: str, context: str, image_paths: List[str]) -> str:
        """Generate summary based on textual context and a list of screenshot paths."""
        raise NotImplementedError("Must implement generate_multimodal_summary")

    def generate_embedding(self, text: str) -> list[float]:
        """Generate a vector embedding for the given text."""
        raise NotImplementedError("Must implement generate_embedding")

