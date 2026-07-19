from __future__ import annotations

from typing import Callable, Optional

from .focal_method import FocalSessionMethod
from .session_baselines import (
    BrainLLMVLMSharedMemorySessionBaseline,
    NaiveLLMAgentSessionBaseline,
)


class SessionExperimentRunner:
    def __init__(self, backend):
        self.backend = backend
        self.method_registry = {
            "focal": FocalSessionMethod(backend),
            "naive_llm_agent_session": NaiveLLMAgentSessionBaseline(backend),
            "brain_vlm_shared_memory_session": BrainLLMVLMSharedMemorySessionBaseline(backend),
        }
        self.last_session_traces: list[dict] = []

    def run_experiment(
        self,
        session_streams,
        method_name: str,
        on_session_complete: Optional[Callable[[int, object, dict, dict], None]] = None,
    ) -> list[dict]:
        if method_name not in self.method_registry:
            raise ValueError(f"Unsupported session method: {method_name}")

        method = self.method_registry[method_name]
        results: list[dict] = []
        self.last_session_traces = []

        for session_index, session in enumerate(session_streams):
            if hasattr(self.backend, "start_session_trace"):
                self.backend.start_session_trace(method_name, session_index, session.session_id, len(session.actions))

            result = method.process_session(session.session_id, session.actions)
            result["catalog_segments"] = session.catalog_segments
            results.append(result)

            trace = {}
            if hasattr(self.backend, "finalize_session_trace"):
                trace = self.backend.finalize_session_trace()
                self.last_session_traces.append(trace)

            if on_session_complete is not None:
                on_session_complete(session_index, session, result, trace)

        return results