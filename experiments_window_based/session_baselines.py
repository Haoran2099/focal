from __future__ import annotations

import json
from typing import Any

from .config import Config
from .core import SessionAction
from .focal_method import FocalSessionMethod, TaskState


class NaiveLLMAgentSessionBaseline:
    """Always sample every action and summarize the full session without task awareness."""

    def __init__(self, llm_backend):
        self.llm_backend = llm_backend
        self.reset_session_state()

    def reset_session_state(self) -> None:
        self.action_history: list[dict[str, Any]] = []
        self.action_trace: list[dict[str, Any]] = []
        self.session_id: str = ""

    def process_session(self, session_id: str, actions: list[SessionAction]) -> dict[str, Any]:
        self.reset_session_state()
        self.session_id = session_id

        descriptions: list[str] = []
        sampled_action_indices: list[int] = []
        action_indices: list[int] = []

        for action in actions:
            observation = ""
            if action.screenshot_path:
                observation = self._observe_action(action)
                if observation:
                    descriptions.append(observation)
                    sampled_action_indices.append(action.stream_index)

            action_indices.append(action.stream_index)
            history_entry = {
                "stream_index": action.stream_index,
                "app": action.app,
                "title": action.title,
                "assigned_task_id": "global_session",
                "assigned_task_label": "Whole session",
                "sampled": bool(observation),
            }
            self.action_history.append(history_entry)
            self.action_trace.append(
                {
                    **history_entry,
                    "decision": "single_stream",
                    "decision_reason": "no task segmentation baseline",
                    "llm_should_sample": bool(action.screenshot_path),
                    "observation": observation,
                }
            )

        task_payloads, session_summary = self._summarize_session(descriptions, sampled_action_indices)
        tasks = self._build_task_outputs(task_payloads, descriptions, sampled_action_indices, action_indices)

        return {
            "session_id": session_id,
            "action_count": len(actions),
            "task_count": len(tasks),
            "tasks": tasks,
            "session_summary": session_summary,
            "action_trace": self.action_trace,
        }

    def _observe_action(self, action: SessionAction) -> str:
        prompt = (
            "Role:\n"
            "You are the VLM note taker for a naive full-coverage desktop session baseline.\n\n"
            "Task:\n"
            "- Describe the current screenshot in one short factual sentence.\n"
            "- Use only the current screenshot.\n\n"
            "Constraints:\n"
            "- Do not use history, memory, or task assumptions.\n"
            "- Do not infer hidden actions, prior steps, or future intent."
        )
        summary = self.llm_backend.generate_multimodal_summary(prompt, "", [action.screenshot_path])
        cleaned = " ".join((summary or "").split())
        if not cleaned or cleaned.startswith("[EMPTY_") or cleaned.startswith("[VLM Error]"):
            return ""
        return cleaned

    def _summarize_session(self, descriptions: list[str], sampled_action_indices: list[int]) -> tuple[list[dict[str, Any]], str]:
        if not descriptions:
            return [], ""

        prompt = (
            "Role:\n"
            "You are the final summary LLM for a naive desktop-session baseline.\n\n"
            "Evidence:\n"
            "- Use only the provided chronological visual descriptions.\n"
            "- These descriptions were generated independently per action with no task memory.\n"
            "- Do not invent hidden actions, intent, or missing steps.\n\n"
            "Task:\n"
            "- Infer coarse task phases from the chronological descriptions.\n"
            "- Group consecutive descriptions that belong to the same task phase.\n"
            "- Write one concise summary for each inferred task phase.\n"
            "- Then write one concise whole-session summary in chronological order.\n\n"
            "Output:\n"
            "Return strict JSON with exactly these keys: task_summaries, session_summary.\n"
            "- task_summaries: a list of objects with keys task_label, start_index, end_index, summary\n"
            "- start_index and end_index are 1-based indices over the description list\n"
            "- session_summary: one concise plain-text paragraph"
        )
        context = (
            "sampled_action_indices:\n- "
            + "\n- ".join(str(value) for value in sampled_action_indices)
            + "\n\nchronological_visual_descriptions:\n"
            + "\n".join(f"{idx + 1}. {item}" for idx, item in enumerate(descriptions))
        )
        raw = self.llm_backend.generate_text_summary(prompt, context)
        parsed = self._extract_json(raw)
        task_summaries = parsed.get("task_summaries", [])
        if not isinstance(task_summaries, list):
            task_summaries = []
        session_summary = " ".join(str(parsed.get("session_summary") or "").split())
        if not session_summary:
            session_summary = " ".join(descriptions)
        return task_summaries, session_summary

    def _build_task_outputs(
        self,
        task_payloads: list[dict[str, Any]],
        descriptions: list[str],
        sampled_action_indices: list[int],
        all_action_indices: list[int],
    ) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        total_descriptions = len(descriptions)

        for task_idx, payload in enumerate(task_payloads, start=1):
            start_index = self._clamp_index(payload.get("start_index"), total_descriptions)
            end_index = self._clamp_index(payload.get("end_index"), total_descriptions)
            if start_index <= 0 or end_index <= 0:
                continue
            if end_index < start_index:
                start_index, end_index = end_index, start_index

            description_slice = descriptions[start_index - 1 : end_index]
            action_slice = sampled_action_indices[start_index - 1 : end_index]
            task_label = " ".join(str(payload.get("task_label") or "").split()) or f"Task phase {task_idx}"
            summary = " ".join(str(payload.get("summary") or "").split()) or " ".join(description_slice)
            task_id = f"naive_task_{task_idx}"

            tasks.append(
                {
                    "task_id": task_id,
                    "task_label": task_label,
                    "action_indices": action_slice,
                    "first_seen": action_slice[0] if action_slice else (all_action_indices[0] if all_action_indices else 0),
                    "last_seen": action_slice[-1] if action_slice else (all_action_indices[-1] if all_action_indices else 0),
                    "observation_count": len(description_slice),
                    "memory_items": description_slice,
                    "summary": summary,
                }
            )

        if tasks:
            return tasks

        if not all_action_indices:
            return []

        return [
            {
                "task_id": "global_session",
                "task_label": "Whole session",
                "action_indices": all_action_indices,
                "first_seen": all_action_indices[0],
                "last_seen": all_action_indices[-1],
                "observation_count": len(descriptions),
                "memory_items": descriptions,
                "summary": " ".join(descriptions),
            }
        ]

    def _clamp_index(self, value: Any, total: int) -> int:
        try:
            index = int(value)
        except (TypeError, ValueError):
            return 0
        if total <= 0:
            return 0
        return max(1, min(total, index))

    def _extract_json(self, text: str) -> dict[str, Any]:
        raw = (text or "").strip()
        if not raw:
            return {}
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

        left = raw.find("{")
        right = raw.rfind("}")
        if left != -1 and right != -1 and right > left:
            try:
                payload = json.loads(raw[left : right + 1])
                if isinstance(payload, dict):
                    return payload
            except json.JSONDecodeError:
                return {}
        return {}


class BrainLLMVLMSharedMemorySessionBaseline(FocalSessionMethod):
    """Keep focal planning/sampling, but replace task-scoped retrieval with one shared memory pool."""

    def reset_session_state(self) -> None:
        super().reset_session_state()
        self.global_memory_entries: list[dict[str, Any]] = []

    def _get_observation_context_memory(self, task_state: TaskState) -> str:
        if not self.global_memory_entries:
            return ""
        history_items = [
            str(entry.get("observation") or "").strip()
            for entry in self.global_memory_entries
            if entry.get("observation")
        ]
        limit = max(1, int(Config.ONLINE_GLOBAL_MEMORY_PROMPT_ITEMS))
        visible_items = history_items[-limit:]
        if len(visible_items) == 1:
            return visible_items[0]
        return "\n".join(f"[{idx}] {item}" for idx, item in enumerate(visible_items, start=1))

    def _observe_action(self, action: SessionAction, task_state: TaskState) -> str:
        prompt = (
            "Role:\n"
            "You are the VLM note taker for online desktop activity tracking with shared global memory.\n\n"
            "Task:\n"
            "- Describe the current screenshot in 1-2 short factual sentences.\n"
            "- Write 1-2 short factual sentences.\n\n"
            "Inference policy:\n"
            "- You may use the screenshot together with the session's historical VLM outputs.\n"
            "- Allow only minimal, evidence-grounded inferences.\n"
            "- Do not invent hidden actions, unseen intent, or unsupported state."
        )
        history_memory = self._get_observation_context_memory(task_state) or "<none>"
        context = (
            f"current_action_index={action.stream_index}\n"
            f"current_app={action.app}\n"
            f"current_title={action.title}\n"
            f"historical_vlm_outputs=\n{history_memory}"
        )
        summary = self.llm_backend.generate_multimodal_summary(prompt, context, [action.screenshot_path])
        cleaned = " ".join((summary or "").split())
        if not cleaned or cleaned.startswith("[EMPTY_") or cleaned.startswith("[VLM Error]"):
            return ""
        return cleaned

    def _store_observation(self, task_state: TaskState, observation: str) -> None:
        if self.global_memory_entries and self.global_memory_entries[-1].get("observation") == observation:
            task_state.latest_memory = observation
            return

        task_state.latest_memory = observation
        task_state.observation_count += 1
        self.global_memory_entries.append(
            {
                "task_id": task_state.task_id,
                "task_label": task_state.task_label,
                "observation": observation,
            }
        )

    def _get_task_memory_items(self, task_state: TaskState) -> list[str]:
        return [
            str(entry.get("observation") or "")
            for entry in self.global_memory_entries
            if entry.get("task_id") == task_state.task_id and entry.get("observation")
        ]