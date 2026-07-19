from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .core import LLMBackendInterface, SessionAction


@dataclass
class TaskState:
    task_id: str
    task_label: str
    first_seen: int
    last_seen: int
    last_app: str
    last_title: str
    latest_memory: str = ""
    action_indices: list[int] = field(default_factory=list)
    memory_items: list[str] = field(default_factory=list)
    observation_count: int = 0


class FocalSessionMethod:
    """One-shot Brain planning followed by sequential VLM observation and task summarization."""

    def __init__(self, llm_backend: LLMBackendInterface):
        self.llm_backend = llm_backend
        self.reset_session_state()

    def reset_session_state(self) -> None:
        self.tasks: OrderedDict[str, TaskState] = OrderedDict()
        self.action_history: list[dict[str, Any]] = []
        self.action_trace: list[dict[str, Any]] = []
        self.session_id: str = ""
        self.planner_task_aliases: dict[str, str] = {}

    def process_session(self, session_id: str, actions: list[SessionAction]) -> dict[str, Any]:
        self.reset_session_state()
        self.session_id = session_id

        action_plans = self._plan_session(actions)

        for action, planned_decision in zip(actions, action_plans):
            task_id, decision = self._resolve_planned_task(action, planned_decision)
            task_state = self.tasks[task_id]
            should_sample = self._should_sample_from_plan(action, task_state, decision)

            observation = ""
            if should_sample:
                observation = self._observe_action(action, task_state)
                if observation:
                    self._append_memory(task_state, observation)

            task_state.last_seen = action.stream_index
            task_state.last_app = action.app
            task_state.last_title = action.title
            task_state.action_indices.append(action.stream_index)

            history_entry = {
                "stream_index": action.stream_index,
                "app": action.app,
                "title": action.title,
                "assigned_task_id": task_id,
                "assigned_task_label": task_state.task_label,
                "sampled": bool(observation),
            }
            self.action_history.append(history_entry)
            self.action_trace.append(
                {
                    **history_entry,
                    "planner_task_id": decision.get("planner_task_id", ""),
                    "decision": decision.get("decision", "unknown"),
                    "decision_reason": decision.get("reason", ""),
                    "llm_should_sample": bool(decision.get("should_sample", False)),
                    "observation": observation,
                }
            )

        task_summaries = self._summarize_tasks()
        session_summary = self._summarize_session(task_summaries)
        ordered_tasks = [task_summaries[task_id] for task_id in self.tasks.keys() if task_id in task_summaries]
        return {
            "session_id": session_id,
            "action_count": len(actions),
            "task_count": len(ordered_tasks),
            "tasks": ordered_tasks,
            "session_summary": session_summary,
            "action_trace": self.action_trace,
        }

    def _plan_session(self, actions: list[SessionAction]) -> list[dict[str, Any]]:
        if not actions:
            return []

        raw_response = self.llm_backend.generate_text_summary(
            self._session_brain_prompt(),
            self._session_brain_context(actions),
            role="planner",
        )
        return self._normalize_session_plan(raw_response, actions)

    def _session_brain_prompt(self) -> str:
        return (
            "Role:\n"
            "You are the Brain for one-shot online desktop session planning.\n\n"
            "Inputs:\n"
            "- the full ordered list of action metadata for one session\n"
            "- each action only contains app, title, stream index, and source action index\n"
            "- you do not see screenshots\n\n"
            "Task routing:\n"
            "- Assign every action to a task.\n"
            "- Reuse the same task_id whenever actions belong to the same ongoing task.\n"
            "- Create a new task only when the user goal, artifact, document, or workspace clearly changes.\n"
            "- Do not split tasks on title noise alone.\n"
            "- Non-consecutive actions may still belong to the same task if the work clearly returns to it.\n\n"
            "Adaptive sampling:\n"
            "- should_sample decides whether the later VLM pass should inspect that action's screenshot.\n"
            "- Set should_sample=true for a new task or an assigned-task title change.\n"
            "- Also set should_sample=true when metadata suggests a meaningful UI phase change inside the same task.\n"
            "- Otherwise keep sampling sparse.\n\n"
            "Output:\n"
            "Return strict JSON with exactly one top-level key: action_plans.\n"
            "- action_plans must be a list with exactly one item per action in the same order as the input.\n"
            "- Each item must be an object with exactly these keys: stream_index, decision, task_id, task_label, should_sample, reason.\n"
            "- decision must be existing_task or new_task.\n"
            "- task_id must be stable across all actions of the same task, for example task_1, task_2.\n"
            "- task_label should be short and human-readable.\n"
            "- reason should briefly justify both routing and sampling."
        )

    def _session_brain_context(self, actions: list[SessionAction]) -> str:
        lines = []
        for action in actions:
            lines.append(
                f"[{action.stream_index}] app={action.app} | title={action.title} | source_action_idx={action.source_action_idx}"
            )
        return f"session_id={self.session_id}\naction_count={len(actions)}\n\nactions:\n" + "\n".join(lines)

    def _normalize_session_plan(self, raw_response: str, actions: list[SessionAction]) -> list[dict[str, Any]]:
        parsed = self._extract_json(raw_response)
        raw_plans = parsed.get("action_plans", []) if isinstance(parsed, dict) else []
        fallback_plans = self._fallback_plan(actions)

        plans_by_stream_index: dict[int, dict[str, Any]] = {}
        if isinstance(raw_plans, list):
            for item in raw_plans:
                if not isinstance(item, dict):
                    continue
                try:
                    normalized_index = int(item.get("stream_index"))
                except (TypeError, ValueError):
                    continue
                plans_by_stream_index[normalized_index] = item

        normalized: list[dict[str, Any]] = []
        for index, action in enumerate(actions):
            source = plans_by_stream_index.get(action.stream_index)
            if source is None and isinstance(raw_plans, list) and index < len(raw_plans) and isinstance(raw_plans[index], dict):
                source = raw_plans[index]
            fallback = fallback_plans[index]
            normalized.append(self._normalize_plan_item(source or {}, action, fallback, index))
        return normalized

    def _normalize_plan_item(
        self,
        payload: dict[str, Any],
        action: SessionAction,
        fallback: dict[str, Any],
        index: int,
    ) -> dict[str, Any]:
        decision = str(payload.get("decision") or fallback["decision"]).strip().lower()
        if decision not in {"existing_task", "new_task"}:
            decision = fallback["decision"]

        planner_task_id = self._normalize_token(str(payload.get("task_id") or fallback["planner_task_id"]))
        if not planner_task_id:
            planner_task_id = f"task_{index + 1}"

        task_label = " ".join(str(payload.get("task_label") or fallback["task_label"]).split())
        if not task_label:
            task_label = self._default_task_label(action)

        should_sample = bool(payload.get("should_sample", fallback["should_sample"]))
        reason = " ".join(str(payload.get("reason") or fallback["reason"]).split())

        return {
            "stream_index": action.stream_index,
            "decision": decision,
            "planner_task_id": planner_task_id,
            "task_label": task_label,
            "should_sample": should_sample,
            "reason": reason or fallback["reason"],
        }

    def _fallback_plan(self, actions: list[SessionAction]) -> list[dict[str, Any]]:
        plans: list[dict[str, Any]] = []
        local_tasks: list[dict[str, Any]] = []
        sample_period = max(1, int(Config.ONLINE_PERIODIC_SAMPLE_EVERY))

        for action in actions:
            chosen_task: dict[str, Any] | None = None
            normalized_app = self._normalize_token(action.app)

            for task in reversed(local_tasks):
                title_overlap = self._title_continuity_score(action.title, task["last_title"])
                same_app_recent = (
                    normalized_app
                    and normalized_app == self._normalize_token(task["last_app"])
                    and action.stream_index - task["last_seen"] <= Config.ONLINE_RECENT_TASK_GAP
                )
                if title_overlap >= Config.ONLINE_TITLE_OVERLAP_THRESHOLD or same_app_recent:
                    chosen_task = task
                    break

            if chosen_task is None:
                planner_task_id = f"task_{len(local_tasks) + 1}"
                chosen_task = {
                    "planner_task_id": planner_task_id,
                    "task_label": self._default_task_label(action),
                    "last_app": action.app,
                    "last_title": action.title,
                    "last_seen": action.stream_index,
                    "seen_count": 0,
                }
                local_tasks.append(chosen_task)
                decision = "new_task"
                should_sample = bool(action.screenshot_path)
                reason = "fallback new task"
            else:
                decision = "existing_task"
                title_changed = chosen_task["last_title"] != action.title
                should_sample = bool(action.screenshot_path) and (
                    title_changed or chosen_task["seen_count"] % sample_period == 0
                )
                reason = "fallback continuity"

            plans.append(
                {
                    "stream_index": action.stream_index,
                    "decision": decision,
                    "planner_task_id": chosen_task["planner_task_id"],
                    "task_label": chosen_task["task_label"],
                    "should_sample": should_sample,
                    "reason": reason,
                }
            )

            chosen_task["last_app"] = action.app
            chosen_task["last_title"] = action.title
            chosen_task["last_seen"] = action.stream_index
            chosen_task["seen_count"] += 1

        return plans

    def _resolve_planned_task(self, action: SessionAction, decision: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        planner_task_id = str(decision.get("planner_task_id") or "").strip()
        task_label = str(decision.get("task_label") or "").strip() or self._default_task_label(action)

        if planner_task_id and planner_task_id in self.planner_task_aliases:
            task_id = self.planner_task_aliases[planner_task_id]
            decision["decision"] = "existing_task"
            return task_id, decision

        requested_task_id = planner_task_id or self._normalize_token(task_label)
        matched_task_id = self._match_existing_task(requested_task_id, action) if requested_task_id else None
        if matched_task_id:
            if planner_task_id:
                self.planner_task_aliases[planner_task_id] = matched_task_id
            decision["decision"] = "existing_task"
            return matched_task_id, decision

        new_task_id = self._create_task_id(action.app, action.title, requested_task_id or task_label)
        self.tasks[new_task_id] = TaskState(
            task_id=new_task_id,
            task_label=task_label,
            first_seen=action.stream_index,
            last_seen=action.stream_index,
            last_app=action.app,
            last_title=action.title,
        )
        if planner_task_id:
            self.planner_task_aliases[planner_task_id] = new_task_id
        decision["decision"] = "new_task"
        return new_task_id, decision

    def _should_sample_from_plan(self, action: SessionAction, task_state: TaskState, decision: dict[str, Any]) -> bool:
        if not action.screenshot_path:
            return False
        if decision.get("should_sample"):
            return True
        if Config.ONLINE_FORCE_SAMPLE_ON_NEW_TASK and not self._task_has_memory(task_state):
            return True
        return False

    def _match_existing_task(self, requested_task_id: str, action: SessionAction) -> str | None:
        if requested_task_id in self.tasks:
            return requested_task_id
        normalized = self._normalize_token(requested_task_id)
        for task_id, task in self.tasks.items():
            if normalized and normalized in {self._normalize_token(task_id), self._normalize_token(task.task_label)}:
                return task_id
        return self._heuristic_existing_task(action)

    def _heuristic_existing_task(self, action: SessionAction) -> str | None:
        normalized_app = self._normalize_token(action.app)

        for task_id, task in reversed(self.tasks.items()):
            if self._title_continuity_score(action.title, task.last_title) >= Config.ONLINE_TITLE_OVERLAP_THRESHOLD:
                return task_id
            if normalized_app and normalized_app == self._normalize_token(task.last_app):
                if action.stream_index - task.last_seen <= Config.ONLINE_RECENT_TASK_GAP:
                    return task_id
        return None

    def _observe_action(self, action: SessionAction, task_state: TaskState) -> str:
        prompt = (
            "Role:\n"
            "You are the VLM note taker for online desktop activity tracking.\n\n"
            "Task:\n"
            "- Describe visible UI state relevant to the assigned task.\n"
            "- Write 1-2 short factual sentences.\n\n"
            "Inference policy:\n"
            "- You may use the screenshot and latest task memory together.\n"
            "- Allow only minimal, evidence-grounded inferences.\n"
            "- Do not invent hidden actions, unseen intent, or unsupported state."
        )
        latest_memory = self._get_observation_context_memory(task_state) or "<none>"
        context = (
            f"assigned_task_id={task_state.task_id}\n"
            f"assigned_task_label={task_state.task_label}\n"
            f"current_action_index={action.stream_index}\n"
            f"current_app={action.app}\n"
            f"current_title={action.title}\n"
            f"latest_task_memory={latest_memory}"
        )
        summary = self.llm_backend.generate_multimodal_summary(prompt, context, [action.screenshot_path])
        cleaned = " ".join((summary or "").split())
        if not cleaned or cleaned.startswith("[EMPTY_") or cleaned.startswith("[VLM Error]"):
            return ""
        return cleaned

    def _append_memory(self, task_state: TaskState, observation: str) -> None:
        self._store_observation(task_state, observation)

    def _task_has_memory(self, task_state: TaskState) -> bool:
        return bool(self._get_task_latest_memory(task_state))

    def _get_task_latest_memory(self, task_state: TaskState) -> str:
        if task_state.latest_memory:
            return task_state.latest_memory
        if task_state.memory_items:
            return task_state.memory_items[-1]
        return ""

    def _get_observation_context_memory(self, task_state: TaskState) -> str:
        memory_items = self._get_task_memory_items(task_state)
        limit = max(1, int(Config.ONLINE_TASK_MEMORY_PROMPT_ITEMS))
        visible_items = memory_items[-limit:]
        if not visible_items:
            return self._get_task_latest_memory(task_state)
        if len(visible_items) == 1:
            return visible_items[0]
        return "\n".join(f"[{index + 1}] {item}" for index, item in enumerate(visible_items))

    def _store_observation(self, task_state: TaskState, observation: str) -> None:
        if task_state.memory_items and task_state.memory_items[-1] == observation:
            task_state.latest_memory = observation
            return
        task_state.latest_memory = observation
        task_state.memory_items.append(observation)
        task_state.memory_items = task_state.memory_items[-Config.ONLINE_TASK_MEMORY_WINDOW :]
        task_state.observation_count += 1

    def _get_task_memory_items(self, task_state: TaskState) -> list[str]:
        return list(task_state.memory_items)

    def _summarize_tasks(self) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        for task_id, task in self.tasks.items():
            memory_items = self._get_task_memory_items(task)
            summary_memory_items = memory_items[-Config.ONLINE_SUMMARY_MAX_MEMORY_ITEMS :]
            if summary_memory_items:
                prompt = (
                    "Role:\n"
                    "You write grounded task summaries for online desktop activity tracking.\n\n"
                    "Evidence:\n"
                    "- Use only the provided observable notes.\n"
                    "- Do not invent hidden actions, intent, or missing steps.\n\n"
                    "Task:\n"
                    "- Write one concise plain-text paragraph summarizing this task.\n"
                    "- Strictly describe what the user did based on the provided memory notes.\n"
                    "- Keep it factual and specific to the observed evidence.\n"
                    "- Do not use markdown headings or bullet points."
                )
                context = (
                    f"task_id={task.task_id}\n"
                    f"task_label={task.task_label}\n"
                    f"last_app={task.last_app}\n"
                    f"last_title={task.last_title}\n"
                    "observable_notes:\n- " + "\n- ".join(summary_memory_items)
                )
                summary = self.llm_backend.generate_text_summary(prompt, context, role="summary")
                summary = " ".join((summary or "").split())
            else:
                summary = (
                    f"The task stayed in {task.last_app} around '{task.last_title}' with limited visual evidence captured online."
                )

            results[task_id] = {
                "task_id": task.task_id,
                "task_label": task.task_label,
                "action_indices": task.action_indices,
                "first_seen": task.first_seen,
                "last_seen": task.last_seen,
                "observation_count": task.observation_count,
                "memory_items": memory_items,
                "summary": summary,
            }
        return results

    def _summarize_session(self, task_summaries: dict[str, dict[str, Any]]) -> str:
        ordered_items = [task_summaries[task_id] for task_id in self.tasks.keys() if task_id in task_summaries]
        if not ordered_items:
            return ""
        return "\n".join(item["summary"] for item in ordered_items if item.get("summary"))

    def _create_task_id(self, app: str, title: str, preferred_label: str = "") -> str:
        base = self._normalize_token(preferred_label) or self._normalize_token(app) or "task"
        if title:
            title_token = self._normalize_token(title)
            if title_token and title_token not in base:
                base = f"{base}_{title_token}"[:64]

        candidate = base[:64] or "task"
        suffix = 2
        while candidate in self.tasks:
            candidate = f"{base[:56]}_{suffix}"
            suffix += 1
        return candidate

    def _default_task_label(self, action: SessionAction) -> str:
        title = (action.title or "").strip()
        app = (action.app or "").strip() or "desktop app"
        return title if title else f"Work in {app}"

    def _title_continuity_score(self, left: str, right: str) -> float:
        left_tokens = self._tokenize_title_for_overlap(left)
        right_tokens = self._tokenize_title_for_overlap(right)
        if not left_tokens or not right_tokens:
            return 0.0
        overlap = left_tokens & right_tokens
        union = left_tokens | right_tokens
        if not union:
            return 0.0
        return len(overlap) / len(union)

    def _tokenize_title_for_overlap(self, value: str) -> set[str]:
        text = (value or "").strip().lower()
        pieces = re.split(r"[^a-z0-9]+", text)
        ignored = {
            "google",
            "chrome",
            "edge",
            "mozilla",
            "firefox",
            "new",
            "tab",
            "home",
            "untitled",
            "window",
            "page",
            "document",
        }
        return {piece for piece in pieces if len(piece) >= 3 and not piece.isdigit() and piece not in ignored}

    def _normalize_token(self, value: str) -> str:
        text = (value or "").strip().lower()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        return text[:48]

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