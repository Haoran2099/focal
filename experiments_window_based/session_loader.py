from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core import SessionAction


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = Path(
    os.environ.get("DESKTOPBENCH_ROOT", str(WORKSPACE_ROOT / "data" / "desktopbench"))
).expanduser().resolve()
SESSION_CATALOG_RELATIVE_PATH = Path(
    "experiments_window_based/data/videogui_sessions/session_catalog_320.json"
)
RAW_RECORDS_RELATIVE_PATH = Path(
    "experiments_window_based/data/videogui_raw/raw_records_with_updated_labels.jsonl"
)
DEFAULT_SESSION_CATALOG_PATH = DEFAULT_DATA_ROOT / SESSION_CATALOG_RELATIVE_PATH
DEFAULT_RAW_RECORDS_PATH = DEFAULT_DATA_ROOT / RAW_RECORDS_RELATIVE_PATH


@dataclass
class SessionStream:
    session_id: str
    actions: list[SessionAction]
    catalog_segments: list[dict[str, Any]]


def resolve_input_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    workspace_candidate = (WORKSPACE_ROOT / path).resolve()
    if workspace_candidate.exists():
        return workspace_candidate

    return workspace_candidate


def resolve_dataset_paths(
    *,
    data_root: str | Path | None = None,
    session_catalog: str | Path | None = None,
    raw_records: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """Resolve explicit file overrides before the configured data root."""
    root = resolve_input_path(data_root or DEFAULT_DATA_ROOT)
    catalog_path = (
        resolve_input_path(session_catalog)
        if session_catalog
        else root / SESSION_CATALOG_RELATIVE_PATH
    )
    raw_records_path = (
        resolve_input_path(raw_records)
        if raw_records
        else root / RAW_RECORDS_RELATIVE_PATH
    )
    return root.resolve(), catalog_path.resolve(), raw_records_path.resolve()


def _resolve_screenshot_path(
    raw_path: str,
    raw_records_path: str | Path | None = None,
) -> str:
    if not raw_path:
        return ""
    path = Path(raw_path)
    if path.is_absolute():
        return str(path)

    candidates: list[Path] = []
    if raw_records_path is not None:
        source_path = resolve_input_path(raw_records_path)
        # Dataset manifests commonly store paths relative either to the raw
        # records directory or to one of its ancestors (for example the
        # bundled desktopbench/experiments_window_based tree).
        for parent in (source_path.parent, *source_path.parents):
            candidates.append((parent / path).resolve())

    candidates.append((WORKSPACE_ROOT / path).resolve())
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return str(candidate)

    # Keep the historical workspace-relative fallback so missing files still
    # produce a deterministic, debuggable path.
    return str((WORKSPACE_ROOT / path).resolve())


def load_session_catalog(session_catalog_path: str | Path) -> dict[str, list[dict[str, Any]]]:
    path = resolve_input_path(session_catalog_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Session catalog must be a JSON object: {path}")

    normalized: dict[str, list[dict[str, Any]]] = {}
    for session_id, items in payload.items():
        if not isinstance(items, list):
            raise ValueError(f"Session {session_id} must map to a list")
        normalized[session_id] = []
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f"Session {session_id} contains a non-object item")
            if "task_id" not in item or "subtask_id" not in item:
                raise ValueError(f"Session {session_id} item missing task_id or subtask_id")
            normalized[session_id].append(
                {
                    "task_id": str(item["task_id"]),
                    "subtask_id": int(item["subtask_id"]),
                    "action_indices": [int(value) for value in item.get("action_indices", [])],
                }
            )
    return normalized


def load_raw_groups(raw_records_path: str | Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    path = resolve_input_path(raw_records_path)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            key = (str(row["gt_task_id"]), int(row["gt_subtask_id"]))
            grouped[key].append(row)

    for rows in grouped.values():
        rows.sort(key=lambda item: int(item.get("action_idx", 0)))
    return grouped


def _materialize_segment_rows(
    item: dict[str, Any],
    raw_groups: dict[tuple[str, int], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    key = (item["task_id"], item["subtask_id"])
    if key not in raw_groups:
        raise KeyError(f"Task not found in raw records: {key}")

    rows = list(raw_groups[key])
    action_indices = item.get("action_indices") or []
    if action_indices:
        allowed = set(int(value) for value in action_indices)
        rows = [row for row in rows if int(row.get("action_idx", 0)) in allowed]

    if not rows:
        raise ValueError(f"No rows found for catalog item: {item}")
    return rows


def materialize_session_streams(
    session_catalog_path: str | Path,
    raw_records_path: str | Path,
    *,
    start_session_index: int = 1,
    max_sessions: int = 0,
) -> list[SessionStream]:
    session_catalog = load_session_catalog(session_catalog_path)
    resolved_raw_records_path = resolve_input_path(raw_records_path)
    raw_groups = load_raw_groups(resolved_raw_records_path)
    start_session_index = max(1, int(start_session_index))
    max_sessions = max(0, int(max_sessions))

    streams: list[SessionStream] = []
    for session_index, (session_id, items) in enumerate(session_catalog.items(), start=1):
        if session_index < start_session_index:
            continue
        if max_sessions > 0 and len(streams) >= max_sessions:
            break

        actions: list[SessionAction] = []
        for item in items:
            rows = _materialize_segment_rows(item, raw_groups)
            for row in rows:
                actions.append(
                    SessionAction(
                        session_id=session_id,
                        stream_index=len(actions),
                        app=str(row.get("app") or "").strip(),
                        title=str(row.get("title") or "").strip(),
                        screenshot_path=_resolve_screenshot_path(
                            str(row.get("screenshot_path") or "").strip(),
                            resolved_raw_records_path,
                        ),
                        source_action_idx=int(row.get("action_idx", 0)),
                        raw_timestamp=str(row.get("timestamp") or "").strip() or None,
                    )
                )

        streams.append(
            SessionStream(
                session_id=session_id,
                actions=actions,
                catalog_segments=[dict(item) for item in items],
            )
        )
    return streams
