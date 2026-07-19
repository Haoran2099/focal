#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_CONFIG_FILE = Path(__file__).resolve().parent / "eval_config.json"
DEFAULT_GT_FILE = PROJECT_ROOT / "data/session_catalog_320_gt_summary_gpt.json"
DEFAULT_ONLINE_FILES = [
    PROJECT_ROOT / "outputs/focal.json",
]
DEFAULT_BASELINES = {
    "naive_llm_agent_session": PROJECT_ROOT / "outputs/naive.json",
    "brain_vlm_shared_memory_session": PROJECT_ROOT / "outputs/global-memory.json",
}
METHOD_ORDER = [
    "focal",
    "naive_llm_agent_session",
    "brain_vlm_shared_memory_session",
]
SCREENSHOT_ACTION_PATTERN = re.compile(r"action(\d+)$")
BERT_SCORER_CACHE: dict[tuple[str, str], Any] = {}


@dataclass
class GTSession:
    session_id: str
    session_summary: str
    task_summaries: list[str]


@dataclass
class SessionRecord:
    session_id: str
    summary: str
    task_count: int
    action_count: int
    result: dict[str, Any]
    trace: dict[str, Any] | None
    source_file: str
    source_rank: int


@dataclass
class MethodDataset:
    method_name: str
    sessions: dict[str, SessionRecord]
    raw_files: list[str]


@dataclass
class EvalConfig:
    config_path: str
    excluded_online_source_files: set[str]
    excluded_session_ids: set[str]


class JudgeClient:
    def __init__(self, api_key: str, model: str, base_url: Optional[str], timeout: float) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for judge-based metrics. Install with: pip install openai") from exc

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.timeout = timeout

    def json_completion(self, prompt: str, max_retries: int = 3) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    timeout=self.timeout,
                )
                content = (response.choices[0].message.content or "").strip()
                return json.loads(content)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt + 1 < max_retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"Judge request failed after retries: {last_error}")


def build_base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHOD_ORDER,
        default=list(METHOD_ORDER),
        help="Subset of methods to evaluate. Defaults to all methods.",
    )
    parser.add_argument(
        "--eval-config",
        default=str(DEFAULT_EVAL_CONFIG_FILE),
        help="Optional evaluation config JSON controlling filters and method-specific behavior.",
    )
    parser.add_argument(
        "--online-files",
        nargs="+",
        default=[str(path) for path in DEFAULT_ONLINE_FILES],
        help="Three partial online-session result JSON files to merge.",
    )
    parser.add_argument(
        "--naive-file",
        default=str(DEFAULT_BASELINES["naive_llm_agent_session"]),
        help="Baseline JSON for naive_llm_agent_session.",
    )
    parser.add_argument(
        "--shared-memory-file",
        default=str(DEFAULT_BASELINES["brain_vlm_shared_memory_session"]),
        help="Baseline JSON for brain_vlm_shared_memory_session.",
    )
    parser.add_argument(
        "--gt-file",
        default=str(DEFAULT_GT_FILE),
        help="Ground-truth session summary JSON file keyed by session_id.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "output"),
        help="Directory for merged files, caches, and reports.",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=0,
        help="Optional cap on evaluated session count for debugging.",
    )
    return parser


def add_judge_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("SESSION_SUMMARY_JUDGE_MODEL", "gpt-5"),
        help="Judge model used for this metric.",
    )
    parser.add_argument(
        "--judge-api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable name containing the judge API key.",
    )
    parser.add_argument(
        "--judge-base-url",
        default=os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE", ""),
        help="Optional OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--judge-timeout",
        type=float,
        default=120.0,
        help="Judge request timeout in seconds.",
    )
    parser.add_argument(
        "--force-refresh-cache",
        action="store_true",
        help="Ignore cached judge results and recompute them.",
    )
    return parser


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_eval_config(config_path: Path) -> EvalConfig:
    resolved_path = config_path.expanduser().resolve()
    if not resolved_path.exists():
        return EvalConfig(config_path=str(resolved_path), excluded_online_source_files=set(), excluded_session_ids=set())

    payload = load_json(resolved_path)
    excluded_files: set[str] = set()
    excluded_session_ids: set[str] = set()
    online_config = payload.get("focal", {}) if isinstance(payload, dict) else {}
    if isinstance(payload, dict):
        for item in payload.get("exclude_session_ids", []) or []:
            session_id = str(item).strip()
            if session_id:
                excluded_session_ids.add(session_id)
    if isinstance(online_config, dict):
        for item in online_config.get("exclude_source_files", []) or []:
            name = Path(str(item)).name.strip()
            if name:
                excluded_files.add(name)
    return EvalConfig(
        config_path=str(resolved_path),
        excluded_online_source_files=excluded_files,
        excluded_session_ids=excluded_session_ids,
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def mean(values: list[float | int]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def clamp_score(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 1.0
    return max(1.0, min(5.0, numeric))


def build_trace_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    traces: dict[str, dict[str, Any]] = {}
    for trace in payload.get("session_traces", []) or []:
        if not isinstance(trace, dict):
            continue
        session_id = str(trace.get("session_id") or "").strip()
        if not session_id:
            continue
        traces[session_id] = trace
    return traces


def extract_action_index_from_screenshot_id(screenshot_id: Any) -> int | None:
    match = SCREENSHOT_ACTION_PATTERN.search(str(screenshot_id or ""))
    if not match:
        return None
    return safe_int(match.group(1))


def build_action_trace_map(result: dict[str, Any]) -> dict[int, dict[str, Any]]:
    action_map: dict[int, dict[str, Any]] = {}
    for action in result.get("action_trace", []) or []:
        if not isinstance(action, dict):
            continue
        stream_index = action.get("stream_index")
        if stream_index is None:
            continue
        action_map[safe_int(stream_index)] = action
    return action_map


def session_quality_score(record: SessionRecord) -> tuple[int, int, int, int, int]:
    return (
        int(bool(record.summary)),
        int(record.trace is not None),
        safe_int(record.task_count),
        safe_int(record.action_count),
        len(record.summary),
    )


def make_session_record(entry: dict[str, Any], trace: dict[str, Any] | None, source_file: Path, source_rank: int) -> SessionRecord | None:
    session_id = str(entry.get("session_id") or "").strip()
    summary = normalize_text(entry.get("session_summary", ""))
    if not session_id or not summary:
        return None
    return SessionRecord(
        session_id=session_id,
        summary=summary,
        task_count=safe_int(entry.get("task_count", 0)),
        action_count=safe_int(entry.get("action_count", 0)),
        result=entry,
        trace=trace,
        source_file=str(source_file),
        source_rank=source_rank,
    )


def aggregate_trace_usage(traces: list[dict[str, Any]]) -> dict[str, int]:
    totals = {
        "brain_prompt_tokens": 0,
        "brain_completion_tokens": 0,
        "brain_total_tokens": 0,
        "vlm_prompt_tokens": 0,
        "vlm_completion_tokens": 0,
        "vlm_total_tokens": 0,
        "brain_calls": 0,
        "vlm_calls": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
    }
    for trace in traces:
        token_usage = trace.get("token_usage", {}) if isinstance(trace, dict) else {}
        for key in list(totals.keys())[:-3]:
            totals[key] += safe_int(token_usage.get(key, 0))
    totals["total_prompt_tokens"] = totals["brain_prompt_tokens"] + totals["vlm_prompt_tokens"]
    totals["total_completion_tokens"] = totals["brain_completion_tokens"] + totals["vlm_completion_tokens"]
    totals["total_tokens"] = totals["brain_total_tokens"] + totals["vlm_total_tokens"]
    return totals


def aggregate_online_vlm_usage_v2(records: list[SessionRecord]) -> dict[str, int]:
    totals = {
        "brain_prompt_tokens": 0,
        "brain_completion_tokens": 0,
        "brain_total_tokens": 0,
        "vlm_prompt_tokens": 0,
        "vlm_completion_tokens": 0,
        "vlm_total_tokens": 0,
        "brain_calls": 0,
        "vlm_calls": 0,
        "filtered_out_vlm_calls": 0,
        "unmatched_vlm_calls": 0,
    }
    for record in records:
        trace = record.trace or {}
        token_usage = trace.get("token_usage", {}) if isinstance(trace, dict) else {}
        totals["brain_prompt_tokens"] += safe_int(token_usage.get("brain_prompt_tokens", 0))
        totals["brain_completion_tokens"] += safe_int(token_usage.get("brain_completion_tokens", 0))
        totals["brain_total_tokens"] += safe_int(token_usage.get("brain_total_tokens", 0))
        totals["brain_calls"] += safe_int(token_usage.get("brain_calls", 0))

        sampled_actions = sorted(
            [
                action
                for action in record.result.get("action_trace", []) or []
                if isinstance(action, dict) and bool(action.get("sampled", False))
            ],
            key=lambda item: safe_int(item.get("stream_index", 0)),
        )
        vlm_calls = sorted(
            [call for call in trace.get("vlm_calls", []) or [] if isinstance(call, dict)],
            key=lambda item: safe_int(item.get("call_seq", 0)),
        )

        for index, vlm_call in enumerate(vlm_calls):
            if index >= len(sampled_actions):
                totals["unmatched_vlm_calls"] += 1
                include_call = True
            else:
                include_call = bool(sampled_actions[index].get("llm_should_sample", False))
            if not include_call:
                totals["filtered_out_vlm_calls"] += 1
                continue
            totals["vlm_prompt_tokens"] += safe_int(vlm_call.get("prompt_tokens", 0))
            totals["vlm_completion_tokens"] += safe_int(vlm_call.get("completion_tokens", 0))
            totals["vlm_total_tokens"] += safe_int(vlm_call.get("total_tokens", 0))
            totals["vlm_calls"] += 1

        if len(sampled_actions) > len(vlm_calls):
            totals["unmatched_vlm_calls"] += len(sampled_actions) - len(vlm_calls)

    totals["total_prompt_tokens"] = totals["brain_prompt_tokens"] + totals["vlm_prompt_tokens"]
    totals["total_completion_tokens"] = totals["brain_completion_tokens"] + totals["vlm_completion_tokens"]
    totals["total_tokens"] = totals["brain_total_tokens"] + totals["vlm_total_tokens"]
    return totals


def build_merged_payload(method_name: str, sessions: dict[str, SessionRecord], source_files: list[str]) -> dict[str, Any]:
    ordered_session_ids = sorted(sessions.keys())
    results = [sessions[session_id].result for session_id in ordered_session_ids]
    traces = [sessions[session_id].trace for session_id in ordered_session_ids if sessions[session_id].trace]
    token_summary = aggregate_trace_usage([sessions[session_id].trace for session_id in ordered_session_ids if sessions[session_id].trace])
    return {
        "mode": "session",
        "method": method_name,
        "merged_from": source_files,
        "session_count": len(ordered_session_ids),
        "completed_session_count": len(ordered_session_ids),
        "results": results,
        "session_traces": traces,
        "token_usage_summary": token_summary,
    }


def load_single_method(method_name: str, json_path: Path, eval_config: EvalConfig | None = None) -> MethodDataset:
    payload = load_json(json_path)
    trace_map = build_trace_map(payload)
    sessions: dict[str, SessionRecord] = {}
    for entry in payload.get("results", []) or []:
        if not isinstance(entry, dict):
            continue
        session_id = str(entry.get("session_id") or "").strip()
        if not session_id:
            continue
        if eval_config and session_id in eval_config.excluded_session_ids:
            continue
        record = make_session_record(entry, trace_map.get(session_id), json_path, 0)
        if record is not None:
            sessions[session_id] = record
    return MethodDataset(method_name=method_name, sessions=sessions, raw_files=[str(json_path)])


def merge_online_method(
    json_paths: list[Path],
    output_dir: Path,
    eval_config: EvalConfig | None = None,
    method_name: str = "focal",
) -> MethodDataset:
    merged_sessions: dict[str, SessionRecord] = {}
    source_files: list[str] = []
    for source_rank, json_path in enumerate(json_paths):
        if eval_config and json_path.name in eval_config.excluded_online_source_files:
            continue
        payload = load_json(json_path)
        source_files.append(str(json_path))
        trace_map = build_trace_map(payload)
        for entry in payload.get("results", []) or []:
            if not isinstance(entry, dict):
                continue
            session_id = str(entry.get("session_id") or "").strip()
            if not session_id:
                continue
            if eval_config and session_id in eval_config.excluded_session_ids:
                continue
            candidate = make_session_record(entry, trace_map.get(session_id), json_path, source_rank)
            if candidate is None:
                continue
            existing = merged_sessions.get(session_id)
            if existing is None:
                merged_sessions[session_id] = candidate
                continue
            existing_score = session_quality_score(existing) + (existing.source_rank,)
            candidate_score = session_quality_score(candidate) + (candidate.source_rank,)
            if candidate_score >= existing_score:
                merged_sessions[session_id] = candidate

    merged_payload = build_merged_payload(method_name, merged_sessions, source_files)
    ensure_dir(output_dir)
    merged_path = output_dir / "focal_merged.json"
    with merged_path.open("w", encoding="utf-8") as handle:
        json.dump(merged_payload, handle, indent=2, ensure_ascii=False)
    return MethodDataset(
        method_name=method_name,
        sessions=merged_sessions,
        raw_files=source_files,
    )


def load_ground_truth(gt_path: Path) -> dict[str, GTSession]:
    payload = load_json(gt_path)
    gt_map: dict[str, GTSession] = {}
    for session_id, item in payload.items():
        if not isinstance(item, dict):
            continue
        summary = normalize_text(item.get("session_summary", ""))
        task_summaries: list[str] = []
        for task in item.get("task_summaries", []) or []:
            if not isinstance(task, dict):
                continue
            task_summary = normalize_text(task.get("task_summary", ""))
            if task_summary:
                task_summaries.append(task_summary)
        if summary:
            gt_map[str(session_id)] = GTSession(
                session_id=str(session_id),
                session_summary=summary,
                task_summaries=task_summaries,
            )
    return gt_map


def load_methods(args: argparse.Namespace) -> dict[str, MethodDataset]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    eval_config = load_eval_config(Path(args.eval_config))
    all_methods = {
        "focal": merge_online_method(
            [Path(path).expanduser().resolve() for path in args.online_files],
            output_dir,
            eval_config,
        ),
        "naive_llm_agent_session": load_single_method(
            "naive_llm_agent_session",
            Path(args.naive_file).expanduser().resolve(),
            eval_config,
        ),
        "brain_vlm_shared_memory_session": load_single_method(
            "brain_vlm_shared_memory_session",
            Path(args.shared_memory_file).expanduser().resolve(),
            eval_config,
        ),
    }
    selected_methods = set(args.methods or METHOD_ORDER)
    return {method_name: dataset for method_name, dataset in all_methods.items() if method_name in selected_methods}


def resolve_method_order(methods: dict[str, MethodDataset]) -> list[str]:
    return [method_name for method_name in METHOD_ORDER if method_name in methods]


def select_session_ids(methods: dict[str, MethodDataset], gt_map: dict[str, GTSession], max_sessions: int = 0) -> tuple[dict[str, list[str]], list[str]]:
    available: dict[str, list[str]] = {}
    for method_name, dataset in methods.items():
        ids = sorted(session_id for session_id in dataset.sessions if session_id in gt_map)
        if max_sessions > 0:
            ids = ids[:max_sessions]
        available[method_name] = ids
    common = sorted(set.intersection(*(set(ids) for ids in available.values()))) if available else []
    if max_sessions > 0:
        common = common[:max_sessions]
    return available, common


def compute_efficiency_metrics(records: list[SessionRecord]) -> dict[str, Any]:
    traces = [record.trace for record in records if record.trace]
    usage = aggregate_trace_usage([trace for trace in traces if trace is not None])
    total_tasks = sum(max(record.task_count, 0) for record in records)
    session_count = len(records)
    # Paper Eq. (13): average VLM calls per session, not per predicted task.
    vcc = (usage["vlm_calls"] / session_count) if session_count > 0 else 0.0
    tcs = (usage["total_tokens"] / session_count) if session_count > 0 else 0.0
    return {
        "session_count": session_count,
        "total_task_count": total_tasks,
        "vcc": vcc,
        "tcs": tcs,
        "avg_brain_tokens_per_session": (usage["brain_total_tokens"] / session_count) if session_count > 0 else 0.0,
        "avg_vlm_tokens_per_session": (usage["vlm_total_tokens"] / session_count) if session_count > 0 else 0.0,
        "avg_total_tokens_per_vlm_call": (usage["vlm_total_tokens"] / usage["vlm_calls"]) if usage["vlm_calls"] > 0 else 0.0,
        "usage_summary": usage,
        "sessions_with_trace": len(traces),
    }


def compute_efficiency_metrics_v2(method_name: str, records: list[SessionRecord]) -> dict[str, Any]:
    session_count = len(records)
    total_tasks = sum(max(record.task_count, 0) for record in records)
    if method_name == "focal":
        usage = aggregate_online_vlm_usage_v2(records)
    else:
        traces = [record.trace for record in records if record.trace]
        usage = aggregate_trace_usage([trace for trace in traces if trace is not None])
        usage["filtered_out_vlm_calls"] = 0
        usage["unmatched_vlm_calls"] = 0
    # Paper Eq. (13): average VLM calls per session, not per predicted task.
    vcc = (usage["vlm_calls"] / session_count) if session_count > 0 else 0.0
    tcs = (usage["total_tokens"] / session_count) if session_count > 0 else 0.0
    return {
        "session_count": session_count,
        "total_task_count": total_tasks,
        "vcc": vcc,
        "tcs": tcs,
        "avg_brain_tokens_per_session": (usage["brain_total_tokens"] / session_count) if session_count > 0 else 0.0,
        "avg_vlm_tokens_per_session": (usage["vlm_total_tokens"] / session_count) if session_count > 0 else 0.0,
        "avg_total_tokens_per_vlm_call": (usage["vlm_total_tokens"] / usage["vlm_calls"]) if usage["vlm_calls"] > 0 else 0.0,
        "usage_summary": usage,
        "sessions_with_trace": sum(1 for record in records if record.trace),
    }


def compute_bertscore(predictions: list[str], references: list[str], device: str) -> dict[str, Any]:
    if not predictions or not references:
        return {
            "available": True,
            "warning": "No sessions available for this scope.",
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
    try:
        from bert_score import BERTScorer
    except ImportError:
        return {
            "available": False,
            "warning": "bert-score is not installed. Install with: pip install bert-score",
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
    cache_key = ("en", device)
    scorer = BERT_SCORER_CACHE.get(cache_key)
    if scorer is None:
        scorer = BERTScorer(lang="en", device=device)
        BERT_SCORER_CACHE[cache_key] = scorer
    precision, recall, f1 = scorer.score(predictions, references, verbose=False)
    return {
        "available": True,
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1": float(f1.mean()),
    }


def cache_key(prefix: str, model: str, payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"{prefix}_{model}_{digest}.json"


def load_or_compute_cache(cache_dir: Path, key: str, force_refresh: bool, compute_fn) -> dict[str, Any]:
    cache_path = cache_dir / key
    if not force_refresh and cache_path.exists():
        return load_json(cache_path)
    payload = compute_fn()
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return payload


def require_judge_from_args(args: argparse.Namespace) -> JudgeClient:
    api_key = os.environ.get(args.judge_api_key_env, "")
    if not api_key:
        raise RuntimeError(
            f"Missing judge API key in environment variable {args.judge_api_key_env}. "
            "Export the key before running this script."
        )
    return JudgeClient(
        api_key=api_key,
        model=args.judge_model,
        base_url=args.judge_base_url or None,
        timeout=args.judge_timeout,
    )


def extract_key_points(judge: JudgeClient, cache_dir: Path, gt_session: GTSession, force_refresh: bool) -> list[str]:
    payload = {
        "session_id": gt_session.session_id,
        "session_summary": gt_session.session_summary,
        "task_summaries": gt_session.task_summaries,
    }
    key = cache_key("key_points", judge.model, payload)

    def _compute() -> dict[str, Any]:
        prompt = (
            "You are extracting atomic key information points for summary evaluation.\n"
            "Return strict JSON only.\n\n"
            "Given the ground-truth session summary and optional task summaries, extract 4 to 12 atomic factual points.\n"
            "Each point should be independently checkable and avoid stylistic duplication.\n"
            "Prefer concrete user actions, selected settings, created artifacts, or outcomes.\n\n"
            f"Input JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
            "Output schema:\n"
            "{\n"
            '  "key_points": ["point 1", "point 2"]\n'
            "}"
        )
        return judge.json_completion(prompt)

    result = load_or_compute_cache(cache_dir, key, force_refresh, _compute)
    key_points = [normalize_text(item) for item in result.get("key_points", []) if normalize_text(item)]
    if key_points:
        return key_points
    fallback = [normalize_text(item) for item in gt_session.task_summaries if normalize_text(item)]
    return fallback if fallback else [gt_session.session_summary]


def judge_kir_for_session(
    judge: JudgeClient,
    cache_dir: Path,
    method_name: str,
    session_id: str,
    prediction: str,
    gt_session: GTSession,
    key_points: list[str],
    force_refresh: bool,
) -> dict[str, Any]:
    payload = {
        "method": method_name,
        "session_id": session_id,
        "prediction": prediction,
        "ground_truth_session_summary": gt_session.session_summary,
        "ground_truth_task_summaries": gt_session.task_summaries,
        "key_points": key_points,
    }
    key = cache_key("kir", judge.model, payload)

    def _compute() -> dict[str, Any]:
        prompt = (
            "You are evaluating Key Information Recall for a generated session summary.\n"
            "Return strict JSON only.\n\n"
            "For each key point, decide whether it is semantically covered by the generated summary.\n"
            "Use a conservative standard: mark covered only if the fact is clearly preserved, possibly via paraphrase.\n"
            "Return the items in the exact same order as the input key_points list.\n"
            "Do not require identical wording.\n\n"
            f"Input JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
            "Output schema:\n"
            "{\n"
            '  "items": [\n'
            '    {"point": "...", "covered": true, "reason": "..."}\n'
            "  ]\n"
            "}"
        )
        return judge.json_completion(prompt)

    result = load_or_compute_cache(cache_dir, key, force_refresh, _compute)
    items = result.get("items", []) if isinstance(result, dict) else []
    normalized_items = []
    for point, item in zip(key_points, items):
        if not isinstance(item, dict):
            continue
        normalized_items.append(
            {
                "point": point,
                "covered": bool(item.get("covered", False)),
                "reason": normalize_text(item.get("reason", "")),
            }
        )
    if len(normalized_items) < len(key_points):
        covered_points = {item["point"] for item in normalized_items}
        for point in key_points:
            if point not in covered_points:
                normalized_items.append({"point": point, "covered": False, "reason": "missing in judge output"})
    covered_count = sum(1 for item in normalized_items if item["covered"])
    score = (covered_count / len(key_points)) if key_points else 0.0
    return {
        "session_id": session_id,
        "key_point_count": len(key_points),
        "covered_count": covered_count,
        "score": score,
        "items": normalized_items,
    }


def judge_geval_for_session(
    judge: JudgeClient,
    cache_dir: Path,
    method_name: str,
    session_id: str,
    prediction: str,
    gt_session: GTSession,
    force_refresh: bool,
) -> dict[str, Any]:
    payload = {
        "method": method_name,
        "session_id": session_id,
        "prediction": prediction,
        "ground_truth_session_summary": gt_session.session_summary,
        "ground_truth_task_summaries": gt_session.task_summaries,
    }
    key = cache_key("geval", judge.model, payload)

    def _compute() -> dict[str, Any]:
        prompt = (
            "You are running G-Eval for a generated session summary.\n"
            "Return strict JSON only.\n\n"
            "Score the generated summary on five dimensions from 1 to 5:\n"
            "- accuracy: factual consistency with the reference session evidence\n"
            "- coverage: completeness of important session content\n"
            "- conciseness: absence of redundant or irrelevant detail\n"
            "- consistency: internal logical coherence\n"
            "- clarity: readability and fluency\n"
            "Provide brief reasons. The overall score is the arithmetic mean of the five dimensions.\n\n"
            f"Input JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
            "Output schema:\n"
            "{\n"
            '  "accuracy": 1,\n'
            '  "coverage": 1,\n'
            '  "conciseness": 1,\n'
            '  "consistency": 1,\n'
            '  "clarity": 1,\n'
            '  "overall": 1.0,\n'
            '  "reasoning": "short rationale"\n'
            "}"
        )
        return judge.json_completion(prompt)

    result = load_or_compute_cache(cache_dir, key, force_refresh, _compute)
    accuracy = clamp_score(result.get("accuracy", 1))
    coverage = clamp_score(result.get("coverage", 1))
    conciseness = clamp_score(result.get("conciseness", 1))
    consistency = clamp_score(result.get("consistency", 1))
    clarity = clamp_score(result.get("clarity", 1))
    overall = float((accuracy + coverage + conciseness + consistency + clarity) / 5.0)
    return {
        "session_id": session_id,
        "accuracy": accuracy,
        "coverage": coverage,
        "conciseness": conciseness,
        "consistency": consistency,
        "clarity": clarity,
        "overall": overall,
        "reasoning": normalize_text(result.get("reasoning", "")),
    }
