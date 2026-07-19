#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from experiments_window_based.config import Config
from experiments_window_based.core import LLMBackendInterface
from experiments_window_based.session_loader import (
    DEFAULT_DATA_ROOT,
    SessionStream,
    materialize_session_streams,
    resolve_dataset_paths,
)
from experiments_window_based.session_runner import SessionExperimentRunner
from major_tom.llm.ollama_backend import OllamaBackend
from major_tom.llm.openai_backend import OpenAIBackend


class SupplementLLMWrapper(LLMBackendInterface):
    def __init__(self, backend: Any, brain_model: str, vlm_model: str, embedding_model: str):
        self.backend = backend
        self.brain_model = brain_model
        self.vlm_model = vlm_model
        self.embedding_model = embedding_model
        self._session_meta: dict[str, Any] = {}
        self._session_llm_calls: list[dict[str, Any]] = []
        self._session_vlm_calls: list[dict[str, Any]] = []
        self._session_usage = self._new_usage_bucket()
        self._total_usage = self._new_usage_bucket()

    def _new_usage_bucket(self) -> dict[str, int]:
        return {
            "brain_prompt_tokens": 0,
            "brain_completion_tokens": 0,
            "brain_total_tokens": 0,
            "vlm_prompt_tokens": 0,
            "vlm_completion_tokens": 0,
            "vlm_total_tokens": 0,
            "brain_calls": 0,
            "vlm_calls": 0,
        }

    def _is_output_usable(self, text: str) -> bool:
        cleaned = " ".join((text or "").split())
        if not cleaned:
            return False
        if cleaned.startswith("[EMPTY_") or cleaned.startswith("[LLM Error]") or cleaned.startswith("[VLM Error]"):
            return False
        if len(cleaned) < int(Config.LLM_MIN_SUMMARY_CHARS):
            return False
        if len(cleaned.split()) < int(Config.LLM_MIN_SUMMARY_WORDS):
            return False
        return True

    def _build_prompt(self, prompt: str, context: str) -> str:
        if (context or "").strip():
            return f"{prompt}\n\nContext:\n{context}"
        return prompt

    def _update_usage(self, channel: str, prompt_tokens: int, completion_tokens: int, total_tokens: int) -> None:
        prefix = "brain" if channel == "brain" else "vlm"
        for bucket in (self._session_usage, self._total_usage):
            bucket[f"{prefix}_prompt_tokens"] += int(prompt_tokens or 0)
            bucket[f"{prefix}_completion_tokens"] += int(completion_tokens or 0)
            bucket[f"{prefix}_total_tokens"] += int(total_tokens or 0)
            bucket[f"{prefix}_calls"] += 1

    def get_usage_summary(self) -> dict[str, int]:
        return dict(self._total_usage)

    def start_session_trace(self, method_name: str, session_index: int, session_id: str, action_count: int) -> None:
        self._session_meta = {
            "method": method_name,
            "session_index": int(session_index),
            "session_id": session_id,
            "action_count": int(action_count),
        }
        self._session_llm_calls = []
        self._session_vlm_calls = []
        self._session_usage = self._new_usage_bucket()

    def finalize_session_trace(self) -> dict[str, Any]:
        return {
            **self._session_meta,
            "token_usage": dict(self._session_usage),
            "llm_calls": list(self._session_llm_calls),
            "vlm_calls": list(self._session_vlm_calls),
        }

    def generate_embedding(self, text: str) -> list[float]:
        response = self.backend.embed(model=self.embedding_model, prompt=text)
        return list(response.vector)

    def generate_text_summary(self, prompt: str, context: str, role: str = "brain") -> str:
        full_prompt = self._build_prompt(prompt, context)
        attempts = max(1, 1 + int(Config.LLM_MAX_RETRIES))
        last_text = ""
        for attempt in range(1, attempts + 1):
            try:
                response = self.backend.generate(
                    model=self.brain_model,
                    prompt=full_prompt,
                    stream=False,
                    options={"temperature": 0.1},
                )
                text = (response.text or "").strip()
                self._update_usage("brain", response.prompt_tokens, response.completion_tokens, response.total_tokens)
                self._session_llm_calls.append(
                    {
                        "role": role,
                        "attempt": attempt,
                        "output": text,
                        "prompt_tokens": int(response.prompt_tokens or 0),
                        "completion_tokens": int(response.completion_tokens or 0),
                        "total_tokens": int(response.total_tokens or 0),
                    }
                )
                last_text = text
                if self._is_output_usable(text):
                    return text
            except Exception as exc:  # noqa: BLE001
                last_text = f"[LLM Error] {exc}"
                self._session_llm_calls.append({"role": role, "attempt": attempt, "output": last_text})
        return last_text

    def generate_multimodal_summary(self, prompt: str, context: str, image_paths: list[str]) -> str:
        if not image_paths:
            return self.generate_text_summary(prompt, context, role="summary")

        selected_paths = list(image_paths)
        max_images = max(1, int(Config.MAX_IMAGES_PER_MULTIMODAL_CALL))
        if len(selected_paths) > max_images:
            stride = max(1, len(selected_paths) // max_images)
            selected_paths = selected_paths[::stride][:max_images]

        image_bytes: list[bytes] = []
        screenshot_ids: list[str] = []
        for image_path in selected_paths:
            path = Path(image_path)
            if not path.exists():
                continue
            image_bytes.append(path.read_bytes())
            screenshot_ids.append(path.stem)

        if not image_bytes:
            return ""

        full_prompt = self._build_prompt(prompt, context)
        attempts = max(1, 1 + int(Config.LLM_MAX_RETRIES))
        last_text = ""
        for attempt in range(1, attempts + 1):
            try:
                response = self.backend.generate(
                    model=self.vlm_model,
                    prompt=full_prompt,
                    images=image_bytes,
                    stream=False,
                    options={"temperature": 0.1},
                )
                text = (response.text or "").strip()
                self._update_usage("vlm", response.prompt_tokens, response.completion_tokens, response.total_tokens)
                self._session_vlm_calls.append(
                    {
                        "attempt": attempt,
                        "screenshot_ids": screenshot_ids,
                        "output": text,
                        "prompt_tokens": int(response.prompt_tokens or 0),
                        "completion_tokens": int(response.completion_tokens or 0),
                        "total_tokens": int(response.total_tokens or 0),
                    }
                )
                last_text = text
                if self._is_output_usable(text):
                    return text
            except Exception as exc:  # noqa: BLE001
                last_text = f"[VLM Error] {exc}"
                self._session_vlm_calls.append(
                    {"attempt": attempt, "screenshot_ids": screenshot_ids, "output": last_text}
                )
        return last_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FOCAL and session baselines on explicitly prepared DesktopBench data."
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "openai"],
        default="ollama",
        help="LLM backend used for both text and vision calls.",
    )
    parser.add_argument(
        "--method",
        choices=[
            "focal",
            "naive_llm_agent_session",
            "brain_vlm_shared_memory_session",
        ],
        default="focal",
        help="Method or baseline to run.",
    )
    parser.add_argument("--brain-model", default=Config.DEFAULT_BRAIN_MODEL)
    parser.add_argument("--vlm-model", default=Config.DEFAULT_VLM_MODEL)
    parser.add_argument("--embedding-model", default=Config.DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="Prepared DesktopBench root (default: DESKTOPBENCH_ROOT or data/desktopbench)",
    )
    parser.add_argument(
        "--session-catalog",
        default="",
        help="Explicit catalog path; overrides --data-root",
    )
    parser.add_argument(
        "--session-raw-records",
        default="",
        help="Explicit raw-record JSONL path; overrides --data-root",
    )
    parser.add_argument("--session-start-index", type=int, default=1)
    parser.add_argument("--session-limit", type=int, default=0)
    parser.add_argument("--output-file", default="")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--openai-base-url", default="")
    return parser.parse_args()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def portable_path(path: Path) -> str:
    """Store reproducible paths without embedding a user's absolute home path."""
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return f"<external>/{resolved.name}"


def build_backend(args: argparse.Namespace) -> SupplementLLMWrapper:
    if args.backend == "openai":
        backend = OpenAIBackend(
            base_url=args.openai_base_url or None,
        )
    else:
        backend = OllamaBackend()
    return SupplementLLMWrapper(backend, args.brain_model, args.vlm_model, args.embedding_model)


def resolve_output_path(args: argparse.Namespace, session_catalog_path: Path, session_count: int) -> Path:
    if args.resume_from:
        candidate = Path(args.resume_from).expanduser()
    elif args.output_file:
        candidate = Path(args.output_file).expanduser()
    else:
        candidate = PROJECT_ROOT / "outputs" / f"{args.method}_{session_catalog_path.stem}_n{session_count}.json"
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def persist_progress(
    output_file: Path,
    payload: dict[str, Any],
    wrapper: SupplementLLMWrapper,
    save_every: int,
    force: bool = False,
) -> None:
    completed = len(payload["results"])
    if not force and completed % max(1, save_every) != 0:
        return
    payload["completed_session_count"] = completed
    payload["token_usage_summary"] = wrapper.get_usage_summary()
    save_json(output_file, payload)


def main() -> None:
    args = parse_args()
    data_root, session_catalog_path, raw_records_path = resolve_dataset_paths(
        data_root=args.data_root,
        session_catalog=args.session_catalog or None,
        raw_records=args.session_raw_records or None,
    )

    if not session_catalog_path.exists():
        raise FileNotFoundError(
            f"Session catalog not found: {session_catalog_path}. "
            "Run `python scripts/prepare_data.py --data-dir data/desktopbench --revision v0.1.0`."
        )
    if not raw_records_path.exists():
        raise FileNotFoundError(
            f"Raw records not found: {raw_records_path}. "
            "Run `python scripts/prepare_data.py --data-dir data/desktopbench --revision v0.1.0`."
        )

    session_streams = materialize_session_streams(
        session_catalog_path,
        raw_records_path,
        start_session_index=args.session_start_index,
        max_sessions=args.session_limit,
    )
    if not session_streams:
        raise ValueError("No sessions were loaded.")

    wrapper = build_backend(args)
    runner = SessionExperimentRunner(wrapper)
    output_file = resolve_output_path(args, session_catalog_path, len(session_streams))

    resumed_results: list[dict[str, Any]] = []
    resumed_traces: list[dict[str, Any]] = []
    completed_session_ids: set[str] = set()
    if args.resume_from:
        payload = json.loads(output_file.read_text(encoding="utf-8"))
        resumed_results = list(payload.get("results", []))
        resumed_traces = list(payload.get("session_traces", []))
        completed_session_ids = {
            str(item.get("session_id"))
            for item in resumed_results
            if item.get("session_id")
        }
        session_streams = [stream for stream in session_streams if stream.session_id not in completed_session_ids]
        if not session_streams:
            return

    output_payload: dict[str, Any] = {
        "mode": "session",
        "method": args.method,
        "backend": args.backend,
        "brain_model": args.brain_model,
        "vlm_model": args.vlm_model,
        "embedding_model": args.embedding_model,
        "data_root": portable_path(data_root),
        "session_catalog": portable_path(session_catalog_path),
        "session_raw_records": portable_path(raw_records_path),
        "session_start_index": max(1, int(args.session_start_index)),
        "session_count": len(session_streams) + len(resumed_results),
        "completed_session_count": len(resumed_results),
        "results": resumed_results,
        "session_traces": resumed_traces,
        "token_usage_summary": {},
        "output_file": portable_path(output_file),
    }

    def on_session_complete(session_index: int, session: SessionStream, result: dict[str, Any], trace: dict[str, Any]) -> None:
        output_payload["results"].append(result)
        if trace:
            output_payload["session_traces"].append(trace)
        persist_progress(output_file, output_payload, wrapper, args.save_every, force=False)

    runner.run_experiment(session_streams, args.method, on_session_complete=on_session_complete)
    persist_progress(output_file, output_payload, wrapper, args.save_every, force=True)
    print(f"Saved results to {output_file}")


if __name__ == "__main__":
    main()
