#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_BASELINES,
    DEFAULT_EVAL_CONFIG_FILE,
    DEFAULT_GT_FILE,
    DEFAULT_ONLINE_FILES,
    add_judge_arguments,
    cache_key,
    compute_bertscore,
    ensure_dir,
    judge_geval_for_session,
    load_eval_config,
    load_ground_truth,
    load_or_compute_cache,
    load_single_method,
    mean,
    merge_online_method,
    normalize_text,
    require_judge_from_args,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
METRIC_CHOICES = ["task_acc", "bertscore", "kir", "geval"]
TASK_KIR_PROMPT_VERSION = "2026-03-31-kir-score-v5-no-match"
TASK_KIR_PROMPT = (
    "You are evaluating task-aware Key Information Recall at the task-summary level.\n"
    "Return strict JSON only.\n\n"
    "You will be given all GT per-task summaries and all predicted per-task summaries together.\n"
    "Do not assume GT tasks and predicted tasks align by order.\n"
    "Do not rely on task labels or task ids.\n"
    "Compare each GT task summary against the full set of predicted per-task summaries.\n\n"
    "For each GT task, assign exactly one of the following statuses:\n"
    "- correct_task: at least one predicted per-task summary substantially captures the main intent, operation, or outcome of this GT task\n"
    "- wrong_task: this GT task is represented, but mainly misplaced or merged into a predicted summary that is primarily describing another task\n"
    "- missing: this GT task is not meaningfully captured in any predicted per-task summary\n\n"
    "Judge by semantic meaning, not exact wording.\n"
    "correct_task should be judged leniently: if a predicted summary clearly reflects the same task and covers its major operation or result, it can be correct even if many low-level UI details are omitted.\n"
    "wrong_task should be used when the GT task content is present but clearly assigned to the wrong predicted task.\n"
    "missing should be used only when the GT task is not meaningfully present anywhere.\n"
    "Do not require exact low-level UI actions unless they are essential for distinguishing one task from another.\n\n"
    "Input JSON:\n{input_json}\n\n"
    "Output schema:\n"
    "{{\n"
    '  "items": [\n'
    '    {{"gt_task_id": "...", "task_summary": "...", "status": "correct_task", "reason": "..."}}\n'
    "  ]\n"
    "}}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified session-summary evaluation for task accuracy, BERTScore, KIR, and G-Eval."
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=METRIC_CHOICES,
        default=list(METRIC_CHOICES),
        help="Metrics to evaluate. Judge-based metrics require API credentials.",
    )
    parser.add_argument(
        "--gt-file",
        default=str(DEFAULT_GT_FILE),
        help="Ground-truth JSON file keyed by session_id.",
    )
    parser.add_argument(
        "--eval-config",
        default=str(DEFAULT_EVAL_CONFIG_FILE),
        help="Evaluation config used for optional session/file exclusions.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory for reports and judge cache.",
    )
    parser.add_argument(
        "--method-file",
        action="append",
        default=[],
        help="Single result file in the form method_name=path/to/result.json. Repeatable.",
    )
    parser.add_argument(
        "--merged-method",
        action="append",
        default=[],
        help="Merged result files in the form method_name=path1,path2,... Repeatable.",
    )
    parser.add_argument(
        "--bert-device",
        default="cpu",
        help="Device passed to bert-score, for example cpu or cuda:0.",
    )
    parser.add_argument(
        "--kir-wrong-task-penalty",
        type=float,
        default=2.0,
        help="Lambda penalty used in KIR = c / (c + lambda*w + m).",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=0,
        help="Optional cap on evaluated completed sessions per method for debugging.",
    )
    parser.add_argument(
        "--report-prefix",
        default="session_summary_metrics",
        help="Prefix used for output JSON/Markdown report files.",
    )
    add_judge_arguments(parser)
    return parser.parse_args()


def _sorted_counter(values: list[int]) -> dict[str, int]:
    return {str(key): count for key, count in sorted(Counter(values).items())}


def _parse_named_file(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid --method-file value: {spec}")
    method_name, raw_path = spec.split("=", 1)
    method_name = method_name.strip()
    raw_path = raw_path.strip()
    if not method_name or not raw_path:
        raise ValueError(f"Invalid --method-file value: {spec}")
    return method_name, Path(raw_path).expanduser().resolve()


def _parse_named_files(spec: str) -> tuple[str, list[Path]]:
    if "=" not in spec:
        raise ValueError(f"Invalid --merged-method value: {spec}")
    method_name, raw_paths = spec.split("=", 1)
    method_name = method_name.strip()
    paths = [Path(item.strip()).expanduser().resolve() for item in raw_paths.split(",") if item.strip()]
    if not method_name or not paths:
        raise ValueError(f"Invalid --merged-method value: {spec}")
    return method_name, paths


def load_selected_methods(args: argparse.Namespace) -> tuple[dict[str, Any], list[str], str]:
    eval_config_path = Path(args.eval_config).expanduser().resolve()
    eval_config = load_eval_config(eval_config_path)
    output_dir = Path(args.output_dir).expanduser().resolve()

    methods: dict[str, Any] = {}
    method_order: list[str] = []

    if args.merged_method or args.method_file:
        for spec in args.merged_method:
            method_name, json_paths = _parse_named_files(spec)
            methods[method_name] = merge_online_method(json_paths, output_dir, eval_config, method_name=method_name)
            method_order.append(method_name)
        for spec in args.method_file:
            method_name, json_path = _parse_named_file(spec)
            methods[method_name] = load_single_method(method_name, json_path, eval_config)
            method_order.append(method_name)
    else:
        methods["focal"] = merge_online_method(
            [Path(path).expanduser().resolve() for path in DEFAULT_ONLINE_FILES],
            output_dir,
            eval_config,
            method_name="focal",
        )
        methods["naive_llm_agent_session"] = load_single_method(
            "naive_llm_agent_session",
            Path(DEFAULT_BASELINES["naive_llm_agent_session"]).expanduser().resolve(),
            eval_config,
        )
        methods["brain_vlm_shared_memory_session"] = load_single_method(
            "brain_vlm_shared_memory_session",
            Path(DEFAULT_BASELINES["brain_vlm_shared_memory_session"]).expanduser().resolve(),
            eval_config,
        )
        method_order = [
            "focal",
            "naive_llm_agent_session",
            "brain_vlm_shared_memory_session",
        ]
    return methods, method_order, str(eval_config_path)


def select_completed_session_ids(dataset, gt_map: dict[str, Any], max_sessions: int = 0) -> list[str]:
    session_ids = sorted(session_id for session_id in dataset.sessions if session_id in gt_map)
    if max_sessions > 0:
        return session_ids[:max_sessions]
    return session_ids


def compute_task_accuracy(dataset, session_ids: list[str], gt_map: dict[str, Any]) -> dict[str, Any]:
    per_session = []
    predicted_counts = []
    gt_counts = []
    for session_id in session_ids:
        record = dataset.sessions[session_id]
        predicted_task_count = int(record.task_count or 0)
        gt_task_count = len(gt_map[session_id].task_summaries)
        predicted_counts.append(predicted_task_count)
        gt_counts.append(gt_task_count)
        per_session.append(
            {
                "session_id": session_id,
                "pred_task_count": predicted_task_count,
                "gt_task_count": gt_task_count,
                "correct": predicted_task_count == gt_task_count,
            }
        )
    correct_sessions = sum(1 for item in per_session if item["correct"])
    session_count = len(per_session)
    return {
        "session_count": session_count,
        "correct_sessions": correct_sessions,
        "accuracy": (correct_sessions / session_count) if session_count > 0 else 0.0,
        "pred_task_count_distribution": _sorted_counter(predicted_counts),
        "gt_task_count_distribution": _sorted_counter(gt_counts),
        "per_session": per_session,
    }


def compute_bertscore_metric(dataset, session_ids: list[str], gt_map: dict[str, Any], device: str) -> dict[str, Any]:
    predictions = [dataset.sessions[session_id].summary for session_id in session_ids]
    references = [gt_map[session_id].session_summary for session_id in session_ids]
    result = compute_bertscore(predictions, references, device)
    result["session_count"] = len(session_ids)
    result["session_ids"] = session_ids
    return result


def load_gt_task_sessions(gt_path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(gt_path.read_text(encoding="utf-8"))
    gt_sessions: dict[str, dict[str, Any]] = {}
    for session_id, item in payload.items():
        if not isinstance(item, dict):
            continue
        task_payloads = item.get("task_summaries", []) or []
        tasks = []
        for index, task in enumerate(task_payloads, start=1):
            if not isinstance(task, dict):
                continue
            task_summary = normalize_text(task.get("task_summary", ""))
            if not task_summary:
                continue
            gt_task_id = str(task.get("task_id") or f"gt_task_{index}").strip()
            tasks.append(
                {
                    "gt_task_id": gt_task_id,
                    "position": task.get("position", index),
                    "subtask_id": task.get("subtask_id"),
                    "subtask_query": normalize_text(task.get("subtask_query", "")),
                    "task_summary": task_summary,
                    "action_indices": task.get("action_indices", []) or [],
                }
            )
        gt_sessions[str(session_id)] = {
            "session_id": str(session_id),
            "session_summary": normalize_text(item.get("session_summary", "")),
            "tasks": tasks,
        }
    return gt_sessions


def build_predicted_tasks(record) -> list[dict[str, Any]]:
    predicted_tasks = []
    for index, task in enumerate(record.result.get("tasks", []) or [], start=1):
        if not isinstance(task, dict):
            continue
        summary = normalize_text(task.get("summary", ""))
        if not summary:
            continue
        predicted_tasks.append(
            {
                "pred_task_id": str(task.get("task_id") or f"pred_task_{index}").strip(),
                "task_label": normalize_text(task.get("task_label", "")),
                "task_summary": summary,
                "action_indices": task.get("action_indices", []) or [],
            }
        )
    if predicted_tasks:
        return predicted_tasks

    paragraphs = [normalize_text(part) for part in str(record.summary or "").splitlines() if normalize_text(part)]
    if len(paragraphs) > 1:
        return [
            {
                "pred_task_id": f"pred_paragraph_{index}",
                "task_label": f"Paragraph {index}",
                "task_summary": paragraph,
                "action_indices": [],
            }
            for index, paragraph in enumerate(paragraphs, start=1)
        ]
    if paragraphs:
        return [
            {
                "pred_task_id": "global_session",
                "task_label": "Whole session",
                "task_summary": paragraphs[0],
                "action_indices": [],
            }
        ]
    return []


def build_gt_task_items(gt_session: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "gt_task_id": task["gt_task_id"],
            "task_summary": task["task_summary"],
        }
        for task in gt_session["tasks"]
        if task.get("task_summary")
    ]


def judge_kir_for_session(
    judge,
    cache_dir: Path,
    method_name: str,
    session_id: str,
    gt_session: dict[str, Any],
    predicted_tasks: list[dict[str, Any]],
    gt_task_items: list[dict[str, Any]],
    wrong_task_penalty: float,
    force_refresh: bool,
) -> dict[str, Any]:
    ensure_dir(cache_dir)
    payload = {
        "prompt_version": TASK_KIR_PROMPT_VERSION,
        "method": method_name,
        "session_id": session_id,
        "gt_tasks": gt_session["tasks"],
        "predicted_tasks": predicted_tasks,
        "gt_task_items": gt_task_items,
        "wrong_task_penalty": wrong_task_penalty,
    }
    key = cache_key("task_kir", judge.model, payload)

    def _compute() -> dict[str, Any]:
        prompt = TASK_KIR_PROMPT.format(input_json=json.dumps(payload, ensure_ascii=False, indent=2))
        return judge.json_completion(prompt)

    result = load_or_compute_cache(cache_dir, key, force_refresh, _compute)
    indexed_result: dict[tuple[str, str], dict[str, Any]] = {}
    for item in result.get("items", []) if isinstance(result, dict) else []:
        if not isinstance(item, dict):
            continue
        gt_task_id = str(item.get("gt_task_id") or "").strip()
        task_summary = normalize_text(item.get("task_summary", ""))
        if gt_task_id and task_summary:
            indexed_result[(gt_task_id, task_summary)] = item

    normalized_items = []
    for gt_task_item in gt_task_items:
        gt_task_id = gt_task_item["gt_task_id"]
        task_summary = gt_task_item["task_summary"]
        raw_item = indexed_result.get((gt_task_id, task_summary), {})
        status = str(raw_item.get("status") or "missing").strip()
        if status not in {"correct_task", "wrong_task", "missing"}:
            status = "missing"
        normalized_items.append(
            {
                "gt_task_id": gt_task_id,
                "task_summary": task_summary,
                "status": status,
                "reason": normalize_text(raw_item.get("reason", "")) or "missing in judge output",
            }
        )

    correct_count = sum(1 for item in normalized_items if item["status"] == "correct_task")
    wrong_task_count = sum(1 for item in normalized_items if item["status"] == "wrong_task")
    missing_count = sum(1 for item in normalized_items if item["status"] == "missing")
    denominator = correct_count + wrong_task_penalty * wrong_task_count + missing_count
    kir_score = (correct_count / denominator) if denominator > 0 else 0.0
    return {
        "session_id": session_id,
        "pred_task_count": len(predicted_tasks),
        "gt_task_count": len(gt_session["tasks"]),
        "correct_count": correct_count,
        "wrong_task_count": wrong_task_count,
        "missing_count": missing_count,
        "task_item_count": len(normalized_items),
        "score": kir_score,
        "items": normalized_items,
    }


def compute_kir_metric(
    dataset,
    method_name: str,
    session_ids: list[str],
    gt_sessions: dict[str, dict[str, Any]],
    judge,
    cache_root: Path,
    wrong_task_penalty: float,
    force_refresh: bool,
) -> dict[str, Any]:
    score_cache_dir = cache_root / "scores"
    ensure_dir(score_cache_dir)

    per_session = []
    for session_id in session_ids:
        if session_id not in gt_sessions:
            continue
        gt_session = gt_sessions[session_id]
        record = dataset.sessions[session_id]
        predicted_tasks = build_predicted_tasks(record)
        gt_task_items = build_gt_task_items(gt_session)
        per_session.append(
            judge_kir_for_session(
                judge,
                score_cache_dir,
                method_name,
                session_id,
                gt_session,
                predicted_tasks,
                gt_task_items,
                wrong_task_penalty,
                force_refresh,
            )
        )
    return {
        "session_count": len(per_session),
        "session_ids": [item["session_id"] for item in per_session],
        "wrong_task_penalty": wrong_task_penalty,
        "mean_score": mean([item["score"] for item in per_session]),
        "mean_correct_count": mean([item["correct_count"] for item in per_session]),
        "mean_wrong_task_count": mean([item["wrong_task_count"] for item in per_session]),
        "mean_missing_count": mean([item["missing_count"] for item in per_session]),
        "per_session": per_session,
    }


def compute_geval_metric(dataset, method_name: str, session_ids: list[str], gt_map: dict[str, Any], judge, cache_dir: Path, force_refresh: bool) -> dict[str, Any]:
    ensure_dir(cache_dir)
    per_session = []
    for session_id in session_ids:
        record = dataset.sessions[session_id]
        per_session.append(
            judge_geval_for_session(
                judge=judge,
                cache_dir=cache_dir,
                method_name=method_name,
                session_id=session_id,
                prediction=record.summary,
                gt_session=gt_map[session_id],
                force_refresh=force_refresh,
            )
        )
    return {
        "session_count": len(per_session),
        "session_ids": [item["session_id"] for item in per_session],
        "overall": mean([item["overall"] for item in per_session]),
        "accuracy": mean([item["accuracy"] for item in per_session]),
        "coverage": mean([item["coverage"] for item in per_session]),
        "conciseness": mean([item["conciseness"] for item in per_session]),
        "consistency": mean([item["consistency"] for item in per_session]),
        "clarity": mean([item["clarity"] for item in per_session]),
        "per_session": per_session,
    }


def build_markdown(report: dict) -> str:
    lines = [
        "# Session Summary Metrics",
        "",
        f"Generated at: {report['generated_at']}",
        f"Ground truth: {report['ground_truth_file']}",
        f"Eval config: {report['eval_config']}",
        "",
        "All summary-quality metrics are computed on completed sessions only.",
        "",
    ]
    for method_name in report["method_order"]:
        method_report = report["methods"][method_name]
        lines.extend(
            [
                f"## {method_name}",
                "",
                f"Completed sessions used as denominator: {method_report['completed_session_count']}",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
            ]
        )
        if "task_acc" in method_report:
            lines.append(f"| Task Accuracy | {method_report['task_acc']['accuracy']:.4f} |")
        if "bertscore" in method_report:
            lines.append(f"| BERTScore F1 | {method_report['bertscore']['f1']:.4f} |")
        if "kir" in method_report:
            lines.append(f"| KIR | {method_report['kir']['mean_score']:.4f} |")
        if "geval" in method_report:
            lines.append(f"| G-Eval | {method_report['geval']['overall']:.4f} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    methods, method_order, eval_config_path = load_selected_methods(args)
    gt_path = Path(args.gt_file).expanduser().resolve()
    gt_map = load_ground_truth(gt_path)
    gt_task_sessions = load_gt_task_sessions(gt_path)

    judge_required = bool({"kir", "geval"} & set(args.metrics))
    judge = require_judge_from_args(args) if judge_required else None

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ground_truth_file": str(gt_path),
        "eval_config": eval_config_path,
        "method_order": method_order,
        "metrics": list(args.metrics),
        "methods": {},
    }
    if judge is not None:
        report["judge_model"] = args.judge_model

    for method_name in method_order:
        dataset = methods[method_name]
        session_ids = select_completed_session_ids(dataset, gt_map, max_sessions=args.max_sessions)
        method_report = {
            "raw_files": dataset.raw_files,
            "completed_session_count": len(session_ids),
            "session_ids": session_ids,
        }
        if "task_acc" in args.metrics:
            method_report["task_acc"] = compute_task_accuracy(dataset, session_ids, gt_map)
        if "bertscore" in args.metrics:
            method_report["bertscore"] = compute_bertscore_metric(dataset, session_ids, gt_map, args.bert_device)
        if "kir" in args.metrics:
            method_report["kir"] = compute_kir_metric(
                dataset,
                method_name,
                session_ids,
                gt_task_sessions,
                judge,
                output_dir / "cache" / method_name / "session_metrics" / "kir",
                args.kir_wrong_task_penalty,
                args.force_refresh_cache,
            )
        if "geval" in args.metrics:
            method_report["geval"] = compute_geval_metric(
                dataset,
                method_name,
                session_ids,
                gt_map,
                judge,
                output_dir / "cache" / method_name / "session_metrics" / "geval",
                args.force_refresh_cache,
            )
        report["methods"][method_name] = method_report

    json_path = output_dir / f"{args.report_prefix}.json"
    md_path = output_dir / f"{args.report_prefix}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(build_markdown(report), encoding="utf-8")

    print(f"Saved unified metrics JSON report to {json_path}")
    print(f"Saved unified metrics Markdown report to {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())