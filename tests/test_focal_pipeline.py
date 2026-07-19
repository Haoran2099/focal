import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from experiments_window_based.core import LLMBackendInterface, SessionAction
from experiments_window_based.focal_method import FocalSessionMethod
from experiments_window_based.session_loader import (
    RAW_RECORDS_RELATIVE_PATH,
    SESSION_CATALOG_RELATIVE_PATH,
    _resolve_screenshot_path,
    materialize_session_streams,
    resolve_dataset_paths,
)
from experiments_window_based.session_summary_eval.common import compute_efficiency_metrics


class FakeBackend(LLMBackendInterface):
    def generate_text_summary(self, prompt: str, context: str, role: str = "brain") -> str:
        if role == "planner":
            return json.dumps(
                {
                    "action_plans": [
                        {"stream_index": 0, "decision": "new_task", "task_id": "task_1", "task_label": "Edit", "should_sample": True, "reason": "start edit"},
                        {"stream_index": 1, "decision": "new_task", "task_id": "task_2", "task_label": "Reference", "should_sample": True, "reason": "short detour"},
                        {"stream_index": 2, "decision": "existing_task", "task_id": "task_1", "task_label": "Edit", "should_sample": True, "reason": "return to edit"},
                    ]
                }
            )
        return "Grounded task summary."

    def generate_multimodal_summary(self, prompt: str, context: str, image_paths: list[str]) -> str:
        return f"Observed {image_paths[0]}."

    def generate_embedding(self, text: str) -> list[float]:
        return [1.0]


class FocalPipelineTest(unittest.TestCase):
    def test_aba_returns_to_isolated_task_memory(self) -> None:
        actions = [
            SessionAction("session", 0, "Editor", "Draft", "action0.png"),
            SessionAction("session", 1, "Browser", "Reference", "action1.png"),
            SessionAction("session", 2, "Editor", "Draft", "action2.png"),
        ]
        result = FocalSessionMethod(FakeBackend()).process_session("session", actions)

        self.assertEqual(result["task_count"], 2)
        self.assertEqual(result["tasks"][0]["action_indices"], [0, 2])
        self.assertEqual(result["tasks"][0]["observation_count"], 2)
        self.assertEqual(result["tasks"][1]["action_indices"], [1])

    def test_vcc_matches_paper_calls_per_session(self) -> None:
        def record(vlm_calls: int):
            return SimpleNamespace(
                task_count=3,
                trace={
                    "token_usage": {
                        "brain_prompt_tokens": 10,
                        "brain_completion_tokens": 5,
                        "brain_total_tokens": 15,
                        "vlm_prompt_tokens": 20,
                        "vlm_completion_tokens": 10,
                        "vlm_total_tokens": 30,
                        "brain_calls": 1,
                        "vlm_calls": vlm_calls,
                    }
                },
            )

        metrics = compute_efficiency_metrics([record(4), record(6)])
        self.assertEqual(metrics["total_task_count"], 6)
        self.assertEqual(metrics["vcc"], 5.0)

    def test_screenshot_path_resolves_from_dataset_ancestor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir) / "desktopbench"
            raw_dir = dataset_root / "experiments_window_based" / "data" / "videogui_raw"
            screenshot = raw_dir / "screenshots" / "action0.jpg"
            screenshot.parent.mkdir(parents=True)
            screenshot.write_bytes(b"image")
            raw_records = raw_dir / "records.jsonl"
            raw_records.write_text("", encoding="utf-8")

            resolved = _resolve_screenshot_path(
                "experiments_window_based/data/videogui_raw/screenshots/action0.jpg",
                raw_records,
            )

        self.assertEqual(Path(resolved), screenshot.resolve())

    def test_data_root_and_explicit_path_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "desktopbench"
            explicit_catalog = Path(temp_dir) / "catalog.json"
            explicit_records = Path(temp_dir) / "records.jsonl"
            resolved_root, catalog, records = resolve_dataset_paths(
                data_root=root,
                session_catalog=explicit_catalog,
                raw_records=explicit_records,
            )

        self.assertEqual(resolved_root, root.resolve())
        self.assertEqual(catalog, explicit_catalog.resolve())
        self.assertEqual(records, explicit_records.resolve())

    def test_data_root_builds_standard_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "desktopbench"
            _, catalog, records = resolve_dataset_paths(data_root=root)

        self.assertEqual(catalog, (root / SESSION_CATALOG_RELATIVE_PATH).resolve())
        self.assertEqual(records, (root / RAW_RECORDS_RELATIVE_PATH).resolve())

    def test_fixture_session_materializes_existing_screenshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            screenshot = data_root / "screenshots" / "action0.jpg"
            screenshot.parent.mkdir(parents=True)
            screenshot.write_bytes(b"image")
            raw_records = data_root / "records.jsonl"
            raw_records.write_text(
                json.dumps(
                    {
                        "gt_task_id": "task",
                        "gt_subtask_id": 0,
                        "action_idx": 0,
                        "app": "Editor",
                        "title": "Draft",
                        "screenshot_path": "screenshots/action0.jpg",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            catalog = data_root / "catalog.json"
            catalog.write_text(
                json.dumps({"session001": [{"task_id": "task", "subtask_id": 0}]}),
                encoding="utf-8",
            )
            streams = materialize_session_streams(catalog, raw_records, max_sessions=1)

        self.assertEqual(len(streams), 1)
        self.assertEqual(len(streams[0].actions), 1)
        self.assertEqual(Path(streams[0].actions[0].screenshot_path), screenshot.resolve())


if __name__ == "__main__":
    unittest.main()
