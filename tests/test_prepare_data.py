import tempfile
from pathlib import Path
from unittest.mock import patch

from scripts.prepare_data import prepare_data


def _fixture_release(root: Path) -> Path:
    release = root / "release"
    scripts = release / "scripts"
    scripts.mkdir(parents=True)
    for name in ("rebuild_desktopbench.py", "validate_release.py"):
        (scripts / name).write_text("# fixture\n", encoding="utf-8")
    return release


def test_local_release_runs_rebuild_and_validation_without_download() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        release = _fixture_release(root)
        with patch("scripts.prepare_data.subprocess.run") as run, patch(
            "scripts.prepare_data.snapshot_download"
        ) as download:
            prepared = prepare_data(root / "data", release_dir=release)

        assert prepared == (root / "data").resolve()
        assert run.call_count == 2
        download.assert_not_called()


def test_download_uses_dataset_type_and_pinned_revision() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        release = _fixture_release(root)
        with patch("scripts.prepare_data.snapshot_download", return_value=str(release)) as download, patch(
            "scripts.prepare_data.subprocess.run"
        ):
            prepare_data(root / "data", revision="v0.1.0")

        kwargs = download.call_args.kwargs
        assert kwargs["repo_id"] == "HaoranYin/desktopbench"
        assert kwargs["repo_type"] == "dataset"
        assert kwargs["revision"] == "v0.1.0"
