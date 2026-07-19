#!/usr/bin/env python3
"""Explicitly download the private DesktopBench release and rebuild inputs."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "HaoranYin/desktopbench"
DEFAULT_REVISION = "v0.1.0"


def run_release_tools(release_dir: Path, data_dir: Path) -> None:
    rebuild = release_dir / "scripts" / "rebuild_desktopbench.py"
    validate = release_dir / "scripts" / "validate_release.py"
    if not rebuild.is_file() or not validate.is_file():
        raise FileNotFoundError(
            "The DesktopBench release is missing scripts/rebuild_desktopbench.py "
            "or scripts/validate_release.py"
        )
    subprocess.run(
        [
            sys.executable,
            str(rebuild),
            "--release-root",
            str(release_dir),
            "--output-dir",
            str(data_dir),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(validate),
            "--release-root",
            str(release_dir),
            "--prepared-root",
            str(data_dir),
        ],
        check=True,
    )


def prepare_data(
    data_dir: Path,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    release_dir: Path | None = None,
) -> Path:
    """Prepare data only after this function is explicitly invoked by the user."""
    data_dir = data_dir.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    if release_dir is not None:
        run_release_tools(release_dir.expanduser().resolve(), data_dir)
        return data_dir

    with tempfile.TemporaryDirectory(prefix="desktopbench-release-") as temp_dir:
        local_release = Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                local_dir=Path(temp_dir) / "release",
            )
        )
        run_release_tools(local_release, data_dir)
    return data_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/desktopbench"))
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--release-dir",
        type=Path,
        help="Use an already downloaded DesktopBench release instead of contacting Hugging Face",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared = prepare_data(
        args.data_dir,
        repo_id=args.repo_id,
        revision=args.revision,
        release_dir=args.release_dir,
    )
    print(f"Prepared DesktopBench at {prepared}")


if __name__ == "__main__":
    main()
