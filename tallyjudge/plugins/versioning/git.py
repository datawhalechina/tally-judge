"""Read experiment revisions without changing the selected Git worktree."""

import os
import subprocess
from pathlib import Path


def git_revision(repository: str | Path, revision: str = "HEAD") -> str:
    """Resolve a named revision to its full commit hash for experiment metadata."""
    return subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "rev-parse",
            "--verify",
            "--end-of-options",
            f"{revision}^{{commit}}",
        ],
        env={
            key: value
            for key, value in os.environ.items()
            if key
            not in {
                "GIT_DIR",
                "GIT_WORK_TREE",
                "GIT_COMMON_DIR",
                "GIT_OBJECT_DIRECTORY",
                "GIT_ALTERNATE_OBJECT_DIRECTORIES",
                "GIT_NAMESPACE",
            }
        },
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
