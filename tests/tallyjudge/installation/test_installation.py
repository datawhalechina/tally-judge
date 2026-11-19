"""Import isolation and a real wheel-install check, runnable from local CI.

uv run python tests/tallyjudge/installation/test_installation.py --wheel-dir <dir>
The wheel check creates a fresh venv outside the checkout and installs only the
wheel's runtime dependencies. It does not import pytest or the source package.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROBE = Path(__file__).with_name("wheel_probe.py")


def test_import_is_independent_of_legacy_and_provider_packages():
    process = subprocess.run(
        [sys.executable, "-c", PROBE.read_text(encoding="utf-8")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stdout + process.stderr


def verify_wheel(wheel_dir: Path, *, text_distance: bool = False) -> None:
    wheels = list(wheel_dir.resolve().glob("tallyjudge-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("Expected exactly one TallyJudge wheel in wheel-dir")
    checkout = Path(__file__).resolve().parents[3]
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")
    }
    with tempfile.TemporaryDirectory(prefix="tallyjudge-wheel-check-") as temp:
        directory = Path(temp)
        venv = directory / "venv"
        subprocess.run(
            ["uv", "venv", "--python", sys.executable, str(venv)],
            cwd=directory,
            env=env,
            check=True,
        )
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                f"tallyjudge[text-distance] @ {wheels[0].as_uri()}"
                if text_distance
                else str(wheels[0]),
            ],
            cwd=directory,
            env=env,
            check=True,
        )
        subprocess.run(
            [
                str(python),
                "-I",
                str(PROBE),
                str(checkout),
                "text-distance" if text_distance else "base",
            ],
            cwd=directory,
            env=env,
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", required=True, type=Path)
    parser.add_argument(
        "--text-distance",
        action="store_true",
        help="Install and verify the optional text metrics",
    )
    args = parser.parse_args()
    verify_wheel(args.wheel_dir, text_distance=args.text_distance)
