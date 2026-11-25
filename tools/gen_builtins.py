import subprocess
import sys
from pathlib import Path

from tallyjudge.sdk.plugin import manifest_source

TARGET = Path(__file__).resolve().parents[1] / "tallyjudge" / "registry" / "builtins.py"

if __name__ == "__main__":
    TARGET.write_text(
        manifest_source("tallyjudge.plugins", "BUILTINS"), encoding="utf-8"
    )
    subprocess.run([sys.executable, "-m", "ruff", "format", str(TARGET)], check=True)
