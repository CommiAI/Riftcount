from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from riftcount.errors import UsageError, VisionError

WEIGHTS_DIRNAME = Path("riftcount") / "weights"


def default_weights_dir() -> Path:
    cache = Path.home() / ".cache"
    return (cache / WEIGHTS_DIRNAME).resolve()


def resolve_weights_dir(override: Path | None) -> Path:
    if override is not None:
        return override.expanduser()
    return default_weights_dir()


def require_weights_dir(override: Path | None) -> Path:
    directory = resolve_weights_dir(override)
    pin = directory / "release.json"
    if pin.is_file():
        return directory
    raise UsageError(
        f"missing weights at {directory}. Run: riftcount update"
    )


def update_weights(directory: Path | None = None) -> Path:
    target = resolve_weights_dir(directory)
    binary = shutil.which("cardvision")
    if binary is None:
        raise VisionError("cardvision is not on PATH")
    target.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [binary, "fetch-weights", "--overwrite", str(target)],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        message = f"cardvision fetch-weights failed for {target}"
        if detail:
            message = f"{message}: {detail}"
        raise VisionError(message)
    if not (target / "release.json").is_file():
        raise VisionError(f"fetch-weights left no release.json in {target}")
    return target
