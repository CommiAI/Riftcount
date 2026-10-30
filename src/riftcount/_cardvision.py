from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from riftcount._media import INFER_STILLS
from riftcount.errors import UsageError, VisionError
from riftcount.inventory import (
    CaptureKind,
    CardName,
    Frame,
    NamedSight,
    PrintingId,
)

FRAME_INTERVAL_S = 0.5
_heif_registered = False


def sample_filter() -> str:
    rate = 1 / FRAME_INTERVAL_S
    return f"fps={rate:g}"


def release_json(release_dir: Path) -> Path:
    if not release_dir.is_dir():
        raise UsageError(f"{release_dir} is not a weights folder")
    document = release_dir / "release.json"
    if not document.is_file():
        raise UsageError(f"missing {document}. Run: riftcount update")
    return document


def release_id(document: Path) -> str:
    try:
        payload = json.loads(document.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        raise VisionError(f"unreadable release file {document}") from err
    if not isinstance(payload, dict):
        raise VisionError(f"{document} has no id")
    ident = payload.get("id")
    if not isinstance(ident, str) or not ident or ident != ident.strip():
        raise VisionError(f"{document} has no id")
    return ident


def infer_argv(binary: str, release_document: Path, images: Sequence[Path]) -> list[str]:
    for image in images:
        if image.suffix.lower() not in INFER_STILLS:
            raise VisionError(f"refusing to pass a non-jpeg to cardvision infer: {image}")
    return [
        binary,
        "infer",
        "--release",
        str(release_document),
        *(str(image) for image in images),
    ]


def read_capture(
    path: Path, kind: CaptureKind, release_dir: Path
) -> tuple[str, tuple[Frame, ...]]:
    document = release_json(release_dir)
    ident = release_id(document)
    binary = _require_binary("cardvision")
    if kind is CaptureKind.VIDEO:
        _require_binary("ffmpeg")
        with tempfile.TemporaryDirectory(prefix="riftcount-") as tmp:
            images = _extract_frames(path, Path(tmp))
            frames = _infer(binary, document, images, source=path)
        return ident, frames
    if path.suffix.lower() in INFER_STILLS:
        frames = _infer(binary, document, [path], source=path)
        return ident, frames
    with tempfile.TemporaryDirectory(prefix="riftcount-") as tmp:
        images = (_heif_to_jpeg(path, Path(tmp)),)
        frames = _infer(binary, document, images, source=path)
    return ident, frames


def _register_heif() -> None:
    global _heif_registered
    if _heif_registered:
        return
    from pillow_heif import register_heif_opener

    register_heif_opener()
    _heif_registered = True


def _heif_to_jpeg(path: Path, dest: Path) -> Path:
    from PIL import Image

    _register_heif()
    jpeg = dest / f"{path.stem}.jpg"
    try:
        with Image.open(path) as image:
            image.convert("RGB").save(jpeg, format="JPEG", quality=95)
    except OSError as err:
        raise VisionError(f"failed to decode still {path.name}") from err
    if not jpeg.is_file():
        raise VisionError(f"failed to decode still {path.name}")
    return jpeg


def _require_binary(name: str) -> str:
    found = shutil.which(name)
    if found is None:
        raise VisionError(f"{name} is not on PATH")
    return found


def _extract_frames(video: Path, dest: Path) -> tuple[Path, ...]:
    pattern = dest / "frame-%04d.jpg"
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-vf",
            sample_filter(),
            "-q:v",
            "2",
            str(pattern),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise VisionError(f"ffmpeg failed on {video.name}")
    images = tuple(sorted(path for path in dest.glob("frame-*.jpg") if path.is_file()))
    if not images:
        raise VisionError(f"ffmpeg extracted no frames from {video.name}")
    return images


def _infer(
    binary: str,
    document: Path,
    images: Sequence[Path],
    *,
    source: Path,
) -> tuple[Frame, ...]:
    proc = subprocess.run(
        infer_argv(binary, document, images),
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise VisionError(f"cardvision infer failed for {source.name}")
    frames = frames_from_stdout(proc.stdout)
    if not frames:
        raise VisionError(f"cardvision infer returned no frames for {source.name}")
    return frames


def frames_from_stdout(text: str) -> tuple[Frame, ...]:
    frames: list[Frame] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("{"):
            continue
        try:
            document = json.loads(line)
        except json.JSONDecodeError as err:
            raise VisionError("cardvision infer returned unreadable output") from err
        frames.append(_frame_from_document(document))
    return tuple(frames)


def _frame_from_document(document: object) -> Frame:
    if not isinstance(document, dict):
        raise VisionError("cardvision infer returned unreadable output")
    boxes = document.get("boxes")
    if not isinstance(boxes, list):
        raise VisionError("cardvision infer returned unreadable output")
    label = document.get("frame")
    if not isinstance(label, str) or not label:
        label = "frame"
    named: list[NamedSight] = []
    unnamed = 0
    backs = 0
    for box in boxes:
        if not isinstance(box, dict):
            raise VisionError("cardvision infer returned unreadable output")
        kind = box.get("class")
        if kind == "back":
            backs += 1
            continue
        if kind != "face":
            raise VisionError("cardvision infer returned an unknown class")
        outcome = box.get("outcome")
        if outcome == "nothing":
            unnamed += 1
            continue
        if outcome != "named":
            raise VisionError("cardvision infer returned an unknown outcome")
        name = box.get("name")
        card_id = box.get("cardId")
        if not isinstance(name, str) or not name or name != name.strip():
            raise VisionError("cardvision infer named a card without a name")
        if not isinstance(card_id, str) or not card_id or card_id != card_id.strip():
            raise VisionError("cardvision infer named a card without a card id")
        named.append(NamedSight(CardName(name), PrintingId(card_id)))
    return Frame(label=label, named=tuple(named), unnamed=unnamed, backs=backs)
