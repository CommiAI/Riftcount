from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
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
_UNREADABLE = "cardvision infer returned unreadable output"


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


@dataclass(frozen=True, slots=True)
class Similarity:
    value: float

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise ValueError("similarity must be a finite number")
        number = float(self.value)
        if not math.isfinite(number):
            raise ValueError("similarity must be a finite number")
        object.__setattr__(self, "value", number)


@dataclass(frozen=True, slots=True)
class Guess:
    """Refused topName. This is not a CardName and cannot enter an inventory line."""

    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text or self.text != self.text.strip():
            raise ValueError("guess must be a non-empty trimmed string")


@dataclass(frozen=True, slots=True)
class PixelBox:
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        x = _finite(self.x, "box")
        y = _finite(self.y, "box")
        width = _finite(self.width, "box")
        height = _finite(self.height, "box")
        if not (width > 0 and height > 0):
            raise ValueError("box width and height must be > 0")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)


@dataclass(frozen=True, slots=True)
class NamedFace:
    name: CardName
    printing: PrintingId
    similarity: Similarity | None = None


@dataclass(frozen=True, slots=True)
class UnnamedFace:
    guess: Guess | None = None
    similarity: Similarity | None = None


@dataclass(frozen=True, slots=True)
class BackFace:
    pass


@dataclass(frozen=True, slots=True)
class Detection:
    face: NamedFace | UnnamedFace | BackFace
    box: PixelBox | None


@dataclass(frozen=True, slots=True)
class ParsedFrame:
    label: str
    detections: tuple[Detection, ...]


@dataclass(frozen=True, slots=True)
class InferredJpeg:
    index: int
    path: Path
    parsed: ParsedFrame

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 1:
            raise ValueError("frame index must be an integer >= 1")


@dataclass(frozen=True, slots=True)
class OpenedCapture:
    release_id: str
    jpegs: tuple[InferredJpeg, ...]

    @property
    def frames(self) -> tuple[Frame, ...]:
        return tuple(_tally(jpeg.parsed) for jpeg in self.jpegs)


def frames_from_stdout(text: str) -> tuple[Frame, ...]:
    return tuple(_tally(parsed) for parsed in _parse_stdout(text))


@contextmanager
def open_capture(
    path: Path, kind: CaptureKind, release_dir: Path
) -> Iterator[OpenedCapture]:
    document = release_json(release_dir)
    ident = release_id(document)
    binary = _require_binary("cardvision")
    if kind is CaptureKind.VIDEO:
        _require_binary("ffmpeg")
        with tempfile.TemporaryDirectory(prefix="riftcount-") as tmp:
            images = _extract_frames(path, Path(tmp))
            yield _opened(ident, binary, document, images, source=path)
        return
    if path.suffix.lower() in INFER_STILLS:
        yield _opened(ident, binary, document, (path,), source=path)
        return
    with tempfile.TemporaryDirectory(prefix="riftcount-") as tmp:
        images = (_heif_to_jpeg(path, Path(tmp)),)
        yield _opened(ident, binary, document, images, source=path)


def read_capture(
    path: Path, kind: CaptureKind, release_dir: Path
) -> tuple[str, tuple[Frame, ...]]:
    with open_capture(path, kind, release_dir) as opened:
        return opened.release_id, opened.frames


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} components must be real numbers")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} components must be finite")
    return number


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


def _opened(
    ident: str,
    binary: str,
    document: Path,
    images: Sequence[Path],
    *,
    source: Path,
) -> OpenedCapture:
    parsed = _infer(binary, document, images, source=source)
    if len(parsed) != len(images):
        raise VisionError(_UNREADABLE)
    # The wire `frame` string is not a path. Position in the infer argv is the order.
    jpegs = tuple(
        InferredJpeg(index=index, path=image, parsed=frame)
        for index, (image, frame) in enumerate(zip(images, parsed), start=1)
    )
    return OpenedCapture(release_id=ident, jpegs=jpegs)


def _infer(
    binary: str,
    document: Path,
    images: Sequence[Path],
    *,
    source: Path,
) -> tuple[ParsedFrame, ...]:
    proc = subprocess.run(
        infer_argv(binary, document, images),
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise VisionError(f"cardvision infer failed for {source.name}")
    frames = _parse_stdout(proc.stdout)
    if not frames:
        raise VisionError(f"cardvision infer returned no frames for {source.name}")
    return frames


def _parse_stdout(text: str) -> tuple[ParsedFrame, ...]:
    frames: list[ParsedFrame] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("{"):
            continue
        try:
            document = json.loads(line)
        except json.JSONDecodeError as err:
            raise VisionError(_UNREADABLE) from err
        frames.append(_parse_document(document))
    return tuple(frames)


def _parse_document(document: object) -> ParsedFrame:
    if not isinstance(document, dict):
        raise VisionError(_UNREADABLE)
    boxes = document.get("boxes")
    if not isinstance(boxes, list):
        raise VisionError(_UNREADABLE)
    label = document.get("frame")
    if not isinstance(label, str) or not label:
        label = "frame"
    detections: list[Detection] = []
    for box in boxes:
        if not isinstance(box, dict):
            raise VisionError(_UNREADABLE)
        detections.append(_detection(box))
    return ParsedFrame(label=label, detections=tuple(detections))


def _detection(box: dict[str, object]) -> Detection:
    kind = box.get("class")
    geometry = _pixel_box(box)
    if kind == "back":
        return Detection(BackFace(), geometry)
    if kind != "face":
        raise VisionError("cardvision infer returned an unknown class")
    outcome = box.get("outcome")
    if outcome == "nothing":
        return Detection(UnnamedFace(_guess(box), _similarity(box)), geometry)
    if outcome != "named":
        raise VisionError("cardvision infer returned an unknown outcome")
    name = box.get("name")
    card_id = box.get("cardId")
    if not isinstance(name, str) or not name or name != name.strip():
        raise VisionError("cardvision infer named a card without a name")
    if not isinstance(card_id, str) or not card_id or card_id != card_id.strip():
        raise VisionError("cardvision infer named a card without a card id")
    return Detection(
        NamedFace(CardName(name), PrintingId(card_id), _similarity(box)),
        geometry,
    )


def _pixel_box(box: dict[str, object]) -> PixelBox | None:
    if "box" not in box or box["box"] is None:
        return None
    raw = box["box"]
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise VisionError(_UNREADABLE)
    try:
        return PixelBox(raw[0], raw[1], raw[2], raw[3])
    except (TypeError, ValueError) as err:
        raise VisionError(_UNREADABLE) from err


def _guess(box: dict[str, object]) -> Guess | None:
    value = box.get("topName")
    if not isinstance(value, str) or not value or value != value.strip():
        return None
    return Guess(value)


def _similarity(box: dict[str, object]) -> Similarity | None:
    if "similarity" not in box or box["similarity"] is None:
        return None
    value = box["similarity"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return Similarity(value)
    except ValueError:
        return None


def _tally(parsed: ParsedFrame) -> Frame:
    named: list[NamedSight] = []
    unnamed = 0
    backs = 0
    for detection in parsed.detections:
        face = detection.face
        if isinstance(face, BackFace):
            backs += 1
            continue
        if isinstance(face, UnnamedFace):
            unnamed += 1
            continue
        named.append(NamedSight(face.name, face.printing))
    return Frame(label=parsed.label, named=tuple(named), unnamed=unnamed, backs=backs)
