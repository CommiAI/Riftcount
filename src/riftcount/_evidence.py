from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from enum import Enum
from fractions import Fraction
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from riftcount._cardvision import (
    FRAME_INTERVAL_S,
    BackFace,
    InferredJpeg,
    NamedFace,
    PixelBox,
    UnnamedFace,
)
from riftcount.errors import UsageError, VisionError
from riftcount.inventory import CaptureCount, CaptureKind, CardName, Frame, Inventory, InventoryLine

JPEG_QUALITY = 95
JPEG_SUBSAMPLING = 0
COUNTED_MP4 = "counted.mp4"

_BAR = (16, 18, 22)
_INK = (245, 245, 245)
_PAD = 8
_GAP = 4


class Caption(Enum):
    COPY = "copy"
    REPEAT = "repeat"
    UNNAMED = "unnamed"
    BACK = "back"


_COLOR: dict[Caption, tuple[int, int, int]] = {
    Caption.COPY: (40, 200, 110),
    Caption.REPEAT: (40, 200, 110),
    Caption.UNNAMED: (240, 170, 40),
    Caption.BACK: (90, 150, 240),
}

_DASH: dict[Caption, tuple[int, int] | None] = {
    Caption.COPY: None,
    Caption.REPEAT: (8, 6),
    Caption.UNNAMED: (8, 6),
    Caption.BACK: (2, 2),
}


def box_label(caption: Caption, face: NamedFace | UnnamedFace | BackFace, inventory: Inventory) -> str:
    if caption is Caption.UNNAMED:
        if not isinstance(face, UnnamedFace):
            raise TypeError("unnamed caption requires an unnamed face")
        text = "unnamed · adds 1"
        if face.guess is not None:
            text += f" · ? {face.guess.text}"
            if face.similarity is not None:
                text += f" · {face.similarity.value:.2f}"
        return text
    if caption is Caption.BACK:
        if not isinstance(face, BackFace):
            raise TypeError("back caption requires a back")
        return "back · adds 1"
    if not isinstance(face, NamedFace) or caption not in (Caption.COPY, Caption.REPEAT):
        raise TypeError("named caption requires a named face")
    line = _line(inventory, face.name)
    word = "copy" if caption is Caption.COPY else "repeat, not added"
    if face.similarity is not None:
        text = f"{face.name.text} · {face.printing.text} · {face.similarity.value:.2f} · {word}"
    else:
        text = f"{face.name.text} · {face.printing.text} · {word}"
    if face.printing != line.card_id:
        text += f" · inventory id {line.card_id.text}"
    return text


def footer_lines(
    order: int,
    release_id: str,
    frame_index: int,
    width: int,
    height: int,
    capture: CaptureCount,
    cited: Mapping[CardName, Sequence[int]],
) -> tuple[str, ...]:
    lines = [
        (
            f"{order:03d}-{capture.path.name} · {capture.kind.value} · "
            f"release {release_id} · frame {frame_index}/{capture.frames_read} · "
            f"{width}×{height} decoded pixels, EXIF ignored"
        ),
        "copies are the busiest frame, not a sum",
    ]
    if not capture.inventory.lines:
        lines.append("  (none)")
    else:
        for line in capture.inventory.lines:
            indexes = tuple(cited[line.name])
            if not indexes:
                raise ValueError(f"no cited frame for {line.name.text}")
            if len(indexes) == 1:
                where = f"frame {indexes[0]}"
            else:
                where = "frames " + ", ".join(str(index) for index in indexes)
            lines.append(f"  {line.name.text} ×{line.copies} ({where})")
    lines.append(
        f"unnamed sum {capture.unnamed_boxes} (each unnamed box adds 1) · "
        f"back sum {capture.back_boxes} (each back box adds 1)"
    )
    return tuple(lines)


def draw_box(image: Image.Image, box: PixelBox, caption: Caption, label: str) -> None:
    color = _COLOR[caption]
    stroke = _stroke_width(image.width)
    draw = ImageDraw.Draw(image)
    _draw_outline(draw, box, color, stroke, _DASH[caption])
    _draw_label(draw, image.size, box, color, label, stroke)


def paint_footer(image: Image.Image, lines: Sequence[str]) -> Image.Image:
    font = ImageFont.load_default()
    probe = ImageDraw.Draw(image)
    heights: list[int] = []
    tops: list[int] = []
    for line in lines:
        _left, top, _right, bottom = probe.textbbox((0, 0), line, font=font)
        heights.append(bottom - top)
        tops.append(top)
    extra = _PAD * 2 + sum(heights) + _GAP * max(0, len(lines) - 1)
    canvas = Image.new("RGB", (image.width, image.height + extra), _BAR)
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    y = image.height + _PAD
    for line, height, top in zip(lines, heights, tops):
        draw.text((_PAD, y - top), line, fill=_INK, font=font)
        y += height + _GAP
    return canvas


def capture_folder(dest: Path, order: int, source: Path) -> Path:
    return dest / f"{order:03d}-{source.name}"


def held_sample_index(time_s: float | Fraction, sample_count: int) -> int:
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
        raise ValueError("sample_count must be an integer >= 1")
    if isinstance(time_s, bool) or not isinstance(time_s, (int, float, Fraction)):
        raise ValueError("time_s must be a finite number >= 0")
    try:
        time = Fraction(time_s)
    except (OverflowError, ValueError) as err:
        raise ValueError("time_s must be a finite number >= 0") from err
    if time < 0:
        raise ValueError("time_s must be a finite number >= 0")
    interval = Fraction(FRAME_INTERVAL_S)
    if time >= sample_count * interval:
        return sample_count - 1
    return math.floor(time / interval)


def write_evidence(
    stage: Path,
    *,
    order: int,
    release_id: str,
    jpegs: Sequence[InferredJpeg],
    frames: Sequence[Frame],
    capture: CaptureCount,
) -> None:
    if tuple(jpeg.index for jpeg in jpegs) != tuple(range(1, len(jpegs) + 1)):
        raise VisionError(f"{capture.path.name} frames are not numbered from 1")
    if len(frames) != len(jpegs):
        raise VisionError(f"{capture.path.name} frame count does not match the infer JPEGs")
    cited = _cited(frames, capture.inventory)
    folder = capture_folder(stage, order, capture.path)
    folder.mkdir()
    if capture.kind is CaptureKind.VIDEO:
        _write_counted_video(folder, order, release_id, jpegs, capture, cited)
        return
    for jpeg in jpegs:
        image = _open_rgb(jpeg.path)
        painted = _paint_sample(image, jpeg, order, release_id, capture, cited)
        destination = folder / f"frame-{jpeg.index:04d}.jpg"
        painted.save(
            destination,
            format="JPEG",
            quality=JPEG_QUALITY,
            subsampling=JPEG_SUBSAMPLING,
        )


@contextmanager
def evidence_run(dest: Path | None, inputs: Sequence[Path]) -> Iterator[Path | None]:
    if dest is None:
        yield None
        return
    _check_dest(dest, inputs)
    partial = _stage(dest)
    try:
        yield partial
    except BaseException:
        _discard(partial)
        raise
    else:
        _install(dest, partial)


def _paint_sample(
    image: Image.Image,
    jpeg: InferredJpeg,
    order: int,
    release_id: str,
    capture: CaptureCount,
    cited: Mapping[CardName, Sequence[int]],
) -> Image.Image:
    for detection in jpeg.parsed.detections:
        box = detection.box
        if box is None:
            raise VisionError(f"{capture.path.name} frame {jpeg.index} has a detection with no box")
        caption = _caption(detection.face, jpeg.index, cited)
        label = box_label(caption, detection.face, capture.inventory)
        draw_box(image, box, caption, label)
    lines = footer_lines(order, release_id, jpeg.index, image.width, image.height, capture, cited)
    return paint_footer(image, lines)


def _write_counted_video(
    folder: Path,
    order: int,
    release_id: str,
    jpegs: Sequence[InferredJpeg],
    capture: CaptureCount,
    cited: Mapping[CardName, Sequence[int]],
) -> None:
    rate = _source_frame_rate(capture.path)
    reference = _open_rgb(jpegs[0].path)
    painted: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="riftcount-") as tmp:
        for index, source in enumerate(_extract_source_frames(capture.path, Path(tmp))):
            image = _open_rgb(source)
            if image.size != reference.size:
                raise VisionError(
                    f"{capture.path.name} frame {index + 1} is {image.width}×{image.height}, "
                    f"not {reference.width}×{reference.height}"
                )
            sample = jpegs[held_sample_index(Fraction(index, 1) / rate, len(jpegs))]
            canvas = _paint_sample(image, sample, order, release_id, capture, cited)
            path = folder / f"frame-{index + 1:08d}.jpg"
            canvas.save(
                path,
                format="JPEG",
                quality=JPEG_QUALITY,
                subsampling=JPEG_SUBSAMPLING,
            )
            painted.append(path)
    _encode_counted_video(folder, capture.path.name, rate, painted)


def _source_frame_rate(video: Path) -> Fraction:
    proc = _ffmpeg_tool(
        [
            "ffprobe",
            "-hide_banner",
            "-loglevel",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate",
            "-of",
            "csv=p=0",
            str(video),
        ],
        video.name,
    )
    if proc.returncode != 0:
        raise VisionError(f"ffmpeg failed on {video.name}")
    try:
        rate = Fraction(proc.stdout.strip())
    except (ValueError, ZeroDivisionError) as err:
        raise VisionError(f"ffmpeg failed on {video.name}") from err
    if rate <= 0:
        raise VisionError(f"ffmpeg failed on {video.name}")
    return rate


def _extract_source_frames(video: Path, dest: Path) -> tuple[Path, ...]:
    proc = _ffmpeg_tool(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-q:v",
            "2",
            str(dest / "frame-%08d.jpg"),
        ],
        video.name,
    )
    if proc.returncode != 0:
        raise VisionError(f"ffmpeg failed on {video.name}")
    frames = tuple(
        sorted(
            (path for path in dest.glob("frame-*.jpg") if path.is_file()),
            key=lambda path: int(path.stem.rsplit("-", 1)[1]),
        )
    )
    if not frames:
        raise VisionError(f"ffmpeg failed on {video.name}")
    return frames


def _ffmpeg_tool(argv: list[str], source_name: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(argv, check=False, capture_output=True, text=True)
    except OSError as err:
        raise VisionError(f"ffmpeg failed on {source_name}") from err


def _counted_video_argv(folder: Path, frame_rate: Fraction) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        f"{frame_rate.numerator}/{frame_rate.denominator}",
        "-start_number",
        "1",
        "-i",
        str(folder / "frame-%08d.jpg"),
        # yuv420p rejects an odd footer height.
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(folder / COUNTED_MP4),
    ]


def _encode_counted_video(
    folder: Path,
    source_name: str,
    frame_rate: Fraction,
    painted: Sequence[Path],
) -> None:
    proc = _ffmpeg_tool(_counted_video_argv(folder, frame_rate), source_name)
    video = folder / COUNTED_MP4
    if proc.returncode != 0 or not video.is_file():
        raise VisionError(f"ffmpeg failed on {source_name}")
    for path in painted:
        path.unlink()


def _line(inventory: Inventory, name: CardName) -> InventoryLine:
    for line in inventory.lines:
        if line.name == name:
            return line
    raise ValueError(f"no inventory line for {name.text}")


def _cited(frames: Sequence[Frame], inventory: Inventory) -> dict[CardName, tuple[int, ...]]:
    cited: dict[CardName, tuple[int, ...]] = {}
    for line in inventory.lines:
        indexes = tuple(
            index
            for index, frame in enumerate(frames, start=1)
            if sum(sight.name == line.name for sight in frame.named) == line.copies
        )
        if not indexes:
            raise ValueError(f"no frame has {line.copies} copies of {line.name.text}")
        cited[line.name] = indexes
    return cited


def _caption(
    face: NamedFace | UnnamedFace | BackFace,
    frame_index: int,
    cited: Mapping[CardName, Sequence[int]],
) -> Caption:
    if isinstance(face, BackFace):
        return Caption.BACK
    if isinstance(face, UnnamedFace):
        return Caption.UNNAMED
    if not isinstance(face, NamedFace):
        raise TypeError(f"unknown face {type(face).__name__}")
    indexes = cited.get(face.name)
    if indexes is None:
        raise ValueError(f"no copy source for {face.name.text}")
    if frame_index in indexes:
        return Caption.COPY
    return Caption.REPEAT


def _stroke_width(image_width: int) -> int:
    return max(2, round(image_width / 640))


def _draw_outline(
    draw: ImageDraw.ImageDraw,
    box: PixelBox,
    color: tuple[int, int, int],
    stroke: int,
    dash: tuple[int, int] | None,
) -> None:
    x0, y0 = box.x, box.y
    x1, y1 = box.x + box.width, box.y + box.height
    if dash is None:
        draw.rectangle((x0, y0, x1, y1), outline=color, width=stroke)
        return
    on, off = dash
    edges = ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0))
    for start_x, start_y, end_x, end_y in edges:
        length = math.hypot(end_x - start_x, end_y - start_y)
        if length <= 0:
            continue
        pos = 0.0
        while pos < length:
            run = min(float(on), length - pos)
            draw.line(
                (
                    start_x + (end_x - start_x) * (pos / length),
                    start_y + (end_y - start_y) * (pos / length),
                    start_x + (end_x - start_x) * ((pos + run) / length),
                    start_y + (end_y - start_y) * ((pos + run) / length),
                ),
                fill=color,
                width=stroke,
            )
            pos += on + off


def _draw_label(
    draw: ImageDraw.ImageDraw,
    size: tuple[int, int],
    box: PixelBox,
    color: tuple[int, int, int],
    label: str,
    stroke: int,
) -> None:
    font = ImageFont.load_default()
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    text_width = right - left
    text_height = bottom - top
    chip_width = text_width + 6
    chip_height = text_height + 6
    x = min(max(0, box.x), max(0, size[0] - chip_width))
    y = box.y - chip_height - 2
    if y < 0:
        y = box.y + stroke
    draw.rectangle((x, y, x + chip_width, y + chip_height), fill=color)
    draw.text((x + 3, y + 3 - top), label, fill=(0, 0, 0), font=font)


def _open_rgb(path: Path) -> Image.Image:
    # Boxes are measured on stored JPEG pixels. EXIF orientation would move them off the cards.
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def _check_dest(dest: Path, inputs: Sequence[Path]) -> None:
    if dest.is_symlink():
        raise UsageError(f"evidence path is a symlink: {dest}")
    if dest.exists() and not dest.is_dir():
        raise UsageError(f"evidence path is a file: {dest}")
    if not dest.parent.is_dir():
        raise UsageError(f"evidence directory not found: {dest.parent}")
    dest_resolved = dest.resolve()
    for raw in inputs:
        if not raw.exists():
            continue
        source = raw.resolve()
        if dest_resolved == source or _contains(dest_resolved, source):
            raise UsageError(f"evidence path would replace a source: {dest}")


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return child != parent


def _present(path: Path) -> bool:
    return path.is_symlink() or path.exists()


def _discard(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.exists():
        shutil.rmtree(path)


def _partial(dest: Path) -> Path:
    return dest.with_name(dest.name + ".partial")


def _previous(dest: Path) -> Path:
    return dest.with_name(dest.name + ".previous")


def _stage(dest: Path) -> Path:
    partial = _partial(dest)
    previous = _previous(dest)
    if not _present(dest) and _present(previous):
        previous.rename(dest)
    _discard(partial)
    partial.mkdir()
    return partial


def _install(dest: Path, partial: Path) -> None:
    previous = _previous(dest)
    moved = False
    try:
        if _present(dest):
            if _present(previous):
                _discard(previous)
            dest.rename(previous)
            moved = True
        partial.rename(dest)
    except Exception:
        if moved and not _present(dest) and _present(previous):
            previous.rename(dest)
        _discard(partial)
        raise
    if _present(previous):
        _discard(previous)
