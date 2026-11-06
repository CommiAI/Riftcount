import json
import shutil
import subprocess
from collections.abc import Callable
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image, ImageOps

from riftcount._cardvision import (
    BackFace,
    Detection,
    Guess,
    InferredJpeg,
    NamedFace,
    ParsedFrame,
    PixelBox,
    Similarity,
    UnnamedFace,
    frames_from_stdout,
    open_capture,
    sample_filter,
)
from riftcount._evidence import (
    COUNTED_MP4,
    JPEG_QUALITY,
    JPEG_SUBSAMPLING,
    Caption,
    box_label,
    draw_box,
    evidence_run,
    footer_lines,
    held_sample_index,
    paint_footer,
    write_evidence,
)
from riftcount.count import count_paths
from riftcount.errors import UsageError, VisionError
from riftcount.inventory import (
    SCHEMA,
    CaptureCount,
    CaptureKind,
    CardName,
    Frame,
    Inventory,
    InventoryLine,
    NamedSight,
    PrintingId,
    dumps_report,
    inventory_from_frames,
)

GREEN = (40, 200, 110)
AMBER = (240, 170, 40)
BLUE = (90, 150, 240)
COPY_SAME = "Fury Rune · RAD-R01A · 0.86 · copy"
COPY_OTHER = "Fury Rune · SFD-R01A · copy · inventory id RAD-R01A"
UNNAMED_GUESS = "unnamed · adds 1 · ? Riptide Rex · 0.42"
UNNAMED_PLAIN = "unnamed · adds 1"
BACK_LABEL = "back · adds 1"


def _inventory(*rows: tuple[str, str, int]) -> Inventory:
    lines = tuple(
        InventoryLine(CardName(name), PrintingId(card_id), copies) for name, card_id, copies in rows
    )
    return Inventory(lines)


def _capture(
    name: str,
    kind: CaptureKind,
    frames_read: int,
    unnamed: int,
    backs: int,
    inventory: Inventory,
) -> CaptureCount:
    return CaptureCount(Path(name), kind, frames_read, unnamed, backs, inventory)


def test_box_label_uses_the_caption_words() -> None:
    inventory = _inventory(("Fury Rune", "RAD-R01A", 2))
    same = NamedFace(CardName("Fury Rune"), PrintingId("RAD-R01A"), Similarity(0.861))
    other = NamedFace(CardName("Fury Rune"), PrintingId("SFD-R01A"), Similarity(0.86))
    plain = NamedFace(CardName("Fury Rune"), PrintingId("RAD-R01A"))
    guessed = UnnamedFace(Guess("Riptide Rex"), Similarity(0.424))
    guessed_only = UnnamedFace(Guess("Riptide Rex"))
    similarity_only = UnnamedFace(None, Similarity(0.9))
    assert box_label(Caption.COPY, same, inventory) == "Fury Rune · RAD-R01A · 0.86 · copy"
    assert box_label(Caption.COPY, other, inventory) == (
        "Fury Rune · SFD-R01A · 0.86 · copy · inventory id RAD-R01A"
    )
    assert box_label(Caption.REPEAT, plain, inventory) == "Fury Rune · RAD-R01A · repeat, not added"
    assert box_label(Caption.REPEAT, other, inventory) == (
        "Fury Rune · SFD-R01A · 0.86 · repeat, not added · inventory id RAD-R01A"
    )
    assert box_label(Caption.UNNAMED, guessed, inventory) == UNNAMED_GUESS
    assert box_label(Caption.UNNAMED, guessed_only, inventory) == "unnamed · adds 1 · ? Riptide Rex"
    assert box_label(Caption.UNNAMED, similarity_only, inventory) == UNNAMED_PLAIN
    assert box_label(Caption.BACK, BackFace(), inventory) == BACK_LABEL


def test_held_sample_index_covers_each_half_second_and_keeps_the_last() -> None:
    assert held_sample_index(0, 3) == 0
    assert held_sample_index(0.49, 3) == 0
    assert held_sample_index(0.5, 3) == 1
    assert held_sample_index(1.0, 2) == 1
    assert held_sample_index(3.0, 2) == 1
    with pytest.raises(ValueError):
        held_sample_index(-0.01, 2)
    with pytest.raises(ValueError):
        held_sample_index(float("nan"), 2)
    with pytest.raises(ValueError):
        held_sample_index(float("inf"), 2)
    with pytest.raises(ValueError):
        held_sample_index(0, 0)


def test_footer_lines_name_one_frame_a_tie_and_the_sums() -> None:
    inventory = _inventory(("Cloud Drake", "VEN-048", 2), ("Otterpus", "OGN-001", 1))
    capture = _capture("clip.mp4", CaptureKind.VIDEO, 5, 4, 0, inventory)
    cited = {
        CardName("Cloud Drake"): (2, 5),
        CardName("Otterpus"): (2,),
    }
    assert footer_lines(1, "rel-1", 3, 80, 60, capture, cited) == (
        "001-clip.mp4 · video · release rel-1 · frame 3/5 · 80×60 decoded pixels, EXIF ignored",
        "copies are the busiest frame, not a sum",
        "  Cloud Drake ×2 (frames 2, 5)",
        "  Otterpus ×1 (frame 2)",
        "unnamed sum 4 (each unnamed box adds 1) · back sum 0 (each back box adds 1)",
    )
    empty = _capture("rune.jpg", CaptureKind.STILL, 1, 2, 1, Inventory(()))
    assert footer_lines(1, "rel-1", 1, 40, 20, empty, {}) == (
        "001-rune.jpg · still · release rel-1 · frame 1/1 · 40×20 decoded pixels, EXIF ignored",
        "copies are the busiest frame, not a sum",
        "  (none)",
        "unnamed sum 2 (each unnamed box adds 1) · back sum 1 (each back box adds 1)",
    )


def _weights(directory: Path) -> Path:
    release = directory / "weights"
    release.mkdir()
    (release / "release.json").write_text('{"id": "rel-1"}\n', encoding="utf-8")
    return release


def _make_mp4(path: Path, size: tuple[int, int], colors: tuple[tuple[int, int, int], ...], fps: int) -> None:
    folder = path.parent / f"{path.stem}-build"
    folder.mkdir()
    for index, color in enumerate(colors, start=1):
        Image.new("RGB", size, color).save(folder / f"frame-{index:04d}.png")
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            f"{fps}/1",
            "-i",
            str(folder / "frame-%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def _probe_video(path: Path, run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> dict[str, object]:
    proc = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_type,codec_name,avg_frame_rate,nb_read_frames",
            "-show_entries",
            "format=nb_streams,duration",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert isinstance(payload, dict)
    return payload


def _decode_frames(
    video: Path,
    dest: Path,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[Image.Image]:
    dest.mkdir()
    proc = run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            str(dest / "frame-%04d.png"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    images: list[Image.Image] = []
    for path in sorted(dest.glob("frame-*.png")):
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    return images


def _rgb(image: Image.Image, x: int, y: int) -> tuple[int, int, int]:
    pixel = image.getpixel((x, y))
    assert isinstance(pixel, tuple)
    return int(pixel[0]), int(pixel[1]), int(pixel[2])


def _far(left: tuple[int, int, int], right: tuple[int, int, int]) -> bool:
    return max(abs(left[channel] - right[channel]) for channel in range(3)) > 40


def _closer(
    pixel: tuple[int, int, int],
    expected: tuple[int, int, int],
    other: tuple[int, int, int],
) -> bool:
    def distance(origin: tuple[int, int, int]) -> int:
        return sum((pixel[channel] - origin[channel]) ** 2 for channel in range(3))

    return distance(expected) < distance(other)


def _bottom_green(image: Image.Image, box: PixelBox) -> int:
    y1 = int(box.y + box.height)
    x0 = int(box.x)
    x1 = int(box.x + box.width)
    hits = 0
    for x in range(max(0, x0), min(image.width, x1 + 1)):
        for y in range(y1 - 3, y1 + 2):
            if y < 0 or y >= image.height:
                continue
            red, green, blue = _rgb(image, x, y)
            if green > 100 and green > red + 20 and green > blue:
                hits += 1
    return hits


def _shows_box(image: Image.Image, box: PixelBox) -> bool:
    return _bottom_green(image, box) >= 8


def _jpeg(path: Path, size: tuple[int, int], color: tuple[int, int, int], exif: int | None = None) -> None:
    image = Image.new("RGB", size, color)
    kwargs: dict[str, object] = {"format": "JPEG", "quality": JPEG_QUALITY, "subsampling": JPEG_SUBSAMPLING}
    if exif is not None:
        payload = image.getexif()
        payload[274] = exif
        kwargs["exif"] = payload
    image.save(path, **kwargs)


def _patch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stdout: str | None = None,
    by_name: dict[str, str] | None = None,
    ffmpeg: Callable[[list[str]], int | None | subprocess.CompletedProcess[str]] | None = None,
) -> None:
    def fake_which(name: str) -> str | None:
        if name == "cardvision":
            return "/usr/bin/cardvision"
        if name == "ffmpeg" and ffmpeg is not None:
            return "/usr/bin/ffmpeg"
        return None

    def fake_run(argv: list[str], **kwargs: object) -> object:
        del kwargs

        class Proc:
            def __init__(self, text: str) -> None:
                self.returncode = 0
                self.stdout = text
                self.stderr = ""

        if argv[0] in {"ffmpeg", "ffprobe"}:
            assert ffmpeg is not None
            outcome = ffmpeg(argv)
            if isinstance(outcome, subprocess.CompletedProcess):
                return outcome
            proc = Proc("")
            proc.returncode = 0 if outcome is None else outcome
            if argv[0] == "ffprobe" and proc.returncode == 0:
                proc.stdout = "30/1\n"
            return proc
        if by_name is not None:
            release_at = argv.index("--release")
            text = "".join(by_name[Path(arg).name] for arg in argv[release_at + 2 :])
        else:
            assert stdout is not None
            text = stdout
        return Proc(text)

    monkeypatch.setattr("riftcount._cardvision.shutil.which", fake_which)
    monkeypatch.setattr("riftcount._cardvision.subprocess.run", fake_run)


def _line(document: dict[str, object]) -> str:
    return json.dumps(document) + "\n"


def _decoded(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def _paint(path: Path, marks: list[tuple[PixelBox, Caption, str]], lines: tuple[str, ...]) -> bytes:
    image = _decoded(path)
    for box, caption, label in marks:
        draw_box(image, box, caption, label)
    painted = paint_footer(image, lines)
    buffer = BytesIO()
    painted.save(buffer, format="JPEG", quality=JPEG_QUALITY, subsampling=JPEG_SUBSAMPLING)
    return buffer.getvalue()


def _roundtrip(color: tuple[int, int, int]) -> tuple[int, int, int]:
    image = Image.new("RGB", (16, 16), color)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY, subsampling=JPEG_SUBSAMPLING)
    buffer.seek(0)
    with Image.open(buffer) as saved:
        pixel = saved.getpixel((8, 8))
    assert isinstance(pixel, tuple)
    return pixel[0], pixel[1], pixel[2]


def _edge_has(path: Path, box: PixelBox, color: tuple[int, int, int]) -> bool:
    with Image.open(path) as image:
        stroke = max(2, round(image.width / 640))
        x0 = int(box.x)
        y0 = int(box.y)
        x1 = int(box.x + box.width)
        y1 = int(box.y + box.height)
        for x in range(x0, min(image.width, x1 + 1)):
            for step in range(stroke):
                for y in (y0 + step, y1 - step):
                    if 0 <= y < image.height and image.getpixel((x, y)) == color:
                        return True
        for y in range(y0, min(image.height, y1 + 1)):
            for step in range(stroke):
                for x in (x0 + step, x1 - step):
                    if 0 <= x < image.width and image.getpixel((x, y)) == color:
                        return True
    return False


def _partial(dest: Path) -> Path:
    return dest.with_name(dest.name + ".partial")


def _previous(dest: Path) -> Path:
    return dest.with_name(dest.name + ".previous")


def _files(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_no_evidence_writes_nothing_and_keeps_the_json_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    photo = tmp_path / "rune.jpg"
    _jpeg(photo, (32, 24), (0, 0, 0))
    release = _weights(tmp_path)
    document = {
        "frame": "rune.jpg",
        "boxes": [
            {
                "class": "face",
                "outcome": "named",
                "name": "Fury Rune",
                "cardId": "RAD-R01A",
                "topName": "Fury Rune",
            }
        ],
    }
    _patch(monkeypatch, stdout=_line(document))
    before = {path.name for path in tmp_path.iterdir()}
    report = count_paths([photo], release=release)
    assert {path.name for path in tmp_path.iterdir()} == before
    text = dumps_report(report)
    payload = json.loads(text)
    assert list(payload) == ["schema", "release_id", "captures"]
    assert payload["schema"] == SCHEMA
    capture = payload["captures"][0]
    assert list(capture) == [
        "path",
        "kind",
        "frames_read",
        "unnamed_boxes",
        "back_boxes",
        "inventory",
    ]
    assert list(capture["inventory"]) == ["lines", "estimated_cards"]
    assert list(capture["inventory"]["lines"][0]) == ["name", "card_id", "copies"]
    assert "topName" not in text
    assert "frame-0001" not in text


def test_evidence_jpeg_shows_each_caption_and_is_taller_than_the_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    width, height = 280, 320
    photo = tmp_path / "rune.jpg"
    _jpeg(photo, (width, height), (0, 0, 0))
    release = _weights(tmp_path)
    same = PixelBox(16, 24, 48, 36)
    other = PixelBox(16, 84, 48, 36)
    guessed = PixelBox(16, 144, 48, 36)
    plain = PixelBox(16, 204, 48, 36)
    back = PixelBox(16, 264, 48, 36)
    document = {
        "frame": "rune.jpg",
        "boxes": [
            {
                "class": "face",
                "outcome": "named",
                "name": "Fury Rune",
                "cardId": "RAD-R01A",
                "similarity": 0.86,
                "box": [16, 24, 48, 36],
            },
            {
                "class": "face",
                "outcome": "named",
                "name": "Fury Rune",
                "cardId": "SFD-R01A",
                "box": [16, 84, 48, 36],
            },
            {
                "class": "face",
                "outcome": "nothing",
                "topName": "Riptide Rex",
                "similarity": 0.424,
                "box": [16, 144, 48, 36],
            },
            {
                "class": "face",
                "outcome": "nothing",
                "topName": "  Not A Name  ",
                "similarity": 0.9,
                "box": [16, 204, 48, 36],
            },
            {"class": "back", "box": [16, 264, 48, 36]},
        ],
    }
    _patch(monkeypatch, stdout=_line(document))
    dest = tmp_path / "evidence"
    report = count_paths([photo], release=release, evidence=dest)
    text = dumps_report(report)
    assert "Riptide Rex" not in text
    assert "topName" not in text
    assert "Not A Name" not in text
    frame = frames_from_stdout(_line(document))[0]
    assert frame.unnamed == 2
    assert frame.backs == 1
    assert [sight.name.text for sight in frame.named] == ["Fury Rune", "Fury Rune"]
    assert "Riptide Rex" not in str(frame)
    capture = report.captures[0]
    assert capture.unnamed_boxes == 2
    assert capture.back_boxes == 1
    assert capture.inventory.lines[0].copies == 2
    assert capture.inventory.lines[0].card_id.text == "RAD-R01A"
    saved_path = dest / "001-rune.jpg" / "frame-0001.jpg"
    assert saved_path.is_file()
    assert not _partial(dest).exists()
    assert not _previous(dest).exists()
    lines = (
        f"001-rune.jpg · still · release rel-1 · frame 1/1 · {width}×{height} decoded pixels, EXIF ignored",
        "copies are the busiest frame, not a sum",
        "  Fury Rune ×2 (frame 1)",
        "unnamed sum 2 (each unnamed box adds 1) · back sum 1 (each back box adds 1)",
    )
    cited = {capture.inventory.lines[0].name: (1,)}
    assert footer_lines(1, report.release_id, 1, width, height, capture, cited) == lines
    marks = [
        (same, Caption.COPY, COPY_SAME),
        (other, Caption.COPY, COPY_OTHER),
        (guessed, Caption.UNNAMED, UNNAMED_GUESS),
        (plain, Caption.UNNAMED, UNNAMED_PLAIN),
        (back, Caption.BACK, BACK_LABEL),
    ]
    assert box_label(Caption.COPY, NamedFace(CardName("Fury Rune"), PrintingId("RAD-R01A"), Similarity(0.86)), capture.inventory) == COPY_SAME
    assert saved_path.read_bytes() == _paint(photo, marks, lines)
    with Image.open(saved_path) as saved, Image.open(photo) as source:
        assert saved.width == source.width == width
        assert saved.height > source.height
        assert saved.getpixel((0, 0)) == source.getpixel((0, 0))
    assert _roundtrip(GREEN) == GREEN
    assert _roundtrip(AMBER) == AMBER
    assert _edge_has(saved_path, same, GREEN)
    assert _edge_has(saved_path, guessed, AMBER)
    assert _edge_has(saved_path, back, _roundtrip(BLUE))


def test_two_captures_are_numbered_in_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = tmp_path / "a.jpg"
    second = tmp_path / "b.jpg"
    _jpeg(first, (40, 32), (0, 0, 0))
    _jpeg(second, (40, 32), (255, 255, 255))
    release = _weights(tmp_path)

    def doc(name: str, card: str) -> str:
        return _line(
            {
                "frame": name,
                "boxes": [
                    {"class": "face", "outcome": "named", "name": card, "cardId": "OGN-001", "box": [4, 8, 16, 16]}
                ],
            }
        )

    _patch(monkeypatch, by_name={"a.jpg": doc("a.jpg", "Otterpus"), "b.jpg": doc("b.jpg", "Cloud Drake")})
    dest = tmp_path / "out"
    report = count_paths([second, first], release=release, evidence=dest)
    assert [capture.path.name for capture in report.captures] == ["b.jpg", "a.jpg"]
    assert (dest / "001-b.jpg" / "frame-0001.jpg").is_file()
    assert (dest / "002-a.jpg" / "frame-0001.jpg").is_file()
    assert sorted(path.name for path in dest.iterdir()) == ["001-b.jpg", "002-a.jpg"]


def test_tie_frames_are_both_copy_and_match_the_extracted_jpegs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    size = (96, 80)
    colors = ((12, 18, 40), (230, 230, 230), (170, 40, 40), (40, 40, 170), (24, 24, 24))
    video = tmp_path / "clip.mp4"
    _make_mp4(video, size, colors, 2)
    release = _weights(tmp_path)
    sources: list[Path] = []
    for index, color in enumerate(colors, start=1):
        path = tmp_path / f"src-{index}.jpg"
        _jpeg(path, size, color)
        sources.append(path)
    one = PixelBox(16, 36, 20, 20)
    two = PixelBox(56, 36, 20, 20)
    real_run = subprocess.run

    def ffmpeg(argv: list[str]) -> int | subprocess.CompletedProcess[str]:
        if sample_filter() in argv:
            dest = Path(argv[-1]).parent
            for index, src in enumerate(sources, start=1):
                (dest / f"frame-{index:04d}.jpg").write_bytes(src.read_bytes())
            return 0
        return real_run(argv, check=False, capture_output=True, text=True)

    documents: list[str] = []
    for index in range(1, 6):
        boxes: list[dict[str, object]] = [
            {
                "class": "face",
                "outcome": "named",
                "name": "Cloud Drake",
                "cardId": "VEN-048",
                "box": [one.x, one.y, one.width, one.height],
            }
        ]
        if index in {2, 5}:
            boxes.append(
                {
                    "class": "face",
                    "outcome": "named",
                    "name": "Cloud Drake",
                    "cardId": "VEN-048",
                    "box": [two.x, two.y, two.width, two.height],
                }
            )
        documents.append(_line({"frame": f"frame-{index:04d}.jpg", "boxes": boxes}))
    _patch(monkeypatch, stdout="".join(documents), ffmpeg=ffmpeg)
    dest = tmp_path / "evidence"
    report = count_paths([video], release=release, evidence=dest)
    folder = dest / "001-clip.mp4"
    assert report.captures[0].frames_read == 5
    assert report.captures[0].inventory.lines[0].copies == 2
    frames = [
        Frame(
            f"frame-{index:04d}.jpg",
            (NamedSight(CardName("Cloud Drake"), PrintingId("VEN-048")),)
            * (2 if index in {2, 5} else 1),
            0,
            0,
        )
        for index in range(1, 6)
    ]
    assert inventory_from_frames(frames).lines[0].copies == 2
    published = folder / COUNTED_MP4
    assert sorted(path.name for path in folder.iterdir()) == [COUNTED_MP4]
    payload = _probe_video(published, real_run)
    assert payload["format"]["nb_streams"] == 1
    streams = payload["streams"]
    assert isinstance(streams, list)
    assert [stream["codec_type"] for stream in streams] == ["video"]
    assert streams[0]["avg_frame_rate"] == "2/1"
    assert streams[0]["nb_read_frames"] == "5"
    assert float(payload["format"]["duration"]) == pytest.approx(2.5, abs=0.05)
    decoded = _decode_frames(published, tmp_path / "tie-decoded", real_run)
    assert len(decoded) == 5
    for index, image in enumerate(decoded):
        assert _shows_box(image, one)
        assert _shows_box(image, two) == (index in {1, 4})
    cited = {CardName("Cloud Drake"): (2, 5)}
    capture = report.captures[0]
    assert footer_lines(1, "rel-1", 2, 96, 80, capture, cited)[2] == "  Cloud Drake ×2 (frames 2, 5)"


def test_shorter_rerun_leaves_only_the_new_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    size = (64, 64)
    camera = ((10, 12, 30), (200, 200, 200))
    video = tmp_path / "clip.mp4"
    _make_mp4(video, size, camera, 2)
    release = _weights(tmp_path)
    holder: dict[str, list[Path]] = {}
    real_run = subprocess.run
    box_a = [8, 32, 16, 16]
    box_b = [40, 32, 16, 16]

    def ffmpeg(argv: list[str]) -> int | subprocess.CompletedProcess[str]:
        if sample_filter() in argv:
            dest = Path(argv[-1]).parent
            for index, src in enumerate(holder["sources"], start=1):
                (dest / f"frame-{index:04d}.jpg").write_bytes(src.read_bytes())
            return 0
        return real_run(argv, check=False, capture_output=True, text=True)

    def use(colors: tuple[tuple[int, int, int], ...], boxes: tuple[list[int], ...]) -> str:
        sources: list[Path] = []
        lines: list[str] = []
        for index, color in enumerate(colors, start=1):
            path = tmp_path / f"hold-{len(colors)}-{index}.jpg"
            _jpeg(path, size, color)
            sources.append(path)
            lines.append(
                _line(
                    {
                        "frame": f"frame-{index:04d}.jpg",
                        "boxes": [
                            {
                                "class": "face",
                                "outcome": "named",
                                "name": "Otterpus",
                                "cardId": "OGN-010",
                                "box": boxes[index - 1],
                            }
                        ],
                    }
                )
            )
        holder["sources"] = sources
        return "".join(lines)

    _patch(monkeypatch, stdout=use(((0, 0, 0), (255, 255, 255)), (box_a, box_b)), ffmpeg=ffmpeg)
    dest = tmp_path / "evidence"
    count_paths([video], release=release, evidence=dest)
    folder = dest / "001-clip.mp4"
    published = folder / COUNTED_MP4
    assert sorted(path.name for path in folder.iterdir()) == [COUNTED_MP4]
    first = _decode_frames(published, tmp_path / "rerun-first", real_run)
    assert _shows_box(first[1], PixelBox(*box_b))
    assert not _shows_box(first[1], PixelBox(*box_a))
    (dest / "stale.txt").write_text("stale", encoding="utf-8")
    _patch(monkeypatch, stdout=use(((240, 170, 40),), (box_a,)), ffmpeg=ffmpeg)
    count_paths([video], release=release, evidence=dest)
    assert sorted(path.name for path in folder.iterdir()) == [COUNTED_MP4]
    assert not (folder / "frame-0002.jpg").exists()
    assert not (dest / "stale.txt").exists()
    assert not _partial(dest).exists()
    assert not _previous(dest).exists()
    payload = _probe_video(published, real_run)
    assert payload["streams"][0]["avg_frame_rate"] == "2/1"
    assert payload["streams"][0]["nb_read_frames"] == "2"
    assert float(payload["format"]["duration"]) == pytest.approx(1.0, abs=0.05)
    second = _decode_frames(published, tmp_path / "rerun-second", real_run)
    assert len(second) == 2
    assert _far(_rgb(second[0], 2, 2), _rgb(second[1], 2, 2))
    assert _closer(_rgb(second[0], 2, 2), camera[0], (240, 170, 40))
    assert _closer(_rgb(second[1], 2, 2), camera[1], (240, 170, 40))
    for image in second:
        assert _shows_box(image, PixelBox(*box_a))
        assert not _shows_box(image, PixelBox(*box_b))


def test_failure_does_not_install_partial_over_a_missing_or_previous_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    photo = tmp_path / "rune.jpg"
    _jpeg(photo, (32, 24), (0, 0, 0))
    release = _weights(tmp_path)
    document = _line(
        {
            "frame": "rune.jpg",
            "boxes": [
                {"class": "face", "outcome": "named", "name": "Otterpus", "cardId": "OGN-010", "box": [4, 4, 12, 12]}
            ],
        }
    )
    dest = tmp_path / "evidence"

    @contextmanager
    def boom(path: Path, kind: object, release_dir: Path):
        del path, kind, release_dir
        assert _partial(dest).is_dir()
        raise VisionError("cardvision infer failed for rune.jpg")
        yield

    monkeypatch.setattr("riftcount.count.open_capture", boom)
    with pytest.raises(VisionError):
        count_paths([photo], release=release, evidence=dest)
    assert not dest.exists()
    assert not _partial(dest).exists()

    monkeypatch.setattr("riftcount.count.open_capture", open_capture)
    _patch(monkeypatch, stdout=document)
    count_paths([photo], release=release, evidence=dest)
    before = _files(dest)
    monkeypatch.setattr("riftcount.count.open_capture", boom)
    with pytest.raises(VisionError):
        count_paths([photo], release=release, evidence=dest)
    assert _files(dest) == before
    assert not _partial(dest).exists()

    (dest / "kept.txt").write_text("kept", encoding="utf-8")
    saved = _files(dest)
    dest.rename(_previous(dest))
    monkeypatch.setattr("riftcount.count.open_capture", boom)
    with pytest.raises(VisionError):
        count_paths([photo], release=release, evidence=dest)
    assert _files(dest) == saved
    assert not _previous(dest).exists()
    assert not _partial(dest).exists()

    dest.rename(_previous(dest))
    monkeypatch.setattr("riftcount.count.open_capture", open_capture)
    _patch(monkeypatch, stdout=document)
    count_paths([photo], release=release, evidence=dest)
    assert (dest / "001-rune.jpg" / "frame-0001.jpg").is_file()
    assert not (dest / "kept.txt").exists()
    assert not _previous(dest).exists()
    assert not _partial(dest).exists()


def test_missing_box_still_counts_until_evidence_is_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    photo = tmp_path / "rune.jpg"
    _jpeg(photo, (32, 24), (0, 0, 0))
    release = _weights(tmp_path)
    document = _line(
        {
            "frame": "rune.jpg",
            "boxes": [{"class": "face", "outcome": "named", "name": "Otterpus", "cardId": "OGN-010"}],
        }
    )
    _patch(monkeypatch, stdout=document)
    report = count_paths([photo], release=release)
    assert report.captures[0].inventory.lines[0].name.text == "Otterpus"
    dest = tmp_path / "evidence"
    with pytest.raises(VisionError):
        count_paths([photo], release=release, evidence=dest)
    assert not dest.exists()
    assert not _partial(dest).exists()


def test_exif_orientation_keeps_the_stored_width_and_height(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    photo = tmp_path / "rune.jpg"
    _jpeg(photo, (40, 20), (0, 0, 0), exif=6)
    with Image.open(photo) as stored:
        assert stored.size == (40, 20)
        assert stored.getpixel((0, 0)) == (0, 0, 0)
        assert ImageOps.exif_transpose(stored).size == (20, 40)
    release = _weights(tmp_path)
    _patch(
        monkeypatch,
        stdout=_line(
            {
                "frame": "rune.jpg",
                "boxes": [
                    {"class": "face", "outcome": "named", "name": "Otterpus", "cardId": "OGN-010", "box": [16, 8, 12, 8]}
                ],
            }
        ),
    )
    dest = tmp_path / "evidence"
    count_paths([photo], release=release, evidence=dest)
    saved_path = dest / "001-rune.jpg" / "frame-0001.jpg"
    with Image.open(saved_path) as saved:
        assert saved.width == 40
        assert saved.height > 20
        assert saved.getpixel((0, 0)) == (0, 0, 0)
    lines = footer_lines(
        1,
        "rel-1",
        1,
        40,
        20,
        _capture("rune.jpg", CaptureKind.STILL, 1, 0, 0, _inventory(("Otterpus", "OGN-010", 1))),
        {CardName("Otterpus"): (1,)},
    )
    assert "40×20 decoded pixels, EXIF ignored" in lines[0]
    assert saved_path.read_bytes() == _paint(photo, [(PixelBox(16, 8, 12, 8), Caption.COPY, "Otterpus · OGN-010 · copy")], lines)


def test_evidence_dest_refuses_a_source_a_file_a_symlink_and_a_missing_parent(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    photos = work / "photos"
    photos.mkdir()
    photo = photos / "a.jpg"
    photo.write_bytes(b"jpeg")
    release = work / "weights"

    def refuse(evidence: Path) -> None:
        with pytest.raises(UsageError):
            count_paths([photo], release=release, evidence=evidence)
        assert photo.read_bytes() == b"jpeg"
        assert not _partial(evidence).exists()

    refuse(photo)
    refuse(photo.resolve())
    refuse(photos)
    with pytest.raises(UsageError):
        count_paths([photos], release=release, evidence=photos)
    assert photo.read_bytes() == b"jpeg"
    with pytest.raises(UsageError):
        count_paths([photos], release=release, evidence=work)
    assert photo.read_bytes() == b"jpeg"

    existing = work / "notes.txt"
    existing.write_text("keep", encoding="utf-8")
    with pytest.raises(UsageError):
        count_paths([photo], release=release, evidence=existing)
    assert existing.read_text(encoding="utf-8") == "keep"

    target = work / "target"
    target.mkdir()
    link = work / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(UsageError):
        count_paths([photo], release=release, evidence=link)
    assert link.is_symlink()

    missing = work / "missing" / "out"
    with pytest.raises(UsageError):
        count_paths([photo], release=release, evidence=missing)
    assert not missing.parent.exists()


def test_evidence_may_be_a_subdirectory_of_a_directory_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    photos = tmp_path / "photos"
    photos.mkdir()
    photo = photos / "a.jpg"
    _jpeg(photo, (40, 32), (0, 0, 0))
    release = _weights(tmp_path)
    _patch(
        monkeypatch,
        stdout=_line(
            {
                "frame": "a.jpg",
                "boxes": [
                    {"class": "face", "outcome": "named", "name": "Otterpus", "cardId": "OGN-010", "box": [4, 8, 16, 16]}
                ],
            }
        ),
    )
    dest = photos / "out"
    count_paths([photos], release=release, evidence=dest)
    assert (dest / "001-a.jpg" / "frame-0001.jpg").is_file()
    assert photo.is_file()
    assert not (photos / "frame-0001.jpg").exists()


def test_encode_failure_leaves_dir_absent_or_on_the_previous_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"mp4")
    release = _weights(tmp_path)
    source = tmp_path / "src.jpg"
    _jpeg(source, (64, 48), (0, 0, 0))
    document = _line(
        {
            "frame": "frame-0001.jpg",
            "boxes": [
                {
                    "class": "face",
                    "outcome": "named",
                    "name": "Otterpus",
                    "cardId": "OGN-010",
                    "box": [8, 16, 20, 20],
                }
            ],
        }
    )
    fail = {"encode": True}

    def ffmpeg(argv: list[str]) -> int:
        if argv[0] == "ffprobe":
            return 0
        output = Path(argv[-1])
        if output.name == COUNTED_MP4:
            if fail["encode"]:
                return 1
            output.write_bytes(b"counted")
            return 0
        (output.parent / "frame-0001.jpg").write_bytes(source.read_bytes())
        return 0

    dest = tmp_path / "evidence"
    _patch(monkeypatch, stdout=document, ffmpeg=ffmpeg)
    with pytest.raises(VisionError, match="ffmpeg failed on clip.mp4"):
        count_paths([video], release=release, evidence=dest)
    assert not dest.exists()
    assert not _partial(dest).exists()

    fail["encode"] = False
    count_paths([video], release=release, evidence=dest)
    before = _files(dest)
    assert (dest / "001-clip.mp4" / COUNTED_MP4).read_bytes() == b"counted"
    assert sorted(path.name for path in (dest / "001-clip.mp4").iterdir()) == [COUNTED_MP4]

    fail["encode"] = True
    with pytest.raises(VisionError, match="ffmpeg failed on clip.mp4"):
        count_paths([video], release=release, evidence=dest)
    assert _files(dest) == before
    assert not _partial(dest).exists()
    assert dest.is_dir()


def test_counted_video_is_an_mp4_a_player_can_open(tmp_path: Path) -> None:
    if shutil.which("ffprobe") is None or shutil.which("ffmpeg") is None:
        pytest.fail("ffmpeg is not on PATH")
    size = (80, 64)
    colors = ((10, 20, 80), (140, 30, 30), (90, 40, 140), (180, 140, 40))
    video = tmp_path / "clip.mp4"
    _make_mp4(video, size, colors, 4)
    scored = []
    for index in (1, 2):
        path = tmp_path / f"sample-{index}.jpg"
        _jpeg(path, size, (128, 128, 128))
        scored.append(path)
    name = CardName("Otterpus")
    printing = PrintingId("OGN-010")
    face = NamedFace(name, printing)
    box_a = PixelBox(8, 32, 18, 16)
    box_b = PixelBox(48, 32, 18, 16)
    jpegs = (
        InferredJpeg(1, scored[0], ParsedFrame("frame-0001.jpg", (Detection(face, box_a),))),
        InferredJpeg(2, scored[1], ParsedFrame("frame-0002.jpg", (Detection(face, box_b),))),
    )
    frames = tuple(Frame(jpeg.parsed.label, (NamedSight(name, printing),), 0, 0) for jpeg in jpegs)
    capture = CaptureCount(video, CaptureKind.VIDEO, len(frames), 0, 0, inventory_from_frames(frames))
    dest = tmp_path / "evidence"
    with evidence_run(dest, [video]) as stage:
        assert stage is not None
        write_evidence(
            stage,
            order=1,
            release_id="rel-1",
            jpegs=jpegs,
            frames=frames,
            capture=capture,
        )
    published = dest / "001-clip.mp4" / COUNTED_MP4
    folder = published.parent
    assert sorted(path.name for path in folder.iterdir()) == [COUNTED_MP4]
    assert list(dest.rglob("frame-*.jpg")) == []
    payload = _probe_video(published)
    assert payload["format"]["nb_streams"] == 1
    streams = payload["streams"]
    assert isinstance(streams, list)
    assert [stream["codec_type"] for stream in streams] == ["video"]
    assert streams[0]["avg_frame_rate"] == "4/1"
    assert streams[0]["nb_read_frames"] == "4"
    assert float(payload["format"]["duration"]) == pytest.approx(1.0, abs=0.05)
    decoded = _decode_frames(published, tmp_path / "counted-decoded")
    assert len(decoded) == 4
    assert _far(_rgb(decoded[0], 2, 2), _rgb(decoded[1], 2, 2))
    assert _closer(_rgb(decoded[0], 2, 2), colors[0], (128, 128, 128))
    assert _closer(_rgb(decoded[1], 2, 2), colors[1], (128, 128, 128))
    assert _shows_box(decoded[0], box_a)
    assert _shows_box(decoded[1], box_a)
    assert not _shows_box(decoded[0], box_b)
    assert not _shows_box(decoded[1], box_b)
    assert _shows_box(decoded[2], box_b)
    assert not _shows_box(decoded[2], box_a)
    assert _far(_rgb(decoded[2], 2, 2), _rgb(decoded[0], 2, 2))
    assert _far(_rgb(decoded[2], 2, 2), _rgb(decoded[1], 2, 2))


def test_source_frame_size_must_match_the_inferred_jpeg(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.fail("ffmpeg is not on PATH")
    video = tmp_path / "clip.mp4"
    _make_mp4(video, (32, 32), ((20, 20, 20),), 4)
    jpeg = tmp_path / "frame-0001.jpg"
    _jpeg(jpeg, (16, 16), (0, 0, 0))
    name = CardName("Otterpus")
    printing = PrintingId("OGN-010")
    face = NamedFace(name, printing)
    jpegs = (
        InferredJpeg(
            1,
            jpeg,
            ParsedFrame("frame-0001.jpg", (Detection(face, PixelBox(2, 2, 8, 8)),)),
        ),
    )
    frames = (Frame(jpegs[0].parsed.label, (NamedSight(name, printing),), 0, 0),)
    capture = CaptureCount(video, CaptureKind.VIDEO, 1, 0, 0, inventory_from_frames(frames))
    dest = tmp_path / "evidence"
    with pytest.raises(VisionError):
        with evidence_run(dest, [video]) as stage:
            assert stage is not None
            write_evidence(
                stage,
                order=1,
                release_id="rel-1",
                jpegs=jpegs,
                frames=frames,
                capture=capture,
            )
    assert not dest.exists()
    assert not _partial(dest).exists()
