import json
from pathlib import Path

import pytest

from riftcount._cardvision import (
    FRAME_INTERVAL_S,
    frames_from_stdout,
    infer_argv,
    sample_filter,
)
from riftcount.errors import VisionError
from riftcount.inventory import CaptureKind


def test_sample_filter_is_two_frames_per_second() -> None:
    assert FRAME_INTERVAL_S == 0.5
    assert sample_filter() == "fps=2"


def test_infer_argv_is_stills_only(tmp_path: Path) -> None:
    photo = tmp_path / "a.jpg"
    photo.write_bytes(b"jpeg")
    release = tmp_path / "release.json"
    argv = infer_argv("cardvision", release, [photo])
    assert argv == ["cardvision", "infer", "--release", str(release), str(photo)]
    assert "--every" not in argv


def test_refused_face_drops_top_name_guess() -> None:
    document = {
        "frame": "frame-0001.jpg",
        "boxes": [
            {
                "class": "face",
                "outcome": "named",
                "name": "Cloud Drake",
                "cardId": "VEN-048",
                "topName": "Cloud Drake",
                "topId": "VEN-048",
            },
            {
                "class": "face",
                "outcome": "nothing",
                "name": "Riptide Rex",
                "cardId": "OGN-092",
                "topName": "Riptide Rex",
                "topId": "OGN-092",
            },
            {"class": "back", "score": 0.9},
        ],
    }
    frame = frames_from_stdout(json.dumps(document) + "\n")[0]
    assert frame.unnamed == 1
    assert frame.backs == 1
    assert len(frame.named) == 1
    assert frame.named[0].name.text == "Cloud Drake"
    assert frame.named[0].printing.text == "VEN-048"


def test_unknown_outcome_is_a_vision_error() -> None:
    document = {"frame": "a.jpg", "boxes": [{"class": "face", "outcome": "maybe", "name": "X", "cardId": "Y"}]}
    with pytest.raises(VisionError):
        frames_from_stdout(json.dumps(document))


def test_video_extracts_jpegs_before_infer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from riftcount._cardvision import read_capture

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"mp4")
    release = tmp_path / "weights"
    release.mkdir()
    (release / "release.json").write_text('{"id": "abc", "detector": {"id": "nope"}}\n', encoding="utf-8")

    calls: list[list[str]] = []

    def fake_which(name: str) -> str | None:
        if name in {"ffmpeg", "cardvision"}:
            return f"/usr/bin/{name}"
        return None

    def fake_run(argv: list[str], **kwargs: object) -> object:
        del kwargs
        calls.append(list(argv))

        class Proc:
            returncode = 0
            stdout = ""
            stderr = ""

        if argv[0] == "ffmpeg":
            pattern = Path(argv[-1])
            pattern.parent.mkdir(parents=True, exist_ok=True)
            jpeg = pattern.parent / "frame-0001.jpg"
            jpeg.write_bytes(b"jpeg")
            return Proc()
        Proc.stdout = json.dumps(
            {
                "frame": "frame-0001.jpg",
                "boxes": [
                    {"class": "face", "outcome": "named", "name": "Otterpus", "cardId": "OGN-010"}
                ],
            }
        ) + "\n"
        return Proc()

    monkeypatch.setattr("riftcount._cardvision.shutil.which", fake_which)
    monkeypatch.setattr("riftcount._cardvision.subprocess.run", fake_run)
    ident, frames = read_capture(video, CaptureKind.VIDEO, release)
    assert ident == "abc"
    assert len(frames) == 1
    assert frames[0].named[0].name.text == "Otterpus"
    assert calls[0][0] == "ffmpeg"
    assert "fps=2" in calls[0]
    infer = calls[1]
    assert infer[0].endswith("cardvision")
    assert infer[1] == "infer"
    assert "--every" not in infer
    assert all(not arg.endswith(".mp4") for arg in infer)
    assert any(arg.endswith(".jpg") for arg in infer)


def test_mov_video_uses_the_same_ffmpeg_extract_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from riftcount._cardvision import read_capture

    video = tmp_path / "clip.MOV"
    video.write_bytes(b"mov")
    release = tmp_path / "weights"
    release.mkdir()
    (release / "release.json").write_text('{"id": "abc"}\n', encoding="utf-8")
    calls: list[list[str]] = []

    def fake_which(name: str) -> str | None:
        if name in {"ffmpeg", "cardvision"}:
            return f"/usr/bin/{name}"
        return None

    def fake_run(argv: list[str], **kwargs: object) -> object:
        del kwargs
        calls.append(list(argv))

        class Proc:
            returncode = 0
            stdout = ""
            stderr = ""

        if argv[0] == "ffmpeg":
            assert argv[argv.index("-i") + 1] == str(video)
            pattern = Path(argv[-1])
            jpeg = pattern.parent / "frame-0001.jpg"
            jpeg.write_bytes(b"jpeg")
            return Proc()
        Proc.stdout = (
            '{"frame": "frame-0001.jpg", "boxes": ['
            '{"class": "face", "outcome": "named", "name": "Otterpus", "cardId": "OGN-010"}'
            "]}\n"
        )
        return Proc()

    monkeypatch.setattr("riftcount._cardvision.shutil.which", fake_which)
    monkeypatch.setattr("riftcount._cardvision.subprocess.run", fake_run)
    _, frames = read_capture(video, CaptureKind.VIDEO, release)
    assert frames[0].named[0].name.text == "Otterpus"
    assert calls[0][0] == "ffmpeg"
    assert str(video) in calls[0]


def test_open_capture_keeps_video_jpegs_until_the_caller_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from riftcount._cardvision import open_capture

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"mp4")
    release = tmp_path / "weights"
    release.mkdir()
    (release / "release.json").write_text('{"id": "abc"}\n', encoding="utf-8")

    def fake_which(name: str) -> str | None:
        if name in {"ffmpeg", "cardvision"}:
            return f"/usr/bin/{name}"
        return None

    def fake_run(argv: list[str], **kwargs: object) -> object:
        del kwargs

        class Proc:
            returncode = 0
            stdout = ""
            stderr = ""

        if argv[0] == "ffmpeg":
            pattern = Path(argv[-1])
            jpeg = pattern.parent / "frame-0001.jpg"
            jpeg.write_bytes(b"jpeg-bytes")
            return Proc()
        Proc.stdout = (
            '{"frame": "frame-0001.jpg", "boxes": ['
            '{"class": "face", "outcome": "named", "name": "Otterpus", "cardId": "OGN-010"}'
            "]}\n"
        )
        return Proc()

    monkeypatch.setattr("riftcount._cardvision.shutil.which", fake_which)
    monkeypatch.setattr("riftcount._cardvision.subprocess.run", fake_run)
    with open_capture(video, CaptureKind.VIDEO, release) as opened:
        held = opened.jpegs[0].path
        assert held.is_file()
        assert held.read_bytes() == b"jpeg-bytes"
        assert opened.frames[0].named[0].name.text == "Otterpus"
    assert video.is_file()
    assert not held.exists()


def test_open_capture_does_not_delete_a_caller_jpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from riftcount._cardvision import UnnamedFace, open_capture

    photo = tmp_path / "card.jpg"
    photo.write_bytes(b"jpeg-bytes")
    release = tmp_path / "weights"
    release.mkdir()
    (release / "release.json").write_text('{"id": "abc"}\n', encoding="utf-8")

    def fake_which(name: str) -> str | None:
        return "/usr/bin/cardvision" if name == "cardvision" else None

    def fake_run(argv: list[str], **kwargs: object) -> object:
        del kwargs

        class Proc:
            returncode = 0
            stdout = (
                '{"frame": "card.jpg", "boxes": ['
                '{"class": "face", "outcome": "nothing", "topName": "Riptide Rex"}'
                "]}\n"
            )
            stderr = ""

        assert Path(argv[-1]) == photo
        return Proc()

    monkeypatch.setattr("riftcount._cardvision.shutil.which", fake_which)
    monkeypatch.setattr("riftcount._cardvision.subprocess.run", fake_run)
    with open_capture(photo, CaptureKind.STILL, release) as opened:
        assert opened.jpegs[0].path == photo
        detection = opened.jpegs[0].parsed.detections[0]
        assert detection.box is None
        assert isinstance(detection.face, UnnamedFace)
        assert detection.face.guess is not None
        assert detection.face.guess.text == "Riptide Rex"
        frame = opened.frames[0]
        assert frame.unnamed == 1
        assert frame.named == ()
        assert "Riptide Rex" not in str(frame)
    assert photo.read_bytes() == b"jpeg-bytes"
