import shutil
import subprocess
from pathlib import Path

import pytest

from riftcount._media import INFER_STILLS, STILLS, VIDEOS, kind_for, media_help
from riftcount.inventory import CaptureKind


def test_kind_for_accepts_iphone_suffixes_case_insensitively(tmp_path: Path) -> None:
    assert kind_for(tmp_path / "pack.MOV") is CaptureKind.VIDEO
    assert kind_for(tmp_path / "pack.mov") is CaptureKind.VIDEO
    assert kind_for(tmp_path / "card.HEIC") is CaptureKind.STILL
    assert kind_for(tmp_path / "card.heic") is CaptureKind.STILL
    assert kind_for(tmp_path / "card.HEIF") is CaptureKind.STILL
    assert kind_for(tmp_path / "card.heif") is CaptureKind.STILL
    assert kind_for(tmp_path / "card.JPG") is CaptureKind.STILL
    assert kind_for(tmp_path / "clip.MP4") is CaptureKind.VIDEO
    assert kind_for(tmp_path / "notes.txt") is None


def test_media_help_lists_new_suffixes() -> None:
    help_text = media_help()
    for suffix in (".jpg", ".jpeg", ".heic", ".heif", ".mp4", ".mov"):
        assert suffix in help_text
    assert STILLS == frozenset({".jpg", ".jpeg", ".heic", ".heif"})
    assert VIDEOS == frozenset({".mp4", ".mov"})
    assert INFER_STILLS == frozenset({".jpg", ".jpeg"})


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_ffmpeg_extracts_frames_from_mov(tmp_path: Path) -> None:
    from riftcount._cardvision import _extract_frames

    video = tmp_path / "clip.MOV"
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=64x64:d=0.6",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    dest = tmp_path / "frames"
    dest.mkdir()
    images = _extract_frames(video, dest)
    assert images
    assert all(path.suffix.lower() == ".jpg" and path.is_file() for path in images)
    assert all(path.stat().st_size > 0 for path in images)


def test_heic_still_is_decoded_to_jpeg_before_infer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pillow_heif import register_heif_opener
    from PIL import Image

    from riftcount._cardvision import read_capture

    register_heif_opener()
    heic = tmp_path / "card.HEIC"
    Image.new("RGB", (32, 32), (12, 34, 56)).save(heic, format="HEIF")

    release = tmp_path / "weights"
    release.mkdir()
    (release / "release.json").write_text('{"id": "abc"}\n', encoding="utf-8")

    infer_paths: list[Path] = []

    def fake_which(name: str) -> str | None:
        if name == "cardvision":
            return "/usr/bin/cardvision"
        return None

    def fake_run(argv: list[str], **kwargs: object) -> object:
        del kwargs

        class Proc:
            returncode = 0
            stdout = ""
            stderr = ""

        assert argv[0].endswith("cardvision")
        image = Path(argv[-1])
        infer_paths.append(image)
        assert image.suffix.lower() == ".jpg"
        assert image.is_file()
        assert image.stat().st_size > 0
        with Image.open(image) as opened:
            assert opened.size == (32, 32)
            assert opened.mode in {"RGB", "L"}
        Proc.stdout = (
            '{"frame": "card.jpg", "boxes": ['
            '{"class": "face", "outcome": "named", "name": "Otterpus", "cardId": "OGN-010"}'
            "]}\n"
        )
        return Proc()

    monkeypatch.setattr("riftcount._cardvision.shutil.which", fake_which)
    monkeypatch.setattr("riftcount._cardvision.subprocess.run", fake_run)
    ident, frames = read_capture(heic, CaptureKind.STILL, release)
    assert ident == "abc"
    assert len(infer_paths) == 1
    assert frames[0].named[0].name.text == "Otterpus"


def test_heif_suffix_follows_the_same_decode_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pillow_heif import register_heif_opener
    from PIL import Image

    from riftcount._cardvision import read_capture

    register_heif_opener()
    heif = tmp_path / "card.heif"
    Image.new("RGB", (16, 16), (1, 2, 3)).save(heif, format="HEIF")
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
                '{"class": "face", "outcome": "named", "name": "Otterpus", "cardId": "OGN-010"}'
                "]}\n"
            )
            stderr = ""

        assert Path(argv[-1]).suffix.lower() == ".jpg"
        return Proc()

    monkeypatch.setattr("riftcount._cardvision.shutil.which", fake_which)
    monkeypatch.setattr("riftcount._cardvision.subprocess.run", fake_run)
    _, frames = read_capture(heif, CaptureKind.STILL, release)
    assert frames[0].named[0].name.text == "Otterpus"
