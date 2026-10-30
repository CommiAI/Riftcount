import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from riftcount.cli import app
from riftcount.errors import VisionError
from riftcount.inventory import CardName, Frame, NamedSight, PrintingId


runner = CliRunner()


def _photo(directory: Path, name: str = "a.jpg") -> Path:
    path = directory / name
    path.write_bytes(b"jpeg")
    return path


def _weights(directory: Path) -> Path:
    release = directory / "weights"
    release.mkdir()
    (release / "release.json").write_text('{"id": "abc"}\n', encoding="utf-8")
    return release


def _sight() -> NamedSight:
    return NamedSight(CardName("Fury Rune"), PrintingId("RAD-R01A"))


def _patch_reader(monkeypatch: pytest.MonkeyPatch, frames_for: dict[Path, tuple[Frame, ...]] | None = None) -> None:
    def fake_read(path: Path, kind: object, release_dir: Path) -> tuple[str, tuple[Frame, ...]]:
        del kind, release_dir
        if frames_for is not None:
            frames = frames_for[path]
        else:
            frames = (Frame(path.name, (_sight(),), 0, 0),)
        return "rel-1", frames

    monkeypatch.setattr("riftcount.count.read_capture", fake_read)


def test_no_verb_prints_help_and_exits_2() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code == 2
    assert "count" in result.output
    assert "sync" not in result.output


def test_help_lists_count_and_update_and_not_sync_or_help_verb() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "count" in result.output
    assert "update" in result.output
    assert "sync" not in result.output
    unknown = runner.invoke(app, ["help"])
    assert unknown.exit_code == 2
    sync = runner.invoke(app, ["sync"])
    assert sync.exit_code == 2


def test_count_help_exits_0() -> None:
    result = runner.invoke(app, ["count", "--help"])
    assert result.exit_code == 0
    assert "--release" in result.output
    assert "--out" in result.output
    assert ".jpg" in result.output
    assert ".mp4" in result.output
    assert "sync" not in result.output
    assert "RIFTCOUNT_RELEASE" not in result.output


def test_count_without_paths_exits_2(tmp_path: Path) -> None:
    release = _weights(tmp_path)
    result = runner.invoke(app, ["count", "--release", str(release)])
    assert result.exit_code == 2


def test_two_paths_are_two_captures_and_a_repeat_is_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _photo(tmp_path, "one.jpg")
    second = _photo(tmp_path, "two.jpg")
    release = _weights(tmp_path)
    _patch_reader(monkeypatch)

    pair = runner.invoke(
        app,
        ["count", "--release", str(release), str(first), str(second)],
    )
    assert pair.exit_code == 0, pair.output
    report = json.loads(pair.stdout)
    assert [Path(row["path"]).name for row in report["captures"]] == ["one.jpg", "two.jpg"]
    assert report["captures"][0]["inventory"]["estimated_cards"] == 1

    repeated = runner.invoke(
        app,
        ["count", "--release", str(release), str(first), str(first)],
    )
    assert repeated.exit_code == 0, repeated.output
    assert len(json.loads(repeated.stdout)["captures"]) == 1


def test_directory_and_same_file_are_one_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    photo = _photo(tmp_path, "one.jpg")
    _photo(tmp_path, "notes.txt")
    release = _weights(tmp_path)
    _patch_reader(monkeypatch)
    result = runner.invoke(
        app,
        ["count", "--release", str(release), str(tmp_path), str(photo)],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert len(report["captures"]) == 1
    assert report["captures"][0]["kind"] == "still"


def test_default_weights_dir_is_used_when_flag_is_omitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    photo = _photo(tmp_path, "one.jpg")
    release = _weights(tmp_path)
    seen: list[Path] = []

    def fake_read(path: Path, kind: object, release_dir: Path) -> tuple[str, tuple[Frame, ...]]:
        del path, kind
        seen.append(release_dir)
        return "rel-1", (Frame("one.jpg", (_sight(),), 2, 1),)

    monkeypatch.setattr("riftcount.weights.default_weights_dir", lambda: release)
    monkeypatch.setattr("riftcount.count.read_capture", fake_read)
    result = runner.invoke(app, ["count", str(photo)])
    assert result.exit_code == 0, result.output
    assert seen == [release]
    capture = json.loads(result.stdout)["captures"][0]
    assert capture["unnamed_boxes"] == 2
    assert capture["back_boxes"] == 1
    assert "unnamed" not in capture["inventory"]


def test_missing_default_weights_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    photo = _photo(tmp_path)
    missing = tmp_path / "missing-weights"
    monkeypatch.setattr("riftcount.weights.default_weights_dir", lambda: missing)
    result = runner.invoke(app, ["count", str(photo)])
    assert result.exit_code == 2
    assert "riftcount update" in result.stderr
    assert result.stdout == ""


def test_update_runs_fetch_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "weights"
    calls: list[list[str]] = []

    def fake_which(name: str) -> str | None:
        return "/usr/bin/cardvision" if name == "cardvision" else None

    def fake_run(argv: list[str], **kwargs: object) -> object:
        del kwargs
        calls.append(list(argv))
        target.mkdir(parents=True, exist_ok=True)
        (target / "release.json").write_text('{"id": "abc"}\n', encoding="utf-8")

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr("riftcount.weights.shutil.which", fake_which)
    monkeypatch.setattr("riftcount.weights.subprocess.run", fake_run)
    result = runner.invoke(app, ["update", "--release", str(target)])
    assert result.exit_code == 0, result.output
    assert calls == [["/usr/bin/cardvision", "fetch-weights", "--overwrite", str(target)]]
    assert "weights ready" in result.stderr


def test_bad_suffix_missing_path_and_empty_directory_exit_2(tmp_path: Path) -> None:
    release = _weights(tmp_path)
    text = tmp_path / "notes.txt"
    text.write_text("nope", encoding="utf-8")
    empty = tmp_path / "empty"
    empty.mkdir()

    bad_suffix = runner.invoke(app, ["count", "--release", str(release), str(text)])
    assert bad_suffix.exit_code == 2
    assert bad_suffix.stdout == ""

    missing = runner.invoke(app, ["count", "--release", str(release), str(tmp_path / "gone.jpg")])
    assert missing.exit_code == 2
    assert missing.stdout == ""

    vacant = runner.invoke(app, ["count", "--release", str(release), str(empty)])
    assert vacant.exit_code == 2
    assert vacant.stdout == ""


def test_release_folder_without_pin_exits_2(tmp_path: Path) -> None:
    photo = _photo(tmp_path)
    release = tmp_path / "weights"
    release.mkdir()
    result = runner.invoke(app, ["count", "--release", str(release), str(photo)])
    assert result.exit_code == 2
    assert "riftcount update" in result.stderr
    assert result.stdout == ""


def test_out_matches_stdout_and_uses_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    photo = _photo(tmp_path)
    release = _weights(tmp_path)
    _patch_reader(monkeypatch)
    destination = tmp_path / "pack.json"
    result = runner.invoke(
        app,
        ["count", "--release", str(release), str(photo), "--out", str(destination)],
    )
    assert result.exit_code == 0, result.output
    assert destination.read_text(encoding="utf-8") == result.stdout
    assert json.loads(result.stdout)["schema"] == "riftcount.inventory.v1"
    assert not destination.with_suffix(".json.tmp").exists()


def test_zero_named_faces_exit_0(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    photo = _photo(tmp_path)
    release = _weights(tmp_path)
    _patch_reader(monkeypatch, {photo.resolve(): (Frame(photo.name, (), 3, 0),)})
    result = runner.invoke(app, ["count", "--release", str(release), str(photo)])
    assert result.exit_code == 0, result.output
    inventory = json.loads(result.stdout)["captures"][0]["inventory"]
    assert inventory["lines"] == []
    assert inventory["estimated_cards"] == 0


def test_vision_error_exits_3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    photo = _photo(tmp_path)
    release = _weights(tmp_path)

    def fake_read(path: Path, kind: object, release_dir: Path) -> tuple[str, tuple[Frame, ...]]:
        del path, kind, release_dir
        raise VisionError("cardvision infer failed for a.jpg")

    monkeypatch.setattr("riftcount.count.read_capture", fake_read)
    result = runner.invoke(app, ["count", "--release", str(release), str(photo)])
    assert result.exit_code == 3
    assert result.stdout == ""
    assert "cardvision infer failed" in result.stderr


def test_bad_out_directory_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    photo = _photo(tmp_path)
    release = _weights(tmp_path)
    _patch_reader(monkeypatch)
    destination = tmp_path / "missing" / "pack.json"
    result = runner.invoke(
        app,
        ["count", "--release", str(release), str(photo), "--out", str(destination)],
    )
    assert result.exit_code == 2
    assert result.stdout == ""
    assert not destination.exists()


def test_infer_argv_rejects_video_before_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from riftcount._cardvision import infer_argv

    def fail(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("subprocess should not start")

    monkeypatch.setattr("riftcount._cardvision.subprocess.run", fail)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"mp4")
    with pytest.raises(VisionError):
        infer_argv("cardvision", tmp_path / "release.json", [video])
