from pathlib import Path

from riftcount.weights import WEIGHTS_DIRNAME, default_weights_dir, resolve_weights_dir


def test_default_weights_dir_is_under_home_cache(monkeypatch) -> None:
    home = Path("/tmp/riftcount-home")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    assert default_weights_dir() == (home / ".cache" / WEIGHTS_DIRNAME).resolve()


def test_resolve_weights_dir_expands_override(tmp_path: Path) -> None:
    override = tmp_path / "custom"
    assert resolve_weights_dir(override) == override
