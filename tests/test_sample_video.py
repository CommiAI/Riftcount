import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from riftcount.cli import app
from riftcount.weights import default_weights_dir

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "examples" / "videos" / "sample-riftbound.mp4"


def weights_dir() -> Path | None:
    for path in (default_weights_dir(), Path("/tmp/rift-weights")):
        if (path / "release.json").is_file():
            return path
    return None


RELEASE = weights_dir()


@pytest.mark.skipif(RELEASE is None or not SAMPLE.is_file(), reason="weights or sample video missing")
def test_sample_video_inventory() -> None:
    assert RELEASE is not None
    result = CliRunner().invoke(
        app,
        ["count", "--release", str(RELEASE), str(SAMPLE)],
    )
    assert result.exit_code == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["schema"] == "riftcount.inventory.v1"
    assert report["release_id"]
    assert len(report["captures"]) == 1
    capture = report["captures"][0]
    assert capture["kind"] == "video"
    assert capture["frames_read"] == 9
    inventory = capture["inventory"]
    assert "printings" not in result.stdout
    assert abs(len(inventory["lines"]) - 24) <= 2
    assert abs(inventory["estimated_cards"] - 29) <= 2
    assert inventory["estimated_cards"] == sum(line["copies"] for line in inventory["lines"])
    for line in inventory["lines"]:
        assert set(line) == {"name", "card_id", "copies"}
        assert line["copies"] >= 1
        assert line["card_id"]
