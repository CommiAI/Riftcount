import json
from pathlib import Path

import pytest

from riftcount.inventory import (
    SCHEMA,
    CaptureCount,
    CaptureKind,
    CardName,
    CountReport,
    Frame,
    Inventory,
    InventoryLine,
    NamedSight,
    PrintingId,
    dumps_report,
    inventory_from_frames,
)


def sight(name: str, card_id: str) -> NamedSight:
    return NamedSight(CardName(name), PrintingId(card_id))


def test_copies_are_the_max_in_one_frame() -> None:
    frames = [
        Frame("a", (sight("Cloud Drake", "VEN-048"),), 0, 0),
        Frame(
            "b",
            (sight("Cloud Drake", "VEN-048"), sight("Cloud Drake", "VEN-048")),
            1,
            0,
        ),
        Frame("c", (sight("Cloud Drake", "VEN-048"),), 0, 0),
    ]
    inventory = inventory_from_frames(frames)
    assert len(inventory.lines) == 1
    line = inventory.lines[0]
    assert line.name.text == "Cloud Drake"
    assert line.copies == 2
    assert line.card_id.text == "VEN-048"
    assert inventory.estimated_cards == 2


def test_unnamed_and_backs_do_not_become_lines() -> None:
    inventory = inventory_from_frames([Frame("a", (), 4, 2)])
    assert inventory.lines == ()
    assert inventory.estimated_cards == 0


def test_empty_frames_have_no_lines() -> None:
    assert inventory_from_frames(()).lines == ()


def test_mode_card_id_is_the_most_frequent_printing() -> None:
    frames = [
        Frame(
            "a",
            (sight("Fury Rune", "RAD-R01A"), sight("Fury Rune", "SFD-R01A")),
            0,
            0,
        ),
        Frame("b", (sight("Fury Rune", "RAD-R01A"),), 0, 0),
    ]
    line = inventory_from_frames(frames).lines[0]
    assert line.copies == 2
    assert line.card_id.text == "RAD-R01A"


def test_tied_printings_break_by_id_text() -> None:
    frames = [
        Frame("a", (sight("Fury Rune", "VEN-R01A"),), 0, 0),
        Frame("b", (sight("Fury Rune", "RAD-R01A"),), 0, 0),
    ]
    line = inventory_from_frames(frames).lines[0]
    assert line.copies == 1
    assert line.card_id.text == "RAD-R01A"


def test_lines_are_sorted_by_name_and_estimated_cards_is_the_sum() -> None:
    frames = [
        Frame(
            "a",
            (
                sight("Otterpus", "OGN-001"),
                sight("Cloud Drake", "VEN-048"),
                sight("Cloud Drake", "VEN-048"),
            ),
            0,
            0,
        )
    ]
    inventory = inventory_from_frames(frames)
    assert [line.name.text for line in inventory.lines] == ["Cloud Drake", "Otterpus"]
    assert inventory.estimated_cards == 3


def test_dumps_report_keeps_unnamed_outside_inventory() -> None:
    report = CountReport(
        release_id="201478a00d6ef6ba",
        captures=(
            CaptureCount(
                path=Path("/abs/examples/photos/photo1-runes.jpg"),
                kind=CaptureKind.STILL,
                frames_read=1,
                unnamed_boxes=2,
                back_boxes=1,
                inventory=Inventory(
                    lines=(
                        InventoryLine(CardName("Body Rune"), PrintingId("VEN-R04B"), 1),
                    )
                ),
            ),
        ),
    )
    text = dumps_report(report)
    payload = json.loads(text)
    assert text.endswith("\n")
    assert list(payload) == ["schema", "release_id", "captures"]
    assert payload["schema"] == SCHEMA
    assert "estimated_cards" not in payload
    capture = payload["captures"][0]
    assert list(capture) == [
        "path",
        "kind",
        "frames_read",
        "unnamed_boxes",
        "back_boxes",
        "inventory",
    ]
    assert capture["unnamed_boxes"] == 2
    assert capture["back_boxes"] == 1
    assert capture["kind"] == "still"
    assert list(capture["inventory"]) == ["lines", "estimated_cards"]
    assert capture["inventory"]["estimated_cards"] == 1
    assert capture["inventory"]["lines"] == [
        {"name": "Body Rune", "card_id": "VEN-R04B", "copies": 1}
    ]
    assert "printings" not in text
    assert "topName" not in text


def test_line_rejects_zero_copies() -> None:
    with pytest.raises(ValueError):
        InventoryLine(CardName("Body Rune"), PrintingId("VEN-R04B"), 0)


def test_blank_name_is_rejected() -> None:
    with pytest.raises(ValueError):
        CardName("  ")
