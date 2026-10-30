from __future__ import annotations

import enum
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "riftcount.inventory.v1"


def _require_text(text: str, label: str) -> None:
    if not isinstance(text, str) or not text or text != text.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")


@dataclass(frozen=True, slots=True)
class CardName:
    text: str

    def __post_init__(self) -> None:
        _require_text(self.text, "card name")


@dataclass(frozen=True, slots=True)
class PrintingId:
    text: str

    def __post_init__(self) -> None:
        _require_text(self.text, "card id")


class CaptureKind(enum.Enum):
    STILL = "still"
    VIDEO = "video"


@dataclass(frozen=True, slots=True)
class NamedSight:
    name: CardName
    printing: PrintingId


@dataclass(frozen=True, slots=True)
class Frame:
    label: str
    named: tuple[NamedSight, ...]
    unnamed: int
    backs: int

    def __post_init__(self) -> None:
        if self.unnamed < 0 or self.backs < 0:
            raise ValueError("unnamed and backs must be >= 0")


@dataclass(frozen=True, slots=True)
class InventoryLine:
    name: CardName
    card_id: PrintingId
    copies: int

    def __post_init__(self) -> None:
        if self.copies < 1:
            raise ValueError("copies must be >= 1")


@dataclass(frozen=True, slots=True)
class Inventory:
    lines: tuple[InventoryLine, ...]

    def __post_init__(self) -> None:
        names = [line.name.text for line in self.lines]
        if names != sorted(names):
            raise ValueError("inventory lines must be sorted by card name")
        if len(names) != len(set(names)):
            raise ValueError("inventory lines must contain each card name once")

    @property
    def estimated_cards(self) -> int:
        return sum(line.copies for line in self.lines)


@dataclass(frozen=True, slots=True)
class CaptureCount:
    path: Path
    kind: CaptureKind
    frames_read: int
    unnamed_boxes: int
    back_boxes: int
    inventory: Inventory

    def __post_init__(self) -> None:
        if self.frames_read < 1:
            raise ValueError("frames_read must be >= 1")
        if self.unnamed_boxes < 0 or self.back_boxes < 0:
            raise ValueError("unnamed_boxes and back_boxes must be >= 0")


@dataclass(frozen=True, slots=True)
class CountReport:
    release_id: str
    captures: tuple[CaptureCount, ...]

    def __post_init__(self) -> None:
        _require_text(self.release_id, "release id")

    @property
    def schema(self) -> str:
        return SCHEMA


def _mode_printing(hits: Mapping[PrintingId, int]) -> PrintingId:
    return min(hits, key=lambda printing: (-hits[printing], printing.text))


def inventory_from_frames(frames: Sequence[Frame]) -> Inventory:
    max_copies: dict[CardName, int] = {}
    printing_hits: dict[CardName, dict[PrintingId, int]] = {}
    for frame in frames:
        tally: dict[CardName, int] = {}
        for sight in frame.named:
            tally[sight.name] = tally.get(sight.name, 0) + 1
            hits = printing_hits.setdefault(sight.name, {})
            hits[sight.printing] = hits.get(sight.printing, 0) + 1
        for name, count in tally.items():
            previous = max_copies.get(name, 0)
            if count > previous:
                max_copies[name] = count
    lines: list[InventoryLine] = []
    for name, copies in max_copies.items():
        if copies < 1:
            continue
        card_id = _mode_printing(printing_hits[name])
        lines.append(InventoryLine(name=name, card_id=card_id, copies=copies))
    lines.sort(key=lambda line: line.name.text)
    return Inventory(lines=tuple(lines))


def dumps_report(report: CountReport) -> str:
    payload = {
        "schema": report.schema,
        "release_id": report.release_id,
        "captures": [_capture_json(capture) for capture in report.captures],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def _capture_json(capture: CaptureCount) -> dict[str, object]:
    return {
        "path": str(capture.path),
        "kind": capture.kind.value,
        "frames_read": capture.frames_read,
        "unnamed_boxes": capture.unnamed_boxes,
        "back_boxes": capture.back_boxes,
        "inventory": {
            "lines": [_line_json(line) for line in capture.inventory.lines],
            "estimated_cards": capture.inventory.estimated_cards,
        },
    }


def _line_json(line: InventoryLine) -> dict[str, object]:
    return {
        "name": line.name.text,
        "card_id": line.card_id.text,
        "copies": line.copies,
    }
