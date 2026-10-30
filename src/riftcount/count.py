from collections.abc import Sequence
from pathlib import Path

from riftcount._cardvision import read_capture
from riftcount._media import kind_for
from riftcount.errors import UsageError, VisionError
from riftcount.inventory import CaptureCount, CountReport, inventory_from_frames


def count_paths(paths: Sequence[Path], *, release: Path) -> CountReport:
    expanded = _expand(paths)
    captures: list[CaptureCount] = []
    release_ids: list[str] = []
    for path in expanded:
        kind = kind_for(path)
        if kind is None:
            raise UsageError(f"unsupported file: {path}")
        ident, frames = read_capture(path, kind, release)
        if not frames:
            raise VisionError(f"cardvision infer returned no frames for {path.name}")
        release_ids.append(ident)
        captures.append(
            CaptureCount(
                path=path,
                kind=kind,
                frames_read=len(frames),
                unnamed_boxes=sum(frame.unnamed for frame in frames),
                back_boxes=sum(frame.backs for frame in frames),
                inventory=inventory_from_frames(frames),
            )
        )
    if len(set(release_ids)) != 1:
        raise VisionError("captures disagree on release id")
    return CountReport(release_id=release_ids[0], captures=tuple(captures))


def _expand(paths: Sequence[Path]) -> tuple[Path, ...]:
    if not paths:
        raise UsageError("no photos or videos to count")
    seen: set[Path] = set()
    ordered: list[Path] = []
    for raw in paths:
        if not raw.exists():
            raise UsageError(f"path not found: {raw}")
        if raw.is_dir():
            accepted = sorted(
                (
                    child
                    for child in raw.iterdir()
                    if child.is_file()
                    and not child.name.startswith(".")
                    and kind_for(child) is not None
                ),
                key=lambda child: child.name,
            )
            if not accepted:
                raise UsageError(f"no photos or videos in {raw}")
            for child in accepted:
                _add(child, seen, ordered)
            continue
        if raw.is_file():
            _add(raw, seen, ordered)
            continue
        raise UsageError(f"path not found: {raw}")
    if not ordered:
        raise UsageError("no photos or videos to count")
    return tuple(ordered)


def _add(path: Path, seen: set[Path], ordered: list[Path]) -> None:
    if kind_for(path) is None:
        raise UsageError(f"unsupported file: {path}")
    resolved = path.resolve()
    if resolved in seen:
        return
    seen.add(resolved)
    ordered.append(resolved)
