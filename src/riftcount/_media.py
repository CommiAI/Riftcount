from pathlib import Path

from riftcount.inventory import CaptureKind

STILLS: frozenset[str] = frozenset({".jpg", ".jpeg"})
VIDEOS: frozenset[str] = frozenset({".mp4"})


def kind_for(path: Path) -> CaptureKind | None:
    suffix = path.suffix.lower()
    if suffix in STILLS:
        return CaptureKind.STILL
    if suffix in VIDEOS:
        return CaptureKind.VIDEO
    return None


def media_help() -> str:
    suffixes = ", ".join(sorted(STILLS | VIDEOS))
    return (
        f"Photos and videos ({suffixes}). "
        "A directory adds the media files directly inside it."
    )
