import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHOTOS = ROOT / "examples" / "photos"
PHOTO_NAMES = ("photo1-runes.jpg", "photo2-fabric.jpg")


def cardvision_bin() -> str:
    found = shutil.which("cardvision")
    if found:
        return found
    sys.exit("cardvision is not on PATH. Install it, then run cardvision fetch-weights.")


def run_infer(binary: str, release: Path, images: list[Path]) -> list[dict]:
    proc = subprocess.run(
        [binary, "infer", "--release", str(release), *[str(p) for p in images]],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            rows.append(json.loads(line))
    return rows


def slim(row: dict) -> dict:
    faces = []
    for box in row.get("boxes", []):
        faces.append(
            {
                "class": box.get("class"),
                "outcome": box.get("outcome"),
                "name": box.get("name"),
                "card_id": box.get("cardId"),
                "detector_score": box.get("score"),
                "similarity": box.get("similarity"),
                "margin": box.get("margin"),
                "rotation_retry": box.get("retried"),
            }
        )
    return {
        "frame": row.get("frame"),
        "width": row.get("frameW"),
        "height": row.get("frameH"),
        "faces": faces,
    }


def near_misses(release_dir: Path, image: Path) -> dict | None:
    try:
        import numpy as np
        from PIL import Image
        from cardvision.detect.detector import Detector
    except ImportError:
        return None
    detector = Detector.open(release_dir / "detector.onnx", threshold=0.05)
    rgb = np.asarray(Image.open(image).convert("RGB"))
    found, best = detector.look(rgb, threshold=0.05)
    return {
        "threshold": 0.05,
        "best_score_before_threshold": best,
        "boxes": [
            {
                "class": item.class_name,
                "score": item.score,
                "box_xywh": list(item.box),
            }
            for item in found
        ],
    }


def crop_named(image: Path, row: dict, name: str):
    from PIL import Image

    match = next(
        box
        for box in row["boxes"]
        if box.get("name") == name and box.get("outcome") == "named"
    )
    x, y, width, height = box_xywh(match["box"])
    src = Image.open(image).convert("RGB")
    card = src.crop((x, y, x + width, y + height))
    return card.resize((520, 678), Image.Resampling.LANCZOS)


def box_xywh(box: list[float]) -> tuple[int, int, int, int]:
    x, y, width, height = box
    return int(x), int(y), int(width), int(height)


def stack_scenes(card, dest: Path) -> list[Path]:
    from PIL import Image

    dest.mkdir(parents=True, exist_ok=True)

    def scene(name: str, offsets: list[tuple[int, int]], size: tuple[int, int]) -> Path:
        canvas = Image.new("RGB", size, (18, 92, 48))
        for dx, dy in offsets:
            canvas.paste(card, (80 + dx, 200 + dy))
        path = dest / f"{name}.jpg"
        canvas.save(path, quality=90)
        return path

    return [
        scene("one", [(0, 0)], (2000, 1600)),
        scene("squared", [(i * 4, i * 3) for i in range(4)], (2000, 1600)),
        scene("fanned-90", [(i * 90, i * 16) for i in range(4)], (2000, 1600)),
        scene("fanned-260", [(i * 260, i * 20) for i in range(4)], (2400, 1400)),
        scene("separated", [(i * 560, 0) for i in range(4)], (2500, 1200)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cardvision on the example photos and a pasted stack.")
    parser.add_argument("--release", type=Path, required=True, help="Folder from cardvision fetch-weights, containing release.json")
    parser.add_argument("--out", type=Path, default=ROOT / "spike" / "results" / "report.json")
    args = parser.parse_args()
    release = args.release / "release.json"
    if not release.is_file():
        sys.exit(f"missing {release}. Run: cardvision fetch-weights {args.release}")

    binary = cardvision_bin()
    photos = [PHOTOS / name for name in PHOTO_NAMES]
    missing = [str(p) for p in photos if not p.is_file()]
    if missing:
        sys.exit("missing photos: " + ", ".join(missing))

    raw_photos = run_infer(binary, release, photos)
    photo_rows = [slim(row) for row in raw_photos]
    card = crop_named(photos[0], raw_photos[0], "Fury Rune")
    stack_dir = args.out.parent / "stacks"
    stack_paths = stack_scenes(card, stack_dir)
    stack_rows = [slim(row) for row in run_infer(binary, release, stack_paths)]
    release_doc = json.loads(release.read_text())
    report = {
        "release_id": release_doc.get("id") or release_doc.get("releaseId"),
        "photos": photo_rows,
        "photo1_detector_at_0_05": near_misses(args.release, photos[0]),
        "pasted_stack": {
            "note": "Four copies of the photo 1 Fury Rune crop pasted on a flat green field. Not a physical stack.",
            "scenes": stack_rows,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
