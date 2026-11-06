# Riftcount

Inventory Riftbound pack photos and videos.

Supported formats: `.jpg`, `.jpeg`, `.heic`, `.heif` photos and `.mp4`, `.mov` videos.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you do not have it yet. Cloud agents already get uv from `.cursor/install.sh`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
uv sync
uv run riftcount update
uv run riftcount count examples/videos/sample-riftbound.mp4
```

`update` downloads the cardvision weights into `~/.cache/riftcount/weights`. `count` uses that folder by default. Pass `--release DIR` only to override it. Add `--out pack.json` to also write the JSON to a file.

Copies of a name are the most times that name appears in any single frame.

## See the frames that were counted

`--evidence DIR` writes the JPEGs that `count` read. Each detection is labeled `copy`, `repeat`, `unnamed`, or `back`.

```bash
uv run riftcount count examples/videos/sample-riftbound.mp4 --evidence evidence/sample
uv run riftcount count examples/photos/photo1-runes.jpg --out pack.json --evidence evidence/runes
```

Stdout stays the inventory JSON. `--out` still writes that JSON. The pictures are files under `DIR`:

```text
evidence/sample/001-sample-riftbound.mp4/frame-0001.jpg
evidence/sample/001-sample-riftbound.mp4/frame-0002.jpg
evidence/runes/001-photo1-runes.jpg/frame-0001.jpg
```

`001` is that capture's place in the JSON, starting at 1. `frame-0001.jpg` is the first JPEG passed to infer. On a video, that file is the first frame ffmpeg extracted at `fps=2`. Leave `--evidence` off and no pictures are written.

Open the frames in filename order. The boxes sit on the stored JPEG pixels, and a bar under the photo repeats the inventory. A later successful count with the same directory replaces it. A failed count leaves `DIR` unchanged.

## Running tests

```bash
uv run pytest
```

## License

Riftcount source is [Apache-2.0](LICENSE). See [NOTICE](NOTICE) for third-party notes.

Riftbound, League of Legends, and related marks are property of Riot Games. This project is unofficial and not endorsed by Riot. Example photos and videos show physical cards for identification testing. Card art stays with its owners.

Identification uses [cardvision](https://pypi.org/project/cardvision/) (Apache-2.0), which fetches RiftSight weights. Card catalog data inside those weights is not relicensed by this repository.
