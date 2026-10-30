# Riftcount

Inventory Riftbound pack photos and videos.

```bash
uv sync
uv run riftcount update
uv run riftcount count examples/videos/sample-riftbound.mp4
```

`update` downloads the cardvision weights into `~/.cache/riftcount/weights`. `count` uses that folder by default. Pass `--release DIR` only to override it. Add `--out pack.json` to also write the JSON to a file.

Copies of a name are the most times that name appears in any single frame.
