# Riftcount

Inventory Riftbound pack photos and videos.

```bash
uv sync
uv run riftcount update
uv run riftcount count examples/videos/sample-riftbound.mp4
```

`update` downloads the cardvision weights into `~/.cache/riftcount/weights`. `count` uses that folder by default. Pass `--release DIR` only to override it. Add `--out pack.json` to also write the JSON to a file.

Copies of a name are the most times that name appears in any single frame.

## License

Riftcount source is [Apache-2.0](LICENSE). See [NOTICE](NOTICE) for third-party notes.

Riftbound, League of Legends, and related marks are property of Riot Games. This project is unofficial and not endorsed by Riot. Example photos and videos show physical cards for identification testing. Card art stays with its owners.

Identification uses [cardvision](https://pypi.org/project/cardvision/) (Apache-2.0), which fetches RiftSight weights. Card catalog data inside those weights is not relicensed by this repository.
