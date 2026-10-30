# OpenRift still-photo spike

Runs OpenRift's shared `readBoard` on `examples/photos/` without Postgres.

## Prerequisites

- Bun
- Public assets under a directory (default `/tmp/openrift-scan`):
  - `scan-bank-50d08713bd8b8fa4.bin`
  - `scan-labels-50d08713bd8b8fa4.json`
  - `scan-encoder-v2.onnx`
  - `scan-board-detector-v2.onnx`
- A checkout of [openriftapp/openrift](https://github.com/openriftapp/openrift) at `/tmp/openrift-src` (imports shared TypeScript)
- Network access to `https://openrift.app/media/cards/{uuid}-400w.webp` for aligned verify

Fetch the scan assets from `https://openrift.app/api/v1/scan/manifest` and the matching `/media/scan/` URLs.

## Run

```bash
cd spike/openrift
bun install
bun run-photos.ts ../../results/openrift-report.json
```

`MAX_SIDE` defaults to 2048. Paths inside `run-photos.ts` currently point at `/tmp/openrift-*` and `/workspace/examples/photos`.
