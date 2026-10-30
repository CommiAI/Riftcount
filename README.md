# riftcount

Inventory helper — start with Riftbound TCG pack photos/videos, grow into a general scan-to-list tool later.

Yuumi keeps a book; this keeps a count.

## Status

Prototype research and spikes (cardvision, OpenRift stills, sample video) live on the research branches and in:

- [docs/research.md](docs/research.md)
- [spike/](spike/)
- [examples/](examples/)

A Python CLI (`uv` + `cardvision`) is the planned next step. Inventory storage (e.g. Notion) is not wired yet.

## Rename note

GitHub repo rename to `CommiAI/riftcount` may need to be done in the repo **Settings → General** (this agent cannot rename the remote). After that, update locals:

```bash
git remote set-url origin https://github.com/CommiAI/riftcount.git
```
