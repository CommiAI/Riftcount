#!/usr/bin/env bash
set -euo pipefail

# Cursor runs install as a non-interactive login shell. Ubuntu's ~/.bashrc
# returns early in that mode, so installer PATH lines never load. Put uv on
# the default PATH via /usr/local/bin so later agent shells find it.

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

uv_bin="$(command -v uv || true)"
if [[ -z "${uv_bin}" && -x "${HOME}/.local/bin/uv" ]]; then
  uv_bin="${HOME}/.local/bin/uv"
fi
if [[ -z "${uv_bin}" ]]; then
  echo "uv install finished but uv was not found on PATH or in ~/.local/bin" >&2
  exit 1
fi
sudo ln -sf "${uv_bin}" /usr/local/bin/uv

uv sync

weights_pin="${HOME}/.cache/riftcount/weights/release.json"
if [[ ! -f "${weights_pin}" ]]; then
  uv run riftcount update
fi
