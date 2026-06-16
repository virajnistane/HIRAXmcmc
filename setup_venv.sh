#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        return
    fi

    echo "uv not found; installing with official installer..."
    if ! command -v curl >/dev/null 2>&1; then
        echo "Error: curl is required to install uv. Install curl and rerun." >&2
        exit 1
    fi

    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Common user-local install paths used by uv installer.
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

    if ! command -v uv >/dev/null 2>&1; then
        echo "Error: uv installation completed but uv is not on PATH." >&2
        echo "Add ~/.local/bin or ~/.cargo/bin to PATH, then rerun." >&2
        exit 1
    fi
}

echo "Bootstrapping HIRAXmcmc development environment..."
ensure_uv

if [[ -d ".venv" ]]; then
    echo "Detected existing .venv; reusing it."
fi

uv sync --group dev

echo "Environment ready."
echo "Use: source .venv/bin/activate"
