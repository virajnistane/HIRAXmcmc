#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -f ".env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source ".env"
    set +a
fi

ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        return
    fi

    echo "uv not found; installing with official installer..."
    if ! command -v curl >/dev/null 2>&1; then
        echo "Error: curl is required to install uv. Install curl and rerun." >&2
        exit 1
    fi

    if [[ -n "${UV_INSTALL_PATH:-}" ]]; then
        echo "Using UV_INSTALL_PATH=${UV_INSTALL_PATH}"
        curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$UV_INSTALL_PATH" sh

        if [[ "${SHELL:-}" == *"fish"* ]]; then
            if [[ -f "$UV_INSTALL_PATH/env.fish" ]]; then
                if [[ -n "${UV_CACHE_DIR:-}" ]]; then
                    if grep -q '^set -gx UV_CACHE_DIR ' "$UV_INSTALL_PATH/env.fish"; then
                        sed -i "s|^set -gx UV_CACHE_DIR .*$|set -gx UV_CACHE_DIR \"$UV_CACHE_DIR\"|" "$UV_INSTALL_PATH/env.fish"
                    else
                        printf '\nset -gx UV_CACHE_DIR "%s"\n' "$UV_CACHE_DIR" >> "$UV_INSTALL_PATH/env.fish"
                    fi
                fi
                # shellcheck disable=SC1090
                source "$UV_INSTALL_PATH/env.fish"
            fi
        else
            if [[ -f "$UV_INSTALL_PATH/env" ]]; then
                if [[ -n "${UV_CACHE_DIR:-}" ]]; then
                    if grep -q '^export UV_CACHE_DIR=' "$UV_INSTALL_PATH/env"; then
                        sed -i "s|^export UV_CACHE_DIR=.*$|export UV_CACHE_DIR=\"$UV_CACHE_DIR\"|" "$UV_INSTALL_PATH/env"
                    else
                        printf '\nexport UV_CACHE_DIR="%s"\n' "$UV_CACHE_DIR" >> "$UV_INSTALL_PATH/env"
                    fi
                fi
                # shellcheck disable=SC1090
                source "$UV_INSTALL_PATH/env"
            fi
        fi

        if ! command -v uv >/dev/null 2>&1; then
            export PATH="$UV_INSTALL_PATH:$UV_INSTALL_PATH/bin:$PATH"
        fi
    else
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi

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
