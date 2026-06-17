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

for optional_var in UV_INSTALL_DIR UV_CACHE_DIR UV_LINK_MODE UV_SYSTEM_SITE_PACKAGES UV_USE_SYSTEM_MPI4PY; do
    if [[ -z "${!optional_var:-}" ]]; then
        unset "$optional_var"
    fi
done

if [[ -n "${UV_INSTALL_DIR:-}" && ! -d "$UV_INSTALL_DIR" ]]; then
    echo "Creating UV_INSTALL_DIR: $UV_INSTALL_DIR"
    mkdir -p "$UV_INSTALL_DIR"
fi

if [[ -n "${UV_CACHE_DIR:-}" && ! -d "$UV_CACHE_DIR" ]]; then
    echo "Creating UV_CACHE_DIR: $UV_CACHE_DIR"
    mkdir -p "$UV_CACHE_DIR"
fi

apply_uv_runtime_env() {
    if [[ -n "${UV_CACHE_DIR:-}" ]]; then
        export UV_CACHE_DIR
    fi

    if [[ -n "${UV_LINK_MODE:-}" ]]; then
        export UV_LINK_MODE
    fi
}

apply_uv_runtime_env

apply_uv_install_env() {
    if [[ -n "${UV_INSTALL_DIR:-}" ]]; then
        if [[ "${SHELL:-}" == *"fish"* ]]; then
            if [[ -f "$UV_INSTALL_DIR/env.fish" ]]; then
                if [[ -n "${UV_CACHE_DIR:-}" ]]; then
                    if grep -q '^set -gx UV_CACHE_DIR ' "$UV_INSTALL_DIR/env.fish"; then
                        sed -i "s|^set -gx UV_CACHE_DIR .*$|set -gx UV_CACHE_DIR \"$UV_CACHE_DIR\"|" "$UV_INSTALL_DIR/env.fish"
                    else
                        printf '\nset -gx UV_CACHE_DIR "%s"\n' "$UV_CACHE_DIR" >> "$UV_INSTALL_DIR/env.fish"
                    fi
                fi
                if [[ -n "${UV_LINK_MODE:-}" ]]; then
                    if grep -q '^set -gx UV_LINK_MODE ' "$UV_INSTALL_DIR/env.fish"; then
                        sed -i "s|^set -gx UV_LINK_MODE .*$|set -gx UV_LINK_MODE \"$UV_LINK_MODE\"|" "$UV_INSTALL_DIR/env.fish"
                    else
                        printf '\nset -gx UV_LINK_MODE "%s"\n' "$UV_LINK_MODE" >> "$UV_INSTALL_DIR/env.fish"
                    fi
                else
                    sed -i '/^set -gx UV_LINK_MODE /d' "$UV_INSTALL_DIR/env.fish"
                fi
                # shellcheck disable=SC1090
                source "$UV_INSTALL_DIR/env.fish"
            fi
        else
            if [[ -f "$UV_INSTALL_DIR/env" ]]; then
                if [[ -n "${UV_CACHE_DIR:-}" ]]; then
                    if grep -q '^export UV_CACHE_DIR=' "$UV_INSTALL_DIR/env"; then
                        sed -i "s|^export UV_CACHE_DIR=.*$|export UV_CACHE_DIR=\"$UV_CACHE_DIR\"|" "$UV_INSTALL_DIR/env"
                    else
                        printf '\nexport UV_CACHE_DIR="%s"\n' "$UV_CACHE_DIR" >> "$UV_INSTALL_DIR/env"
                    fi
                fi
                if [[ -n "${UV_LINK_MODE:-}" ]]; then
                    if grep -q '^export UV_LINK_MODE=' "$UV_INSTALL_DIR/env"; then
                        sed -i "s|^export UV_LINK_MODE=.*$|export UV_LINK_MODE=\"$UV_LINK_MODE\"|" "$UV_INSTALL_DIR/env"
                    else
                        printf '\nexport UV_LINK_MODE="%s"\n' "$UV_LINK_MODE" >> "$UV_INSTALL_DIR/env"
                    fi
                else
                    sed -i '/^export UV_LINK_MODE=/d' "$UV_INSTALL_DIR/env"
                fi
                # shellcheck disable=SC1090
                source "$UV_INSTALL_DIR/env"
            fi
        fi

        if ! command -v uv >/dev/null 2>&1; then
            export PATH="$UV_INSTALL_DIR:$UV_INSTALL_DIR/bin:$PATH"
        fi
    else
        if [[ -n "${UV_CACHE_DIR:-}" ]]; then
            _shell_rc=""
            if [[ -f "$HOME/.bash_profile" ]]; then
                _shell_rc="$HOME/.bash_profile"
            elif [[ -f "$HOME/.bashrc" ]]; then
                _shell_rc="$HOME/.bashrc"
            fi
            if [[ -n "$_shell_rc" ]]; then
                if grep -q '^export UV_CACHE_DIR=' "$_shell_rc"; then
                    sed -i "s|^export UV_CACHE_DIR=.*$|export UV_CACHE_DIR=\"$UV_CACHE_DIR\"|" "$_shell_rc"
                else
                    printf '\nexport UV_CACHE_DIR="%s"\n' "$UV_CACHE_DIR" >> "$_shell_rc"
                fi
                echo "Set UV_CACHE_DIR in $_shell_rc"
            fi
            unset _shell_rc
        fi
    fi
}

ensure_uv() {
    apply_uv_install_env

    if command -v uv >/dev/null 2>&1; then
        return
    fi

    echo "uv not found; installing with official installer..."
    if ! command -v curl >/dev/null 2>&1; then
        echo "Error: curl is required to install uv. Install curl and rerun." >&2
        exit 1
    fi

    if [[ -n "${UV_INSTALL_DIR:-}" ]]; then
        echo "Using UV_INSTALL_DIR=${UV_INSTALL_DIR}"
        curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$UV_INSTALL_DIR" sh
        apply_uv_install_env
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

is_true() {
    local value="${1:-}"
    [[ "$value" == "1" || "$value" == "true" || "$value" == "yes" ]]
}

echo "Bootstrapping HIRAXmcmc development environment..."
ensure_uv

_uv_flags=()
[[ -n "${UV_CACHE_DIR:-}" ]] && _uv_flags+=(--cache-dir "$UV_CACHE_DIR")
[[ -n "${UV_LINK_MODE:-}" ]] && _uv_flags+=(--link-mode "$UV_LINK_MODE")

if is_true "${UV_SYSTEM_SITE_PACKAGES:-}"; then
    if [[ -d ".venv" ]]; then
        echo "Recreating .venv with --system-site-packages enabled."
        rm -rf .venv
    fi
    uv venv "${_uv_flags[@]}" --system-site-packages .venv
elif [[ -d ".venv" ]]; then
    echo "Detected existing .venv; reusing it."
fi

_cc_wrapper="" _cxx_wrapper="" _saved_cc="${CC:-}" _saved_cxx="${CXX:-}"
if [[ -f ".venv/bin/python" ]]; then
    _sysconfig_cflags="$(".venv/bin/python" -c \
        'import sysconfig; print(sysconfig.get_config_var("CFLAGS") or "")' \
        2>/dev/null)"
    if [[ "$_sysconfig_cflags" == *"-fdebug-default-version"* ]] && \
       ! echo "" | ${CXX:-c++} -fdebug-default-version=4 -x c++ - -o /dev/null 2>/dev/null; then
        _cc_wrapper="$(mktemp /tmp/uv_cc_wrap_XXXXXX)"
        _cxx_wrapper="$(mktemp /tmp/uv_cxx_wrap_XXXXXX)"
        printf '#!/usr/bin/env bash\nargs=()\nfor arg in "$@"; do [[ "$arg" == -fdebug-default-version=* ]] || args+=("$arg"); done\nexec "%s" "${args[@]}"\n' "${CC:-cc}" > "$_cc_wrapper"
        printf '#!/usr/bin/env bash\nargs=()\nfor arg in "$@"; do [[ "$arg" == -fdebug-default-version=* ]] || args+=("$arg"); done\nexec "%s" "${args[@]}"\n' "${CXX:-c++}" > "$_cxx_wrapper"
        chmod +x "$_cc_wrapper" "$_cxx_wrapper"
        export CC="$_cc_wrapper" CXX="$_cxx_wrapper"
        echo "Note: wrapping compiler to filter -fdebug-default-version (unsupported by system compiler)."
    fi
fi

uv sync "${_uv_flags[@]}" --group dev --group analysis

if [[ -n "$_cc_wrapper" ]]; then
    rm -f "$_cc_wrapper" "$_cxx_wrapper"
    if [[ -n "$_saved_cc" ]]; then export CC="$_saved_cc"; else unset CC; fi
    if [[ -n "$_saved_cxx" ]]; then export CXX="$_saved_cxx"; else unset CXX; fi
fi
unset _cc_wrapper _cxx_wrapper _saved_cc _saved_cxx _sysconfig_cflags

USE_SYSTEM_MPI4PY="${UV_USE_SYSTEM_MPI4PY:-}"
if [[ -z "$USE_SYSTEM_MPI4PY" && "${LOADEDMODULES:-}" == *"mpi4py"* ]]; then
    USE_SYSTEM_MPI4PY="true"
    echo "Detected loaded mpi4py module in LOADEDMODULES; preferring system/module mpi4py."
fi

if is_true "$USE_SYSTEM_MPI4PY"; then
    if ! is_true "${UV_SYSTEM_SITE_PACKAGES:-}"; then
        echo "Warning: UV_USE_SYSTEM_MPI4PY is enabled without UV_SYSTEM_SITE_PACKAGES; system mpi4py may not be visible in .venv."
    fi

    echo "Using system/module mpi4py: removing mpi4py from .venv if present."
    uv pip uninstall -y mpi4py >/dev/null 2>&1 || true
fi

if .venv/bin/python -m ipykernel --version >/dev/null 2>&1; then
    .venv/bin/python -m ipykernel install --user --name hiraxmcmc --display-name "HIRAXmcmc"
    echo "Jupyter kernel 'HIRAXmcmc' registered."
fi

echo "Environment ready."
echo "Use: source .venv/bin/activate"
