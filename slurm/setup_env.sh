#!/bin/bash
# ============================================================================
# One-time environment setup for the supercomputer.
# Run this interactively (NOT via sbatch): bash slurm/setup_env.sh
# ============================================================================
set -euo pipefail

# --- Initialize the module system (try common locations) ---
if ! command -v module &> /dev/null; then
    for init_script in \
        /etc/profile.d/modules.sh \
        /etc/profile.d/lmod.sh \
        /usr/share/lmod/lmod/init/bash \
        /usr/share/modules/init/bash \
        /opt/modules/init/bash; do
        if [ -f "$init_script" ]; then
            source "$init_script"
            break
        fi
    done
fi

# --- Install uv if not already available ---
if ! command -v uv &> /dev/null; then
    echo "=== Installing uv ==="
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "=== uv version: $(uv --version) ==="

# --- Create virtual environment and install all dependencies ---
echo "=== Installing dependencies with uv sync ==="
cd "$(dirname "$0")/.."
uv sync

echo "=== Setup complete ==="
echo "To activate the environment manually: source .venv/bin/activate"
