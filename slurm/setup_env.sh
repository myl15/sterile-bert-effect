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

# --- Customize these for your cluster ---
MODULE_CONDA="miniconda3"   # or: anaconda, anaconda3, miniforge, etc.
MODULE_CUDA="cuda/12.4"     # or: cuda/12.1, cuda/11.8, etc.
# ----------------------------------------

echo "=== Loading modules ==="
module purge
module load "$MODULE_CONDA"
module load "$MODULE_CUDA"

echo "=== Creating conda environment ==="
conda env create -f environment.yml || {
    echo "environment.yml failed, installing manually..."
    conda create -n sterile-lang python=3.11 -y
    conda activate sterile-lang
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    conda install numpy scipy scikit-learn matplotlib seaborn pyyaml tqdm tensorboard pytest -y
    pip install "transformers>=4.40" "datasets>=3.0" "tokenizers>=0.19" "accelerate>=0.30" evaluate seqeval pandas
}

echo "=== Verifying GPU access ==="
conda activate sterile-lang
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo "=== Setup complete ==="
