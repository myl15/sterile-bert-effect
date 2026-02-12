#!/bin/bash
#SBATCH --job-name=sterile-analyze
#SBATCH --output=slurm/logs/04_analyze_%j.out
#SBATCH --error=slurm/logs/04_analyze_%j.out
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --qos=matrix
# ============================================================================
# Steps 7-8: Compare results + CKA structural probing.
# Requires both control and sterile finetune jobs to have completed.
# ============================================================================
set -euo pipefail

export TRANSFORMERS_OFFLINE=1

# --- Customize these for your cluster ---
MODULE_CUDA="cuda/12.4"
# ----------------------------------------

module load "$MODULE_CUDA"

source "$SLURM_SUBMIT_DIR/.venv/bin/activate"

cd "$SLURM_SUBMIT_DIR"

echo "$(date) | Starting analysis"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

echo "=== Step 7: Compare results ==="
python src/analysis/compare_results.py --config configs/base_config.yaml

echo "=== Step 8: Structural probing (CKA) ==="
python src/analysis/structural_probing.py --config configs/base_config.yaml

echo "$(date) | Analysis complete"
