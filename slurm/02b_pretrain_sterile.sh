#!/bin/bash
#SBATCH --job-name=pretrain-sterile
#SBATCH --output=slurm/logs/02b_pretrain_sterile_%j.out
#SBATCH --error=slurm/logs/02b_pretrain_sterile_%j.out
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --qos=matrix
# ============================================================================
# Step 5b: MLM pre-training for the STERILE model.
# ============================================================================
set -euo pipefail

export TRANSFORMERS_OFFLINE=1

# --- Customize these for your cluster ---
MODULE_CUDA="cuda/12.4"
# ----------------------------------------

module load "$MODULE_CUDA"

source "$SLURM_SUBMIT_DIR/.venv/bin/activate"

cd "$SLURM_SUBMIT_DIR"

echo "$(date) | Starting MLM pre-training (sterile)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python src/models/pretrain_mlm.py --config configs/base_config.yaml --variant sterile

echo "$(date) | Pre-training (sterile) complete"
