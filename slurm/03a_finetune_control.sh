#!/bin/bash
#SBATCH --job-name=finetune-control
#SBATCH --output=slurm/logs/03a_finetune_control_%j.out
#SBATCH --error=slurm/logs/03a_finetune_control_%j.out
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --qos=matrix
# ============================================================================
# Step 6a: POS fine-tuning + zero-shot eval for the CONTROL model.
# ============================================================================
set -euo pipefail

export TRANSFORMERS_OFFLINE=1

# --- Customize these for your cluster ---
MODULE_CUDA="cuda/12.4"
# ----------------------------------------

module load "$MODULE_CUDA"

source "$SLURM_SUBMIT_DIR/.venv/bin/activate"

cd "$SLURM_SUBMIT_DIR"

echo "$(date) | Starting POS fine-tuning (control)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python src/models/finetune_pos.py --config configs/base_config.yaml --variant control

echo "$(date) | Fine-tuning (control) complete"
