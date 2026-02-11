#!/bin/bash
#SBATCH --job-name=sterile-pretrain
#SBATCH --output=slurm/logs/02_pretrain_%j.out
#SBATCH --error=slurm/logs/02_pretrain_%j.out
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu              # Adjust for your cluster
# ============================================================================
# Step 5: MLM pre-training for both control and sterile models.
# Requires 1 GPU.
# ============================================================================
set -euo pipefail

# --- Initialize the module system ---
for init_script in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh \
    /usr/share/lmod/lmod/init/bash /usr/share/modules/init/bash; do
    [ -f "$init_script" ] && { source "$init_script"; break; }
done

# --- Customize these for your cluster ---
MODULE_CONDA="miniconda3"
MODULE_CUDA="cuda/12.4"
# ----------------------------------------

module purge
module load "$MODULE_CONDA"
module load "$MODULE_CUDA"
conda activate sterile-lang

cd "$SLURM_SUBMIT_DIR"

echo "$(date) | Starting MLM pre-training"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python src/models/pretrain_mlm.py --config configs/base_config.yaml --variant both

echo "$(date) | Pre-training complete"
