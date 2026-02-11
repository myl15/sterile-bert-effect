#!/bin/bash
#SBATCH --job-name=sterile-finetune
#SBATCH --output=slurm/logs/03_finetune_%j.out
#SBATCH --error=slurm/logs/03_finetune_%j.out
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu              # Adjust for your cluster
# ============================================================================
# Steps 6-8: POS fine-tuning, evaluation, comparison, and CKA probing.
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

echo "$(date) | Starting fine-tuning and evaluation"

echo "=== Step 6: POS fine-tuning + zero-shot eval ==="
python src/models/finetune_pos.py --config configs/base_config.yaml --variant both

echo "=== Step 7: Compare results ==="
python src/analysis/compare_results.py --config configs/base_config.yaml

echo "=== Step 8: Structural probing (CKA) ==="
python src/analysis/structural_probing.py --config configs/base_config.yaml

echo "$(date) | All evaluation complete"
