#!/bin/bash
#SBATCH --job-name=sterile-data
#SBATCH --output=slurm/logs/01_data_prep_%j.out
#SBATCH --error=slurm/logs/01_data_prep_%j.out
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=compute          # CPU-only partition (adjust for your cluster)
# ============================================================================
# Steps 1-4: Download Wikipedia, sterilize, train tokenizers, prepare MLM data
# No GPU needed.
# ============================================================================
set -euo pipefail

# --- Initialize the module system ---
for init_script in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh \
    /usr/share/lmod/lmod/init/bash /usr/share/modules/init/bash; do
    [ -f "$init_script" ] && { source "$init_script"; break; }
done

# --- Customize these for your cluster ---
MODULE_CONDA="miniconda3"
# ----------------------------------------

module purge
module load "$MODULE_CONDA"
conda activate sterile-lang

cd "$SLURM_SUBMIT_DIR"

echo "$(date) | Starting data preparation pipeline"

echo "=== Step 1: Download Wikipedia ==="
python src/data/download_wiki.py --config configs/base_config.yaml

echo "=== Step 2: Sterilize text ==="
python src/data/sterilize.py --config configs/base_config.yaml

echo "=== Step 3: Train tokenizers ==="
python src/data/train_tokenizer.py --config configs/base_config.yaml

echo "=== Step 4: Prepare MLM datasets ==="
python src/data/prepare_mlm_data.py --config configs/base_config.yaml

echo "$(date) | Data preparation complete"
