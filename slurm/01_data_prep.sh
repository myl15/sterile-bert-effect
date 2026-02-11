#!/bin/bash
#SBATCH --job-name=sterile-data
#SBATCH --output=slurm/logs/01_data_prep_%j.out
#SBATCH --error=slurm/logs/01_data_prep_%j.out
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
# ============================================================================
# Steps 2-4: Sterilize text, train tokenizers, prepare MLM data
# (Step 1 Wikipedia download runs on the login node via scripts/00_download_cache.sh)
# No GPU needed.
# ============================================================================
set -euo pipefail

export TRANSFORMERS_OFFLINE=1

source "$SLURM_SUBMIT_DIR/.venv/bin/activate"

cd "$SLURM_SUBMIT_DIR"

echo "$(date) | Starting data preparation pipeline"

echo "=== Step 2: Sterilize text ==="
python src/data/sterilize.py --config configs/base_config.yaml

echo "=== Step 3: Train tokenizers ==="
python src/data/train_tokenizer.py --config configs/base_config.yaml

echo "=== Step 4: Prepare MLM datasets ==="
python src/data/prepare_mlm_data.py --config configs/base_config.yaml

echo "$(date) | Data preparation complete"
