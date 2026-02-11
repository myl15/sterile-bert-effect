#!/bin/bash
# ============================================================================
# Submit the full pipeline with job dependencies.
# Each step waits for the previous one to complete successfully.
#
# Usage:  bash slurm/run_all.sh
# ============================================================================
set -euo pipefail

# PREREQUISITE: Run the following on the login node first (requires internet):
#   bash scripts/00_download_cache.sh
#   bash slurm/setup_env.sh

echo "Submitting pipeline..."

# Step 1-4: Data preparation (CPU)
JOB1=$(sbatch --parsable slurm/01_data_prep.sh)
echo "  Submitted data prep:    job $JOB1"

# Step 5: Pre-training (GPU) — depends on data prep
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/02_pretrain.sh)
echo "  Submitted pre-training: job $JOB2 (after $JOB1)"

# Steps 6-8: Fine-tune + eval (GPU) — depends on pre-training
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 slurm/03_finetune_eval.sh)
echo "  Submitted fine-tune:    job $JOB3 (after $JOB2)"

echo ""
echo "Pipeline submitted! Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f slurm/logs/01_data_prep_${JOB1}.out"
echo "  tail -f slurm/logs/02_pretrain_${JOB2}.out"
echo "  tail -f slurm/logs/03_finetune_${JOB3}.out"
