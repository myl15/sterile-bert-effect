#!/bin/bash
# ============================================================================
# Submit the full pipeline with job dependencies.
# Control and sterile variants run in parallel at each stage.
#
# Usage:
#   bash slurm/run_all.sh                  # submit all jobs (steps 1-4)
#   bash slurm/run_all.sh --from-step 2   # skip data prep, start at pre-training
#   bash slurm/run_all.sh --from-step 3   # skip to fine-tuning (both variants)
#   bash slurm/run_all.sh --from-step 4   # run analysis only
# ============================================================================
set -euo pipefail

# PREREQUISITE: Run the following on the login node first (requires internet):
#   bash slurm/setup_env.sh
#   bash scripts/00_download_cache.sh

FROM_STEP=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from-step)
            FROM_STEP="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: bash slurm/run_all.sh [--from-step 1|2|3|4]" >&2
            exit 1
            ;;
    esac
done

if [[ "$FROM_STEP" -lt 1 || "$FROM_STEP" -gt 4 ]]; then
    echo "Error: --from-step must be 1, 2, 3, or 4" >&2
    exit 1
fi

echo "Submitting pipeline from step $FROM_STEP..."

# Track last job for each variant independently
PREV_CONTROL=""
PREV_STERILE=""
declare -a LOG_FILES=()

# ── Step 1: Data preparation (CPU) ───────────────────────────────────────────
if [[ "$FROM_STEP" -le 1 ]]; then
    JOB1=$(sbatch --parsable slurm/01_data_prep.sh)
    echo "  Submitted data prep:        job $JOB1"
    PREV_CONTROL="$JOB1"
    PREV_STERILE="$JOB1"
    LOG_FILES+=("slurm/logs/01_data_prep_${JOB1}.out")
fi

# ── Step 2: Pre-training (parallel) ──────────────────────────────────────────
if [[ "$FROM_STEP" -le 2 ]]; then
    if [[ -n "$PREV_CONTROL" ]]; then
        JOB2A=$(sbatch --parsable --dependency=afterok:$PREV_CONTROL slurm/02a_pretrain_control.sh)
        echo "  Submitted pretrain control: job $JOB2A (after $PREV_CONTROL)"
    else
        JOB2A=$(sbatch --parsable slurm/02a_pretrain_control.sh)
        echo "  Submitted pretrain control: job $JOB2A"
    fi

    if [[ -n "$PREV_STERILE" ]]; then
        JOB2B=$(sbatch --parsable --dependency=afterok:$PREV_STERILE slurm/02b_pretrain_sterile.sh)
        echo "  Submitted pretrain sterile: job $JOB2B (after $PREV_STERILE)"
    else
        JOB2B=$(sbatch --parsable slurm/02b_pretrain_sterile.sh)
        echo "  Submitted pretrain sterile: job $JOB2B"
    fi

    PREV_CONTROL="$JOB2A"
    PREV_STERILE="$JOB2B"
    LOG_FILES+=("slurm/logs/02a_pretrain_control_${JOB2A}.out" "slurm/logs/02b_pretrain_sterile_${JOB2B}.out")
fi

# ── Step 3: Fine-tuning (parallel, each variant depends on its own pretrain) ─
if [[ "$FROM_STEP" -le 3 ]]; then
    if [[ -n "$PREV_CONTROL" ]]; then
        JOB3A=$(sbatch --parsable --dependency=afterok:$PREV_CONTROL slurm/03a_finetune_control.sh)
        echo "  Submitted finetune control: job $JOB3A (after $PREV_CONTROL)"
    else
        JOB3A=$(sbatch --parsable slurm/03a_finetune_control.sh)
        echo "  Submitted finetune control: job $JOB3A"
    fi

    if [[ -n "$PREV_STERILE" ]]; then
        JOB3B=$(sbatch --parsable --dependency=afterok:$PREV_STERILE slurm/03b_finetune_sterile.sh)
        echo "  Submitted finetune sterile: job $JOB3B (after $PREV_STERILE)"
    else
        JOB3B=$(sbatch --parsable slurm/03b_finetune_sterile.sh)
        echo "  Submitted finetune sterile: job $JOB3B"
    fi

    PREV_CONTROL="$JOB3A"
    PREV_STERILE="$JOB3B"
    LOG_FILES+=("slurm/logs/03a_finetune_control_${JOB3A}.out" "slurm/logs/03b_finetune_sterile_${JOB3B}.out")
fi

# ── Step 4: Analysis (waits for BOTH finetune variants) ──────────────────────
if [[ -n "$PREV_CONTROL" && -n "$PREV_STERILE" ]]; then
    JOB4=$(sbatch --parsable --dependency=afterok:$PREV_CONTROL:$PREV_STERILE slurm/04_analyze.sh)
    echo "  Submitted analysis:         job $JOB4 (after $PREV_CONTROL,$PREV_STERILE)"
else
    JOB4=$(sbatch --parsable slurm/04_analyze.sh)
    echo "  Submitted analysis:         job $JOB4"
fi
LOG_FILES+=("slurm/logs/04_analyze_${JOB4}.out")

echo ""
echo "Pipeline submitted! Monitor with:"
echo "  squeue -u \$USER"
for log in "${LOG_FILES[@]}"; do
    echo "  tail -f $log"
done
