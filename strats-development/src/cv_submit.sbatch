#!/usr/bin/env bash
#SBATCH --job-name=physio_unbal_cv_allfolds
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=work
#SBATCH --array=0-209%10

set -euo pipefail

cd /home/hpc/iwbn/iwbn127h/projects/STRaTS_christel/strats-development/src

PCTS=(1 2 3 4 5 6 7 8 9 10 20 30 40 50)
MODELS=(gru grud tcn sand strats)
FOLDS=(0 1 2)

NUM_PCTS=${#PCTS[@]}
NUM_MODELS=${#MODELS[@]}
NUM_FOLDS=${#FOLDS[@]}

TASK_ID=${SLURM_ARRAY_TASK_ID}

COMBOS_PER_FOLD=$((NUM_PCTS * NUM_MODELS))

FOLD_INDEX=$((TASK_ID / COMBOS_PER_FOLD))
REMAINDER=$((TASK_ID % COMBOS_PER_FOLD))

PCT_INDEX=$((REMAINDER / NUM_MODELS))
MODEL_INDEX=$((REMAINDER % NUM_MODELS))

FOLD="${FOLDS[$FOLD_INDEX]}"
PCT="${PCTS[$PCT_INDEX]}"
MODEL="${MODELS[$MODEL_INDEX]}"

DATASET="physionet_2012"
MAX_EPOCHS="50"

echo "=============================================="
echo "SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"
echo "DATASET             = $DATASET"
echo "FOLD                = $FOLD"
echo "PCT                 = $PCT"
echo "MODEL               = $MODEL"
echo "MAX_EPOCHS          = $MAX_EPOCHS"
echo "HOST                = $(hostname)"
echo "=============================================="

bash run_unbalanced_cv.sh "$DATASET" "$MODEL" "$PCT" "$FOLD" "$MAX_EPOCHS"