#!/usr/bin/env bash
#SBATCH --job-name=mimic_iii_grud_missing
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=work
#SBATCH --array=0-9%10

set -euo pipefail

mkdir -p logs

cd /home/hpc/iwbn/iwbn127h/projects/STRaTS_christel/strats-development/src

if [[ -f "$HOME/.conda/etc/profile.d/conda.sh" ]]; then
  source "$HOME/.conda/etc/profile.d/conda.sh"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found"
  exit 1
fi

conda activate strats

echo "Using python3: $(which python3)"
python3 -c "import sys; print('Executable:', sys.executable)"
python3 -c "import pandas as pd; print('pandas OK:', pd.__version__)"

# Explicitly list all 10 missing combinations as parallel arrays
FOLDS=(0 0 0 0 1 1 1 1 2 2)
PCTS=( 1 2 3 4 1 2 3 4 1 2)

TASK_ID=${SLURM_ARRAY_TASK_ID}

FOLD="${FOLDS[$TASK_ID]}"
PCT="${PCTS[$TASK_ID]}"

DATASET="mimic_iii"
MODEL="grud"
MAX_EPOCHS="50"

echo "=============================================="
echo "SLURM_ARRAY_TASK_ID = $TASK_ID"
echo "DATASET             = $DATASET"
echo "MODEL               = $MODEL"
echo "FOLD                = $FOLD"
echo "PCT                 = $PCT"
echo "MAX_EPOCHS          = $MAX_EPOCHS"
echo "HOST                = $(hostname)"
echo "PYTHON              = $(which python3)"
echo "START_TIME          = $(date)"
echo "=============================================="

bash run_unbalanced_cv.sh "$DATASET" "$MODEL" "$PCT" "$FOLD" "$MAX_EPOCHS"

echo "=============================================="
echo "DONE"
echo "DATASET    = $DATASET"
echo "MODEL      = $MODEL"
echo "FOLD       = $FOLD"
echo "PCT        = $PCT"
echo "END_TIME   = $(date)"
echo "=============================================="