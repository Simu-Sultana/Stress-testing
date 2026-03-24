#!/usr/bin/env bash
#SBATCH --job-name=all_sparse_array
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-199%50
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
mkdir -p logs

cd /home/hpc/iwbn/iwbn127h/projects/STRaTS_christel/strats-development/src

chmod +x run_perturbations_training.sh

# Load conda
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found"
  exit 1
fi

conda activate strats

echo "======================================"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Conda env: $CONDA_DEFAULT_ENV"
which python
python --version
nvidia-smi || true
echo "======================================"

MAX_EPOCHS=50

DATASETS=(physionet_2012 mimic_iii)
PERTURBS=(subsampled sparsified-tsid-varid)
MODELS=(gru grud tcn sand strats)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

i="${SLURM_ARRAY_TASK_ID}"

n_datasets=${#DATASETS[@]}
n_perts=${#PERTURBS[@]}
n_models=${#MODELS[@]}
n_seeds=${#SEEDS[@]}

total_jobs=$(( n_datasets * n_perts * n_models * n_seeds ))

if (( i >= total_jobs )); then
  echo "Array index $i is out of range for total_jobs=$total_jobs"
  exit 1
fi

model_idx=$(( i % n_models ))
tmp=$(( i / n_models ))

seed_idx=$(( tmp % n_seeds ))
tmp=$(( tmp / n_seeds ))

pert_idx=$(( tmp % n_perts ))
dataset_idx=$(( tmp / n_perts ))

DATASET="${DATASETS[$dataset_idx]}"
PERTURB="${PERTURBS[$pert_idx]}"
MODEL="${MODELS[$model_idx]}"
SEED="${SEEDS[$seed_idx]}"

echo "Running:"
echo "  DATASET=$DATASET"
echo "  PERTURB=$PERTURB"
echo "  MODEL=$MODEL"
echo "  SEED=$SEED"
echo "  MAX_EPOCHS=$MAX_EPOCHS"
echo

./run_perturbations_training.sh "$DATASET" "$PERTURB" "$MODEL" "$SEED" "$MAX_EPOCHS"