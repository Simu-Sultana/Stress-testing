#!/usr/bin/env bash
#SBATCH --job-name=strats_physio_array
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-29%16
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
mkdir -p logs

cd /home/hpc/iwbn/iwbn127h/projects/STRaTS_christel/strats-development/src
chmod +x run_perturbations_training.sh

# If you need conda:
source ~/miniconda3/etc/profile.d/conda.sh
conda activate strats
echo "Conda env: $CONDA_DEFAULT_ENV"
which python
python --version

DATASET="physionet_2012"
MAX_EPOCHS=50

SEEDS=(0 2)
PERTURBS=(subsampled sparsified-patientwise sparsified-tsid-varid)
MODELS=(gru grud tcn sand strats)

# total tasks = 2*3*5 = 30 => array 0-29
i="${SLURM_ARRAY_TASK_ID}"

seed_idx=$(( i / ( ${#PERTURBS[@]} * ${#MODELS[@]} ) ))
rem=$(( i % ( ${#PERTURBS[@]} * ${#MODELS[@]} ) ))
pert_idx=$(( rem / ${#MODELS[@]} ))
model_idx=$(( rem % ${#MODELS[@]} ))

SEED="${SEEDS[$seed_idx]}"
PERTURB="${PERTURBS[$pert_idx]}"
MODEL="${MODELS[$model_idx]}"

echo "Node: $(hostname)"
echo "Job: ${SLURM_JOB_ID}  ArrayTask: ${SLURM_ARRAY_TASK_ID}"
echo "Running: dataset=$DATASET perturb=$PERTURB model=$MODEL seed=$SEED max_epochs=$MAX_EPOCHS"
nvidia-smi || true

# Call bash in single-combo mode (matches your bash file)
./run_perturbations_training.sh "$DATASET" "$PERTURB" "$MODEL" "$SEED" "$MAX_EPOCHS"
