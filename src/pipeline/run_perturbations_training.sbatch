#!/usr/bin/env bash
#SBATCH --job-name=perturb_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4

set -euo pipefail
mkdir -p logs

# (Optional) load modules if your cluster uses them
# module load anaconda cuda

# Make sure we are in the directory where this sbatch file lives
cd "$SLURM_SUBMIT_DIR"

# Run your loop script
chmod +x run_experiments.sh run_perturbations_training.sh
./run_perturbations_training.sh
