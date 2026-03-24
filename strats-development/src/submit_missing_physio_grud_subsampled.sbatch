#!/usr/bin/env bash
#SBATCH --job-name=physio_grud_sub_missing
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=work
#SBATCH --array=0-12

set -euo pipefail
mkdir -p logs
cd /home/hpc/iwbn/iwbn127h/projects/STRaTS_christel/strats-development/src

source /home/hpc/iwbn/iwbn127h/.bashrc
conda activate strats

MAX_EPOCHS=50
DATASET="physionet_2012"
PERTURB="subsampled"
MODEL="grud"

COMBOS=(
"1 7" "2 7" "5 7" "10 7" "20 7" "30 7" "40 7" "50 7" "60 7" "70 7" "80 7" "90 7" "100 7"
)

pair="${COMBOS[$SLURM_ARRAY_TASK_ID]}"
read -r PCT SEED <<< "$pair"

TMP_SCRIPT=$(mktemp /tmp/run_perturbations_training.XXXXXX.sh)
cp ./run_perturbations_training.sh "$TMP_SCRIPT"
sed -i "s/^PCTS=(.*)$/PCTS=($PCT)/" "$TMP_SCRIPT"
chmod +x "$TMP_SCRIPT"

"$TMP_SCRIPT" "$DATASET" "$PERTURB" "$MODEL" "$SEED" "$MAX_EPOCHS"