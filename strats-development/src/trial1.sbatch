#!/usr/bin/env bash
#SBATCH --job-name=all_mimic_grud_missing
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=work
#SBATCH --array=0-134%50

set -euo pipefail
mkdir -p logs

cd /home/hpc/iwbn/iwbn127h/projects/STRaTS_christel/strats-development/src

source /home/hpc/iwbn/iwbn127h/miniconda3/etc/profile.d/conda.sh
conda activate strats

# Format: MODEL PERTURBATION PCT SEED
COMBOS=(
  "grud sparsified-tsid-varid 1 0"
  "grud sparsified-tsid-varid 1 1"
  "grud sparsified-tsid-varid 1 2"
  "grud sparsified-tsid-varid 1 3"
  "grud sparsified-tsid-varid 1 4"
  "grud sparsified-tsid-varid 1 5"
  "grud sparsified-tsid-varid 1 6"
  "grud sparsified-tsid-varid 1 7"
  "grud sparsified-tsid-varid 1 8"
  "grud sparsified-tsid-varid 1 9"
  "grud sparsified-tsid-varid 2 0"
  "grud sparsified-tsid-varid 2 1"
  "grud sparsified-tsid-varid 2 2"
  "grud sparsified-tsid-varid 2 3"
  "grud sparsified-tsid-varid 2 4"
  "grud sparsified-tsid-varid 2 5"
  "grud sparsified-tsid-varid 2 6"
  "grud sparsified-tsid-varid 2 7"
  "grud sparsified-tsid-varid 2 8"
  "grud sparsified-tsid-varid 2 9"
  "grud sparsified-tsid-varid 5 0"
  "grud sparsified-tsid-varid 5 1"
  "grud sparsified-tsid-varid 5 2"
  "grud sparsified-tsid-varid 5 3"
  "grud sparsified-tsid-varid 5 4"
  "grud sparsified-tsid-varid 5 5"
  "grud sparsified-tsid-varid 5 6"
  "grud sparsified-tsid-varid 5 7"
  "grud sparsified-tsid-varid 5 8"
  "grud sparsified-tsid-varid 5 9"
  "grud sparsified-tsid-varid 10 1"
  "grud sparsified-tsid-varid 10 3"
  "grud sparsified-tsid-varid 10 4"
  "grud sparsified-tsid-varid 10 5"
  "grud sparsified-tsid-varid 10 6"
  "grud sparsified-tsid-varid 10 7"
  "grud sparsified-tsid-varid 10 8"
  "grud sparsified-tsid-varid 10 9"
  "grud sparsified-tsid-varid 20 1"
  "grud sparsified-tsid-varid 20 3"
  "grud sparsified-tsid-varid 20 4"
  "grud sparsified-tsid-varid 20 5"
  "grud sparsified-tsid-varid 20 6"
  "grud sparsified-tsid-varid 20 7"
  "grud sparsified-tsid-varid 20 8"
  "grud sparsified-tsid-varid 20 9"
  "grud sparsified-tsid-varid 30 1"
  "grud sparsified-tsid-varid 30 3"
  "grud sparsified-tsid-varid 30 4"
  "grud sparsified-tsid-varid 30 5"
  "grud sparsified-tsid-varid 30 6"
  "grud sparsified-tsid-varid 30 7"
  "grud sparsified-tsid-varid 30 8"
  "grud sparsified-tsid-varid 30 9"
  "grud sparsified-tsid-varid 40 1"
  "grud sparsified-tsid-varid 40 3"
  "grud sparsified-tsid-varid 40 4"
  "grud sparsified-tsid-varid 40 5"
  "grud sparsified-tsid-varid 40 6"
  "grud sparsified-tsid-varid 40 7"
  "grud sparsified-tsid-varid 40 8"
  "grud sparsified-tsid-varid 40 9"
  "grud sparsified-tsid-varid 50 1"
  "grud sparsified-tsid-varid 50 3"
  "grud sparsified-tsid-varid 50 4"
  "grud sparsified-tsid-varid 50 5"
  "grud sparsified-tsid-varid 50 6"
  "grud sparsified-tsid-varid 50 7"
  "grud sparsified-tsid-varid 50 8"
  "grud sparsified-tsid-varid 50 9"
  "grud sparsified-tsid-varid 60 1"
  "grud sparsified-tsid-varid 60 3"
  "grud sparsified-tsid-varid 60 4"
  "grud sparsified-tsid-varid 60 5"
  "grud sparsified-tsid-varid 60 6"
  "grud sparsified-tsid-varid 60 7"
  "grud sparsified-tsid-varid 60 8"
  "grud sparsified-tsid-varid 60 9"
  "grud sparsified-tsid-varid 70 1"
  "grud sparsified-tsid-varid 70 3"
  "grud sparsified-tsid-varid 70 4"
  "grud sparsified-tsid-varid 70 5"
  "grud sparsified-tsid-varid 70 6"
  "grud sparsified-tsid-varid 70 7"
  "grud sparsified-tsid-varid 70 8"
  "grud sparsified-tsid-varid 70 9"
  "grud sparsified-tsid-varid 80 1"
  "grud sparsified-tsid-varid 80 3"
  "grud sparsified-tsid-varid 80 4"
  "grud sparsified-tsid-varid 80 5"
  "grud sparsified-tsid-varid 80 6"
  "grud sparsified-tsid-varid 80 7"
  "grud sparsified-tsid-varid 80 8"
  "grud sparsified-tsid-varid 80 9"
  "grud sparsified-tsid-varid 90 1"
  "grud sparsified-tsid-varid 90 3"
  "grud sparsified-tsid-varid 90 4"
  "grud sparsified-tsid-varid 90 5"
  "grud sparsified-tsid-varid 90 6"
  "grud sparsified-tsid-varid 90 7"
  "grud sparsified-tsid-varid 90 8"
  "grud sparsified-tsid-varid 90 9"
  "grud sparsified-tsid-varid 100 0"
  "grud sparsified-tsid-varid 100 1"
  "grud sparsified-tsid-varid 100 2"
  "grud sparsified-tsid-varid 100 3"
  "grud sparsified-tsid-varid 100 4"
  "grud sparsified-tsid-varid 100 5"
  "grud sparsified-tsid-varid 100 6"
  "grud sparsified-tsid-varid 100 7"
  "grud sparsified-tsid-varid 100 8"
  "grud sparsified-tsid-varid 100 9"

  "grud subsampled 1 0"
  "grud subsampled 1 1"
  "grud subsampled 1 2"
  "grud subsampled 1 3"
  "grud subsampled 1 5"
  "grud subsampled 1 6"
  "grud subsampled 1 7"
  "grud subsampled 1 9"
  "grud subsampled 2 2"
  "grud subsampled 2 3"
  "grud subsampled 2 5"
  "grud subsampled 5 8"
  "grud subsampled 50 7"
  "grud subsampled 70 5"
  "grud subsampled 70 9"
  "grud subsampled 80 9"
  "grud subsampled 90 1"
  "grud subsampled 90 5"
  "grud subsampled 100 1"
  "grud subsampled 100 2"
  "grud subsampled 100 3"
  "grud subsampled 100 4"
  "grud subsampled 100 7"
)

read -r MODEL PERTURBATION PCT SEED <<< "${COMBOS[$SLURM_ARRAY_TASK_ID]}"

echo "Running:"
echo "dataset=mimic_iii"
echo "target=in_hospital_mortality"
echo "model=$MODEL"
echo "perturbation=$PERTURBATION"
echo "pct=$PCT"
echo "seed=$SEED"

DATASET="mimic_iii"

# Hyperparameters
HID_DIM=60
DROPOUT=0.2
ATTN_DROPOUT=0.2
LR="0.0001"
NUM_LAYERS=2
NUM_HEADS=4
KERNEL_SIZE=4
R=24
M=12

PREPROCESS_SCRIPT="preprocess_${DATASET}_${PERTURBATION}.py"
python "$PREPROCESS_SCRIPT" \
  --data_dir ../data/processed \
  --out_dir ../data/processed \
  --seed "$SEED" \
  --pct "$PCT"

python main.py \
  --dataset "$DATASET" \
  --target "in_hospital_mortality" \
  --file "${DATASET}_${PERTURBATION}_${PCT}_${SEED}" \
  --model_type "$MODEL" \
  --hid_dim "$HID_DIM" \
  --dropout "$DROPOUT" \
  --attention_dropout "$ATTN_DROPOUT" \
  --lr "$LR" \
  --num_layers "$NUM_LAYERS" \
  --num_heads "$NUM_HEADS" \
  --kernel_size "$KERNEL_SIZE" \
  --r "$R" \
  --M "$M" \
  --train_frac 1 \
  --max_epochs 50 \
  --seed "$SEED"

rm -f ../data/processed/"${DATASET}_${PERTURBATION}_${PCT}_${SEED}"*