#!/usr/bin/env bash
#SBATCH --job-name=all_mimic_grud_missing
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=work
#SBATCH --array=0-134%20

set -euo pipefail
mkdir -p logs

cd /home/hpc/iwbn/iwbn127h/projects/STRaTS_christel/strats-development/src

source /home/hpc/iwbn/iwbn127h/miniconda3/etc/profile.d/conda.sh
conda activate strats

# Format: MODEL PERTURBATION PCT SEED
COMBOS=(
  "grud sparsified-tsid-varid 100 3"
  "grud sparsified-tsid-varid 100 4"
  "grud sparsified-tsid-varid 100 5"
  "grud sparsified-tsid-varid 100 6"
  "grud sparsified-tsid-varid 100 7"
  "grud sparsified-tsid-varid 100 8"
  "grud sparsified-tsid-varid 100 9"
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