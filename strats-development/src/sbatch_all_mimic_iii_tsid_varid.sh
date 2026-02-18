#!/usr/bin/env bash
#SBATCH --job-name=mimic_tsidvarid
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=work

set -euo pipefail
mkdir -p logs
# make sure we run from the repo src folder
cd /home/hpc/iwbn/iwbn127h/projects/STRaTS_christel/strats-development/src

# activate conda env (batch jobs don't inherit your interactive shell)
source /home/hpc/iwbn/iwbn127h/.bashrc
conda activate strats

# sanity checks (will print into logs)
which python
python -V
ls -lh main.py



DATASET="mimic_iii"
TARGET="in_hospital_mortality"
DATA_DIR="../data/processed"
SEED=2
MAX_EPOCHS=50

PCTS=(90)
MODELS=(gru grud tcn sand strats)

NUM_PCTS=${#PCTS[@]}
NUM_MODELS=${#MODELS[@]}
TOTAL=$((NUM_PCTS * NUM_MODELS))  # 45

# Map array id -> pct/model
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID not set. Submit with --array=0-$((TOTAL-1))"
  exit 1
fi
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TOTAL )); then
  echo "ERROR: task id out of range"
  exit 1
fi

pct_idx=$(( SLURM_ARRAY_TASK_ID / NUM_MODELS ))
model_idx=$(( SLURM_ARRAY_TASK_ID % NUM_MODELS ))
PCT="${PCTS[$pct_idx]}"
MODEL="${MODELS[$model_idx]}"

PKL_NAME="mimic_iii_sparsified-tsid-varid_${PCT}_${SEED}.pkl"
PKL_PATH="${DATA_DIR}/${PKL_NAME}"
FILE_TAG="${PKL_NAME%.pkl}"

if [[ ! -f "$PKL_PATH" ]]; then
  echo "ERROR: missing PKL: $PKL_PATH"
  exit 1
fi

echo "=================================="
echo "TASK_ID=${SLURM_ARRAY_TASK_ID}/${TOTAL}"
echo "MODEL=${MODEL}  PCT=${PCT}  SEED=${SEED}"
echo "FILE_TAG=${FILE_TAG}"
echo "PKL=${PKL_PATH}"
echo "=================================="

# Hyperparameters (MIMIC-III)
HID_DIM=32
DROPOUT=0.2
ATTN_DROPOUT=0.2
LR="1e-4"
NUM_LAYERS=2
NUM_HEADS=4
KERNEL_SIZE=4
R=24
M=12
HE=8

case "$MODEL" in
  gru)    HID_DIM=50;  DROPOUT=0.2; LR="0.0001" ;;
  grud)   HID_DIM=60;  DROPOUT=0.2; LR="0.0001" ;;
  tcn)    NUM_LAYERS=4; HID_DIM=128; KERNEL_SIZE=4; DROPOUT=0.1; LR="0.0001" ;;
  sand)   NUM_LAYERS=4; R=24; M=12; HID_DIM=64; NUM_HEADS=2; HE=8; DROPOUT=0.3; ATTN_DROPOUT=0.3; LR="0.0005" ;;
  strats) HID_DIM=50; M=2; NUM_HEADS=4; DROPOUT=0.2; ATTN_DROPOUT=0.2; LR="0.0005" ;;
  *) echo "ERROR: invalid model $MODEL"; exit 1 ;;
esac

python main.py \
  --dataset "$DATASET" \
  --target "$TARGET" \
  --file "$FILE_TAG" \
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
  --max_epochs "$MAX_EPOCHS" \
  --seed "$SEED"
EOF

chmod +x sbatch_all_mimic_iii_tsid_varid.sh
