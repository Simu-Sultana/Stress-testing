#!/usr/bin/env bash
#SBATCH --job-name=mimic_pw_grud
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=work

#SBATCH --output=/home/hpc/iwbn/iwbn127h/projects/STRaTS_christel/strats-development/src/logs/%x_%A_%a.out
#SBATCH --error=/home/hpc/iwbn/iwbn127h/projects/STRaTS_christel/strats-development/src/logs/%x_%A_%a.err

set -euo pipefail
set -x

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
cd /home/hpc/iwbn/iwbn127h/projects/STRaTS_christel/strats-development/src
mkdir -p logs

PY=/home/hpc/iwbn/iwbn127h/.conda/envs/strats/bin/python
$PY -V

DATASET="mimic_iii"
TARGET="in_hospital_mortality"
PROCESSED_DIR="../data/processed"
RESULTS_ROOT="../results"
MAX_EPOCHS=50

# ------------------------------------------------------------
# Generator (your file)
# ------------------------------------------------------------
GEN_SCRIPT="preprocess_mimic_iii_sparsified-tsid-varid.py"
if [[ ! -f "$GEN_SCRIPT" ]]; then
  echo "ERROR: generator script not found in src/: $GEN_SCRIPT"
  echo "PWD=$(pwd)"
  echo "Files matching preprocess_mimic_iii*: "
  ls -lh preprocess_mimic_iii* || true
  exit 1
fi

SEEDS=(0 2)
PCTS=(10 20 30 40 50 60 70 80 90)

NUM_SEEDS=${#SEEDS[@]}
NUM_PCTS=${#PCTS[@]}
TOTAL=$((NUM_SEEDS * NUM_PCTS))   # 18

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID not set. Submit with --array=0-$((TOTAL-1))"
  exit 1
fi
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TOTAL )); then
  echo "ERROR: task id out of range (0..$((TOTAL-1)))"
  exit 1
fi

seed_idx=$(( SLURM_ARRAY_TASK_ID / NUM_PCTS ))
pct_idx=$(( SLURM_ARRAY_TASK_ID % NUM_PCTS ))

SEED="${SEEDS[$seed_idx]}"
PCT="${PCTS[$pct_idx]}"

MODEL="grud"
FILE_TAG="mimic_iii_sparsified-tsid-varid_${PCT}_${SEED}"
PKL_PATH="${PROCESSED_DIR}/${FILE_TAG}.pkl"


PERT="sparsified-tsid-varid"
EXP_DIR="${RESULTS_ROOT}/${DATASET}/${TARGET}/${MODEL}/${PERT}/${FILE_TAG}"
CSV_PATH="${EXP_DIR}/${FILE_TAG}.csv"

echo "=================================="
echo "JOB=${SLURM_JOB_ID} TASK=${SLURM_ARRAY_TASK_ID}/${TOTAL}"
echo "SEED=${SEED} PCT=${PCT} MODEL=${MODEL}"
echo "FILE_TAG=${FILE_TAG}"
echo "PKL_PATH=${PKL_PATH}"
echo "EXP_DIR=${EXP_DIR}"
echo "CSV_PATH=${CSV_PATH}"
echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "=================================="

TMPBASE="${SLURM_TMPDIR:-/tmp}"
JOB_TMP="${TMPBASE}/mimic_pw_grud_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$JOB_TMP"

TMP_PKL="${JOB_TMP}/${FILE_TAG}.pkl"

echo "[1/3] Generating PKL into: $JOB_TMP"
$PY "$GEN_SCRIPT" \
  --data_dir "$PROCESSED_DIR" \
  --out_dir "$JOB_TMP" \
  --pct "$PCT" \
  --seed "$SEED"

echo "Contents of JOB_TMP after generation:"
ls -lh "$JOB_TMP" || true

if [[ ! -f "$TMP_PKL" ]]; then
  echo "ERROR: Generator did not produce expected file: $TMP_PKL"
  echo "It produced these files instead:"
  ls -lh "$JOB_TMP" || true
  exit 1
fi

rm -f "$PKL_PATH"
ln -s "$TMP_PKL" "$PKL_PATH"
ls -l "$PKL_PATH"

HID_DIM=60
DROPOUT=0.2
LR="0.0001"

set +e
$PY main.py \
  --dataset "$DATASET" \
  --target "$TARGET" \
  --file "$FILE_TAG" \
  --model_type "$MODEL" \
  --hid_dim "$HID_DIM" \
  --dropout "$DROPOUT" \
  --lr "$LR" \
  --train_frac 1 \
  --max_epochs "$MAX_EPOCHS" \
  --patience 10 \
  --seed "$SEED"
STATUS=$?
set -e

if [[ $STATUS -eq 0 && -f "$CSV_PATH" ]]; then
  echo "[3/3] Success + CSV found. Cleaning up PKL."
  rm -f "$PKL_PATH"   
  rm -f "$TMP_PKL"    
  rmdir "$JOB_TMP" 2>/dev/null || true
  echo "Cleanup done."
else
  echo "Run failed or CSV missing -> keeping PKL for debugging."
  echo "STATUS=$STATUS"
  echo "PKL symlink: $PKL_PATH"
  echo "TMP PKL    : $TMP_PKL"
  echo "CSV        : $CSV_PATH"
  exit $STATUS
fi