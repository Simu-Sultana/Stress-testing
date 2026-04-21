#!/usr/bin/env bash
set -euo pipefail

source /home/hpc/iwbn/iwbn127h/miniconda3/etc/profile.d/conda.sh
conda activate strats

DATASET="${1:?ERROR: dataset required (physionet_2012 | mimic_iii)}"
MODEL="${2:?ERROR: model required (gru | grud | tcn | sand | strats)}"
FOLD="${3:?ERROR: fold required (0 | 1 | 2)}"
MAX_EPOCHS="${4:-50}"

TARGET_FOLDER="unbalanced"
PERTURB="unbalanced"
TASK_TARGET="in_hospital_mortality"

CV_ROOT="../data/cv_splits"
RESULTS_ROOT="../results_cv"

set_hparams () {
  local DATASET="$1"
  local MODEL="$2"

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

  if [[ "$DATASET" == "physionet_2012" ]]; then
    case "$MODEL" in
      gru)    HID_DIM=43; DROPOUT=0.2; LR="0.0001" ;;
      grud)   HID_DIM=49; DROPOUT=0.2; LR="0.0001" ;;
      tcn)    NUM_LAYERS=6; HID_DIM=64;  KERNEL_SIZE=4; DROPOUT=0.1; LR="0.0005" ;;
      sand)   NUM_LAYERS=4; R=24; M=12; HID_DIM=64; NUM_HEADS=2; HE=8; DROPOUT=0.3; ATTN_DROPOUT=0.3; LR="0.0005" ;;
      strats) HID_DIM=50; M=2;  NUM_HEADS=4; DROPOUT=0.2; ATTN_DROPOUT=0.2; LR="0.0005" ;;
      *) echo "ERROR: unknown model '$MODEL'"; exit 1 ;;
    esac
  elif [[ "$DATASET" == "mimic_iii" ]]; then
    case "$MODEL" in
      gru)    HID_DIM=50; DROPOUT=0.2; LR="0.0001" ;;
      grud)   HID_DIM=60; DROPOUT=0.2; LR="0.0001" ;;
      tcn)    NUM_LAYERS=4; HID_DIM=128; KERNEL_SIZE=4; DROPOUT=0.1; LR="0.0001" ;;
      sand)   NUM_LAYERS=4; R=24; M=12; HID_DIM=64; NUM_HEADS=2; HE=8; DROPOUT=0.3; ATTN_DROPOUT=0.3; LR="0.0005" ;;
      strats) HID_DIM=50; M=2;  NUM_HEADS=4; DROPOUT=0.2; ATTN_DROPOUT=0.2; LR="0.0005" ;;
      *) echo "ERROR: unknown model '$MODEL'"; exit 1 ;;
    esac
  else
    echo "ERROR: unknown dataset '$DATASET'"
    exit 1
  fi
}

run_one () {
  set_hparams "$DATASET" "$MODEL"

  local FOLD_DIR="${CV_ROOT}/${DATASET}_in_hospital_mortality_unbalanced"
  local FILE="${DATASET}_fold_${FOLD}"
  local PKL_PATH="${FOLD_DIR}/${FILE}.pkl"

  local RUN_DIR="${RESULTS_ROOT}/${DATASET}/${TARGET_FOLDER}/${MODEL}/${PERTURB}/fold_${FOLD}"
  mkdir -p "$RUN_DIR"
  local LOG_FILE="${RUN_DIR}/wrapper.log"

  {
    echo "=================================================="
    echo "DATASET=$DATASET"
    echo "MODEL=$MODEL"
    echo "FOLD=$FOLD"
    echo "TASK_TARGET=$TASK_TARGET"
    echo "TARGET_FOLDER=$TARGET_FOLDER"
    echo "PERTURB=$PERTURB"
    echo "FILE=$FILE"
    echo "FOLD_DIR=$FOLD_DIR"
    echo "Expect PKL: $PKL_PATH"
    echo "hid_dim=$HID_DIM dropout=$DROPOUT attn_dropout=$ATTN_DROPOUT lr=$LR num_layers=$NUM_LAYERS num_heads=$NUM_HEADS kernel_size=$KERNEL_SIZE r=$R M=$M"
    echo "max_epochs=$MAX_EPOCHS"
    echo "Host: $(hostname)"
    echo "=================================================="
  } | tee -a "$LOG_FILE"

  if [[ ! -f "$PKL_PATH" ]]; then
    echo "❌ PKL not found: $PKL_PATH" | tee -a "$LOG_FILE"
    exit 1
  fi

  PYTHON="$HOME/.conda/envs/strats/bin/python"

  "$PYTHON" main.py \
    --dataset "$DATASET" \
    --target "$TASK_TARGET" \
    --model_type "$MODEL" \
    --file "$FILE" \
    --output_dir "$RUN_DIR" \
    --output_dir_prefix "$FOLD_DIR" \
    --train_frac 1.0 \
    --max_epochs "$MAX_EPOCHS" \
    --hid_dim "$HID_DIM" \
    --dropout "$DROPOUT" \
    --attention_dropout "$ATTN_DROPOUT" \
    --lr "$LR" \
    --num_layers "$NUM_LAYERS" \
    --num_heads "$NUM_HEADS" \
    --kernel_size "$KERNEL_SIZE" \
    --r "$R" \
    --M "$M" \
    >> "$LOG_FILE" 2>&1 || { echo "❌ Python failed (see $LOG_FILE)"; exit 1; }
}

echo "Running ONE CV unbalanced experiment..."
echo "dataset=$DATASET model=$MODEL fold=$FOLD max_epochs=$MAX_EPOCHS"
echo

run_one
echo "✅ Done."