#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?ERROR: dataset required (physionet_2012 | mimic_iii)}"
MODEL="${2:?ERROR: model required (gru | grud | tcn | sand | strats)}"
PCT="${3:?ERROR: pct required}"
FOLD="${4:?ERROR: fold required}"
MAX_EPOCHS="${5:-50}"

DATA_DIR="../data/processed"
OUT_DIR="../data/processed"

TARGET="unbalanced"
PERTURB="unbalanced"
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

ensure_pkl () {
  local FILE="${DATASET}_fold_${FOLD}_${PERTURB}_${PCT}"
  local PKL_PATH="${OUT_DIR}/${FILE}.pkl"

  if [[ -f "$PKL_PATH" ]]; then
    echo "✔ PKL already exists: $PKL_PATH"
    return 0
  fi

  echo "PKL missing. Generating: $PKL_PATH"
  python3 preprocess_unbalanced_cv.py \
    --dataset "$DATASET" \
    --data_dir "$DATA_DIR" \
    --out_dir "$OUT_DIR" \
    --fold "$FOLD" \
    --pct "$PCT"
}

run_one () {
  set_hparams "$DATASET" "$MODEL"

  local FILE="${DATASET}_fold_${FOLD}_${PERTURB}_${PCT}"
  local PKL_PATH="${OUT_DIR}/${FILE}.pkl"

  local RUN_DIR="${RESULTS_ROOT}/${DATASET}/${TARGET}/${MODEL}/${PERTURB}/fold_${FOLD}/${FILE}"
  mkdir -p "$RUN_DIR"
  local LOG_FILE="${RUN_DIR}/wrapper.log"

  {
    echo "=================================================="
    echo "DATASET=$DATASET  TARGET=$TARGET  MODEL=$MODEL"
    echo "PERTURB=$PERTURB  FOLD=$FOLD  PCT=$PCT"
    echo "FILE=$FILE"
    echo "RUN_DIR=$RUN_DIR"
    echo "Expect PKL: $PKL_PATH"
    echo "hid_dim=$HID_DIM dropout=$DROPOUT attn_dropout=$ATTN_DROPOUT lr=$LR"
    echo "num_layers=$NUM_LAYERS num_heads=$NUM_HEADS kernel_size=$KERNEL_SIZE r=$R M=$M"
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
    --target "$TARGET" \
    --model_type "$MODEL" \
    --file "$FILE" \
    --output_dir "$RUN_DIR" \
    --output_dir_prefix "$RUN_DIR" \
    --train_frac 0.8 \
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

echo "Running ONE fold-based UNBALANCED training job..."
echo "dataset=$DATASET model=$MODEL pct=$PCT fold=$FOLD max_epochs=$MAX_EPOCHS"
echo

ensure_pkl
run_one

PKL_PATH="${OUT_DIR}/${DATASET}_fold_${FOLD}_${PERTURB}_${PCT}.pkl"
rm -f "$PKL_PATH"
echo "🗑 Deleted PKL: $PKL_PATH"
echo "✅ Done."