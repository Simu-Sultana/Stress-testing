#!/usr/bin/env bash
# =============================================================================
# run_physionet_unbalanced.sh
#
# Generates PhysioNet unbalanced PKLs and launches training jobs for all
# combinations of fold × pct × model, with a concurrency limit of N parallel
# jobs at a time.
#
# Usage:
#   bash run_physionet_unbalanced.sh
#
# Config (edit the block below):
#   FOLDS       — which folds to run
#   PCTS        — which pct values to run
#   MODELS      — which models to run
#   MAX_JOBS    — max parallel jobs at a time
#   MAX_EPOCHS  — training epochs
# =============================================================================
set -euo pipefail

# ── User config ───────────────────────────────────────────────────────────────
FOLDS=(0 1 2)
PCTS=(1 2 3 4 5 6 7 8 9 10 20 30 40 50)
MODELS=(gru grud tcn sand strats)
MAX_JOBS=20
MAX_EPOCHS=50

DATASET="physionet_2012"
TARGET="unbalanced"
PERTURB="unbalanced"

# Paths — adjust if your layout differs
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${REPO_DIR}/data/processed"
RESULTS_ROOT="${REPO_DIR}/results_cv"
SRC_DIR="${REPO_DIR}/src"
PREPROCESS_SCRIPT="${SRC_DIR}/preprocess_unbalanced_cv.py"
MAIN_SCRIPT="${SRC_DIR}/main.py"
PYTHON="$HOME/.conda/envs/strats/bin/python"
LOG_ROOT="${RESULTS_ROOT}/${DATASET}/unbalanced"

# ── Hyperparameters per model ─────────────────────────────────────────────────
get_hparams () {
  local MODEL="$1"
  # Defaults
  HID_DIM=32; DROPOUT=0.2; ATTN_DROPOUT=0.2; LR="1e-4"
  NUM_LAYERS=2; NUM_HEADS=4; KERNEL_SIZE=4; R=24; M=12; HE=8

  case "$MODEL" in
    gru)    HID_DIM=43;  DROPOUT=0.2; LR="0.0001" ;;
    grud)   HID_DIM=49;  DROPOUT=0.2; LR="0.0001" ;;
    tcn)    NUM_LAYERS=6; HID_DIM=64;  KERNEL_SIZE=4; DROPOUT=0.1; LR="0.0005" ;;
    sand)   NUM_LAYERS=4; R=24; M=12; HID_DIM=64; NUM_HEADS=2; HE=8; DROPOUT=0.3; ATTN_DROPOUT=0.3; LR="0.0005" ;;
    strats) HID_DIM=50;  M=2;  NUM_HEADS=4; DROPOUT=0.2; ATTN_DROPOUT=0.2; LR="0.0005" ;;
    *) echo "ERROR: unknown model '$MODEL'"; exit 1 ;;
  esac
}

# ── Concurrency helpers ───────────────────────────────────────────────────────
# Waits until the number of background jobs drops below MAX_JOBS.
wait_for_slot () {
  while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
    sleep 2
  done
}

# ── PKL generation ────────────────────────────────────────────────────────────
generate_pkl () {
  local FOLD="$1"
  local PCT="$2"
  local PKL_PATH="${DATA_DIR}/${DATASET}_fold_${FOLD}_unbalanced_${PCT}.pkl"

  if [[ -f "$PKL_PATH" ]]; then
    echo "[PKL] Already exists, skipping: $(basename $PKL_PATH)"
    return 0
  fi

  echo "[PKL] Generating: fold=${FOLD} pct=${PCT}"
  "$PYTHON" "$PREPROCESS_SCRIPT" \
    --dataset  "$DATASET" \
    --data_dir "$DATA_DIR" \
    --out_dir  "$DATA_DIR" \
    --fold     "$FOLD" \
    --pct      "$PCT" \
  

  if [[ ! -f "$PKL_PATH" ]]; then
    echo "ERROR: PKL generation failed for fold=${FOLD} pct=${PCT}"
    exit 1
  fi
  echo "[PKL] Done: $(basename $PKL_PATH)"
}

# ── Single training job ───────────────────────────────────────────────────────
run_training () {
  local FOLD="$1"
  local PCT="$2"
  local MODEL="$3"

  get_hparams "$MODEL"

  local FILE="${DATASET}_fold_${FOLD}_unbalanced_${PCT}"
  local PKL_PATH="${DATA_DIR}/${FILE}.pkl"
  local RUN_DIR="${RESULTS_ROOT}/${DATASET}/unbalanced/${MODEL}/unbalanced/fold_${FOLD}/${FILE}"
  local LOG_FILE="${RUN_DIR}/wrapper.log"

  mkdir -p "$RUN_DIR"

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
    echo "ERROR: PKL not found: $PKL_PATH" | tee -a "$LOG_FILE"
    return 1
  fi

  "$PYTHON" "$MAIN_SCRIPT" \
    --dataset          "$DATASET" \
    --target           "$TARGET" \
    --model_type       "$MODEL" \
    --file             "$FILE" \
    --output_dir       "$RUN_DIR" \
    --output_dir_prefix "$DATA_DIR" \
    --train_frac       1.0 \
    --max_epochs       "$MAX_EPOCHS" \
    --hid_dim          "$HID_DIM" \
    --dropout          "$DROPOUT" \
    --attention_dropout "$ATTN_DROPOUT" \
    --lr               "$LR" \
    --num_layers       "$NUM_LAYERS" \
    --num_heads        "$NUM_HEADS" \
    --kernel_size      "$KERNEL_SIZE" \
    --r                "$R" \
    --M                "$M" \
    >> "$LOG_FILE" 2>&1 \
    && echo "[DONE] fold=${FOLD} pct=${PCT} model=${MODEL}" \
    || echo "[FAILED] fold=${FOLD} pct=${PCT} model=${MODEL} — see ${LOG_FILE}"
}

export -f run_training get_hparams wait_for_slot
export DATASET TARGET PERTURB MAX_EPOCHS DATA_DIR RESULTS_ROOT PYTHON MAIN_SCRIPT

# ── Main loop ─────────────────────────────────────────────────────────────────
source /home/hpc/iwbn/iwbn127h/miniconda3/etc/profile.d/conda.sh
conda activate strats

echo "============================================================"
echo " PhysioNet Unbalanced CV — Full Experiment"
echo " Folds:      ${FOLDS[*]}"
echo " PCTs:       ${PCTS[*]}"
echo " Models:     ${MODELS[*]}"
echo " Max jobs:   ${MAX_JOBS}"
echo " Max epochs: ${MAX_EPOCHS}"
echo " Data dir:   ${DATA_DIR}"
echo " Results:    ${RESULTS_ROOT}"
echo "============================================================"
echo ""

TOTAL=$(( ${#FOLDS[@]} * ${#PCTS[@]} * ${#MODELS[@]} ))
COUNT=0

for FOLD in "${FOLDS[@]}"; do
  for PCT in "${PCTS[@]}"; do

    # ── Generate PKL first (sequential, fast) ────────────────────────────────
    generate_pkl "$FOLD" "$PCT"

    # ── Launch one training job per model (parallel, rate-limited) ───────────
    for MODEL in "${MODELS[@]}"; do
      COUNT=$(( COUNT + 1 ))
      echo "[${COUNT}/${TOTAL}] Queuing: fold=${FOLD} pct=${PCT} model=${MODEL}"
      wait_for_slot
      run_training "$FOLD" "$PCT" "$MODEL" &
    done

  done
done

# ── Wait for all remaining jobs to finish ─────────────────────────────────────
echo ""
echo "All jobs queued. Waiting for remaining ${MAX_JOBS} slots to finish..."
wait
echo ""
echo "============================================================"
echo " ALL DONE"
echo " Total jobs: ${TOTAL}"
echo " Results in: ${RESULTS_ROOT}/${DATASET}/unbalanced/"
echo "============================================================"