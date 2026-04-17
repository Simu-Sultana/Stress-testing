#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Usage:
#   1) Run one combo:
#      ./run_experiments.sh <dataset> <perturbation> <model> [seed] [max_epochs]
#
#   2) Run EVERYTHING:
#      ./run_experiments.sh all [seed] [max_epochs]
#
# Examples:
#   ./run_experiments.sh physionet_2012 subsampled gru
#   ./run_experiments.sh mimic_iii sparsified-tsid-varid tcn 2 50
#   ./run_experiments.sh all
#   ./run_experiments.sh all 0 50
# -----------------------------

MODE="${1:-}"

# Default single seed + epochs
DEFAULT_SEED=0
DEFAULT_MAX_EPOCHS=50

# Percentages to loop over (your PKLs must exist for these)
PCTS=(1 2 5 10 20 30 40 50 60 70 80 90 100)

# Helper: run a single experiment combo
run_one () {
  local DATASET="$1"
  local PERTURB="$2"
  local MODEL="$3"
  local SEED="${4:-$DEFAULT_SEED}"
  local MAX_EPOCHS="${5:-$DEFAULT_MAX_EPOCHS}"

  case "$PERTURB" in
    subsampled|sparsified-patientwise|sparsified-tsid-varid) ;;
    *) echo "ERROR: invalid perturbation '$PERTURB'"; exit 1 ;;
  esac

  case "$MODEL" in
    gru|grud|tcn|sand|strats) ;;
    *) echo "ERROR: invalid model '$MODEL'"; exit 1 ;;
  esac

  # ------------------------------------------------------------
  # Hyperparameters from Table 3 (paper) for the 5 models:
  # GRU, GRU-D, TCN, SaND, STraTS on MIMIC-III and PhysioNet-2012
  # ------------------------------------------------------------
  local DATASET_KEY="$DATASET"
  if [[ "$DATASET_KEY" != "physionet_2012" && "$DATASET_KEY" != "mimic_iii" ]]; then
    echo "⚠️ Dataset '$DATASET' not in paper Table 3. Falling back to PhysioNet-2012 hyperparameters."
    DATASET_KEY="physionet_2012"
  fi

  # Defaults (match your argparse names)
  local HID_DIM=32
  local DROPOUT=0.2
  local ATTN_DROPOUT=0.2
  local LR="1e-4"
  local NUM_LAYERS=2
  local NUM_HEADS=4
  local KERNEL_SIZE=4
  local R=24
  local M=12
  local HE=8

  if [[ "$DATASET_KEY" == "physionet_2012" ]]; then
    case "$MODEL" in
      gru)    HID_DIM=43; DROPOUT=0.2; LR="0.0001" ;;
      grud)   HID_DIM=49; DROPOUT=0.2; LR="0.0001" ;;
      tcn)    NUM_LAYERS=6; HID_DIM=64; KERNEL_SIZE=4; DROPOUT=0.1; LR="0.0005" ;;
      sand)   NUM_LAYERS=4; R=24; M=12; HID_DIM=64; NUM_HEADS=2; HE=8; DROPOUT=0.3; ATTN_DROPOUT=0.3; LR="0.0005" ;;
      strats) HID_DIM=50; M=2; NUM_HEADS=4; DROPOUT=0.2; ATTN_DROPOUT=0.2; LR="0.0005" ;;
    esac
  else
    # mimic_iii
    case "$MODEL" in
      gru)    HID_DIM=50; DROPOUT=0.2; LR="0.0001" ;;
      grud)   HID_DIM=60; DROPOUT=0.2; LR="0.0001" ;;
      tcn)    NUM_LAYERS=4; HID_DIM=128; KERNEL_SIZE=4; DROPOUT=0.1; LR="0.0001" ;;
      sand)   NUM_LAYERS=4; R=24; M=12; HID_DIM=64; NUM_HEADS=2; HE=8; DROPOUT=0.3; ATTN_DROPOUT=0.3; LR="0.0005" ;;
      strats) HID_DIM=50; M=2; NUM_HEADS=4; DROPOUT=0.2; ATTN_DROPOUT=0.2; LR="0.0005" ;;
    esac
  fi

  echo "=============================="
  echo "DATASET=$DATASET  PERTURB=$PERTURB  MODEL=$MODEL  SEED=$SEED  max_epochs=$MAX_EPOCHS"
  echo "hid_dim=$HID_DIM dropout=$DROPOUT attn_dropout=$ATTN_DROPOUT lr=$LR num_layers=$NUM_LAYERS num_heads=$NUM_HEADS kernel_size=$KERNEL_SIZE r=$R M=$M he=$HE"
  echo "=============================="

  for pct in "${PCTS[@]}"; do
    echo "---- pct=$pct ----"

    # ------------------------------------------------------------
    # PREPROCESS STEP (COMMENTED OUT because you already have PKLs)
    # ------------------------------------------------------------
    PREPROCESS_SCRIPT="preprocess_${DATASET}_${PERTURB}.py"
    if [[ ! -f "$PREPROCESS_SCRIPT" ]]; then
      echo "ERROR: preprocess script not found: $PREPROCESS_SCRIPT"
      exit 1
    fi
    python "$PREPROCESS_SCRIPT" \
       --data_dir ../data/processed \
       --out_dir ../data/processed \
       --seed "$SEED" \
       --pct "$pct"

    # Train (your code loads ./data/processed/<file>.pkl)
    python main.py \
      --dataset "$DATASET" \
      --target "in_hospital_mortality" \
      --file "${DATASET}_${PERTURB}_${pct}_${SEED}" \
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

    # ------------------------------------------------------------
    # CLEANUP STEP (COMMENTED OUT for now)
    # ------------------------------------------------------------
    rm -f ../data/processed/"${DATASET}_${PERTURB}_${pct}_${SEED}"*
  done
}

# -----------------------------
# Main entry
# -----------------------------
if [[ -z "$MODE" ]]; then
  echo "Usage:"
  echo "  $0 <dataset> <perturbation> <model> [seed] [max_epochs]"
  echo "  $0 all [seed] [max_epochs]"
  exit 1
fi

if [[ "$MODE" == "all" ]]; then
  SEED="${2:-$DEFAULT_SEED}"
  MAX_EPOCHS="${3:-$DEFAULT_MAX_EPOCHS}"

  for d in physionet_2012 mimic_iii; do
    for p in subsampled sparsified-patientwise sparsified-tsid-varid; do
      for m in gru grud tcn sand strats; do
        run_one "$d" "$p" "$m" "$SEED" "$MAX_EPOCHS"
      done
    done
  done
else
  DATASET="$MODE"
  PERTURB="${2:-}"
  MODEL="${3:-}"
  SEED="${4:-$DEFAULT_SEED}"
  MAX_EPOCHS="${5:-$DEFAULT_MAX_EPOCHS}"

  if [[ -z "$PERTURB" || -z "$MODEL" ]]; then
    echo "Usage: $0 <dataset> <perturbation> <model> [seed] [max_epochs]"
    exit 1
  fi

  run_one "$DATASET" "$PERTURB" "$MODEL" "$SEED" "$MAX_EPOCHS"
fi
