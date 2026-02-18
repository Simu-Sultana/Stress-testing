#!/usr/bin/env bash
set -euo pipefail

# This script runs PhysioNet 2012:
# - IHM experiments in parallel across multiple GPUs
# - then Unbalanced once (it trains all models internally)
#
# It calls the BIG script: ./run_perturbations_training.sh
# (Your big script is the "runner" that knows targets/perturbations)

DATASET="physionet_2012"
MAX_EPOCHS="${MAX_EPOCHS:-50}"

# Must match SBATCH: --gres=gpu:<NGPUS>
NGPUS="${NGPUS:-4}"

SEEDS=(0 2)
PERTS=(subsampled sparsified-patientwise sparsified-tsid-varid)
MODELS=(gru grud tcn sand strats)

wait_for_slot () {
  while true; do
    local n
    n="$(jobs -rp | wc -l | tr -d ' ')"
    (( n < NGPUS )) && break
    sleep 2
  done
}

GPU_NEXT=0
next_gpu () {
  local g="$GPU_NEXT"
  GPU_NEXT=$(( (GPU_NEXT + 1) % NGPUS ))
  echo "$g"
}

launch () {
  local gpu="$1"; shift
  echo "▶ Launch on GPU $gpu: $*"
  CUDA_VISIBLE_DEVICES="$gpu" "$@" &
}

# -----------------------------
# 1) IHM in parallel
# -----------------------------
TARGET="in_hospital_mortality"
for SEED in "${SEEDS[@]}"; do
  for PERT in "${PERTS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
      wait_for_slot
      GPU="$(next_gpu)"
      launch "$GPU" ./run_perturbations_training.sh "$DATASET" "$TARGET" "$PERT" "$MODEL" "$SEED" "$MAX_EPOCHS"
    done
  done
done

# Wait all IHM jobs
wait
echo "✅ IHM finished."

# -----------------------------
# 2) Unbalanced (no seed)
# Your big script in unbalanced mode trains ALL models internally
# -----------------------------
CUDA_VISIBLE_DEVICES=0 ./run_perturbations_training.sh "$DATASET" unbalanced unbalanced gru "$MAX_EPOCHS"

echo "✅ Finished: PhysioNet 2012 (IHM + Unbalanced)."

