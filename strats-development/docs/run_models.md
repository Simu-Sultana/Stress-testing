To Run all the models at once

# Run from repo root (main.py).

DATA_DIR="data/processed"
RESULTS_ROOT="results"
MAX_EPOCHS=50
TRAIN_FRAC=1.0

DATASETS=(physionet_2012 mimic_iii)
MODELS=(gru grud tcn sand strats)
PCTS=(10 20 30 40 50 60 70 80 90)
SEEDS=(0 2)
PERTS=(subsampled sparsified-patientwise sparsified-tsid-varid)

hparams () {
  local d="$1" m="$2"
  if [[ "$d" == physionet_2012 ]]; then
    case "$m" in
      gru)    echo "43 0.2 0.2 0.0001 2 4 4 24 12" ;;
      grud)   echo "49 0.2 0.2 0.0001 2 4 4 24 12" ;;
      tcn)    echo "64 0.1 0.2 0.0005 6 4 4 24 12" ;;
      sand)   echo "64 0.3 0.3 0.0005 4 2 4 24 12" ;;
      strats) echo "50 0.2 0.2 0.0005 2 4 4 24 2"  ;;
    esac
  else # mimic_iii
    case "$m" in
      gru)    echo "50 0.2 0.2 0.0001 2 4 4 24 12" ;;
      grud)   echo "60 0.2 0.2 0.0001 2 4 4 24 12" ;;
      tcn)    echo "128 0.1 0.2 0.0001 4 4 4 24 12" ;;
      sand)   echo "64 0.3 0.3 0.0005 4 2 4 24 12" ;;
      strats) echo "50 0.2 0.2 0.0005 2 4 4 24 2"  ;;
    esac
  fi
}

run_main () {
  local dataset="$1" target="$2" model="$3" pert="$4" file="$5"
  read -r HID DROPOUT ATTN LR NL NH KS R M <<<"$(hparams "$dataset" "$model")"
  RUN_DIR="${RESULTS_ROOT}/${dataset}/${target}/${model}/${pert}/${file}"
  mkdir -p "$RUN_DIR"
  python3 src/main.py \
    --dataset "$dataset" --target "$target" --model_type "$model" \
    --file "$file" --output_dir "$RUN_DIR" \
    --train_frac "$TRAIN_FRAC" --max_epochs "$MAX_EPOCHS" \
    --hid_dim "$HID" --dropout "$DROPOUT" --attention_dropout "$ATTN" \
    --lr "$LR" --num_layers "$NL" --num_heads "$NH" --kernel_size "$KS" --r "$R" --M "$M"
}

# ---- 3 perturbations (with seed) -> target length_of_stay ----
TARGET="length_of_stay"
for d in "${DATASETS[@]}"; do
  for p in "${PERTS[@]}"; do
    for m in "${MODELS[@]}"; do
      for s in "${SEEDS[@]}"; do
        for pct in "${PCTS[@]}"; do
          file="${d}_${p}_${pct}_${s}"
          run_main "$d" "$TARGET" "$m" "$p" "$file"
        done
      done
    done
  done
done

# ---- unbalanced (no seed) -> target unbalanced ----
TARGET="unbalanced"
p="unbalanced"
for d in "${DATASETS[@]}"; do
  for m in "${MODELS[@]}"; do
    for pct in "${PCTS[@]}"; do
      file="${d}_${p}_${pct}"
      run_main "$d" "$TARGET" "$m" "$p" "$file"
    done
  done
done