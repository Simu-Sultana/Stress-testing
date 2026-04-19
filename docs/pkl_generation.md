Notes about PKL generation

# =========================
# Commands for PKl genetaion from mimic_iii:
# =========================

python3 src/preprocess_mimic_iii_subsampled.py \
  --data_dir data/processed \
  --out_dir data/processed \
  --seed 2 \
  --pct 20

python3 src/preprocess_mimic_iii_sparsified-patientwise.py \
  --data_dir data/processed \
  --out_dir data/processed \
  --seed 0 \
  --pct 40

python3 src/preprocess_mimic_iii_sparsified-tsid-varid.py \
  --data_dir data/processed \
  --out_dir data/processed \
  --seed 2 \
  --pct 80

python3 src/preprocess_mimic_iii_unbalanced.py \
  --data_dir data/processed \
  --out_dir data/processed \
  --pct 80

# =========================
# Commands for PKl genetaion from physionet_2012:
# =========================

python3 src/preprocess_physionet_2012_subsampled.py \
  --data_dir data/processed \
  --out_dir data/processed \
  --seed 0 \
  --pct 20

python3 src/preprocess_physionet_2012_sparsified-patientwise.py \
  --data_dir data/processed \
  --out_dir data/processed \
  --seed 2 \
  --pct 40

python3 src/preprocess_physionet_2012_sparsified-tsid-varid.py \
  --data_dir data/processed \
  --out_dir data/processed \
  --seed 2 \
  --pct 80

python3 src/preprocess_physionet_2012_unbalanced.py \
  --data_dir data/processed \
  --out_dir data/processed \
  --pct 80


# =========================
# LOOP 1: MIMIC-III
# subsampled / sparsified-*  -> run for seed {0,2} and pct 10..90
# unbalanced                 -> run for pct 10..90 (NO seed)
# =========================

PCTS=(10 20 30 40 50 60 70 80 90)
SEEDS=(0 2)
PERTS_WITH_SEED=("subsampled" "sparsified-patientwise" "sparsified-tsid-varid")

for pert in "${PERTS_WITH_SEED[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for pct in "${PCTS[@]}"; do
      python3 "src/preprocess_mimic_iii_${pert}.py" \
        --data_dir data/processed \
        --out_dir data/processed \
        --seed "$seed" \
        --pct "$pct"
    done
  done
done

for pct in "${PCTS[@]}"; do
  python3 src/preprocess_mimic_iii_unbalanced.py \
    --data_dir data/processed \
    --out_dir data/processed \
    --pct "$pct"
done


# =========================
# LOOP 2: PHYSIONET-2012
# subsampled / sparsified-*  -> run for seed {0,2} and pct 10..90
# unbalanced                 -> run for pct 10..90 (NO seed)
# =========================


PCTS=(10 20 30 40 50 60 70 80 90)
SEEDS=(0 2)
PERTS_WITH_SEED=("subsampled" "sparsified-patientwise" "sparsified-tsid-varid")

for pert in "${PERTS_WITH_SEED[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for pct in "${PCTS[@]}"; do
      python3 "src/preprocess_physionet_2012_${pert}.py" \
        --data_dir data/processed \
        --out_dir data/processed \
        --seed "$seed" \
        --pct "$pct"
    done
  done
done

for pct in "${PCTS[@]}"; do
  python3 src/preprocess_physionet_2012_unbalanced.py \
    --data_dir data/processed \
    --out_dir data/processed \
    --pct "$pct"
done