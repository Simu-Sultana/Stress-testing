#!/usr/bin/env bash
#SBATCH --job-name=mimic_grud_tsid_varid
#SBATCH --output=/home/hpc/iwbn/iwbn127h/projects/STRaTS_christel/strats-development/logs/%x_%A_%a.out
#SBATCH --error=/home/hpc/iwbn/iwbn127h/projects/STRaTS_christel/strats-development/logs/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=work
#SBATCH --array=0-111%20

set -euxo pipefail
export PYTHONUNBUFFERED=1

cd /home/hpc/iwbn/iwbn127h/projects/STRaTS_christel/strats-development

source ~/miniconda3/etc/profile.d/conda.sh
conda activate strats

echo "Conda activated"
echo "PWD=$(pwd)"

COMBOS=(
  "1 0" "1 1" "1 2" "1 3" "1 4" "1 5" "1 6" "1 7" "1 8" "1 9"
  "2 0" "2 1" "2 2" "2 3" "2 4" "2 5" "2 6" "2 7" "2 8" "2 9"
  "5 0" "5 1" "5 2" "5 3" "5 4" "5 5" "5 6" "5 7" "5 8" "5 9"
  "10 1" "10 3" "10 4" "10 5" "10 6" "10 7" "10 8" "10 9"
  "20 1" "20 3" "20 4" "20 5" "20 6" "20 7" "20 8" "20 9"
  "30 1" "30 3" "30 4" "30 5" "30 6" "30 7" "30 8" "30 9"
  "40 1" "40 3" "40 4" "40 5" "40 6" "40 7" "40 8" "40 9"
  "50 1" "50 3" "50 4" "50 5" "50 6" "50 7" "50 8" "50 9"
  "60 1" "60 3" "60 4" "60 5" "60 6" "60 7" "60 8" "60 9"
  "70 1" "70 3" "70 4" "70 5" "70 6" "70 7" "70 8" "70 9"
  "80 1" "80 3" "80 4" "80 5" "80 6" "80 7" "80 8" "80 9"
  "90 1" "90 3" "90 4" "90 5" "90 6" "90 7" "90 8" "90 9"
  "100 0" "100 1" "100 2" "100 3" "100 4" "100 5" "100 6" "100 7" "100 8" "100 9"
)

combo="${COMBOS[$SLURM_ARRAY_TASK_ID]}"
set -- $combo
pct=$1
seed=$2

PKL="data/processed/mimic_iii_sparsified-tsid-varid_${pct}_${seed}.pkl"

echo "Running pct=$pct seed=$seed"
echo "PKL: $PKL"

python3 -u src/preprocess_mimic_iii_sparsified-tsid-varid.py \
  --data_dir data/processed \
  --out_dir data/processed \
  --seed "$seed" \
  --pct "$pct"

python3 -u src/main.py \
  --dataset mimic_iii \
  --target in_hospital_mortality \
  --file mimic_iii_sparsified-tsid-varid_${pct}_${seed} \
  --model_type grud \
  --hid_dim 60 \
  --dropout 0.2 \
  --attention_dropout 0.2 \
  --lr 0.0001 \
  --num_layers 2 \
  --num_heads 4 \
  --kernel_size 4 \
  --r 24 \
  --M 12 \
  --train_frac 1 \
  --max_epochs 50 \
  --seed "$seed"

rm -f "$PKL"
echo "Deleted $PKL"