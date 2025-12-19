#!/bin/bash

# Usage: ./run_experiments.sh physionet_2012 random

DATASET=$1          # first argument
PERTURB=$2          # second argument (type of perturbation)

for seed in 2; do
    for pct in 10 50 90; do

        # Preprocess data
        python preprocess_${DATASET}_${PERTURB}.py \
            --data_dir ../data/processed \
            --out_dir ../data/processed \
            --seed $seed \
            --pct $pct

        # Train model
        python main.py \
            --dataset $DATASET \
            --model_type gru \
            --hid_dim 43 \
            --dropout 0.2 \
            --lr 1e-4 \
            --file ${DATASET}_${PERTURB}_${pct}_${seed} \
            --train_frac 1 \
            --max_epochs 2 \
            --seed $seed

        # Delete preprocessed data to free space
        rm ../data/processed/${DATASET}_${PERTURB}_${pct}_${seed}*

    done
done
