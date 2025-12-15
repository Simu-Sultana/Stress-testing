#!/bin/bash

# Usage: ./run_experiments.sh physionet_2012 random

DATASET=$1          # first argument

for pct in {20..60..20}; do

    # Preprocess data
    python preprocess_${DATASET}_unbalanced.py \
        --data_dir ../data/processed \
        --out_dir ../data/processed \
        --pct $pct

    # Train model
    python main.py \
        --dataset $DATASET \
        --target unbalanced \
        --model_type gru \
        --hid_dim 64 \
        --dropout 0.2 \
        --lr 5e-4 \
        --file ${DATASET}_unbalanced_${pct} \
        --train_frac 1 \
        --max_epochs 2

    # Delete preprocessed data to free space
    rm ../data/processed/${DATASET}_unbalanced_${pct}*

done

