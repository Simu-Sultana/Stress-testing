#!/usr/bin/env bash
#SBATCH --job-name=strats_unbalanced
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-27%20
#SBATCH --chdir=/home/hpc/iwbn/iwbn127h/projects/STRaTS_christel/strats-development/src
#SBATCH --output=/home/hpc/iwbn/iwbn127h/projects/STRaTS_christel/strats-development/src/logs/%x_%A_%a.out
#SBATCH --error=/home/hpc/iwbn/iwbn127h/projects/STRaTS_christel/strats-development/src/logs/%x_%A_%a.err

set -euo pipefail

mkdir -p logs
chmod +x run_unbalanced_training.sh

./run_unbalanced_training.sh 50
