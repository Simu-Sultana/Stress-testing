# Now write the full content into the file

content = """# Reproducing All Experiments

This guide provides a complete, step-by-step pipeline to reproduce all experiments, including preprocessing, training, and evaluation.

---

## 1. Environment Setup

### Create environment

```bash
conda create -n strats python=3.10.9 -y
conda activate strats

Install dependencies
pip install -r requirement.txt
Verify installation
python -c "import torch, pandas, numpy; print('Environment OK')"
2. Dataset Setup

Place the base dataset files inside:

data/processed/
├── mimic_iii.pkl
├── physionet_2012.pkl
Important
These are the base datasets
All generated files will also be stored here
3. Generate Perturbed Data
Settings
Percentages: 10–90
Seeds: 0, 2
Perturbations:
subsampled
sparsified-tsid-varid
unbalanced
Subsampled + Sparsified
PCTS=(10 20 30 40 50 60 70 80 90)
SEEDS=(0 2)
PERTS=(subsampled sparsified-tsid-varid)

for p in "${PERTS[@]}"; do
  for s in "${SEEDS[@]}"; do
    for pct in "${PCTS[@]}"; do
      python3 src/preprocess_mimic_iii_${p}.py \
        --data_dir data/processed \
        --out_dir data/processed \
        --seed $s \
        --pct $pct

      python3 src/preprocess_physionet_2012_${p}.py \
        --data_dir data/processed \
        --out_dir data/processed \
        --seed $s \
        --pct $pct
    done
  done
done
Unbalanced (no seed)
for pct in "${PCTS[@]}"; do
  python3 src/preprocess_mimic_iii_unbalanced.py \
    --data_dir data/processed \
    --out_dir data/processed \
    --pct $pct

  python3 src/preprocess_physionet_2012_unbalanced.py \
    --data_dir data/processed \
    --out_dir data/processed \
    --pct $pct
done
4. Training
bash run_all_models.sh
This will
train all models
run all perturbations
save outputs in results/
5. Results Structure
results/<dataset>/<target>/<model>/<perturbation>/<run_name>/

Each run contains:

CSV metrics
logs
checkpoint (best.pt)
6. Plot Generation
python3 src/plot_metrics.py

Outputs will be stored in:

plots_per_metric/
7. HPC Execution (Optional)
conda activate strats
srun --exclusive -N1 -n1 --gres=gpu:1 bash -lc "bash run_perturbations_training.sh"
8. Reproducibility Notes
Seeds: 0, 2
Percentages: 10–90
Unbalanced has no seed
Ensure correct file naming
Ensure all .pkl files exist before training
9. Common Issues
Missing files → run preprocessing
Wrong naming → check format
GRU-D fails → data too sparse
10. Workflow Summary
Setup environment
Add datasets
Generate perturbed data
Train models
Generate plots